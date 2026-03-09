"""Utilities for loading LuxTTS models and running audio processing and generation."""

# start zipvoice/modeling_utils.py
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import orjson
import stamina
import structlog
import torch
from huggingface_hub import snapshot_download
from linacodec.vocoder.vocos import Vocos
from torch.nn.utils import parametrize
from transformers import pipeline

from zipvoice.constants import (
    DEFAULT_FEAT_SCALE,
    DEFAULT_SPEED_FACTOR,
    SAMPLE_RATE_INPUT,
    SAMPLE_RATE_OUTPUT,
)
from zipvoice.exceptions import ModelLoadError
from zipvoice.generation_types import GenerationConfig, PromptContext
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.onnx_modeling import OnnxModel
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import rms_norm

log = structlog.get_logger()


@dataclass
class LuxTTSConfig:
    """Low-level configuration for model file paths and tokenizer selection.

    Attributes:
        model_dir: Path to a local model directory. When ``None``, the model
            is downloaded from HuggingFace Hub.
        checkpoint_name: Filename of the PyTorch model checkpoint inside
            ``model_dir``.
        vocoder_path: Optional override path for the Vocos vocoder weights.
        trt_engine_path: Optional path to a TensorRT engine file for
            accelerated inference.
        tokenizer: Tokenizer variant to use. Choices: ``"emilia"``,
            ``"libritts"``, ``"espeak"``, ``"simple"``.
        lang: BCP-47 language code passed to the tokenizer.
    """

    # Model Setup
    model_dir: str | None = None
    checkpoint_name: str = "model.pt"
    vocoder_path: str | None = None
    trt_engine_path: str | None = None

    # Tokenizer & Language
    tokenizer: str = "emilia"  # choices: ["emilia", "libritts", "espeak", "simple"]
    lang: str = "en-us"


@stamina.retry(on=Exception, attempts=3)
def _download_model(repo_id: str) -> str:
    """Download model from HuggingFace Hub with retry on failure.

    Args:
        repo_id: HuggingFace repository ID.

    Returns:
        Local path to downloaded model directory.

    Raises:
        ModelLoadError: If download fails after 3 attempts.
    """
    try:
        return snapshot_download(repo_id)
    except Exception as exc:
        msg = f"Failed to download model '{repo_id}' after 3 attempts"
        raise ModelLoadError(msg) from exc


def _remove_vocos_parametrizations(vocos: Any) -> None:
    """Remove parametrizations from Vocos model in-place."""
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[0], "weight")
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[1], "weight")


def load_models(  # noqa: PLR0913
    model_path: str | None = None, device: str = "cuda", num_thread: int = 2
) -> tuple[Any, VocosFbank, Any, EmiliaTokenizer, Any]:
    """Load LuxTTS models for inference on any device.

    Args:
        model_path: Local path to model directory. Downloads from HuggingFace if None.
        device: Target device ('cuda', 'mps', 'cpu').
        num_thread: Number of threads for ONNX inference (CPU only).

    Returns:
        Tuple of (model, feature_extractor, vocos, tokenizer, transcriber).

    Raises:
        ModelLoadError: If model loading or download fails.
    """
    if not model_path:
        model_path = _download_model("YatharthS/LuxTTS")
    try:
        token_file = f"{model_path}/tokens.txt"
        model_config_path = f"{model_path}/config.json"

        tokenizer = EmiliaTokenizer(token_file=token_file)
        tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

        with open(model_config_path, "rb") as f:
            model_config = orjson.loads(f.read())

        model: Any
        if device == "cpu":
            text_encoder_path = f"{model_path}/text_encoder.onnx"
            fm_decoder_path = f"{model_path}/fm_decoder.onnx"
            transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device="cpu")
            model = OnnxModel(text_encoder_path, fm_decoder_path, num_thread=num_thread)
        else:
            model_ckpt = f"{model_path}/model.pt"
            transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)
            model = ZipVoiceDistill(  # type: ignore[no-untyped-call]
                **model_config["model"],
                **tokenizer_config,
            )
            load_checkpoint(filename=Path(model_ckpt), model=model, strict=True)
            torch_device = torch.device(device, 0)
            model = model.to(torch_device).eval()

        vocos = Vocos.from_hparams(f"{model_path}/vocoder/config.yaml")
        vocos = vocos.to(device) if device != "cpu" else vocos.eval()

        _remove_vocos_parametrizations(vocos)

        map_device = torch.device(device, 0) if device != "cpu" else torch.device("cpu")
        vocos.load_state_dict(torch.load(f"{model_path}/vocoder/vocos.bin", map_location=map_device))

        feature_extractor = VocosFbank()

    except ModelLoadError:
        raise
    except Exception as exc:
        msg = f"Failed to load model from {model_path}"
        raise ModelLoadError(msg) from exc
    else:
        return model, feature_extractor, vocos, tokenizer, transcriber


@torch.inference_mode()
def process_audio(  # noqa: PLR0913
    audio: str,
    transcriber: Any,
    tokenizer: Any,
    feature_extractor: Any,
    device: str,
    target_rms: float = 0.1,
    duration: int = 4,
    feat_scale: float = DEFAULT_FEAT_SCALE,
) -> PromptContext:
    """Process an audio file into a PromptContext for generation.

    Args:
        audio: Path to audio file.
        transcriber: ASR pipeline for transcribing the prompt.
        tokenizer: Tokenizer for encoding the transcription.
        feature_extractor: Feature extractor for computing filterbank features.
        device: Target device for tensors.
        target_rms: Target RMS level for volume normalisation.
        duration: Maximum duration in seconds to load from the audio file.
        feat_scale: Scale factor applied to extracted features.

    Returns:
        PromptContext with encoded tokens, features, and RMS value.
    """
    prompt_wav_arr, _ = librosa.load(audio, sr=SAMPLE_RATE_OUTPUT, duration=duration)
    prompt_wav2, _ = librosa.load(audio, sr=SAMPLE_RATE_INPUT, duration=duration)
    prompt_text = transcriber(prompt_wav2)["text"]
    log.debug("prompt_transcribed", prompt_text=prompt_text)

    prompt_wav = torch.from_numpy(prompt_wav_arr).unsqueeze(0)
    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)

    prompt_features = feature_extractor.extract(prompt_wav, sampling_rate=SAMPLE_RATE_OUTPUT).to(device)
    prompt_features = prompt_features.unsqueeze(0) * feat_scale
    prompt_features_lens = torch.tensor([prompt_features.size(1)], device=device)
    prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])
    return PromptContext(
        tokens=prompt_tokens,
        features_lens=prompt_features_lens,
        features=prompt_features,
        rms=prompt_rms,
    )


def generate(  # noqa: PLR0913
    prompt: PromptContext,
    text: str,
    model: Any,
    vocoder: Any,
    tokenizer: Any,
    config: GenerationConfig | None = None,
) -> torch.Tensor:
    """Generate speech for the given text using a voice prompt.

    Args:
        prompt: Encoded voice prompt produced by process_audio().
        text: Text to synthesise.
        model: ZipVoice flow-matching model.
        vocoder: Vocos vocoder for converting features to waveform.
        tokenizer: Tokenizer for encoding the input text.
        config: Generation hyperparameters; defaults to GenerationConfig().

    Returns:
        Generated waveform tensor of shape (1, samples).
    """
    if config is None:
        config = GenerationConfig()
    tokens = tokenizer.texts_to_token_ids([text])
    speed = config.speed * DEFAULT_SPEED_FACTOR

    with torch.inference_mode():
        (pred_features, _, _, _) = model.sample(
            tokens=tokens,
            prompt_tokens=prompt.tokens,
            prompt_features=prompt.features,
            prompt_features_lens=prompt.features_lens,
            speed=speed,
            t_shift=config.t_shift,
            duration="predict",
            num_step=config.num_step,
            guidance_scale=config.guidance_scale,
        )

    # Convert to waveform
    pred_features = pred_features.permute(0, 2, 1) / DEFAULT_FEAT_SCALE
    wav: torch.Tensor = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)

    # Volume matching
    if prompt.rms < config.target_rms:
        wav = wav * (prompt.rms / config.target_rms)

    return wav


# end zipvoice/modeling_utils.py
