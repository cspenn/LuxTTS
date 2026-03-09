# start zipvoice/bin/infer_zipvoice.py
#!/usr/bin/env python3
# Copyright         2025  Xiaomi Corp.        (authors: Han Zhu)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Generate speech with pre-trained ZipVoice or ZipVoice-Distill models.

If no local model is specified, required files will be automatically
downloaded from HuggingFace.

Usage:

Note: If you having trouble connecting to HuggingFace,
    try switching endpoint to mirror site:
export HF_ENDPOINT=https://hf-mirror.com

(1) Inference of a single sentence:

python3 -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --prompt-wav prompt.wav \
    --prompt-text "I am a prompt." \
    --text "I am a sentence." \
    --res-wav-path result.wav

(2) Inference of a list of sentences:

python3 -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --test-list test.tsv \
    --res-dir results

`--model-name` can be `zipvoice` or `zipvoice_distill`,
    which are the models before and after distillation, respectively.

Each line of `test.tsv` is in the format of
    `{wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}`.


(3) Inference with TensorRT:

python3 -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice_distill \
    --prompt-wav prompt.wav \
    --prompt-text "I am a prompt." \
    --text "I am a sentence." \
    --res-wav-path result.wav \
    --trt-engine-path models/zipvoice_distill_onnx_trt/fm_decoder.fp16.plan
"""

import datetime as dt
import os
from pathlib import Path

import numpy as np
import orjson
import safetensors.torch
import structlog
import torch
import torchaudio
import typer
from huggingface_hub import hf_hub_download
from lhotse.utils import fix_random_seed
from vocos import Vocos

from zipvoice.constants import HUGGINGFACE_REPO_ZIPVOICE
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import (
    add_punctuation,
    batchify_tokens,
    chunk_tokens_punctuation,
    load_prompt_wav,
    merge_chunked_wavs,
    remove_silence,
    rms_norm,
)
from zipvoice.utils.tensorrt import load_trt

log = structlog.get_logger()

HUGGINGFACE_REPO = HUGGINGFACE_REPO_ZIPVOICE
MODEL_DIR = {
    "zipvoice": "zipvoice",
    "zipvoice_distill": "zipvoice_distill",
}

app = typer.Typer(
    help="Generate speech with pre-trained ZipVoice or ZipVoice-Distill models.",
    add_completion=False,
)


def get_vocoder(vocos_local_path: str | None = None):
    """Load a Vocos vocoder from a local path or HuggingFace.

    Args:
        vocos_local_path: Optional path to local vocoder checkpoint directory.

    Returns:
        A Vocos vocoder model instance.
    """
    if vocos_local_path:
        vocoder = Vocos.from_hparams(f"{vocos_local_path}/config.yaml")
        state_dict = torch.load(
            f"{vocos_local_path}/pytorch_model.bin",
            weights_only=True,
            map_location="cpu",
        )
        vocoder.load_state_dict(state_dict)
    else:
        vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    return vocoder


def generate_sentence_raw_evaluation(  # noqa: PLR0913
    save_path: str,
    prompt_text: str,
    prompt_wav: str,
    text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: EmiliaTokenizer,
    feature_extractor: VocosFbank,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
):
    """Generate waveform of a text based on a given prompt waveform and its transcription.

    This function directly feed the prompt_text, prompt_wav and text to the model.
    It is not efficient and can have poor results for some inappropriate inputs.
    (e.g., prompt wav contains long silence, text to be generated is too long)
    This function can be used to evaluate the "raw" performance of the model.

    Args:
        save_path (str): Path to save the generated wav.
        prompt_text (str): Transcription of the prompt wav.
        prompt_wav (str): Path to the prompt wav file.
        text (str): Text to be synthesized into a waveform.
        model (torch.nn.Module): The model used for generation.
        vocoder (torch.nn.Module): The vocoder used to convert features to waveforms.
        tokenizer (EmiliaTokenizer): The tokenizer used to convert text to tokens.
        feature_extractor (VocosFbank): The feature extractor used to
            extract acoustic features.
        device (torch.device): The device on which computations are performed.
        num_step (int, optional): Number of steps for decoding. Defaults to 16.
        guidance_scale (float, optional): Scale for classifier-free guidance.
            Defaults to 1.0.
        speed (float, optional): Speed control. Defaults to 1.0.
        t_shift (float, optional): Time shift. Defaults to 0.5.
        target_rms (float, optional): Target RMS for waveform normalization.
            Defaults to 0.1.
        feat_scale (float, optional): Scale for features.
            Defaults to 0.1.
        sampling_rate (int, optional): Sampling rate for the waveform.
            Defaults to 24000.

    Returns:
        metrics (dict): Dictionary containing time and real-time
            factor metrics for processing.
    """
    # Load and process prompt wav
    prompt_wav = load_prompt_wav(prompt_wav, sampling_rate=sampling_rate)
    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)

    # Extract features from prompt wav
    prompt_features = feature_extractor.extract(prompt_wav, sampling_rate=sampling_rate).to(device)

    prompt_features = prompt_features.unsqueeze(0) * feat_scale
    prompt_features_lens = torch.tensor([prompt_features.size(1)], device=device)

    # Convert text to tokens
    tokens = tokenizer.texts_to_token_ids([text])
    prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])

    # Start timing
    start_t = dt.datetime.now()

    # Generate features
    (
        pred_features,
        pred_features_lens,
        pred_prompt_features,
        pred_prompt_features_lens,
    ) = model.sample(
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        prompt_features=prompt_features,
        prompt_features_lens=prompt_features_lens,
        speed=speed,
        t_shift=t_shift,
        duration="predict",
        num_step=num_step,
        guidance_scale=guidance_scale,
    )

    # Postprocess predicted features
    pred_features = pred_features.permute(0, 2, 1) / feat_scale  # (B, C, T)

    # Start vocoder processing
    start_vocoder_t = dt.datetime.now()
    wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)

    # Calculate processing times and real-time factors
    t = (dt.datetime.now() - start_t).total_seconds()
    t_no_vocoder = (start_vocoder_t - start_t).total_seconds()
    t_vocoder = (dt.datetime.now() - start_vocoder_t).total_seconds()
    wav_seconds = wav.shape[-1] / sampling_rate
    rtf = t / wav_seconds
    rtf_no_vocoder = t_no_vocoder / wav_seconds
    rtf_vocoder = t_vocoder / wav_seconds
    metrics = {
        "t": t,
        "t_no_vocoder": t_no_vocoder,
        "t_vocoder": t_vocoder,
        "wav_seconds": wav_seconds,
        "rtf": rtf,
        "rtf_no_vocoder": rtf_no_vocoder,
        "rtf_vocoder": rtf_vocoder,
    }

    # Adjust wav volume if necessary
    if prompt_rms < target_rms:
        wav = wav * prompt_rms / target_rms
    torchaudio.save(save_path, wav.cpu(), sample_rate=sampling_rate)

    return metrics


def generate_sentence(  # noqa: PLR0913
    save_path: str,
    prompt_text: str,
    prompt_wav: str,
    text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: EmiliaTokenizer,
    feature_extractor: VocosFbank,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    max_duration: float = 100,
    remove_long_sil: bool = False,
):
    """Generate waveform of a text based on a given prompt waveform and its transcription.

    This function will do the following to improve the generation quality:
    1. chunk the text according to punctuations.
    2. process chunked texts in batches.
    3. remove long silences in the prompt audio.
    4. add punctuation to the end of prompt text and text if there is not.

    Args:
        save_path (str): Path to save the generated wav.
        prompt_text (str): Transcription of the prompt wav.
        prompt_wav (str): Path to the prompt wav file.
        text (str): Text to be synthesized into a waveform.
        model (torch.nn.Module): The model used for generation.
        vocoder (torch.nn.Module): The vocoder used to convert features to waveforms.
        tokenizer (EmiliaTokenizer): The tokenizer used to convert text to tokens.
        feature_extractor (VocosFbank): The feature extractor used to
            extract acoustic features.
        device (torch.device): The device on which computations are performed.
        num_step (int, optional): Number of steps for decoding. Defaults to 16.
        guidance_scale (float, optional): Scale for classifier-free guidance.
            Defaults to 1.0.
        speed (float, optional): Speed control. Defaults to 1.0.
        t_shift (float, optional): Time shift. Defaults to 0.5.
        target_rms (float, optional): Target RMS for waveform normalization.
            Defaults to 0.1.
        feat_scale (float, optional): Scale for features.
            Defaults to 0.1.
        sampling_rate (int, optional): Sampling rate for the waveform.
            Defaults to 24000.
        max_duration (float, optional): The maximum duration to process in each
            batch. Used to control memory consumption when generating long audios.
        remove_long_sil (bool, optional): Whether to remove long silences in the
            middle of the generated speech (edge silences will be removed by default).

    Returns:
        metrics (dict): Dictionary containing time and real-time
            factor metrics for processing.
    """
    # Load and process prompt wav
    prompt_wav = load_prompt_wav(prompt_wav, sampling_rate=sampling_rate)

    # Remove edge and long silences in the prompt wav.
    # Add 0.2s trailing silence to avoid leaking prompt to generated speech.
    prompt_wav = remove_silence(prompt_wav, sampling_rate, only_edge=False, trail_sil=200)

    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)

    prompt_duration = prompt_wav.shape[-1] / sampling_rate

    if prompt_duration > 20:
        log.warning(
            "prompt_wav_too_long",
            prompt_duration=prompt_duration,
        )
    elif prompt_duration > 10:
        log.warning(
            "prompt_wav_long",
            prompt_duration=prompt_duration,
        )

    # Extract features from prompt wav
    prompt_features = feature_extractor.extract(prompt_wav, sampling_rate=sampling_rate).to(device)

    prompt_features = prompt_features.unsqueeze(0) * feat_scale

    # Add punctuation in the end if there is not
    text = add_punctuation(text)
    prompt_text = add_punctuation(prompt_text)

    # Tokenize text (str tokens), punctuations will be preserved.
    tokens_str = tokenizer.texts_to_tokens([text])[0]
    prompt_tokens_str = tokenizer.texts_to_tokens([prompt_text])[0]

    # chunk text so that each len(prompt wav + generated wav) is around 25 seconds.
    token_duration = (prompt_wav.shape[-1] / sampling_rate) / (len(prompt_tokens_str) * speed)
    max_tokens = int((25 - prompt_duration) / token_duration)
    chunked_tokens_str = chunk_tokens_punctuation(tokens_str, max_tokens=max_tokens)

    # Tokenize text (int tokens)
    chunked_tokens = tokenizer.tokens_to_token_ids(chunked_tokens_str)
    prompt_tokens = tokenizer.tokens_to_token_ids([prompt_tokens_str])

    # Batchify chunked texts for faster processing
    tokens_batches, chunked_index = batchify_tokens(chunked_tokens, max_duration, prompt_duration, token_duration)

    # Start predicting features
    chunked_features = []
    start_t = dt.datetime.now()

    for batch_tokens in tokens_batches:
        batch_prompt_tokens = prompt_tokens * len(batch_tokens)

        batch_prompt_features = prompt_features.repeat(len(batch_tokens), 1, 1)
        batch_prompt_features_lens = torch.full((len(batch_tokens),), prompt_features.size(1), device=device)

        # Generate features
        (
            pred_features,
            pred_features_lens,
            pred_prompt_features,
            pred_prompt_features_lens,
        ) = model.sample(
            tokens=batch_tokens,
            prompt_tokens=batch_prompt_tokens,
            prompt_features=batch_prompt_features,
            prompt_features_lens=batch_prompt_features_lens,
            speed=speed,
            t_shift=t_shift,
            duration="predict",
            num_step=num_step,
            guidance_scale=guidance_scale,
        )

        # Postprocess predicted features
        pred_features = pred_features.permute(0, 2, 1) / feat_scale  # (B, C, T)
        chunked_features.append((pred_features, pred_features_lens))

    # Start vocoder processing
    chunked_wavs = []
    start_vocoder_t = dt.datetime.now()

    for pred_features, pred_features_lens in chunked_features:
        batch_wav = []
        for i in range(pred_features.size(0)):
            wav = vocoder.decode(pred_features[i][None, :, : pred_features_lens[i]]).squeeze(1).clamp(-1, 1)
            # Adjust wav volume if necessary
            if prompt_rms < target_rms:
                wav = wav * prompt_rms / target_rms
            batch_wav.append(wav)
        chunked_wavs.extend(batch_wav)

    # Finish model generation
    t = (dt.datetime.now() - start_t).total_seconds()

    # Merge chunked wavs
    final_wav = merge_chunked_wavs(chunked_wavs, chunked_index, remove_long_sil, sampling_rate)

    # Calculate processing time metrics
    t_no_vocoder = (start_vocoder_t - start_t).total_seconds()
    t_vocoder = (dt.datetime.now() - start_vocoder_t).total_seconds()
    wav_seconds = final_wav.shape[-1] / sampling_rate
    rtf = t / wav_seconds
    rtf_no_vocoder = t_no_vocoder / wav_seconds
    rtf_vocoder = t_vocoder / wav_seconds
    metrics = {
        "t": t,
        "t_no_vocoder": t_no_vocoder,
        "t_vocoder": t_vocoder,
        "wav_seconds": wav_seconds,
        "rtf": rtf,
        "rtf_no_vocoder": rtf_no_vocoder,
        "rtf_vocoder": rtf_vocoder,
    }

    torchaudio.save(save_path, final_wav.cpu(), sample_rate=sampling_rate)
    return metrics


def generate_list(  # noqa: PLR0913
    res_dir: str,
    test_list: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: EmiliaTokenizer,
    feature_extractor: VocosFbank,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    raw_evaluation: bool = False,
    max_duration: float = 100,
    remove_long_sil: bool = False,
):
    """Generate speech for a list of test samples and report metrics.

    Args:
        res_dir: Directory to save generated wavs.
        test_list: Path to TSV file with test samples.
        model: The model for generation.
        vocoder: The vocoder model.
        tokenizer: The tokenizer for text.
        feature_extractor: Feature extractor for audio.
        device: The computation device.
        num_step: Number of decoding steps.
        guidance_scale: Classifier-free guidance scale.
        speed: Speed control factor.
        t_shift: Time shift for ODE solver.
        target_rms: Target RMS normalization value.
        feat_scale: Feature scale factor.
        sampling_rate: Audio sampling rate.
        raw_evaluation: Whether to use raw evaluation mode.
        max_duration: Maximum batch duration in seconds.
        remove_long_sil: Whether to remove long silences.
    """
    total_t = []
    total_t_no_vocoder = []
    total_t_vocoder = []
    total_wav_seconds = []

    with open(test_list) as fr:
        lines = fr.readlines()

    for i, line in enumerate(lines):
        wav_name, prompt_text, prompt_wav, text = line.strip().split("\t")
        save_path = f"{res_dir}/{wav_name}.wav"

        common_params = {
            "save_path": save_path,
            "prompt_text": prompt_text,
            "prompt_wav": prompt_wav,
            "text": text,
            "model": model,
            "vocoder": vocoder,
            "tokenizer": tokenizer,
            "feature_extractor": feature_extractor,
            "device": device,
            "num_step": num_step,
            "guidance_scale": guidance_scale,
            "speed": speed,
            "t_shift": t_shift,
            "target_rms": target_rms,
            "feat_scale": feat_scale,
            "sampling_rate": sampling_rate,
        }

        if raw_evaluation:
            metrics = generate_sentence_raw_evaluation(**common_params)
        else:
            metrics = generate_sentence(
                **common_params,
                max_duration=max_duration,
                remove_long_sil=remove_long_sil,
            )
        log.info("sentence_saved", sentence=i, save_path=save_path)
        log.info("sentence_rtf", sentence=i, rtf=round(metrics["rtf"], 4))
        total_t.append(metrics["t"])
        total_t_no_vocoder.append(metrics["t_no_vocoder"])
        total_t_vocoder.append(metrics["t_vocoder"])
        total_wav_seconds.append(metrics["wav_seconds"])

    log.info("average_rtf", rtf=round(np.sum(total_t) / np.sum(total_wav_seconds), 4))
    log.info(
        "average_rtf_no_vocoder",
        rtf=round(np.sum(total_t_no_vocoder) / np.sum(total_wav_seconds), 4),
    )
    log.info(
        "average_rtf_vocoder",
        rtf=round(np.sum(total_t_vocoder) / np.sum(total_wav_seconds), 4),
    )


@app.command()
@torch.inference_mode()
def main(  # noqa: PLR0912, PLR0913, PLR0915, C901
    model_name: str = typer.Option(  # noqa: B008
        "zipvoice",
        "--model-name",
        help="The model used for inference",
    ),
    model_dir: str | None = typer.Option(
        None,
        "--model-dir",
        help="The model directory that contains model checkpoint, configuration "
        "file model.json, and tokens file tokens.txt. Will download pre-trained "
        "checkpoint from huggingface if not specified.",
    ),
    checkpoint_name: str = typer.Option(
        "model.pt",
        "--checkpoint-name",
        help="The name of model checkpoint.",
    ),
    vocoder_path: str | None = typer.Option(
        None,
        "--vocoder-path",
        help="The vocoder checkpoint. Will download pre-trained vocoder from huggingface if not specified.",
    ),
    tokenizer: str = typer.Option(
        "emilia",
        "--tokenizer",
        help="Tokenizer type.",
    ),
    lang: str = typer.Option(
        "en-us",
        "--lang",
        help="Language identifier, used when tokenizer type is espeak. see"
        "https://github.com/rhasspy/espeak-ng/blob/master/docs/languages.md",
    ),
    test_list: str | None = typer.Option(
        None,
        "--test-list",
        help="The list of prompt speech, prompt_transcription, "
        "and text to synthesize in the format of "
        "'{wav_name}\\t{prompt_transcription}\\t{prompt_wav}\\t{text}'.",
    ),
    prompt_wav: str | None = typer.Option(
        None,
        "--prompt-wav",
        help="The prompt wav to mimic",
    ),
    prompt_text: str | None = typer.Option(
        None,
        "--prompt-text",
        help="The transcription of the prompt wav",
    ),
    text: str | None = typer.Option(
        None,
        "--text",
        help="The text to synthesize",
    ),
    res_dir: str = typer.Option(
        "results",
        "--res-dir",
        help="Path name of the generated wavs dir, used when test-list is not None",
    ),
    res_wav_path: str = typer.Option(
        "result.wav",
        "--res-wav-path",
        help="Path name of the generated wav path, used when test-list is None",
    ),
    guidance_scale: float | None = typer.Option(
        None,
        "--guidance-scale",
        help="The scale of classifier-free guidance during inference.",
    ),
    num_step: int | None = typer.Option(
        None,
        "--num-step",
        help="The number of sampling steps.",
    ),
    feat_scale: float = typer.Option(
        0.1,
        "--feat-scale",
        help="The scale factor of fbank feature",
    ),
    speed: float = typer.Option(
        1.0,
        "--speed",
        help="Control speech speed, 1.0 means normal, >1.0 means speed up",
    ),
    t_shift: float = typer.Option(
        0.5,
        "--t-shift",
        help="Shift t to smaller ones if t_shift < 1.0",
    ),
    target_rms: float = typer.Option(
        0.1,
        "--target-rms",
        help="Target speech normalization rms value, set to 0 to disable normalization",
    ),
    seed: int = typer.Option(
        666,
        "--seed",
        help="Random seed",
    ),
    num_thread: int = typer.Option(
        1,
        "--num-thread",
        help="Number of threads to use for PyTorch on CPU.",
    ),
    raw_evaluation: bool = typer.Option(
        False,
        "--raw-evaluation",
        help="Whether to use the 'raw' evaluation mode where provided "
        "prompts and text are fed to the model without pre-processing",
    ),
    max_duration: float = typer.Option(
        100,
        "--max-duration",
        help="Maximum duration (seconds) in a single batch, including "
        "durations of the prompt and generated wavs. You can reduce it "
        "if it causes CUDA OOM.",
    ),
    remove_long_sil: bool = typer.Option(
        False,
        "--remove-long-sil",
        help="Whether to remove long silences in the middle of the generated "
        "speech (edge silences will be removed by default).",
    ),
    trt_engine_path: str | None = typer.Option(
        None,
        "--trt-engine-path",
        help="The path to the TensorRT engine file.",
    ),
) -> None:
    """Run speech synthesis inference using ZipVoice or ZipVoice-Distill."""
    torch.set_num_threads(num_thread)
    torch.set_num_interop_threads(num_thread)

    fix_random_seed(seed)

    model_defaults = {
        "zipvoice": {
            "num_step": 16,
            "guidance_scale": 1.0,
        },
        "zipvoice_distill": {
            "num_step": 8,
            "guidance_scale": 3.0,
        },
    }

    model_specific_defaults = model_defaults.get(model_name, {})

    if num_step is None:
        num_step = model_specific_defaults.get("num_step", 16)
        log.info("setting_default_param", param="num_step", value=num_step)
    if guidance_scale is None:
        guidance_scale = model_specific_defaults.get("guidance_scale", 1.0)
        log.info("setting_default_param", param="guidance_scale", value=guidance_scale)

    if not ((test_list is not None) ^ ((prompt_wav and prompt_text and text) is not None)):
        msg = (
            "For inference, please provide prompts and text with either '--test-list'"
            " or '--prompt-wav, --prompt-text and --text'."
        )
        raise ValueError(msg)

    if model_dir is not None:
        model_dir_path = Path(model_dir)
        if not model_dir_path.is_dir():
            msg = f"{model_dir_path} does not exist"
            raise FileNotFoundError(msg)
        for filename in [checkpoint_name, "model.json", "tokens.txt"]:
            if not (model_dir_path / filename).is_file():
                msg = f"{model_dir_path / filename} does not exist"
                raise FileNotFoundError(msg)
        model_ckpt = model_dir_path / checkpoint_name
        model_config_path = model_dir_path / "model.json"
        token_file = model_dir_path / "tokens.txt"
        log.info(
            "using_local_model",
            model_name=model_name,
            model_dir=model_dir,
            checkpoint=checkpoint_name,
        )
    else:
        log.info("using_pretrained_model", model_name=model_name)
        model_ckpt = hf_hub_download(HUGGINGFACE_REPO, filename=f"{MODEL_DIR[model_name]}/model.pt")
        model_config_path = hf_hub_download(HUGGINGFACE_REPO, filename=f"{MODEL_DIR[model_name]}/model.json")

        token_file = hf_hub_download(HUGGINGFACE_REPO, filename=f"{MODEL_DIR[model_name]}/tokens.txt")

    if tokenizer == "emilia":
        tokenizer_obj = EmiliaTokenizer(token_file=token_file)
    elif tokenizer == "libritts":
        tokenizer_obj = LibriTTSTokenizer(token_file=token_file)
    elif tokenizer == "espeak":
        tokenizer_obj = EspeakTokenizer(token_file=token_file, lang=lang)
    else:
        if tokenizer != "simple":
            msg = f"Unsupported tokenizer: {tokenizer}"
            raise ValueError(msg)
        tokenizer_obj = SimpleTokenizer(token_file=token_file)

    tokenizer_config = {"vocab_size": tokenizer_obj.vocab_size, "pad_id": tokenizer_obj.pad_id}

    with open(model_config_path, "rb") as f:
        model_config = orjson.loads(f.read())

    if model_name == "zipvoice":
        model_obj = ZipVoice(
            **model_config["model"],
            **tokenizer_config,
        )
    else:
        if model_name != "zipvoice_distill":
            msg = f"Unsupported model name: {model_name}"
            raise ValueError(msg)
        model_obj = ZipVoiceDistill(
            **model_config["model"],
            **tokenizer_config,
        )

    if str(model_ckpt).endswith(".safetensors"):
        safetensors.torch.load_model(model_obj, model_ckpt)
    elif str(model_ckpt).endswith(".pt"):
        load_checkpoint(filename=model_ckpt, model=model_obj, strict=True)
    else:
        msg = f"Unsupported model checkpoint format: {model_ckpt}"
        raise ValueError(msg)

    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info("device", device=str(device))

    model_obj = model_obj.to(device)
    model_obj.eval()

    if trt_engine_path:
        load_trt(model_obj, trt_engine_path)

    vocoder = get_vocoder(vocoder_path)
    vocoder = vocoder.to(device)
    vocoder.eval()

    if model_config["feature"]["type"] == "vocos":
        feature_extractor = VocosFbank()
    else:
        msg = f"Unsupported feature type: {model_config['feature']['type']}"
        raise ValueError(msg)
    sampling_rate = model_config["feature"]["sampling_rate"]

    log.info("start_generating")
    if test_list:
        os.makedirs(res_dir, exist_ok=True)
        generate_list(
            res_dir=res_dir,
            test_list=test_list,
            model=model_obj,
            vocoder=vocoder,
            tokenizer=tokenizer_obj,
            feature_extractor=feature_extractor,
            device=device,
            num_step=num_step,
            guidance_scale=guidance_scale,
            speed=speed,
            t_shift=t_shift,
            target_rms=target_rms,
            feat_scale=feat_scale,
            sampling_rate=sampling_rate,
            raw_evaluation=raw_evaluation,
            max_duration=max_duration,
            remove_long_sil=remove_long_sil,
        )
    else:
        if raw_evaluation:
            msg = "Raw evaluation is only valid with --test-list"
            raise ValueError(msg)
        generate_sentence(
            save_path=res_wav_path,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav,
            text=text,
            model=model_obj,
            vocoder=vocoder,
            tokenizer=tokenizer_obj,
            feature_extractor=feature_extractor,
            device=device,
            num_step=num_step,
            guidance_scale=guidance_scale,
            speed=speed,
            t_shift=t_shift,
            target_rms=target_rms,
            feat_scale=feat_scale,
            sampling_rate=sampling_rate,
            max_duration=max_duration,
            remove_long_sil=remove_long_sil,
        )
        log.info("saved", path=res_wav_path)
    log.info("done")


if __name__ == "__main__":
    app()

# end zipvoice/bin/infer_zipvoice.py
