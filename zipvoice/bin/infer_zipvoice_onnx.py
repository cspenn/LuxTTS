# start zipvoice/bin/infer_zipvoice_onnx.py
#!/usr/bin/env python3
# Copyright         2025  Xiaomi Corp.        (authors: Han Zhu,
#                                                       Zengwei Yao)
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

r"""Generate speech with pre-trained ZipVoice or ZipVoice-Distill ONNX models.

If no local model is specified, required files will be automatically
downloaded from HuggingFace.

Usage:

Note: If you having trouble connecting to HuggingFace,
    try switching endpoint to mirror site:
export HF_ENDPOINT=https://hf-mirror.com

(1) Inference of a single sentence:

python3 -m zipvoice.bin.infer_zipvoice_onnx \
    --onnx-int8 False \
    --model-name zipvoice \
    --prompt-wav prompt.wav \
    --prompt-text "I am a prompt." \
    --text "I am a sentence." \
    --res-wav-path result.wav

(2) Inference of a list of sentences:
python3 -m zipvoice.bin.infer_zipvoice_onnx \
    --onnx-int8 False \
    --model-name zipvoice \
    --test-list test.tsv \
    --res-dir results

`--model-name` can be `zipvoice` or `zipvoice_distill`,
    which are the models before and after distillation, respectively.

Each line of `test.tsv` is in the format of
    `{wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}`.

Set `--onnx-int8 True` to use int8 quantizated ONNX model.
Quantizated model has faster but lower quality.
"""

import datetime as dt
import os
from pathlib import Path

import orjson
import structlog
import typer

log = structlog.get_logger()

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from lhotse.utils import fix_random_seed  # noqa: E402
from torch import Tensor, nn  # noqa: E402

from zipvoice.bin.infer_zipvoice import get_vocoder  # noqa: E402
from zipvoice.constants import HUGGINGFACE_REPO_ZIPVOICE  # noqa: E402
from zipvoice.models.modules.solver import get_time_steps  # noqa: E402
from zipvoice.tokenizer.tokenizer import (  # noqa: E402
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
from zipvoice.utils.feature import VocosFbank  # noqa: E402
from zipvoice.utils.infer import (  # noqa: E402
    add_punctuation,
    chunk_tokens_punctuation,
    load_prompt_wav,
    merge_chunked_wavs,
    remove_silence,
    rms_norm,
)

HUGGINGFACE_REPO = HUGGINGFACE_REPO_ZIPVOICE
MODEL_DIR = {
    "zipvoice": "zipvoice",
    "zipvoice_distill": "zipvoice_distill",
}

app = typer.Typer(
    help="Generate speech with pre-trained ZipVoice or ZipVoice-Distill ONNX models.",
    add_completion=False,
)


class OnnxModel:
    """ONNX-based inference model for ZipVoice."""

    def __init__(
        self,
        text_encoder_path: str,
        fm_decoder_path: str,
        num_thread: int = 1,
    ):
        """Initialize the ONNX model with text encoder and flow-matching decoder.

        Args:
            text_encoder_path: Path to the text encoder ONNX model.
            fm_decoder_path: Path to the flow-matching decoder ONNX model.
            num_thread: Number of threads for inference.
        """
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = num_thread
        session_opts.intra_op_num_threads = num_thread

        self.session_opts = session_opts

        self.init_text_encoder(text_encoder_path)
        self.init_fm_decoder(fm_decoder_path)

    def init_text_encoder(self, model_path: str):
        """Initialize the text encoder ONNX session.

        Args:
            model_path: Path to the text encoder ONNX model file.
        """
        self.text_encoder = ort.InferenceSession(
            model_path,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

    def init_fm_decoder(self, model_path: str):
        """Initialize the flow-matching decoder ONNX session.

        Args:
            model_path: Path to the flow-matching decoder ONNX model file.
        """
        self.fm_decoder = ort.InferenceSession(
            model_path,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )
        meta = self.fm_decoder.get_modelmeta().custom_metadata_map
        self.feat_dim = int(meta["feat_dim"])

    def run_text_encoder(
        self,
        tokens: Tensor,
        prompt_tokens: Tensor,
        prompt_features_len: Tensor,
        speed: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run the text encoder on the given inputs.

        Args:
            tokens: Text token tensor.
            prompt_tokens: Prompt text token tensor.
            prompt_features_len: Length of prompt features.
            speed: Speed control tensor.

        Returns:
            Text condition tensor from the encoder.
        """
        out = self.text_encoder.run(
            [
                self.text_encoder.get_outputs()[0].name,
            ],
            {
                self.text_encoder.get_inputs()[0].name: tokens.numpy(),
                self.text_encoder.get_inputs()[1].name: prompt_tokens.numpy(),
                self.text_encoder.get_inputs()[2].name: prompt_features_len.numpy(),
                self.text_encoder.get_inputs()[3].name: speed.numpy(),
            },
        )
        return torch.from_numpy(out[0])

    def run_fm_decoder(
        self,
        t: Tensor,
        x: Tensor,
        text_condition: Tensor,
        speech_condition: torch.Tensor,
        guidance_scale: Tensor,
    ) -> Tensor:
        """Run the flow-matching decoder on the given inputs.

        Args:
            t: Current timestep tensor.
            x: Current noisy features tensor.
            text_condition: Text conditioning tensor.
            speech_condition: Speech conditioning tensor.
            guidance_scale: Classifier-free guidance scale tensor.

        Returns:
            Predicted velocity tensor.
        """
        out = self.fm_decoder.run(
            [
                self.fm_decoder.get_outputs()[0].name,
            ],
            {
                self.fm_decoder.get_inputs()[0].name: t.numpy(),
                self.fm_decoder.get_inputs()[1].name: x.numpy(),
                self.fm_decoder.get_inputs()[2].name: text_condition.numpy(),
                self.fm_decoder.get_inputs()[3].name: speech_condition.numpy(),
                self.fm_decoder.get_inputs()[4].name: guidance_scale.numpy(),
            },
        )
        return torch.from_numpy(out[0])


def sample(  # noqa: PLR0913
    model: OnnxModel,
    tokens: list[list[int]],
    prompt_tokens: list[list[int]],
    prompt_features: Tensor,
    speed: float = 1.0,
    t_shift: float = 0.5,
    guidance_scale: float = 1.0,
    num_step: int = 16,
) -> torch.Tensor:
    """Generate acoustic features given text tokens, prompt features and prompt tokens.

    Args:
        model: The ONNX model used for inference.
        tokens: A list of list of text tokens.
        prompt_tokens: A list of list of prompt tokens.
        prompt_features: The prompt feature with the shape
            (batch_size, seq_len, feat_dim).
        speed: Speed control.
        t_shift: Time shift.
        guidance_scale: The guidance scale for classifier-free guidance.
        num_step: The number of steps to use in the ODE solver.

    Returns:
        Generated acoustic feature tensor.
    """
    # Run text encoder
    if len(tokens) != 1 or len(prompt_tokens) != 1:
        msg = f"Expected len(tokens)==1 and len(prompt_tokens)==1, got {len(tokens)} and {len(prompt_tokens)}"
        raise ValueError(msg)
    tokens = torch.tensor(tokens, dtype=torch.int64)
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.int64)
    prompt_features_len = torch.tensor(prompt_features.size(1), dtype=torch.int64)
    speed = torch.tensor(speed, dtype=torch.float32)

    text_condition = model.run_text_encoder(tokens, prompt_tokens, prompt_features_len, speed)

    batch_size, num_frames, _ = text_condition.shape
    if batch_size != 1:
        msg = f"Expected batch_size == 1, got {batch_size}"
        raise ValueError(msg)
    feat_dim = model.feat_dim

    # Run flow matching model
    timesteps = get_time_steps(
        t_start=0.0,
        t_end=1.0,
        num_step=num_step,
        t_shift=t_shift,
    )
    x = torch.randn(batch_size, num_frames, feat_dim)
    speech_condition = torch.nn.functional.pad(
        prompt_features, (0, 0, 0, num_frames - prompt_features.shape[1])
    )  # (B, T, F)
    guidance_scale = torch.tensor(guidance_scale, dtype=torch.float32)

    for step in range(num_step):
        v = model.run_fm_decoder(
            t=timesteps[step],
            x=x,
            text_condition=text_condition,
            speech_condition=speech_condition,
            guidance_scale=guidance_scale,
        )
        x = x + v * (timesteps[step + 1] - timesteps[step])

    x = x[:, prompt_features_len.item() :, :]
    return x


# Copied from zipvoice/bin/infer_zipvoice.py, but call an external sample function
def generate_sentence_raw_evaluation(  # noqa: PLR0913
    save_path: str,
    prompt_text: str,
    prompt_wav: str,
    text: str,
    model: OnnxModel,
    vocoder: nn.Module,
    tokenizer: EmiliaTokenizer,
    feature_extractor: VocosFbank,
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
    prompt_features = feature_extractor.extract(prompt_wav, sampling_rate=sampling_rate)

    prompt_features = prompt_features.unsqueeze(0) * feat_scale

    # Convert text to tokens
    tokens = tokenizer.texts_to_token_ids([text])
    prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])

    # Start timing
    start_t = dt.datetime.now()

    # Generate features
    pred_features = sample(
        model=model,
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        prompt_features=prompt_features,
        speed=speed,
        t_shift=t_shift,
        guidance_scale=guidance_scale,
        num_step=num_step,
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
    model: OnnxModel,
    vocoder: nn.Module,
    tokenizer: EmiliaTokenizer,
    feature_extractor: VocosFbank,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    remove_long_sil: bool = False,
):
    """Generate waveform of a text based on a given prompt waveform and its transcription.

    This function will do the following to improve the generation quality:
    1. chunk the text according to punctuations.
    2. process chunked texts sequentially.
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
    prompt_features = feature_extractor.extract(prompt_wav, sampling_rate=sampling_rate)

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
    log.debug("chunked_tokens_count", count=len(chunked_tokens_str))
    log.debug("chunked_tokens", tokens=chunked_tokens_str)

    # Tokenize text (int tokens)
    chunked_tokens = tokenizer.tokens_to_token_ids(chunked_tokens_str)
    prompt_tokens = tokenizer.tokens_to_token_ids([prompt_tokens_str])

    # Start predicting features
    chunked_features = []
    start_t = dt.datetime.now()
    for tokens in chunked_tokens:
        # Generate features
        pred_features = sample(
            model=model,
            tokens=[tokens],
            prompt_tokens=prompt_tokens,
            prompt_features=prompt_features,
            speed=speed,
            t_shift=t_shift,
            guidance_scale=guidance_scale,
            num_step=num_step,
        )

        # Postprocess predicted features
        pred_features = pred_features.permute(0, 2, 1) / feat_scale  # (B, C, T)
        chunked_features.append(pred_features)

    # Start vocoder processing
    chunked_wavs = []
    start_vocoder_t = dt.datetime.now()

    for pred_features in chunked_features:
        wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)
        # Adjust wav volume if necessary
        if prompt_rms < target_rms:
            wav = wav * prompt_rms / target_rms
        chunked_wavs.append(wav)

    # Finish model generation
    t = (dt.datetime.now() - start_t).total_seconds()

    # Merge chunked wavs
    final_wav = merge_chunked_wavs(
        chunked_wavs, chunked_index=None, remove_long_sil=remove_long_sil, sampling_rate=sampling_rate
    )

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
    model: OnnxModel,
    vocoder: nn.Module,
    tokenizer: EmiliaTokenizer,
    feature_extractor: VocosFbank,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    raw_evaluation: bool = False,
    remove_long_sil: bool = False,
):
    """Generate speech for a list of test samples using ONNX models and report metrics.

    Args:
        res_dir: Directory to save generated wavs.
        test_list: Path to TSV file with test samples.
        model: The ONNX model for generation.
        vocoder: The vocoder model.
        tokenizer: The tokenizer for text.
        feature_extractor: Feature extractor for audio.
        num_step: Number of decoding steps.
        guidance_scale: Classifier-free guidance scale.
        speed: Speed control factor.
        t_shift: Time shift for ODE solver.
        target_rms: Target RMS normalization value.
        feat_scale: Feature scale factor.
        sampling_rate: Audio sampling rate.
        raw_evaluation: Whether to use raw evaluation mode.
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
    onnx_int8: bool = typer.Option(  # noqa: B008
        False,
        "--onnx-int8",
        help="Whether to use the int8 model",
    ),
    model_name: str = typer.Option(
        "zipvoice",
        "--model-name",
        help="The model used for inference",
    ),
    model_dir: str | None = typer.Option(
        None,
        "--model-dir",
        help="The path to the local onnx model. "
        "Will download pre-trained checkpoint from huggingface if not specified.",
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
        help="Number of threads to use for ONNX Runtime and PyTorch.",
    ),
    raw_evaluation: bool = typer.Option(
        False,
        "--raw-evaluation",
        help="Whether to use the 'raw' evaluation mode where provided "
        "prompts and text are fed to the model without pre-processing",
    ),
    remove_long_sil: bool = typer.Option(
        False,
        "--remove-long-sil",
        help="Whether to remove long silences in the middle of the generated "
        "speech (edge silences will be removed by default).",
    ),
) -> None:
    """Run speech synthesis inference using ZipVoice ONNX models."""
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

    if onnx_int8:
        text_encoder_name = "text_encoder_int8.onnx"
        fm_decoder_name = "fm_decoder_int8.onnx"
    else:
        text_encoder_name = "text_encoder.onnx"
        fm_decoder_name = "fm_decoder.onnx"

    if model_dir is not None:
        model_dir_path = Path(model_dir)
        if not model_dir_path.is_dir():
            msg = f"{model_dir_path} does not exist"
            raise FileNotFoundError(msg)

        for filename in [
            text_encoder_name,
            fm_decoder_name,
            "model.json",
            "tokens.txt",
        ]:
            if not (model_dir_path / filename).is_file():
                msg = f"{model_dir_path / filename} does not exist"
                raise FileNotFoundError(msg)
        text_encoder_path = model_dir_path / text_encoder_name
        fm_decoder_path = model_dir_path / fm_decoder_name
        model_config_path = model_dir_path / "model.json"
        token_file = model_dir_path / "tokens.txt"
        log.info("using_local_model_dir", model_dir=model_dir)
    else:
        log.info("using_pretrained_model")
        text_encoder_path = hf_hub_download(
            HUGGINGFACE_REPO,
            filename=f"{MODEL_DIR[model_name]}/{text_encoder_name}",
        )
        fm_decoder_path = hf_hub_download(
            HUGGINGFACE_REPO,
            filename=f"{MODEL_DIR[model_name]}/{fm_decoder_name}",
        )
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

    with open(model_config_path, "rb") as f:
        model_config = orjson.loads(f.read())

    model_obj = OnnxModel(text_encoder_path, fm_decoder_path, num_thread=num_thread)

    vocoder = get_vocoder(vocoder_path)
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
            num_step=num_step,
            guidance_scale=guidance_scale,
            speed=speed,
            t_shift=t_shift,
            target_rms=target_rms,
            feat_scale=feat_scale,
            sampling_rate=sampling_rate,
            raw_evaluation=raw_evaluation,
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
            num_step=num_step,
            guidance_scale=guidance_scale,
            speed=speed,
            t_shift=t_shift,
            target_rms=target_rms,
            feat_scale=feat_scale,
            sampling_rate=sampling_rate,
            remove_long_sil=remove_long_sil,
        )
        log.info("saved", path=res_wav_path)
    log.info("done")


if __name__ == "__main__":
    app()

# end zipvoice/bin/infer_zipvoice_onnx.py
