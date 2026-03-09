# start zipvoice/bin/infer_zipvoice_dialog.py
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

r"""Generate speech with pre-trained ZipVoice-Dialog or ZipVoice-Dialog-Stereo models.

If no local model is specified, required files will be automatically
downloaded from HuggingFace.

Usage:

Note: If you having trouble connecting to HuggingFace,
    try switching endpoint to mirror site:
export HF_ENDPOINT=https://hf-mirror.com

python3 -m zipvoice.bin.infer_zipvoice_dialog \
    --model-name zipvoice_dialog \
    --test-list test.tsv \
    --res-dir results

`--model-name` can be `zipvoice_dialog` or `zipvoice_dialog_stereo`,
    which generate mono and stereo dialogues, respectively.

Each line of `test.tsv` is in the format of merged conversation:
    '{wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}'
    or splited conversation:
    '{wav_name}\t{spk1_prompt_transcription}\t{spk2_prompt_transcription}
        \t{spk1_prompt_wav}\t{spk2_prompt_wav}\t{text}'
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
from zipvoice.models.zipvoice_dialog import ZipVoiceDialog, ZipVoiceDialogStereo
from zipvoice.tokenizer.tokenizer import DialogTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import (
    add_punctuation,
    batchify_tokens,
    chunk_tokens_dialog,
    load_prompt_wav,
    merge_chunked_wavs,
    remove_silence,
    rms_norm,
)

log = structlog.get_logger()

HUGGINGFACE_REPO = HUGGINGFACE_REPO_ZIPVOICE
MODEL_DIR = {
    "zipvoice_dialog": "zipvoice_dialog",
    "zipvoice_dialog_stereo": "zipvoice_dialog_stereo",
}

app = typer.Typer(
    help="Generate speech with pre-trained ZipVoice-Dialog or ZipVoice-Dialog-Stereo models.",
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
    prompt_wav: str | list[str],
    text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: DialogTokenizer,
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
        prompt_wav (str | list[str]): Path to the prompt wav file, can be
            one or two wav files, which corresponding to a merged conversational
            speech or two seperate speaker's speech.
        text (str): Text to be synthesized into a waveform.
        model (torch.nn.Module): The model used for generation.
        vocoder (torch.nn.Module): The vocoder used to convert features to waveforms.
        tokenizer (DialogTokenizer): The tokenizer used to convert text to tokens.
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
    # Load and preprocess prompt wav
    if isinstance(prompt_wav, str):
        prompt_wav = [
            prompt_wav,
        ]
    else:
        if not (len(prompt_wav) == 2 and isinstance(prompt_wav[0], str)):
            msg = "prompt_wav must be a list of exactly 2 strings"
            raise ValueError(msg)

    loaded_prompt_wavs = prompt_wav
    for i in range(len(prompt_wav)):
        loaded_prompt_wavs[i] = load_prompt_wav(loaded_prompt_wavs[i], sampling_rate=sampling_rate)
        if loaded_prompt_wavs[i].size(0) != 1:
            loaded_prompt_wavs[i] = loaded_prompt_wavs[i].mean(0, keepdim=True)

    prompt_wav = loaded_prompt_wavs[0] if len(loaded_prompt_wavs) == 1 else torch.cat(loaded_prompt_wavs, dim=1)

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


def generate_sentence(  # noqa: PLR0912, PLR0913, PLR0915, C901
    save_path: str,
    prompt_text: str,
    prompt_wav: str | list[str],
    text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: DialogTokenizer,
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
    1. chunk the text according to speaker-turn symbol [S1].
    2. process chunked texts in batches.
    3. remove long silences in the prompt audio.
    4. add punctuation to the end of prompt text and text if there is not.

    Args:
        save_path (str): Path to save the generated wav.
        prompt_text (str): Transcription of the prompt wav.
        prompt_wav (str | list[str]): Path to the prompt wav file, can be
            one or two wav files, which corresponding to a merged conversational
            speech or two seperate speaker's speech.
        text (str): Text to be synthesized into a waveform.
        model (torch.nn.Module): The model used for generation.
        vocoder (torch.nn.Module): The vocoder used to convert features to waveforms.
        tokenizer (DialogTokenizer): The tokenizer used to convert text to tokens.
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
    # Load and preprocess prompt wav
    if isinstance(prompt_wav, str):
        prompt_wav = [
            prompt_wav,
        ]
    else:
        if not (len(prompt_wav) == 2 and isinstance(prompt_wav[0], str)):
            msg = "prompt_wav must be a list of exactly 2 strings"
            raise ValueError(msg)

    loaded_prompt_wavs = prompt_wav
    for i in range(len(prompt_wav)):
        loaded_prompt_wavs[i] = load_prompt_wav(loaded_prompt_wavs[i], sampling_rate=sampling_rate)
        if loaded_prompt_wavs[i].size(0) != 1:
            loaded_prompt_wavs[i] = loaded_prompt_wavs[i].mean(0, keepdim=True)

    prompt_wav = loaded_prompt_wavs[0] if len(loaded_prompt_wavs) == 1 else torch.cat(loaded_prompt_wavs, dim=1)

    # Remove edge and long silences in the prompt wav.
    # Add 0.2s trailing silence to avoid leaking prompt to generated speech.
    prompt_wav = remove_silence(prompt_wav, sampling_rate, only_edge=False, trail_sil=200)

    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)

    prompt_duration = prompt_wav.shape[-1] / sampling_rate

    if prompt_duration > 40:
        log.warning(
            "prompt_wav_too_long",
            prompt_duration=prompt_duration,
        )
    elif prompt_duration > 20:
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

    # chunk text so that each len(prompt wav + generated wav) is around 40 seconds.
    token_duration = (prompt_wav.shape[-1] / sampling_rate) / (len(prompt_tokens_str) * speed)
    max_tokens = int((40 - prompt_duration) / token_duration)
    chunked_tokens_str = chunk_tokens_dialog(tokens_str, max_tokens=max_tokens)

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


def generate_sentence_stereo_raw_evaluation(  # noqa: PLR0912, PLR0913, PLR0915, C901
    save_path: str,
    prompt_text: str,
    prompt_wav: str | list[str],
    text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: DialogTokenizer,
    feature_extractor: VocosFbank,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    silence_wav: str | None = None,
):
    """Generate waveform of a text based on a given prompt waveform and its transcription.

    This function directly feed the prompt_text, prompt_wav and text to the model.
    It is not efficient and can have poor results for some inappropriate inputs.
    (e.g., prompt wav contains long silence, text to be generated is too long)
    This function can be used to evaluate the "raw" performance of the model.

    Args:
        save_path (str): Path to save the generated wav.
        prompt_text (str): Transcription of the prompt wav.
        prompt_wav (str | list[str]): Path to the prompt wav file, can be
            one or two wav files, which corresponding to a merged conversational
            speech or two seperate speaker's speech.
        text (str): Text to be synthesized into a waveform.
        model (torch.nn.Module): The model used for generation.
        vocoder (torch.nn.Module): The vocoder used to convert features to waveforms.
        tokenizer (DialogTokenizer): The tokenizer used to convert text to tokens.
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
        silence_wav (str): Path of the silence wav file, used in two-channel
            generation with single-channel prompts
    Returns:
        metrics (dict): Dictionary containing time and real-time
            factor metrics for processing.
    """
    # Load and preprocess prompt wav
    if isinstance(prompt_wav, str):
        prompt_wav = [
            prompt_wav,
        ]
    else:
        if not (len(prompt_wav) == 2 and isinstance(prompt_wav[0], str)):
            msg = "prompt_wav must be a list of exactly 2 strings"
            raise ValueError(msg)

    loaded_prompt_wavs = prompt_wav
    for i in range(len(prompt_wav)):
        loaded_prompt_wavs[i] = load_prompt_wav(loaded_prompt_wavs[i], sampling_rate=sampling_rate)

    if len(loaded_prompt_wavs) == 1:
        if loaded_prompt_wavs[0].size(0) != 2:
            msg = "Merged prompt wav must be stereo for stereo dialogue generation"
            raise ValueError(msg)
        prompt_wav = loaded_prompt_wavs[0]

    else:
        if len(loaded_prompt_wavs) != 2:
            msg = f"Expected 2 loaded prompt wavs, got {len(loaded_prompt_wavs)}"
            raise RuntimeError(msg)
        if loaded_prompt_wavs[0].size(0) == 2:
            prompt_wav = torch.cat(loaded_prompt_wavs, dim=1)
        else:
            if loaded_prompt_wavs[0].size(0) != 1:
                msg = f"Expected mono prompt wav, got {loaded_prompt_wavs[0].size(0)} channels"
                raise ValueError(msg)
            silence_wav, silence_sampling_rate = torchaudio.load(silence_wav)
            if silence_sampling_rate != sampling_rate:
                msg = f"Silence wav sampling rate {silence_sampling_rate} != expected {sampling_rate}"
                raise ValueError(msg)
            prompt_wav = silence_wav[:, : loaded_prompt_wavs[0].size(1) + loaded_prompt_wavs[1].size(1)]
            prompt_wav[0, : loaded_prompt_wavs[0].size(1)] = loaded_prompt_wavs[0]
            prompt_wav[1, loaded_prompt_wavs[0].size(1) :] = loaded_prompt_wavs[1]

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
    feat_dim = pred_features.size(1) // 2
    wav_left = vocoder.decode(pred_features[:, :feat_dim]).squeeze(1).clamp(-1, 1)
    wav_right = vocoder.decode(pred_features[:, feat_dim : feat_dim * 2]).squeeze(1).clamp(-1, 1)

    wav = torch.cat([wav_left, wav_right], dim=0)

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


def generate_sentence_stereo(  # noqa: PLR0912, PLR0913, PLR0915, C901
    save_path: str,
    prompt_text: str,
    prompt_wav: str | list[str],
    text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: DialogTokenizer,
    feature_extractor: VocosFbank,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    silence_wav: str | None = None,
    max_duration: float = 100,
    remove_long_sil: bool = False,
):
    """Generate waveform of a text based on a given prompt waveform and its transcription.

    This function will do the following to improve the generation quality:
    1. chunk the text according to speaker-turn symbol [S1].
    2. process chunked texts in batches.
    3. remove long silences in the prompt audio.
    4. add punctuation to the end of prompt text and text if there is not.

    Args:
        save_path (str): Path to save the generated wav.
        prompt_text (str): Transcription of the prompt wav.
        prompt_wav (str | list[str]): Path to the prompt wav file, can be
            one or two wav files, which corresponding to a merged conversational
            speech or two seperate speaker's speech.
        text (str): Text to be synthesized into a waveform.
        model (torch.nn.Module): The model used for generation.
        vocoder (torch.nn.Module): The vocoder used to convert features to waveforms.
        tokenizer (DialogTokenizer): The tokenizer used to convert text to tokens.
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
        silence_wav (str): Path of the silence wav file, used in two-channel
            generation with single-channel prompts
        max_duration (float, optional): The maximum duration to process in each
            batch. Used to control memory consumption when generating long audios.
        remove_long_sil (bool, optional): Whether to remove long silences in the
            middle of the generated speech (edge silences will be removed by default).

    Returns:
        metrics (dict): Dictionary containing time and real-time
            factor metrics for processing.
    """
    # Load and preprocess prompt wav
    if isinstance(prompt_wav, str):
        prompt_wav = [
            prompt_wav,
        ]
    else:
        if not (len(prompt_wav) == 2 and isinstance(prompt_wav[0], str)):
            msg = "prompt_wav must be a list of exactly 2 strings"
            raise ValueError(msg)

    loaded_prompt_wavs = prompt_wav
    for i in range(len(prompt_wav)):
        loaded_prompt_wavs[i] = load_prompt_wav(loaded_prompt_wavs[i], sampling_rate=sampling_rate)

    if len(loaded_prompt_wavs) == 1:
        if loaded_prompt_wavs[0].size(0) != 2:
            msg = "Merged prompt wav must be stereo for stereo dialogue generation"
            raise ValueError(msg)
        prompt_wav = loaded_prompt_wavs[0]

    else:
        if len(loaded_prompt_wavs) != 2:
            msg = f"Expected 2 loaded prompt wavs, got {len(loaded_prompt_wavs)}"
            raise RuntimeError(msg)
        if loaded_prompt_wavs[0].size(0) == 2:
            prompt_wav = torch.cat(loaded_prompt_wavs, dim=1)
        else:
            if loaded_prompt_wavs[0].size(0) != 1:
                msg = f"Expected mono prompt wav, got {loaded_prompt_wavs[0].size(0)} channels"
                raise ValueError(msg)
            silence_wav, silence_sampling_rate = torchaudio.load(silence_wav)
            if silence_sampling_rate != sampling_rate:
                msg = f"Silence wav sampling rate {silence_sampling_rate} != expected {sampling_rate}"
                raise ValueError(msg)
            prompt_wav = silence_wav[:, : loaded_prompt_wavs[0].size(1) + loaded_prompt_wavs[1].size(1)]
            prompt_wav[0, : loaded_prompt_wavs[0].size(1)] = loaded_prompt_wavs[0]
            prompt_wav[1, loaded_prompt_wavs[0].size(1) :] = loaded_prompt_wavs[1]

    # Remove edge and long silences in the prompt wav.
    # Add 0.2s trailing silence to avoid leaking prompt to generated speech.
    prompt_wav = remove_silence(prompt_wav, sampling_rate, only_edge=False, trail_sil=200)

    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)

    prompt_duration = prompt_wav.shape[-1] / sampling_rate

    if prompt_duration > 40:
        log.warning(
            "prompt_wav_too_long",
            prompt_duration=prompt_duration,
        )
    elif prompt_duration > 20:
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

    # chunk text so that each len(prompt wav + generated wav) is around 40 seconds.
    token_duration = (prompt_wav.shape[-1] / sampling_rate) / (len(prompt_tokens_str) * speed)
    max_tokens = int((40 - prompt_duration) / token_duration)
    chunked_tokens_str = chunk_tokens_dialog(tokens_str, max_tokens=max_tokens)

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
            feat_dim = pred_features.size(1) // 2
            wav_left = (
                vocoder.decode(pred_features[i][None, :feat_dim, : pred_features_lens[i]]).squeeze(1).clamp(-1, 1)
            )
            wav_right = (
                vocoder.decode(pred_features[i][None, feat_dim : feat_dim * 2, : pred_features_lens[i]])
                .squeeze(1)
                .clamp(-1, 1)
            )
            wav = torch.cat([wav_left, wav_right], dim=0)

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
    model_name: str,
    res_dir: str,
    test_list: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: DialogTokenizer,
    feature_extractor: VocosFbank,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.5,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    silence_wav: str | None = None,
    raw_evaluation: bool = False,
    max_duration: float = 100,
    remove_long_sil: bool = False,
):
    """Generate speech for a list of dialog test samples and report metrics.

    Args:
        model_name: Name of the model being used.
        res_dir: Directory to save generated wavs.
        test_list: Path to TSV file with test samples.
        model: The model for generation.
        vocoder: The vocoder model.
        tokenizer: The dialog tokenizer for text.
        feature_extractor: Feature extractor for audio.
        device: The computation device.
        num_step: Number of decoding steps.
        guidance_scale: Classifier-free guidance scale.
        speed: Speed control factor.
        t_shift: Time shift for ODE solver.
        target_rms: Target RMS normalization value.
        feat_scale: Feature scale factor.
        sampling_rate: Audio sampling rate.
        silence_wav: Optional path to silence wav for stereo generation.
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
        items = line.strip().split("\t")
        if len(items) == 6:
            (
                wav_name,
                prompt_text_1,
                prompt_text_2,
                prompt_wav_1,
                prompt_wav_2,
                text,
            ) = items
            prompt_text = f"[S1]{prompt_text_1}[S2]{prompt_text_2}"
            prompt_wav = [prompt_wav_1, prompt_wav_2]
        elif len(items) == 4:
            wav_name, prompt_text, prompt_wav, text = items
        else:
            msg = f"Invalid line: {line}"
            raise ValueError(msg)
        if not text.startswith("[S1]"):
            msg = f"Dialog text must start with '[S1]', got: {text[:50]!r}"
            raise ValueError(msg)

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

        if model_name == "zipvoice_dialog":
            if raw_evaluation:
                metrics = generate_sentence_raw_evaluation(**common_params)
            else:
                metrics = generate_sentence(
                    **common_params,
                    max_duration=max_duration,
                    remove_long_sil=remove_long_sil,
                )
        else:
            if model_name != "zipvoice_dialog_stereo":
                msg = f"Unsupported model name: {model_name}"
                raise ValueError(msg)
            if raw_evaluation:
                metrics = generate_sentence_stereo_raw_evaluation(
                    **common_params,
                    silence_wav=silence_wav,
                )
            else:
                metrics = generate_sentence_stereo(
                    **common_params,
                    silence_wav=silence_wav,
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
        "zipvoice_dialog",
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
    test_list: str | None = typer.Option(
        None,
        "--test-list",
        help="The list of prompt speech, prompt_transcription, "
        "and text to synthesize in the format of merged conversation: "
        "'{wav_name}\\t{prompt_transcription}\\t{prompt_wav}\\t{text}' "
        "or splited conversation: "
        "'{wav_name}\\t{spk1_prompt_transcription}\\t{spk2_prompt_transcription}"
        "\\t{spk1_prompt_wav}\\t{spk2_prompt_wav}\\t{text}'.",
    ),
    res_dir: str = typer.Option(
        "results",
        "--res-dir",
        help="Path name of the generated wavs dir, used when test-list is not None",
    ),
    guidance_scale: float = typer.Option(
        1.5,
        "--guidance-scale",
        help="The scale of classifier-free guidance during inference.",
    ),
    num_step: int = typer.Option(
        16,
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
    silence_wav: str = typer.Option(
        "assets/silence.wav",
        "--silence-wav",
        help="Path of the silence wav file, used in two-channel generation with single-channel prompts",
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
) -> None:
    """Run speech synthesis inference using ZipVoice-Dialog or ZipVoice-Dialog-Stereo."""
    torch.set_num_threads(num_thread)
    torch.set_num_interop_threads(num_thread)

    fix_random_seed(seed)

    if test_list is None:
        msg = "For inference, please provide prompts and text with '--test-list'"
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

    tokenizer_obj = DialogTokenizer(token_file=token_file)

    tokenizer_config = {
        "vocab_size": tokenizer_obj.vocab_size,
        "pad_id": tokenizer_obj.pad_id,
        "spk_a_id": tokenizer_obj.spk_a_id,
        "spk_b_id": tokenizer_obj.spk_b_id,
    }

    with open(model_config_path, "rb") as f:
        model_config = orjson.loads(f.read())

    if model_name == "zipvoice_dialog":
        model_obj = ZipVoiceDialog(
            **model_config["model"],
            **tokenizer_config,
        )
    else:
        if model_name != "zipvoice_dialog_stereo":
            msg = f"Unsupported model name: {model_name}"
            raise ValueError(msg)
        model_obj = ZipVoiceDialogStereo(
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

    vocoder = get_vocoder(vocoder_path)
    vocoder = vocoder.to(device)
    vocoder.eval()

    if model_config["feature"]["type"] == "vocos":
        if model_name == "zipvoice_dialog":
            num_channels = 1
        else:
            if model_name != "zipvoice_dialog_stereo":
                msg = f"Unsupported model name: {model_name}"
                raise ValueError(msg)
            num_channels = 2
        feature_extractor = VocosFbank(num_channels=num_channels)
    else:
        msg = f"Unsupported feature type: {model_config['feature']['type']}"
        raise ValueError(msg)
    sampling_rate = model_config["feature"]["sampling_rate"]

    log.info("start_generating")
    os.makedirs(res_dir, exist_ok=True)
    generate_list(
        model_name=model_name,
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
        silence_wav=silence_wav,
        raw_evaluation=raw_evaluation,
        max_duration=max_duration,
        remove_long_sil=remove_long_sil,
    )
    log.info("done")


if __name__ == "__main__":
    app()

# end zipvoice/bin/infer_zipvoice_dialog.py
