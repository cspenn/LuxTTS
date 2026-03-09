# start zipvoice/utils/infer.py
"""Audio inference utilities for chunking, batching, and cross-fade merging."""

import numpy as np
import torch
import torchaudio
from pydub import AudioSegment
from pydub.silence import detect_leading_silence, split_on_silence

punctuation = {";", ":", ",", ".", "!", "?", "；", "：", "，", "。", "！", "？"}

# Named constants replacing magic numbers
SILENCE_THRESHOLD_MS = 1000
SILENCE_THRESHOLD_DB = -50
AUDIO_NORM_FACTOR = 32768.0
AUDIO_MAX_INT16 = 32767


def chunk_tokens_punctuation(tokens_list: list[str], max_tokens: int = 100):
    """Split tokens into chunks at punctuation boundaries.

    Splits the token list at punctuation marks, then merges short chunks until
    each chunk contains at most ``max_tokens`` tokens.

    Args:
        tokens_list: The list of tokens to be split.
        max_tokens: Maximum number of tokens per chunk.

    Returns:
        A list of token-list chunks.
    """
    # 1. Split the tokens according to punctuations.
    sentences = []
    current_sentence = []
    for token in tokens_list:
        # If the first token of current sentence is punctuation or blank,
        # append it to the end of the previous sentence.
        if (
            len(current_sentence) == 0 and len(sentences) != 0 and (token in punctuation or token == " ")  # noqa: S105
        ):
            sentences[-1].append(token)
        # Otherwise, append the current token to the current sentence.
        else:
            current_sentence.append(token)
            # Split the sentence in positions of punctuations.
            if token in punctuation:
                sentences.append(current_sentence)
                current_sentence = []
    # Assume the last few tokens are also a sentence
    if len(current_sentence) != 0:
        sentences.append(current_sentence)

    # 2. Merge short sentences.
    chunks = []
    current_chunk = []
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_tokens:
            current_chunk.extend(sentence)
        else:
            if len(current_chunk) > 0:
                chunks.append(current_chunk)
            current_chunk = sentence

    if len(current_chunk) > 0:
        chunks.append(current_chunk)

    return chunks


def chunk_tokens_dialog(tokens_list: list[str], max_tokens: int = 100):
    """Split dialog tokens into chunks at speaker-turn boundaries.

    Splits the token list at ``[S1]`` speaker-turn markers, then merges short
    dialog turns until each chunk contains at most ``max_tokens`` tokens.

    Args:
        tokens_list: The list of tokens to be split.
        max_tokens: Maximum number of tokens per chunk.

    Returns:
        A list of token-list chunks.
    """
    # 1. Split the tokens according to speaker-turn symbol [S1].
    dialogs = []
    current_dialog = []
    for token in tokens_list:
        if token == "[S1]":  # noqa: S105
            if len(current_dialog) != 0:
                dialogs.append(current_dialog)
            current_dialog = []
        current_dialog.append(token)
    # Assume the last few tokens are also a dialog
    if len(current_dialog) != 0:
        dialogs.append(current_dialog)

    # 2. Merge short dialogs.
    chunks = []
    current_chunk = []
    for dialog in dialogs:
        if len(current_chunk) + len(dialog) <= max_tokens:
            current_chunk.extend(dialog)
        else:
            if len(current_chunk) > 0:
                chunks.append(current_chunk)
            current_chunk = dialog

    if len(current_chunk) > 0:
        chunks.append(current_chunk)

    return chunks


def batchify_tokens(
    tokens_list: list[list[int]],
    max_duration: float,
    prompt_duration: float,
    token_duration: float,
):
    """Sort and group token sequences into duration-limited batches.

    Sequences are first sorted by length to minimise padding, then greedily
    grouped so that each batch's estimated audio duration stays within
    ``max_duration``.

    Args:
        tokens_list: A list of token sequences to batch.
        max_duration: Maximum allowed total duration (seconds) for each batch.
        prompt_duration: Estimated duration cost (seconds) per prompt in a
            batch.
        token_duration: Estimated duration (seconds) contributed by each
            individual token.

    Returns:
        A tuple ``(batches, index)`` where ``batches`` is a list of batches
        (each batch is a list of token sequences) and ``index`` is the list of
        original positions of each sequence, used to restore sequential order
        after parallel generation.
    """
    # Create index for each sentence
    indexed_tokens = list(enumerate(tokens_list))

    # Sort according to sentence length (for less padding)
    indexed_sorted_tokens = sorted(indexed_tokens, key=lambda x: len(x[1]))
    index = [indexed_sorted_tokens[i][0] for i in range(len(indexed_sorted_tokens))]
    sorted_tokens = [indexed_sorted_tokens[i][1] for i in range(len(indexed_sorted_tokens))]

    batches = []
    batch = []
    batch_size = 0  # Total number of tokens in current batch

    for tokens in sorted_tokens:
        # Calculate if adding current token sequence would exceed max duration
        # Formula considers: existing tokens' duration + existing
        # prompts' duration + new tokens' duration
        if batch_size * token_duration + len(batch) * prompt_duration + len(tokens) * token_duration <= max_duration:
            # Add to current batch if within duration limit
            batch.append(tokens)
            batch_size += len(tokens)
        else:
            # If exceeding limit, finalize current batch (if not empty)
            if len(batch) > 0:
                batches.append(batch)
            # Start new batch with current token sequence
            batch = [tokens]
            batch_size = len(tokens)

    # Add the last batch if it's not empty
    if len(batch) > 0:
        batches.append(batch)

    return batches, index


def cross_fade_concat(chunks: list[torch.Tensor], fade_duration: float = 0.1, sample_rate: int = 24000) -> torch.Tensor:
    """Concatenates audio chunks with cross-fading between consecutive chunks.

    Args:
        chunks: List of audio tensors, each with shape (C, T) where
                C = number of channel, T = time dimension (samples)
        fade_duration: Duration of cross-fade in seconds
        sample_rate: Audio sample rate in Hz

    Returns:
        Concatenated audio tensor with shape (N, T_total)
    """
    # Handle edge cases: empty input or single chunk
    if len(chunks) <= 1:
        return chunks[0] if chunks else torch.tensor([])

    # Calculate total fade samples from duration and sample rate
    fade_samples = int(fade_duration * sample_rate)

    # Use simple concatenation if fade duration is non-positive
    if fade_samples <= 0:
        return torch.cat(chunks, dim=-1)

    # Initialize final tensor with the first chunk
    final = chunks[0]

    # Iterate through remaining chunks to apply cross-fading
    for next_chunk in chunks[1:]:
        # Calculate safe fade length (cannot exceed either chunk's duration)
        k = min(fade_samples, final.shape[-1], next_chunk.shape[-1])

        # Fall back to simple concatenation if safe fade length is invalid
        if k <= 0:
            final = torch.cat([final, next_chunk], dim=-1)
            continue

        # Create fade curve (1 -> 0) with shape (1, k) for broadcasting
        fade = torch.linspace(1, 0, k, device=final.device)[None]

        # Concatenate three parts:
        # 1. Non-overlapping part of previous audio
        # 2. Cross-faded overlapping region
        # 3. Non-overlapping part of next audio
        final = torch.cat(
            [
                final[..., :-k],  # All samples except last k from previous
                final[..., -k:] * fade + next_chunk[..., :k] * (1 - fade),  # Cross-fade region
                next_chunk[..., k:],  # All samples except first k from next
            ],
            dim=-1,
        )

    return final


def add_punctuation(text: str):
    """Append a period to ``text`` if it does not already end with punctuation."""
    text = text.strip()
    if text[-1] not in punctuation:
        text += "."
    return text


def load_prompt_wav(prompt_wav: str, sampling_rate: int):
    """Load a waveform from disk and resample to the target rate if needed.

    Args:
        prompt_wav: Path to the prompt audio file.
        sampling_rate: Target sample rate in Hz.

    Returns:
        Audio tensor of shape (C, T) at the requested sample rate.
    """
    prompt_wav, prompt_sampling_rate = torchaudio.load(prompt_wav)

    if prompt_sampling_rate != sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=prompt_sampling_rate, new_freq=sampling_rate)
        prompt_wav = resampler(prompt_wav)
    return prompt_wav


def rms_norm(prompt_wav: torch.Tensor, target_rms: float):
    """Normalise a waveform's RMS level up to ``target_rms`` if it is below it.

    The waveform is only scaled upward; waveforms louder than ``target_rms``
    are returned unchanged.

    Args:
        prompt_wav: Audio tensor of shape (C, T).
        target_rms: Target RMS amplitude level.

    Returns:
        A tuple ``(normalised_wav, original_rms)`` where ``normalised_wav``
        has shape (C, T) and ``original_rms`` is the RMS of the input before
        normalisation (used to restore output volume).
    """
    prompt_rms = torch.sqrt(torch.mean(torch.square(prompt_wav)))
    if prompt_rms < target_rms:
        prompt_wav = prompt_wav * target_rms / prompt_rms
    return prompt_wav, prompt_rms


def remove_silence(
    audio: torch.Tensor,
    sampling_rate: int,
    only_edge: bool = False,
    trail_sil: float = 0,
):
    """Remove silence from a waveform tensor.

    By default, removes interior silences longer than 1 second and edge
    silences longer than 0.1 seconds.  When ``only_edge`` is ``True``, only
    edge silences are removed.

    Args:
        audio: Audio tensor of shape (C, T).
        sampling_rate: Sample rate of the audio in Hz.
        only_edge: If ``True``, skip interior silence removal and only trim
            leading and trailing silence.
        trail_sil: Duration in milliseconds of silence to append after
            trimming.

    Returns:
        Processed audio tensor of shape (C, T').
    """
    # Load audio file
    wave = tensor_to_audiosegment(audio, sampling_rate)

    if not only_edge:
        # Split audio using silences longer than 1 second
        non_silent_segs = split_on_silence(
            wave,
            min_silence_len=SILENCE_THRESHOLD_MS,  # Silences longer than 1 second (1000ms)
            silence_thresh=SILENCE_THRESHOLD_DB,
            keep_silence=SILENCE_THRESHOLD_MS,  # Keep 1.0 second of silence around segments
            seek_step=10,
        )

        # Concatenate all non-silent segments
        wave = AudioSegment.silent(duration=0)
        for seg in non_silent_segs:
            wave += seg

    # Remove silence longer than 0.1 seconds in the begining and ending of wave
    wave = remove_silence_edges(wave, 100, SILENCE_THRESHOLD_DB)

    # Add trailing silence to avoid leaking prompt to generated speech.
    wave = wave + AudioSegment.silent(duration=trail_sil)

    # Convert to PyTorch tensor
    return audiosegment_to_tensor(wave)


def remove_silence_edges(audio: AudioSegment, keep_silence: int = 100, silence_threshold: float = SILENCE_THRESHOLD_DB):
    """Trim leading and trailing silence from an AudioSegment.

    Args:
        audio: The AudioSegment to trim.
        keep_silence: Milliseconds of silence to retain at each edge after
            trimming.
        silence_threshold: dBFS threshold below which audio is considered
            silent.

    Returns:
        Trimmed AudioSegment with at most ``keep_silence`` ms of silence at
        each edge.
    """
    # Remove leading silence
    start_idx = detect_leading_silence(audio, silence_threshold=silence_threshold)
    start_idx = max(0, start_idx - keep_silence)
    audio = audio[start_idx:]

    # Remove trailing silence
    audio = audio.reverse()
    start_idx = detect_leading_silence(audio, silence_threshold=silence_threshold)
    start_idx = max(0, start_idx - keep_silence)
    audio = audio[start_idx:]
    audio = audio.reverse()

    return audio


def audiosegment_to_tensor(aseg):
    """Convert a pydub AudioSegment to a PyTorch audio tensor.

    Args:
        aseg: The AudioSegment to convert.

    Returns:
        Float32 tensor of shape (1, T) for mono or (C, T) for multi-channel
        audio, with values normalised to the range [-1, 1].
    """
    audio_data = np.array(aseg.get_array_of_samples())

    # Convert to float32 and normalize to [-1, 1] range
    audio_data = audio_data.astype(np.float32) / AUDIO_NORM_FACTOR

    # Handle channels
    if aseg.channels == 1:
        # Mono channel: add channel dimension (T) -> (1, T)
        tensor_data = torch.from_numpy(audio_data).unsqueeze(0)
    else:
        # Multi-channel: reshape to (C, T)
        tensor_data = torch.from_numpy(audio_data.reshape(-1, aseg.channels).T)

    return tensor_data


def tensor_to_audiosegment(tensor, sample_rate):
    """Convert a PyTorch audio tensor to a pydub AudioSegment.

    Args:
        tensor: Float32 tensor of shape (C, T) with values in [-1, 1], where
            C is the number of channels and T is the number of samples.
        sample_rate: Audio sample rate in Hz.

    Returns:
        AudioSegment with 16-bit PCM encoding at the given sample rate.
    """
    # Convert tensor to numpy array
    audio_np = tensor.cpu().numpy()

    # Add channel dimension if single channel
    if audio_np.ndim == 1:
        audio_np = audio_np[np.newaxis, :]

    # Convert to int16 type (common format for pydub)
    # Assumes tensor values are in [-1, 1] range as floating point
    audio_np = (audio_np * AUDIO_NORM_FACTOR).clip(-AUDIO_NORM_FACTOR, AUDIO_MAX_INT16).astype(np.int16)

    # Convert to byte stream
    # For multi-channel audio, pydub requires interleaved format
    # (e.g., left-right-left-right)
    if audio_np.shape[0] > 1:
        # Convert to interleaved format
        audio_np = audio_np.transpose(1, 0).flatten()
    audio_bytes = audio_np.tobytes()

    # Create AudioSegment
    audio_segment = AudioSegment(
        data=audio_bytes,
        sample_width=2,
        frame_rate=sample_rate,
        channels=tensor.shape[0],
    )

    return audio_segment


def merge_chunked_wavs(
    chunked_wavs: list[torch.Tensor],
    chunked_index: list[int] | None = None,
    remove_long_sil: bool = False,
    sampling_rate: int = 24000,
) -> torch.Tensor:
    """Merge a list of chunked audio tensors into a single waveform.

    When chunks were produced by a batching step that reordered them (e.g.,
    ``batchify_tokens``), pass the corresponding ``chunked_index`` so the
    chunks are sorted back into their original sequential order before
    concatenation.  When chunks were produced in sequential order (no
    reordering), leave ``chunked_index`` as ``None``.

    After reordering, chunks are joined with a 0.1-second cross-fade via
    ``cross_fade_concat``, and edge silences (plus, optionally, long interior
    silences) are removed via ``remove_silence``.

    Args:
        chunked_wavs: List of audio tensors, each with shape (C, T).
        chunked_index: Original sequential indices corresponding to each
            element of ``chunked_wavs``, as returned by ``batchify_tokens``.
            When ``None``, ``chunked_wavs`` is treated as already in order.
        remove_long_sil: If ``True``, also remove long silences in the
            interior of the merged audio (edge silences are always removed).
        sampling_rate: Audio sample rate in Hz.

    Returns:
        A single merged audio tensor with shape (C, T_total).
    """
    if chunked_index is not None:
        # Restore the original sequential order that was shuffled by batchify_tokens
        indexed_chunked_wavs = list(zip(chunked_index, chunked_wavs, strict=False))
        indexed_chunked_wavs.sort(key=lambda x: x[0])
        sequential_chunked_wavs = [wav for _, wav in indexed_chunked_wavs]
    else:
        sequential_chunked_wavs = chunked_wavs

    final_wav = cross_fade_concat(sequential_chunked_wavs, fade_duration=0.1, sample_rate=sampling_rate)
    final_wav = remove_silence(final_wav, sampling_rate, only_edge=(not remove_long_sil), trail_sil=0)
    return final_wav


# end zipvoice/utils/infer.py
