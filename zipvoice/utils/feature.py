# start zipvoice/utils/feature.py
#!/usr/bin/env python3
"""Log-mel filterbank feature extraction for ZipVoice TTS training and inference."""
# Copyright         2024  Xiaomi Corp.        (authors: Han Zhu)
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

from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from lhotse.features.base import FeatureExtractor, register_extractor
from lhotse.utils import Seconds, compute_num_frames


@dataclass
class VocosFbankConfig:
    """Configuration for the VocosFbank feature extractor.

    Attributes:
        sampling_rate: Expected audio sample rate in Hz.
        n_mels: Number of mel filterbank channels.
        n_fft: FFT window size.
        hop_length: Hop size between consecutive STFT frames.
    """

    sampling_rate: int = 24000
    n_mels: int = 100
    n_fft: int = 1024
    hop_length: int = 256


@register_extractor
class VocosFbank(FeatureExtractor):
    """Log-mel filterbank feature extractor compatible with the Vocos vocoder.

    Wraps ``torchaudio.transforms.MelSpectrogram`` in a Lhotse
    ``FeatureExtractor`` so that features can be computed during both training
    data preparation and real-time inference.  Output values are the natural
    logarithm of the mel spectrogram magnitudes, matching the format expected
    by the Vocos decoder.
    """

    name = "VocosFbank"
    config_type = VocosFbankConfig

    def __init__(self, num_channels: int = 1):
        """Initialize the VocosFbank feature extractor.

        Args:
            num_channels: Number of audio channels (1 for mono, 2 for stereo).
        """
        config = VocosFbankConfig
        super().__init__(config=config)
        if num_channels not in (1, 2):
            msg = f"num_channels must be 1 or 2, got {num_channels}"
            raise ValueError(msg)
        self.num_channels = num_channels
        self.fbank = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sampling_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            center=True,
            power=1,
        )

    def _feature_fn(self, sample):
        """Compute log-mel spectrogram for a single audio sample tensor.

        Args:
            sample: Audio tensor of shape (C, T).

        Returns:
            Log-mel spectrogram tensor of shape (C, n_mels, T').
        """
        mel = self.fbank(sample)
        logmel = mel.clamp(min=1e-7).log()

        return logmel

    @property
    def device(self) -> str | torch.device:
        """Return the device used by the feature extractor."""
        return self.config.device

    def feature_dim(self, _sampling_rate: int) -> int:
        """Return the feature dimension (number of mel bins).

        Args:
            _sampling_rate: Audio sample rate (unused; present for API compat).

        Returns:
            Number of mel filterbank channels.
        """
        return self.config.n_mels

    def extract(  # noqa: C901, PLR0912
        self,
        samples: np.ndarray | torch.Tensor,
        sampling_rate: int,
    ) -> np.ndarray | torch.Tensor:
        """Extract log-mel filterbank features from audio samples.

        Args:
            samples: Audio samples as a 1-D array (mono) or 2-D array of
                shape (C, T). NumPy arrays are converted to tensors internally
                and the result is returned as a NumPy array; tensors are
                returned as tensors.
            sampling_rate: Sample rate of the input audio. Must match
                ``config.sampling_rate``.

        Returns:
            Log-mel features of shape (T, n_mels) for mono or
            (T, 2 * n_mels) for stereo input.

        Raises:
            ValueError: If ``sampling_rate`` does not match the extractor's
                expected rate, if ``samples`` is not 1-D or 2-D, or if a
                stereo extractor receives mono input (or vice versa).
        """
        # Check for sampling rate compatibility.
        expected_sr = self.config.sampling_rate
        if sampling_rate != expected_sr:
            msg = f"Mismatched sampling rate: extractor expects {expected_sr}, got {sampling_rate}"
            raise ValueError(msg)
        is_numpy = False
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
            is_numpy = True

        if len(samples.shape) == 1:
            samples = samples.unsqueeze(0)
        else:
            if samples.ndim != 2:
                msg = f"Expected 2D samples array, got shape {samples.shape}"
                raise ValueError(msg)

        if self.num_channels == 1:
            if samples.shape[0] == 2:
                samples = samples.mean(dim=0, keepdims=True)
        else:
            if samples.shape[0] != 2:
                msg = f"Expected 2-channel (stereo) samples, got shape {samples.shape}"
                raise ValueError(msg)

        mel = self._feature_fn(samples)
        # (1, n_mels, time) or (2, n_mels, time)
        mel = mel.reshape(-1, mel.shape[-1]).t()
        # (time, n_mels) or (time, 2 * n_mels)

        num_frames = compute_num_frames(samples.shape[1] / sampling_rate, self.frame_shift, sampling_rate)

        if mel.shape[0] > num_frames:
            mel = mel[:num_frames]
        elif mel.shape[0] < num_frames:
            mel = mel.unsqueeze(0)
            mel = torch.nn.functional.pad(mel, (0, 0, 0, num_frames - mel.shape[1]), mode="replicate").squeeze(0)

        if is_numpy:
            return mel.cpu().numpy()
        else:
            return mel

    @property
    def frame_shift(self) -> Seconds:
        """Frame shift duration in seconds between consecutive feature frames."""
        return self.config.hop_length / self.config.sampling_rate


# end zipvoice/utils/feature.py
