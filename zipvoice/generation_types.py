# start zipvoice/generation_types.py
"""Shared dataclasses for generation inputs and configuration.

These are kept in a dedicated module to avoid circular imports between
modeling_utils (which imports OnnxModel) and onnx_modeling (which needs
PromptContext / GenerationConfig).
"""

from dataclasses import dataclass

import torch

from zipvoice.constants import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_STEPS,
    DEFAULT_SPEED,
    DEFAULT_T_SHIFT,
    DEFAULT_TARGET_RMS,
)


@dataclass
class PromptContext:
    """Encoded voice prompt for use in generation.

    Attributes:
        tokens: Token ID sequences for the prompt transcription, one list per
            batch item.
        features_lens: Length of each prompt feature sequence, shape (B,).
        features: Extracted log-mel filterbank features for the prompt,
            shape (B, T, feat_dim).
        rms: RMS level of the original (pre-normalised) prompt waveform, used
            to restore the output volume.
    """

    tokens: list[list[int]]
    features_lens: torch.Tensor
    features: torch.Tensor
    rms: float | torch.Tensor


@dataclass
class GenerationConfig:
    """Parameters controlling speech generation.

    Attributes:
        num_step: Number of ODE solver steps.
        guidance_scale: Classifier-free guidance scale.
        speed: Speaking rate multiplier relative to the prompt.
        t_shift: Time-shift parameter for the ODE solver schedule.
        target_rms: Target RMS level used when normalising the output volume.
    """

    num_step: int = DEFAULT_NUM_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    speed: float = DEFAULT_SPEED
    t_shift: float = DEFAULT_T_SHIFT
    target_rms: float = DEFAULT_TARGET_RMS


# end zipvoice/generation_types.py
