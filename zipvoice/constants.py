# start zipvoice/constants.py
"""Shared constants for LuxTTS.

Single source of truth for numeric constants and type aliases used
across multiple modules.
"""

# Audio sample rates
SAMPLE_RATE_OUTPUT: int = 24000
SAMPLE_RATE_INPUT: int = 16000

# Feature extraction defaults
DEFAULT_FEAT_SCALE: float = 0.1
DEFAULT_SPEED_FACTOR: float = 1.3

# HuggingFace repository for ZipVoice upstream models
HUGGINGFACE_REPO_ZIPVOICE: str = "k2-fsa/ZipVoice"

# Generation defaults (single source of truth for both config.py and generation_types.py)
DEFAULT_NUM_STEPS: int = 4
DEFAULT_GUIDANCE_SCALE: float = 3.0
DEFAULT_SPEED: float = 1.0
DEFAULT_T_SHIFT: float = 0.5
DEFAULT_TARGET_RMS: float = 0.1

# end zipvoice/constants.py
