# start zipvoice/config.py
"""LuxTTS runtime configuration loaded from config.yml.

Uses pydantic-settings for validation. The config file is read once at import
time; override individual values by passing keyword arguments to LuxTTS().
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from zipvoice.constants import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_STEPS,
    DEFAULT_SPEED,
    DEFAULT_T_SHIFT,
    DEFAULT_TARGET_RMS,
)


class ModelConfig(BaseModel):
    """Model loading settings."""

    model_config = ConfigDict(strict=True, extra="forbid")

    repo_id: str = "YatharthS/LuxTTS"
    device: str = "auto"
    threads: int = Field(default=4, ge=1, le=64)


class GenerationConfig(BaseModel):
    """Default generation parameters (settings layer, not runtime dataclass).

    Note: ``zipvoice.generation_types.GenerationConfig`` is a separate dataclass
    used at runtime to pass parameters into the model. This class holds the
    *default* values read from ``config.yml`` or environment variables.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    num_steps: int = Field(default=DEFAULT_NUM_STEPS, ge=1, le=100)
    guidance_scale: float = Field(default=DEFAULT_GUIDANCE_SCALE, ge=0.0)
    speed: float = Field(default=DEFAULT_SPEED, ge=0.1, le=5.0)
    t_shift: float = Field(default=DEFAULT_T_SHIFT, ge=0.0, le=1.0)
    target_rms: float = Field(default=DEFAULT_TARGET_RMS, ge=0.0)


class AudioConfig(BaseModel):
    """Audio processing settings."""

    model_config = ConfigDict(strict=True, extra="forbid")

    freq_range: int = 12000
    default_rms: float = 0.001
    default_prompt_duration: int = 5


class LuxTTSSettings(BaseSettings):
    """Top-level settings for LuxTTS, loadable from config.yml."""

    model: ModelConfig = ModelConfig()
    generation: GenerationConfig = GenerationConfig()
    audio: AudioConfig = AudioConfig()

    model_config = SettingsConfigDict(extra="forbid")


def load_settings(config_path: Path | None = None) -> LuxTTSSettings:
    """Load settings from config.yml, falling back to defaults.

    Args:
        config_path: Path to config.yml. Defaults to project-root config.yml.

    Returns:
        Validated LuxTTSSettings instance.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yml"

    if not config_path.exists():
        return LuxTTSSettings()

    try:
        import yaml

        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return LuxTTSSettings(**raw)
    except ModuleNotFoundError:
        return LuxTTSSettings()


# end zipvoice/config.py
