# start tests/test_config.py
"""Unit tests for zipvoice.config module.

Tests cover:
- load_settings() returning defaults when no config.yml exists.
- load_settings() applying overrides from a real YAML file.
- load_settings() with an explicit nonexistent path returning defaults.
- ModelConfig, GenerationConfig, and AudioConfig default values.
- ModelConfig validation constraints (extra fields, threads bounds).
"""

import pytest
from pathlib import Path
from pydantic import ValidationError

from zipvoice.config import (
    AudioConfig,
    GenerationConfig,
    LuxTTSSettings,
    ModelConfig,
    load_settings,
)


# ---------------------------------------------------------------------------
# load_settings tests
# ---------------------------------------------------------------------------


class TestLoadSettings:
    """Tests for the load_settings() function."""

    def test_defaults_when_no_config_file(self, tmp_path: Path) -> None:
        """load_settings() returns defaults when the config file does not exist.

        Args:
            tmp_path: Pytest temporary directory fixture.
        """
        nonexistent = tmp_path / "missing.yml"
        settings = load_settings(config_path=nonexistent)

        assert isinstance(settings, LuxTTSSettings)
        assert settings.model.repo_id == "YatharthS/LuxTTS"
        assert settings.model.device == "auto"
        assert settings.model.threads == 4

    def test_defaults_when_path_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """load_settings() with path=None falls back to defaults when project config.yml absent.

        This patches the expected default path to a nonexistent location so the
        test is hermetic regardless of what is in the repo root config.yml.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
        """
        # Point the default path resolution to something that doesn't exist
        import zipvoice.config as cfg_module

        fake_parent = Path("/nonexistent/path/that/does/not/exist")
        monkeypatch.setattr(cfg_module.Path, "parent", property(lambda self: fake_parent))

        # When the resolved path doesn't exist, defaults are returned
        settings = load_settings(config_path=Path("/nonexistent/config.yml"))
        assert isinstance(settings, LuxTTSSettings)

    def test_overrides_from_yaml(self, tmp_path: Path) -> None:
        """load_settings() applies values from a real YAML file.

        Args:
            tmp_path: Pytest temporary directory fixture.
        """
        config_file = tmp_path / "config.yml"
        config_file.write_text(
            "model:\n  repo_id: myorg/MyModel\n  device: cpu\n  threads: 8\n",
            encoding="utf-8",
        )

        settings = load_settings(config_path=config_file)

        assert settings.model.repo_id == "myorg/MyModel"
        assert settings.model.device == "cpu"
        assert settings.model.threads == 8

    def test_generation_overrides_from_yaml(self, tmp_path: Path) -> None:
        """load_settings() applies generation section overrides from YAML.

        Args:
            tmp_path: Pytest temporary directory fixture.
        """
        config_file = tmp_path / "config.yml"
        config_file.write_text(
            "generation:\n  num_steps: 10\n  guidance_scale: 5.0\n  speed: 1.2\n",
            encoding="utf-8",
        )

        settings = load_settings(config_path=config_file)

        assert settings.generation.num_steps == 10
        assert settings.generation.guidance_scale == pytest.approx(5.0)
        assert settings.generation.speed == pytest.approx(1.2)

    def test_nonexistent_explicit_path_returns_defaults(self, tmp_path: Path) -> None:
        """load_settings() with a nonexistent explicit path returns defaults.

        Args:
            tmp_path: Pytest temporary directory fixture.
        """
        settings = load_settings(config_path=tmp_path / "no_such_file.yml")
        assert settings.model.repo_id == "YatharthS/LuxTTS"


# ---------------------------------------------------------------------------
# ModelConfig tests
# ---------------------------------------------------------------------------


class TestModelConfig:
    """Tests for the ModelConfig Pydantic model."""

    def test_default_values(self) -> None:
        """ModelConfig has correct default values."""
        cfg = ModelConfig()

        assert cfg.repo_id == "YatharthS/LuxTTS"
        assert cfg.device == "auto"
        assert cfg.threads == 4

    def test_custom_values(self) -> None:
        """ModelConfig accepts valid custom values."""
        cfg = ModelConfig(repo_id="org/Repo", device="cuda", threads=16)

        assert cfg.repo_id == "org/Repo"
        assert cfg.device == "cuda"
        assert cfg.threads == 16

    def test_extra_field_raises_validation_error(self) -> None:
        """ModelConfig with extra='forbid' rejects unknown fields."""
        with pytest.raises(ValidationError):
            ModelConfig(repo_id="org/Repo", unknown_field="oops")  # type: ignore[call-arg]

    def test_threads_below_minimum_raises(self) -> None:
        """ModelConfig.threads=0 violates ge=1 constraint."""
        with pytest.raises(ValidationError):
            ModelConfig(threads=0)

    def test_threads_above_maximum_raises(self) -> None:
        """ModelConfig.threads=65 violates le=64 constraint."""
        with pytest.raises(ValidationError):
            ModelConfig(threads=65)

    def test_threads_at_minimum_boundary(self) -> None:
        """ModelConfig.threads=1 is at the minimum allowed boundary."""
        cfg = ModelConfig(threads=1)
        assert cfg.threads == 1

    def test_threads_at_maximum_boundary(self) -> None:
        """ModelConfig.threads=64 is at the maximum allowed boundary."""
        cfg = ModelConfig(threads=64)
        assert cfg.threads == 64


# ---------------------------------------------------------------------------
# GenerationConfig (config layer) tests
# ---------------------------------------------------------------------------


class TestGenerationConfigSettings:
    """Tests for the config-layer GenerationConfig Pydantic model."""

    def test_default_values(self) -> None:
        """GenerationConfig defaults match the shared constants."""
        from zipvoice.constants import (
            DEFAULT_GUIDANCE_SCALE,
            DEFAULT_NUM_STEPS,
            DEFAULT_SPEED,
            DEFAULT_T_SHIFT,
            DEFAULT_TARGET_RMS,
        )

        cfg = GenerationConfig()

        assert cfg.num_steps == DEFAULT_NUM_STEPS
        assert cfg.guidance_scale == pytest.approx(DEFAULT_GUIDANCE_SCALE)
        assert cfg.speed == pytest.approx(DEFAULT_SPEED)
        assert cfg.t_shift == pytest.approx(DEFAULT_T_SHIFT)
        assert cfg.target_rms == pytest.approx(DEFAULT_TARGET_RMS)

    def test_custom_values(self) -> None:
        """GenerationConfig accepts valid custom values."""
        cfg = GenerationConfig(num_steps=10, guidance_scale=2.5, speed=1.2)

        assert cfg.num_steps == 10
        assert cfg.guidance_scale == pytest.approx(2.5)
        assert cfg.speed == pytest.approx(1.2)

    def test_extra_field_raises_validation_error(self) -> None:
        """GenerationConfig rejects unknown fields."""
        with pytest.raises(ValidationError):
            GenerationConfig(unknown="bad")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# AudioConfig tests
# ---------------------------------------------------------------------------


class TestAudioConfig:
    """Tests for the AudioConfig Pydantic model."""

    def test_default_values(self) -> None:
        """AudioConfig has correct default values."""
        cfg = AudioConfig()

        assert cfg.freq_range == 12000
        assert cfg.default_rms == pytest.approx(0.001)
        assert cfg.default_prompt_duration == 5

    def test_custom_values(self) -> None:
        """AudioConfig accepts valid custom values."""
        cfg = AudioConfig(freq_range=8000, default_rms=0.01, default_prompt_duration=10)

        assert cfg.freq_range == 8000
        assert cfg.default_rms == pytest.approx(0.01)
        assert cfg.default_prompt_duration == 10

    def test_extra_field_raises_validation_error(self) -> None:
        """AudioConfig rejects unknown fields."""
        with pytest.raises(ValidationError):
            AudioConfig(unsupported_key=True)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# LuxTTSSettings tests
# ---------------------------------------------------------------------------


class TestLuxTTSSettings:
    """Tests for the top-level LuxTTSSettings model."""

    def test_default_construction(self) -> None:
        """LuxTTSSettings constructs with nested default sub-models."""
        settings = LuxTTSSettings()

        assert isinstance(settings.model, ModelConfig)
        assert isinstance(settings.generation, GenerationConfig)
        assert isinstance(settings.audio, AudioConfig)

    def test_nested_override(self) -> None:
        """LuxTTSSettings accepts nested dict overrides for sub-models."""
        settings = LuxTTSSettings(model={"repo_id": "x/y", "device": "cpu", "threads": 2})  # type: ignore[arg-type]

        assert settings.model.repo_id == "x/y"
        assert settings.model.device == "cpu"
        assert settings.model.threads == 2
# end tests/test_config.py
