# start tests/test_generation_types.py
"""Unit tests for zipvoice.generation_types module.

Tests cover:
- GenerationConfig default values match the shared constants.
- GenerationConfig accepts custom values.
- GenerationConfig is a proper dataclass.
- PromptContext can be instantiated with mock torch tensors.
- PromptContext.rms accepts both float and torch.Tensor.
"""

import dataclasses

import pytest
import torch

from zipvoice.constants import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_STEPS,
    DEFAULT_SPEED,
    DEFAULT_T_SHIFT,
    DEFAULT_TARGET_RMS,
)
from zipvoice.generation_types import GenerationConfig, PromptContext


# ---------------------------------------------------------------------------
# GenerationConfig tests
# ---------------------------------------------------------------------------


class TestGenerationConfig:
    """Tests for the GenerationConfig dataclass."""

    def test_default_num_step(self) -> None:
        """GenerationConfig.num_step defaults to DEFAULT_NUM_STEPS."""
        cfg = GenerationConfig()
        assert cfg.num_step == DEFAULT_NUM_STEPS

    def test_default_guidance_scale(self) -> None:
        """GenerationConfig.guidance_scale defaults to DEFAULT_GUIDANCE_SCALE."""
        cfg = GenerationConfig()
        assert cfg.guidance_scale == pytest.approx(DEFAULT_GUIDANCE_SCALE)

    def test_default_speed(self) -> None:
        """GenerationConfig.speed defaults to DEFAULT_SPEED."""
        cfg = GenerationConfig()
        assert cfg.speed == pytest.approx(DEFAULT_SPEED)

    def test_default_t_shift(self) -> None:
        """GenerationConfig.t_shift defaults to DEFAULT_T_SHIFT."""
        cfg = GenerationConfig()
        assert cfg.t_shift == pytest.approx(DEFAULT_T_SHIFT)

    def test_default_target_rms(self) -> None:
        """GenerationConfig.target_rms defaults to DEFAULT_TARGET_RMS."""
        cfg = GenerationConfig()
        assert cfg.target_rms == pytest.approx(DEFAULT_TARGET_RMS)

    def test_custom_num_step_and_speed(self) -> None:
        """GenerationConfig accepts custom num_step and speed values."""
        cfg = GenerationConfig(num_step=10, speed=1.5)

        assert cfg.num_step == 10
        assert cfg.speed == pytest.approx(1.5)

    def test_all_defaults_match_constants(self) -> None:
        """All GenerationConfig defaults exactly match their constant values."""
        cfg = GenerationConfig()

        assert cfg.num_step == DEFAULT_NUM_STEPS
        assert cfg.guidance_scale == pytest.approx(DEFAULT_GUIDANCE_SCALE)
        assert cfg.speed == pytest.approx(DEFAULT_SPEED)
        assert cfg.t_shift == pytest.approx(DEFAULT_T_SHIFT)
        assert cfg.target_rms == pytest.approx(DEFAULT_TARGET_RMS)

    def test_is_dataclass(self) -> None:
        """GenerationConfig is a proper dataclass."""
        assert dataclasses.is_dataclass(GenerationConfig)

    def test_field_names(self) -> None:
        """GenerationConfig has the expected field names."""
        field_names = {f.name for f in dataclasses.fields(GenerationConfig)}
        expected = {"num_step", "guidance_scale", "speed", "t_shift", "target_rms"}
        assert field_names == expected

    def test_equality(self) -> None:
        """Two GenerationConfig instances with the same values are equal."""
        a = GenerationConfig(num_step=8, speed=1.2)
        b = GenerationConfig(num_step=8, speed=1.2)
        assert a == b

    def test_inequality(self) -> None:
        """Two GenerationConfig instances with different values are not equal."""
        a = GenerationConfig(num_step=4)
        b = GenerationConfig(num_step=8)
        assert a != b

    def test_full_custom_construction(self) -> None:
        """GenerationConfig can be constructed with all fields overridden."""
        cfg = GenerationConfig(
            num_step=20,
            guidance_scale=5.0,
            speed=0.9,
            t_shift=0.3,
            target_rms=0.2,
        )

        assert cfg.num_step == 20
        assert cfg.guidance_scale == pytest.approx(5.0)
        assert cfg.speed == pytest.approx(0.9)
        assert cfg.t_shift == pytest.approx(0.3)
        assert cfg.target_rms == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# PromptContext tests
# ---------------------------------------------------------------------------


class TestPromptContext:
    """Tests for the PromptContext dataclass."""

    def _make_prompt_context(self, rms: float | torch.Tensor) -> PromptContext:
        """Build a minimal PromptContext for testing.

        Args:
            rms: RMS value to set; may be a float or a torch.Tensor.

        Returns:
            A PromptContext populated with small mock tensors.
        """
        tokens: list[list[int]] = [[1, 2, 3]]
        features_lens = torch.tensor([10])
        features = torch.zeros(1, 10, 80)
        return PromptContext(
            tokens=tokens,
            features_lens=features_lens,
            features=features,
            rms=rms,
        )

    def test_instantiation_with_float_rms(self) -> None:
        """PromptContext can be instantiated when rms is a float."""
        ctx = self._make_prompt_context(rms=0.05)

        assert ctx.rms == pytest.approx(0.05)

    def test_instantiation_with_tensor_rms(self) -> None:
        """PromptContext can be instantiated when rms is a torch.Tensor."""
        rms_tensor = torch.tensor(0.07)
        ctx = self._make_prompt_context(rms=rms_tensor)

        assert isinstance(ctx.rms, torch.Tensor)

    def test_tokens_stored_correctly(self) -> None:
        """PromptContext stores the tokens list-of-lists as provided."""
        ctx = self._make_prompt_context(rms=0.1)

        assert ctx.tokens == [[1, 2, 3]]

    def test_features_lens_is_tensor(self) -> None:
        """PromptContext.features_lens is a torch.Tensor."""
        ctx = self._make_prompt_context(rms=0.1)

        assert isinstance(ctx.features_lens, torch.Tensor)

    def test_features_is_tensor(self) -> None:
        """PromptContext.features is a torch.Tensor."""
        ctx = self._make_prompt_context(rms=0.1)

        assert isinstance(ctx.features, torch.Tensor)

    def test_features_shape(self) -> None:
        """PromptContext.features has the expected (B, T, feat_dim) shape."""
        ctx = self._make_prompt_context(rms=0.1)

        assert ctx.features.shape == (1, 10, 80)

    def test_is_dataclass(self) -> None:
        """PromptContext is a proper dataclass."""
        assert dataclasses.is_dataclass(PromptContext)

    def test_field_names(self) -> None:
        """PromptContext has the expected field names."""
        field_names = {f.name for f in dataclasses.fields(PromptContext)}
        expected = {"tokens", "features_lens", "features", "rms"}
        assert field_names == expected

    def test_multiple_batch_items(self) -> None:
        """PromptContext supports batch sizes greater than one."""
        tokens: list[list[int]] = [[1, 2], [3, 4, 5]]
        features_lens = torch.tensor([8, 10])
        features = torch.zeros(2, 10, 80)
        ctx = PromptContext(
            tokens=tokens,
            features_lens=features_lens,
            features=features,
            rms=0.05,
        )

        assert len(ctx.tokens) == 2
        assert ctx.features.shape[0] == 2
# end tests/test_generation_types.py
