# start tests/test_modeling_utils.py
"""Unit tests for zipvoice.modeling_utils.

Tests cover:
- LuxTTSConfig dataclass creation and default values (pure dataclass, no deps).
- load_models path-selection logic, verified by asserting that snapshot_download
  is NOT called when an explicit model_path is provided.

Heavy model-loading functions (ZipVoiceDistill, Vocos, OnnxModel, etc.) are
mocked out so that no GPU, disk weights, or network access is required.
"""

from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest

# All heavy third-party dependencies required by modeling_utils are pre-mocked
# in conftest.py, which pytest loads before any test module is collected.


# ---------------------------------------------------------------------------
# LuxTTSConfig tests
# ---------------------------------------------------------------------------


class TestLuxTTSConfig:
    """Tests for the LuxTTSConfig dataclass."""

    def test_import_succeeds(self) -> None:
        """LuxTTSConfig can be imported without errors."""
        from zipvoice.modeling_utils import LuxTTSConfig  # noqa: F401
        assert LuxTTSConfig is not None

    def test_default_values(self) -> None:
        """LuxTTSConfig initialises with the documented default values."""
        from zipvoice.modeling_utils import LuxTTSConfig

        cfg = LuxTTSConfig()
        assert cfg.model_dir is None
        assert cfg.checkpoint_name == "model.pt"
        assert cfg.vocoder_path is None
        assert cfg.trt_engine_path is None
        assert cfg.tokenizer == "emilia"
        assert cfg.lang == "en-us"

    def test_custom_values(self) -> None:
        """LuxTTSConfig accepts and stores custom values for all fields."""
        from zipvoice.modeling_utils import LuxTTSConfig

        cfg = LuxTTSConfig(
            model_dir="/tmp/model",
            checkpoint_name="ckpt.pt",
            vocoder_path="/tmp/vocoder",
            trt_engine_path="/tmp/engine",
            tokenizer="espeak",
            lang="fr-fr",
        )
        assert cfg.model_dir == "/tmp/model"
        assert cfg.checkpoint_name == "ckpt.pt"
        assert cfg.vocoder_path == "/tmp/vocoder"
        assert cfg.trt_engine_path == "/tmp/engine"
        assert cfg.tokenizer == "espeak"
        assert cfg.lang == "fr-fr"

    def test_is_dataclass(self) -> None:
        """LuxTTSConfig is a proper dataclass with the expected fields."""
        from zipvoice.modeling_utils import LuxTTSConfig
        import dataclasses

        assert dataclasses.is_dataclass(LuxTTSConfig)
        field_names = {f.name for f in fields(LuxTTSConfig)}
        expected = {"model_dir", "checkpoint_name", "vocoder_path", "trt_engine_path", "tokenizer", "lang"}
        assert expected == field_names

    def test_equality(self) -> None:
        """Two LuxTTSConfig instances with identical values are equal."""
        from zipvoice.modeling_utils import LuxTTSConfig

        a = LuxTTSConfig(model_dir="/m", lang="en-us")
        b = LuxTTSConfig(model_dir="/m", lang="en-us")
        assert a == b

    def test_constants(self) -> None:
        """Module-level constants have expected values."""
        from zipvoice.modeling_utils import (
            SAMPLE_RATE_OUTPUT,
            SAMPLE_RATE_INPUT,
            DEFAULT_FEAT_SCALE,
            DEFAULT_SPEED_FACTOR,
        )
        assert SAMPLE_RATE_OUTPUT == 24000
        assert SAMPLE_RATE_INPUT == 16000
        assert DEFAULT_FEAT_SCALE == pytest.approx(0.1)
        assert DEFAULT_SPEED_FACTOR == pytest.approx(1.3)


# ---------------------------------------------------------------------------
# load_models_cpu path-selection tests
# ---------------------------------------------------------------------------


class TestLoadModelsPathSelection:
    """Tests that load_models respects a provided model_path.

    When a model_path is explicitly given, snapshot_download must NOT be called.
    This verifies the bug-fix where downloads are skipped for local paths.
    """

    def _build_mocks(self, tmp_path):
        """Set up the minimal mock objects needed for load_models to run.

        Args:
            tmp_path: pytest tmp_path fixture providing a writable directory.

        Returns:
            A dict of mock objects and the path string to pass as model_path.
        """
        import orjson

        # Create a minimal directory with the files load_models expects
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "tokens.txt").write_text("_\t0\na\t1\n", encoding="utf-8")

        # Minimal config.json with the structure load_models reads
        config = {
            "feature": {"sampling_rate": 24000},
            "model": {},
        }
        (model_dir / "config.json").write_bytes(orjson.dumps(config))

        return str(model_dir)

    def test_snapshot_download_not_called_when_path_provided(self, tmp_path) -> None:
        """snapshot_download is not invoked when model_path is explicitly set.

        Args:
            tmp_path: Pytest temporary directory fixture.
        """
        model_dir = self._build_mocks(tmp_path)

        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 2
        mock_tokenizer.pad_id = 0

        mock_model = MagicMock()
        mock_vocos = MagicMock()
        mock_feature_extractor = MagicMock()
        mock_transcriber = MagicMock()

        with (
            patch("zipvoice.modeling_utils.snapshot_download") as mock_dl,
            patch("zipvoice.modeling_utils.EmiliaTokenizer", return_value=mock_tokenizer),
            patch("zipvoice.modeling_utils.pipeline", return_value=mock_transcriber),
            patch("zipvoice.modeling_utils.OnnxModel", return_value=mock_model),
            patch("zipvoice.modeling_utils.VocosFbank", return_value=mock_feature_extractor),
            patch("zipvoice.modeling_utils.Vocos") as mock_vocos_cls,
            patch("zipvoice.modeling_utils.parametrize"),
            patch("zipvoice.modeling_utils.torch.load", return_value={}),
        ):
            mock_vocos_cls.from_hparams.return_value = mock_vocos

            from zipvoice.modeling_utils import load_models
            load_models(model_path=model_dir, device="cpu")

            mock_dl.assert_not_called()

    def test_snapshot_download_called_when_no_path(self, tmp_path) -> None:
        """snapshot_download IS called when model_path is None (default).

        Args:
            tmp_path: Pytest temporary directory fixture.
        """
        import orjson

        # Make the download return a valid directory
        model_dir = tmp_path / "downloaded"
        model_dir.mkdir()
        (model_dir / "tokens.txt").write_text("_\t0\na\t1\n", encoding="utf-8")
        config = {"feature": {"sampling_rate": 24000}, "model": {}}
        (model_dir / "config.json").write_bytes(orjson.dumps(config))

        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 2
        mock_tokenizer.pad_id = 0

        mock_model = MagicMock()
        mock_vocos = MagicMock()
        mock_feature_extractor = MagicMock()
        mock_transcriber = MagicMock()

        with (
            patch("zipvoice.modeling_utils.snapshot_download", return_value=str(model_dir)) as mock_dl,
            patch("zipvoice.modeling_utils.EmiliaTokenizer", return_value=mock_tokenizer),
            patch("zipvoice.modeling_utils.pipeline", return_value=mock_transcriber),
            patch("zipvoice.modeling_utils.OnnxModel", return_value=mock_model),
            patch("zipvoice.modeling_utils.VocosFbank", return_value=mock_feature_extractor),
            patch("zipvoice.modeling_utils.Vocos") as mock_vocos_cls,
            patch("zipvoice.modeling_utils.parametrize"),
            patch("zipvoice.modeling_utils.torch.load", return_value={}),
        ):
            mock_vocos_cls.from_hparams.return_value = mock_vocos

            from zipvoice.modeling_utils import load_models
            load_models(model_path=None, device="cpu")

            mock_dl.assert_called_once_with("YatharthS/LuxTTS")


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for zipvoice.constants module values."""

    def test_sample_rate_output(self) -> None:
        """SAMPLE_RATE_OUTPUT is 24000 Hz."""
        from zipvoice.constants import SAMPLE_RATE_OUTPUT

        assert SAMPLE_RATE_OUTPUT == 24000

    def test_sample_rate_input(self) -> None:
        """SAMPLE_RATE_INPUT is 16000 Hz."""
        from zipvoice.constants import SAMPLE_RATE_INPUT

        assert SAMPLE_RATE_INPUT == 16000

    def test_default_feat_scale(self) -> None:
        """DEFAULT_FEAT_SCALE is approximately 0.1."""
        from zipvoice.constants import DEFAULT_FEAT_SCALE

        assert DEFAULT_FEAT_SCALE == pytest.approx(0.1)

    def test_default_speed_factor(self) -> None:
        """DEFAULT_SPEED_FACTOR is approximately 1.3."""
        from zipvoice.constants import DEFAULT_SPEED_FACTOR

        assert DEFAULT_SPEED_FACTOR == pytest.approx(1.3)

    def test_default_num_steps(self) -> None:
        """DEFAULT_NUM_STEPS is 4."""
        from zipvoice.constants import DEFAULT_NUM_STEPS

        assert DEFAULT_NUM_STEPS == 4

    def test_default_guidance_scale(self) -> None:
        """DEFAULT_GUIDANCE_SCALE is approximately 3.0."""
        from zipvoice.constants import DEFAULT_GUIDANCE_SCALE

        assert DEFAULT_GUIDANCE_SCALE == pytest.approx(3.0)

    def test_default_speed(self) -> None:
        """DEFAULT_SPEED is approximately 1.0."""
        from zipvoice.constants import DEFAULT_SPEED

        assert DEFAULT_SPEED == pytest.approx(1.0)

    def test_default_t_shift(self) -> None:
        """DEFAULT_T_SHIFT is approximately 0.5."""
        from zipvoice.constants import DEFAULT_T_SHIFT

        assert DEFAULT_T_SHIFT == pytest.approx(0.5)

    def test_default_target_rms(self) -> None:
        """DEFAULT_TARGET_RMS is approximately 0.1."""
        from zipvoice.constants import DEFAULT_TARGET_RMS

        assert DEFAULT_TARGET_RMS == pytest.approx(0.1)

    def test_huggingface_repo_zipvoice(self) -> None:
        """HUGGINGFACE_REPO_ZIPVOICE is the k2-fsa ZipVoice repo string."""
        from zipvoice.constants import HUGGINGFACE_REPO_ZIPVOICE

        assert HUGGINGFACE_REPO_ZIPVOICE == "k2-fsa/ZipVoice"

    def test_all_constants_importable(self) -> None:
        """All expected constants can be imported from zipvoice.constants."""
        from zipvoice.constants import (  # noqa: F401
            DEFAULT_FEAT_SCALE,
            DEFAULT_GUIDANCE_SCALE,
            DEFAULT_NUM_STEPS,
            DEFAULT_SPEED,
            DEFAULT_SPEED_FACTOR,
            DEFAULT_T_SHIFT,
            DEFAULT_TARGET_RMS,
            HUGGINGFACE_REPO_ZIPVOICE,
            SAMPLE_RATE_INPUT,
            SAMPLE_RATE_OUTPUT,
        )

        assert True  # import itself is the assertion
# end tests/test_modeling_utils.py
