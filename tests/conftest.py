# start tests/conftest.py
"""Shared pytest fixtures for LuxTTS test suite."""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest


def _make_mock_piper_phonemize() -> ModuleType:
    """Create a minimal mock for the piper_phonemize module.

    Returns:
        A mock module that exposes phonemize_espeak as a no-op.
    """
    mock_module = MagicMock(spec=ModuleType)
    mock_module.phonemize_espeak = MagicMock(return_value=[])
    return mock_module


def _ensure_mock(module_name: str) -> None:
    """Insert a MagicMock into sys.modules if the module is not already present.

    The mock is given a ``__spec__`` attribute (a real ``types.ModuleType``
    object) so that code using ``importlib.util.find_spec`` — e.g. the
    ``transformers`` library checking for optional packages — does not raise
    a ``ValueError`` when querying whether the package is available.

    Args:
        module_name: Dotted module path to mock.
    """
    import types as _types

    if module_name not in sys.modules:
        mock = MagicMock()
        mock.__spec__ = _types.ModuleType(module_name)
        sys.modules[module_name] = mock


# ---------------------------------------------------------------------------
# Inject mocks before any project module is imported.
# conftest.py is loaded at collection time, before test files import anything.
# ---------------------------------------------------------------------------

# piper_phonemize — optional binary not available in test environments
if "piper_phonemize" not in sys.modules:
    sys.modules["piper_phonemize"] = _make_mock_piper_phonemize()

# Modules that tokenizer.py (and its transitive imports) need at import time.
# normalizer.py (imported by tokenizer.py) requires cn2an and inflect.
# We mock the entire zipvoice.tokenizer.normalizer module so that tokenizer.py
# can be imported without those packages being installed.
_TOKENIZER_DEPS = [
    "jieba",
    "lhotse",
    "pypinyin",
    "pypinyin.contrib",
    "pypinyin.contrib.tone_convert",
    "cn2an",
    "inflect",
    "zipvoice.tokenizer.normalizer",
]
for _mod in _TOKENIZER_DEPS:
    _ensure_mock(_mod)

# Modules that infer.py needs at import time
_INFER_DEPS = [
    "torchaudio",
    "pydub",
    "pydub.silence",
]
for _mod in _INFER_DEPS:
    _ensure_mock(_mod)

# Modules that modeling_utils.py needs at import time (beyond torch itself,
# which is a real package in the test environment).
_MODELING_DEPS = [
    "safetensors",
    "safetensors.torch",
    "librosa",
    "lhotse.utils",
    "linacodec",
    "linacodec.vocoder",
    "linacodec.vocoder.vocos",
    "zipvoice.models",
    "zipvoice.models.zipvoice_distill",
    "zipvoice.utils.checkpoint",
    "zipvoice.utils.feature",
    "zipvoice.utils.common",
    "zipvoice.onnx_modeling",
    "tensorboard",
    "torch.utils.tensorboard",
]
for _mod in _MODELING_DEPS:
    _ensure_mock(_mod)


# ---------------------------------------------------------------------------
# Token file fixtures
# ---------------------------------------------------------------------------

#: Minimal vocabulary used across multiple tests.
SAMPLE_VOCAB: dict[str, int] = {
    "_": 0,  # padding token (required by SimpleTokenizer)
    "a": 1,
    "b": 2,
    "c": 3,
    "h": 4,
    "e": 5,
    "l": 6,
    "o": 7,
    " ": 8,
}


@pytest.fixture()
def sample_token_file(tmp_path: Path) -> str:
    """Write a minimal token file and return its path.

    The file uses tab-separated ``{token}\\t{token_id}`` format, one entry
    per line, as expected by ``_load_token_file``.

    Args:
        tmp_path: Pytest-provided temporary directory (unique per test).

    Returns:
        Absolute path to the token file as a string.
    """
    token_path = tmp_path / "tokens.txt"
    lines = [f"{token}\t{token_id}\n" for token, token_id in SAMPLE_VOCAB.items()]
    token_path.write_text("".join(lines), encoding="utf-8")
    return str(token_path)


@pytest.fixture()
def duplicate_token_file(tmp_path: Path) -> str:
    """Write a token file with a duplicate entry and return its path.

    Args:
        tmp_path: Pytest-provided temporary directory.

    Returns:
        Absolute path to the duplicate-token file as a string.
    """
    token_path = tmp_path / "tokens_dup.txt"
    content = "a\t1\nb\t2\na\t3\n"  # 'a' appears twice
    token_path.write_text(content, encoding="utf-8")
    return str(token_path)
# end tests/conftest.py
