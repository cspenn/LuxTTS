# start tests/test_exceptions.py
"""Unit tests for zipvoice.exceptions module.

Tests cover:
- All three public exceptions are importable.
- Each inherits from the LuxTTSError base and ultimately from Exception.
- Each can be raised and caught as expected.
- Each carries its message string.
- Exception chaining works correctly for TokenizerError.
"""

import pytest

from zipvoice.exceptions import AudioProcessingError, LuxTTSError, ModelLoadError, TokenizerError


# ---------------------------------------------------------------------------
# Import and inheritance tests
# ---------------------------------------------------------------------------


class TestExceptionImports:
    """Tests that all custom exceptions are importable and properly typed."""

    def test_model_load_error_importable(self) -> None:
        """ModelLoadError can be imported without errors."""
        assert ModelLoadError is not None

    def test_audio_processing_error_importable(self) -> None:
        """AudioProcessingError can be imported without errors."""
        assert AudioProcessingError is not None

    def test_tokenizer_error_importable(self) -> None:
        """TokenizerError can be imported without errors."""
        assert TokenizerError is not None


class TestExceptionHierarchy:
    """Tests that exceptions inherit from the correct base classes."""

    def test_model_load_error_is_lux_tts_error(self) -> None:
        """ModelLoadError is a subclass of LuxTTSError."""
        assert issubclass(ModelLoadError, LuxTTSError)

    def test_audio_processing_error_is_lux_tts_error(self) -> None:
        """AudioProcessingError is a subclass of LuxTTSError."""
        assert issubclass(AudioProcessingError, LuxTTSError)

    def test_tokenizer_error_is_lux_tts_error(self) -> None:
        """TokenizerError is a subclass of LuxTTSError."""
        assert issubclass(TokenizerError, LuxTTSError)

    def test_lux_tts_error_is_exception(self) -> None:
        """LuxTTSError is a subclass of the built-in Exception."""
        assert issubclass(LuxTTSError, Exception)

    def test_model_load_error_is_exception(self) -> None:
        """ModelLoadError is a subclass of Exception (transitively)."""
        assert issubclass(ModelLoadError, Exception)

    def test_audio_processing_error_is_exception(self) -> None:
        """AudioProcessingError is a subclass of Exception (transitively)."""
        assert issubclass(AudioProcessingError, Exception)

    def test_tokenizer_error_is_exception(self) -> None:
        """TokenizerError is a subclass of Exception (transitively)."""
        assert issubclass(TokenizerError, Exception)


# ---------------------------------------------------------------------------
# Raise and catch tests
# ---------------------------------------------------------------------------


class TestModelLoadError:
    """Tests for raising and catching ModelLoadError."""

    def test_raises_and_catches(self) -> None:
        """ModelLoadError can be raised and caught."""
        with pytest.raises(ModelLoadError):
            raise ModelLoadError("weights not found")

    def test_carries_message(self) -> None:
        """ModelLoadError stores the message string."""
        exc = ModelLoadError("bad checkpoint path")
        assert str(exc) == "bad checkpoint path"

    def test_caught_as_lux_tts_error(self) -> None:
        """ModelLoadError is caught by an except LuxTTSError clause."""
        with pytest.raises(LuxTTSError):
            raise ModelLoadError("network failure")

    def test_caught_as_exception(self) -> None:
        """ModelLoadError is caught by a bare except Exception clause."""
        with pytest.raises(Exception):
            raise ModelLoadError("generic failure")


class TestAudioProcessingError:
    """Tests for raising and catching AudioProcessingError."""

    def test_raises_and_catches(self) -> None:
        """AudioProcessingError can be raised and caught."""
        with pytest.raises(AudioProcessingError):
            raise AudioProcessingError("invalid sample rate")

    def test_carries_message(self) -> None:
        """AudioProcessingError stores the message string."""
        exc = AudioProcessingError("corrupt audio data")
        assert str(exc) == "corrupt audio data"

    def test_caught_as_lux_tts_error(self) -> None:
        """AudioProcessingError is caught by an except LuxTTSError clause."""
        with pytest.raises(LuxTTSError):
            raise AudioProcessingError("unsupported format")


class TestTokenizerError:
    """Tests for raising and catching TokenizerError."""

    def test_raises_and_catches(self) -> None:
        """TokenizerError can be raised and caught."""
        with pytest.raises(TokenizerError):
            raise TokenizerError("missing token file")

    def test_carries_message(self) -> None:
        """TokenizerError stores the message string."""
        exc = TokenizerError("unknown language code")
        assert str(exc) == "unknown language code"

    def test_caught_as_lux_tts_error(self) -> None:
        """TokenizerError is caught by an except LuxTTSError clause."""
        with pytest.raises(LuxTTSError):
            raise TokenizerError("encoding error")

    def test_exception_chaining(self) -> None:
        """TokenizerError can be chained from a cause exception.

        Verifies that raise ... from ... correctly sets __cause__.
        """
        cause = ValueError("bad input encoding")
        try:
            raise TokenizerError("tokenisation failed") from cause
        except TokenizerError as exc:
            assert exc.__cause__ is cause
            assert isinstance(exc.__cause__, ValueError)

    def test_chained_cause_message(self) -> None:
        """TokenizerError chained cause retains its own message."""
        cause = ValueError("the root cause")
        try:
            raise TokenizerError("wrapper message") from cause
        except TokenizerError as exc:
            assert str(exc.__cause__) == "the root cause"
# end tests/test_exceptions.py
