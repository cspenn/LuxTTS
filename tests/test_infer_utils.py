# start tests/test_infer_utils.py
"""Unit tests for pure-Python utility functions in zipvoice.utils.infer.

Only functions that require no GPU, no audio files, and no heavy model
dependencies are tested here. Functions that operate on torch.Tensor audio
data (cross_fade_concat, rms_norm, remove_silence, etc.) are intentionally
excluded from this module because they require non-trivial audio fixtures that
would slow the test suite and couple it to torch availability.

Tested:
- Module-level constants
- add_punctuation
- chunk_tokens_punctuation
- chunk_tokens_dialog
- batchify_tokens
"""

import pytest

# pydub and torchaudio are pre-mocked in conftest.py before any test module
# is imported, so infer.py can be imported without those binaries installed.

# Now safe to import from infer
from zipvoice.utils.infer import (  # noqa: E402
    AUDIO_MAX_INT16,
    AUDIO_NORM_FACTOR,
    SILENCE_THRESHOLD_DB,
    SILENCE_THRESHOLD_MS,
    add_punctuation,
    batchify_tokens,
    chunk_tokens_dialog,
    chunk_tokens_punctuation,
)


# ---------------------------------------------------------------------------
# Constant value tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module-level numeric constants have their documented values."""

    def test_silence_threshold_ms(self) -> None:
        """SILENCE_THRESHOLD_MS is 1000 ms (1 second)."""
        assert SILENCE_THRESHOLD_MS == 1000

    def test_silence_threshold_db(self) -> None:
        """SILENCE_THRESHOLD_DB is -50 dB."""
        assert SILENCE_THRESHOLD_DB == -50

    def test_audio_norm_factor(self) -> None:
        """AUDIO_NORM_FACTOR is 32768.0 (2^15), matching int16 range."""
        assert AUDIO_NORM_FACTOR == pytest.approx(32768.0)

    def test_audio_max_int16(self) -> None:
        """AUDIO_MAX_INT16 is 32767, the maximum positive int16 value."""
        assert AUDIO_MAX_INT16 == 32767


# ---------------------------------------------------------------------------
# add_punctuation tests
# ---------------------------------------------------------------------------


class TestAddPunctuation:
    """Tests for the add_punctuation helper."""

    def test_adds_period_when_no_trailing_punct(self) -> None:
        """A period is appended when the text has no trailing punctuation."""
        result = add_punctuation("Hello world")
        assert result == "Hello world."

    def test_preserves_existing_period(self) -> None:
        """Text ending with a period is left unchanged."""
        result = add_punctuation("Hello world.")
        assert result == "Hello world."

    def test_preserves_exclamation(self) -> None:
        """Text ending with '!' is left unchanged."""
        result = add_punctuation("Hello!")
        assert result == "Hello!"

    def test_preserves_question_mark(self) -> None:
        """Text ending with '?' is left unchanged."""
        result = add_punctuation("Hello?")
        assert result == "Hello?"

    def test_strips_trailing_whitespace(self) -> None:
        """Leading/trailing whitespace is stripped before the check."""
        result = add_punctuation("  Hello  ")
        assert result == "Hello."

    def test_preserves_chinese_punct(self) -> None:
        """Text ending with a Chinese punctuation mark is left unchanged."""
        result = add_punctuation("你好。")
        assert result == "你好。"


# ---------------------------------------------------------------------------
# chunk_tokens_punctuation tests
# ---------------------------------------------------------------------------


class TestChunkTokensPunctuation:
    """Tests for chunk_tokens_punctuation."""

    def test_basic_split_on_period(self) -> None:
        """Tokens that fit within max_tokens are merged into one chunk.

        chunk_tokens_punctuation first splits on punctuation boundaries, then
        re-merges adjacent short sentences. Both "Hello." (6 tokens) and
        " World." (7 tokens) fit inside max_tokens=100 so they are merged into
        a single chunk.
        """
        tokens = list("Hello. World.")
        chunks = chunk_tokens_punctuation(tokens, max_tokens=100)
        assert len(chunks) == 1
        # All tokens must be present in the single merged chunk
        assert chunks[0] == tokens

    def test_single_sentence_no_split(self) -> None:
        """A single sentence below max_tokens stays in one chunk."""
        tokens = list("Hello.")
        chunks = chunk_tokens_punctuation(tokens, max_tokens=100)
        assert len(chunks) == 1
        assert chunks[0] == list("Hello.")

    def test_max_tokens_forces_split(self) -> None:
        """When a sentence exceeds max_tokens the overflow starts a new chunk."""
        # Two sentences that together exceed max_tokens=5
        tokens = list("Hi. Go.")
        chunks = chunk_tokens_punctuation(tokens, max_tokens=3)
        assert len(chunks) == 2

    def test_no_punctuation_returns_one_chunk(self) -> None:
        """Tokens with no punctuation are returned as a single chunk."""
        tokens = list("hello world")
        chunks = chunk_tokens_punctuation(tokens, max_tokens=100)
        assert len(chunks) == 1
        assert chunks[0] == tokens

    def test_empty_list_returns_empty(self) -> None:
        """An empty token list returns an empty list of chunks."""
        chunks = chunk_tokens_punctuation([], max_tokens=100)
        assert chunks == []

    def test_punctuation_set_includes_expected_chars(self) -> None:
        """Common punctuation marks trigger sentence boundaries."""
        from zipvoice.utils.infer import punctuation as PUNCT

        for char in (".", ",", "!", "?", ";", ":"):
            assert char in PUNCT


# ---------------------------------------------------------------------------
# chunk_tokens_dialog tests
# ---------------------------------------------------------------------------


class TestChunkTokensDialog:
    """Tests for chunk_tokens_dialog."""

    def test_basic_split_on_s1(self) -> None:
        """Dialog turns that together fit within max_tokens are merged into one chunk.

        chunk_tokens_dialog splits on [S1] markers first, then merges turns back
        together if the combined length is within max_tokens. With max_tokens=100
        the two turns (3 + 4 = 7 tokens total) fit into a single chunk.
        """
        tokens = ["[S1]", "H", "i", "[S1]", "B", "y", "e"]
        chunks = chunk_tokens_dialog(tokens, max_tokens=100)
        assert len(chunks) == 1
        assert chunks[0] == tokens

    def test_single_turn_no_split(self) -> None:
        """A single dialog turn below max_tokens stays in one chunk."""
        tokens = ["[S1]", "H", "e", "l", "l", "o"]
        chunks = chunk_tokens_dialog(tokens, max_tokens=100)
        assert len(chunks) == 1

    def test_max_tokens_forces_new_chunk(self) -> None:
        """When merging would exceed max_tokens, a new chunk is started."""
        tokens = ["[S1]", "A", "B", "[S1]", "C", "D"]
        # max_tokens=3: first turn ["[S1]","A","B"] is 3 tokens (ok),
        # adding second turn ["[S1]","C","D"] would be 6 > 3 -> new chunk
        chunks = chunk_tokens_dialog(tokens, max_tokens=3)
        assert len(chunks) == 2

    def test_empty_list_returns_empty(self) -> None:
        """An empty token list returns an empty list of chunks."""
        chunks = chunk_tokens_dialog([], max_tokens=100)
        assert chunks == []

    def test_no_s1_marker_single_chunk(self) -> None:
        """Without any [S1] marker the entire token list is one chunk."""
        tokens = ["H", "e", "l", "l", "o"]
        chunks = chunk_tokens_dialog(tokens, max_tokens=100)
        assert len(chunks) == 1
        assert chunks[0] == tokens


# ---------------------------------------------------------------------------
# batchify_tokens tests
# ---------------------------------------------------------------------------


class TestBatchifyTokens:
    """Tests for batchify_tokens."""

    def test_single_sequence_single_batch(self) -> None:
        """A single short token sequence produces exactly one batch."""
        tokens_list = [[1, 2, 3]]
        batches, index = batchify_tokens(tokens_list, max_duration=10.0, prompt_duration=1.0, token_duration=0.1)
        assert len(batches) == 1
        assert len(index) == 1

    def test_index_tracks_original_positions(self) -> None:
        """The returned index correctly maps sorted position back to original."""
        # Three sequences of lengths 3, 1, 2 — will be sorted to [1,2,3]
        tokens_list = [[10, 20, 30], [10], [10, 20]]
        _, index = batchify_tokens(tokens_list, max_duration=100.0, prompt_duration=0.0, token_duration=0.1)
        # index should be a permutation of [0, 1, 2]
        assert sorted(index) == [0, 1, 2]

    def test_batches_respect_max_duration(self) -> None:
        """No batch exceeds max_duration given the token/prompt costs."""
        # Each token costs 1.0, each prompt costs 0.0, max_duration=3.0
        # Sequences: [1,2,3] (3 tokens), [4,5,6] (3 tokens)
        tokens_list = [[1, 2, 3], [4, 5, 6]]
        batches, _ = batchify_tokens(tokens_list, max_duration=3.0, prompt_duration=0.0, token_duration=1.0)
        # Each 3-token sequence exactly fits; with two sequences both fitting
        # into 3.0 would require 6 tokens total > 3.0, so they must split
        assert len(batches) == 2

    def test_empty_input_returns_empty(self) -> None:
        """An empty tokens_list produces empty batches and empty index."""
        batches, index = batchify_tokens([], max_duration=10.0, prompt_duration=1.0, token_duration=0.1)
        assert batches == []
        assert index == []

    def test_all_fit_in_one_batch(self) -> None:
        """All sequences fit into a single batch when duration is generous."""
        tokens_list = [[1], [2], [3]]
        batches, _ = batchify_tokens(tokens_list, max_duration=100.0, prompt_duration=0.0, token_duration=0.1)
        assert len(batches) == 1

    def test_index_length_matches_input_length(self) -> None:
        """The index list has the same length as the input tokens_list."""
        tokens_list = [[1, 2], [3], [4, 5, 6]]
        _, index = batchify_tokens(tokens_list, max_duration=100.0, prompt_duration=0.0, token_duration=0.1)
        assert len(index) == len(tokens_list)
# end tests/test_infer_utils.py
