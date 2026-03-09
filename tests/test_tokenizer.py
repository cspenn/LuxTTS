# start tests/test_tokenizer.py
"""Unit tests for zipvoice.tokenizer.tokenizer pure-Python helpers.

These tests cover _load_token_file, _tokens_to_token_ids, and SimpleTokenizer.
They do NOT exercise EspeakTokenizer, EmiliaTokenizer, LibriTTSTokenizer, or
DialogTokenizer because those require piper_phonemize/espeak, jieba, or
tacotron_cleaner binaries that are not available in a lightweight test env.

The piper_phonemize module is pre-mocked in conftest.py before this file is
imported, so the top-level `from piper_phonemize import phonemize_espeak` in
tokenizer.py succeeds without the real binary.
"""

import pytest

from zipvoice.tokenizer.tokenizer import (
    SimpleTokenizer,
    _load_token_file,
    _tokens_to_token_ids,
)


# ---------------------------------------------------------------------------
# _load_token_file tests
# ---------------------------------------------------------------------------


class TestLoadTokenFile:
    """Tests for the _load_token_file helper."""

    def test_load_token_file_valid(self, sample_token_file: str) -> None:
        """Loading a well-formed token file returns a complete token→id mapping.

        Args:
            sample_token_file: Fixture providing path to a valid token file.
        """
        token2id = _load_token_file(sample_token_file)
        # Must contain every entry written in SAMPLE_VOCAB
        assert "_" in token2id
        assert token2id["_"] == 0
        assert token2id["a"] == 1
        assert token2id["b"] == 2
        assert token2id["c"] == 3

    def test_load_token_file_returns_dict(self, sample_token_file: str) -> None:
        """Return type is dict[str, int].

        Args:
            sample_token_file: Fixture providing path to a valid token file.
        """
        token2id = _load_token_file(sample_token_file)
        assert isinstance(token2id, dict)
        for key, value in token2id.items():
            assert isinstance(key, str)
            assert isinstance(value, int)

    def test_load_token_file_full_size(self, sample_token_file: str) -> None:
        """All entries from the token file are loaded.

        Args:
            sample_token_file: Fixture providing path to a valid token file.
        """
        token2id = _load_token_file(sample_token_file)
        assert len(token2id) == 9  # SAMPLE_VOCAB has 9 entries: _, a, b, c, h, e, l, o, space

    def test_load_token_file_duplicate_raises(self, duplicate_token_file: str) -> None:
        """A file with duplicate tokens raises TokenizerError.

        Args:
            duplicate_token_file: Fixture providing path to a file with a
                duplicated token entry.
        """
        from zipvoice.exceptions import TokenizerError
        with pytest.raises(TokenizerError):
            _load_token_file(duplicate_token_file)


# ---------------------------------------------------------------------------
# _tokens_to_token_ids tests
# ---------------------------------------------------------------------------


class TestTokensToTokenIds:
    """Tests for the _tokens_to_token_ids helper."""

    def test_known_inputs_returns_correct_ids(self) -> None:
        """Known tokens map to their correct integer IDs."""
        token2id = {"a": 1, "b": 2, "c": 3}
        tokens_list = [["a", "b", "c"]]
        result = _tokens_to_token_ids(tokens_list, token2id)
        assert result == [[1, 2, 3]]

    def test_multiple_sequences(self) -> None:
        """Multiple token sequences are all converted correctly."""
        token2id = {"x": 10, "y": 20}
        tokens_list = [["x", "y"], ["y", "x", "y"]]
        result = _tokens_to_token_ids(tokens_list, token2id)
        assert result == [[10, 20], [20, 10, 20]]

    def test_oov_tokens_are_skipped(self) -> None:
        """Out-of-vocabulary tokens are silently dropped from the output."""
        token2id = {"a": 1}
        tokens_list = [["a", "z", "a"]]  # 'z' is OOV
        result = _tokens_to_token_ids(tokens_list, token2id)
        assert result == [[1, 1]]

    def test_all_oov_produces_empty_sequence(self) -> None:
        """A sequence composed entirely of OOV tokens yields an empty list."""
        token2id = {"a": 1}
        tokens_list = [["z", "q"]]
        result = _tokens_to_token_ids(tokens_list, token2id)
        assert result == [[]]

    def test_empty_token_list(self) -> None:
        """An empty input list returns an empty output list."""
        token2id = {"a": 1}
        result = _tokens_to_token_ids([], token2id)
        assert result == []

    def test_empty_sequence_in_list(self) -> None:
        """An inner empty sequence produces an empty inner list in the output."""
        token2id = {"a": 1}
        result = _tokens_to_token_ids([[]], token2id)
        assert result == [[]]


# ---------------------------------------------------------------------------
# SimpleTokenizer tests
# ---------------------------------------------------------------------------


class TestSimpleTokenizer:
    """Tests for the SimpleTokenizer class."""

    def test_instantiation_with_token_file(self, sample_token_file: str) -> None:
        """SimpleTokenizer initialised with a file sets has_tokens=True.

        Args:
            sample_token_file: Fixture providing path to a valid token file.
        """
        tok = SimpleTokenizer(token_file=sample_token_file)
        assert tok.has_tokens is True

    def test_instantiation_without_token_file(self) -> None:
        """SimpleTokenizer initialised without a file sets has_tokens=False."""
        tok = SimpleTokenizer()
        assert tok.has_tokens is False

    def test_vocab_size_matches_file(self, sample_token_file: str) -> None:
        """vocab_size equals the number of entries in the token file.

        Args:
            sample_token_file: Fixture providing path to a valid token file.
        """
        tok = SimpleTokenizer(token_file=sample_token_file)
        assert tok.vocab_size == 9  # SAMPLE_VOCAB has 9 entries: _, a, b, c, h, e, l, o, space

    def test_pad_id_is_underscore_token(self, sample_token_file: str) -> None:
        """pad_id matches the id assigned to the '_' padding token.

        Args:
            sample_token_file: Fixture providing path to a valid token file.
        """
        tok = SimpleTokenizer(token_file=sample_token_file)
        assert tok.pad_id == tok.token2id["_"]
        assert tok.pad_id == 0

    def test_texts_to_tokens_splits_chars(self, sample_token_file: str) -> None:
        """texts_to_tokens converts each string to a list of individual characters.

        Args:
            sample_token_file: Fixture providing path to a valid token file.
        """
        tok = SimpleTokenizer(token_file=sample_token_file)
        result = tok.texts_to_tokens(["abc"])
        assert result == [["a", "b", "c"]]

    def test_tokens_to_token_ids_correct(self, sample_token_file: str) -> None:
        """tokens_to_token_ids returns correct IDs for in-vocabulary tokens.

        Args:
            sample_token_file: Fixture providing path to a valid token file.
        """
        tok = SimpleTokenizer(token_file=sample_token_file)
        result = tok.tokens_to_token_ids([["a", "b", "c"]])
        assert result == [[1, 2, 3]]

    def test_tokens_to_token_ids_oov_skipped(self, sample_token_file: str) -> None:
        """tokens_to_token_ids skips OOV tokens silently.

        Args:
            sample_token_file: Fixture providing path to a valid token file.
        """
        tok = SimpleTokenizer(token_file=sample_token_file)
        result = tok.tokens_to_token_ids([["a", "Z", "b"]])  # 'Z' is OOV
        assert result == [[1, 2]]

    def test_tokens_to_token_ids_requires_token_file(self) -> None:
        """tokens_to_token_ids raises TokenizerError when has_tokens is False."""
        from zipvoice.exceptions import TokenizerError
        tok = SimpleTokenizer()
        with pytest.raises(TokenizerError):
            tok.tokens_to_token_ids([["a"]])

    def test_texts_to_token_ids_end_to_end(self, sample_token_file: str) -> None:
        """texts_to_token_ids correctly chains texts_to_tokens and tokens_to_token_ids.

        Args:
            sample_token_file: Fixture providing path to a valid token file.
        """
        tok = SimpleTokenizer(token_file=sample_token_file)
        result = tok.texts_to_token_ids(["hello"])
        # 'h'=4, 'e'=5, 'l'=6, 'l'=6, 'o'=7
        assert result == [[4, 5, 6, 6, 7]]

    def test_texts_to_token_ids_empty_string(self, sample_token_file: str) -> None:
        """An empty text string results in an empty token-id list.

        Args:
            sample_token_file: Fixture providing path to a valid token file.
        """
        tok = SimpleTokenizer(token_file=sample_token_file)
        result = tok.texts_to_token_ids([""])
        assert result == [[]]
# end tests/test_tokenizer.py
