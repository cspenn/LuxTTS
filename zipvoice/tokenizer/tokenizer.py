# start zipvoice/tokenizer/tokenizer.py
"""Tokenizers for ZipVoice TTS: character, espeak, Emilia (zh+en), and LibriTTS variants."""

# Copyright      2023-2024  Xiaomi Corp.        (authors: Zengwei Yao
#                                                         Han Zhu,
#                                                         Wei Kang)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from abc import ABC, abstractmethod
from functools import reduce

import jieba
import structlog
from lhotse import CutSet
from pypinyin import Style, lazy_pinyin
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials

from zipvoice.exceptions import TokenizerError
from zipvoice.tokenizer.normalizer import ChineseTextNormalizer, EnglishTextNormalizer

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    msg = f"{ex}\nPlease run\npip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html"
    raise TokenizerError(msg) from ex

log = structlog.get_logger()

jieba.default_logger.setLevel(20)  # logging.INFO = 20


def _load_token_file(token_file_path: str) -> dict[str, int]:
    r"""Read a token file and return a mapping from token string to integer id.

    The file is expected to have one '{token}\\t{token_id}' entry per line.
    """
    token2id: dict[str, int] = {}
    with open(token_file_path, encoding="utf-8") as f:
        for line in f:
            info = line.rstrip().split("\t")
            token, token_id = info[0], int(info[1])
            if token in token2id:
                raise TokenizerError(token)
            token2id[token] = token_id
    return token2id


def _tokens_to_token_ids(
    tokens_list: list[list[str]],
    token2id: dict[str, int],
) -> list[list[int]]:
    """Convert a list of token sequences to a list of token-id sequences.

    Out-of-vocabulary tokens are silently skipped with a debug log message.
    This logic is shared by SimpleTokenizer, EspeakTokenizer, EmiliaTokenizer,
    and LibriTTSTokenizer (non-BPE path).
    """
    token_ids_list = []
    for tokens in tokens_list:
        token_ids = []
        for t in tokens:
            if t not in token2id:
                log.debug("skip_oov_token", token=t)
                continue
            token_ids.append(token2id[t])
        token_ids_list.append(token_ids)
    return token_ids_list


class Tokenizer(ABC):
    """Abstract base class for tokenizers, defining common interface."""

    @abstractmethod
    def texts_to_token_ids(self, texts: list[str]) -> list[list[int]]:
        """Convert list of texts to list of token id sequences."""
        raise NotImplementedError

    @abstractmethod
    def texts_to_tokens(self, texts: list[str]) -> list[list[str]]:
        """Convert list of texts to list of token sequences."""
        raise NotImplementedError

    @abstractmethod
    def tokens_to_token_ids(self, tokens: list[list[str]]) -> list[list[int]]:
        """Convert list of token sequences to list of token id sequences."""
        raise NotImplementedError


class SimpleTokenizer(Tokenizer):
    """The simplest tokenizer, treating every character as a token without text normalization."""

    def __init__(self, token_file: str | None = None):
        r"""Initialize SimpleTokenizer.

        Args:
            token_file: The file that contains information that maps tokens to ids,
              which is a text file with '{token}\t{token_id}' per line.
        """
        # Parse token file
        self.has_tokens = False
        if token_file is None:
            log.debug("tokenizer_no_tokens_file", note="will fail when map to ids")
            return
        self.token2id = _load_token_file(token_file)
        self.pad_id = self.token2id["_"]  # padding
        self.vocab_size = len(self.token2id)
        self.has_tokens = True

    def texts_to_token_ids(
        self,
        texts: list[str],
    ) -> list[list[int]]:
        """Convert texts to token id sequences.

        Args:
            texts: List of input text strings.

        Returns:
            List of token id sequences.
        """
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def texts_to_tokens(
        self,
        texts: list[str],
    ) -> list[list[str]]:
        """Convert texts to character token sequences.

        Args:
            texts: List of input text strings.

        Returns:
            List of character token sequences.
        """
        tokens_list = [list(texts[i]) for i in range(len(texts))]
        return tokens_list

    def tokens_to_token_ids(
        self,
        tokens_list: list[list[str]],
    ) -> list[list[int]]:
        """Convert token sequences to token id sequences.

        Args:
            tokens_list: List of token sequences.

        Returns:
            List of token id sequences.
        """
        if not self.has_tokens:
            msg = "Please initialize Tokenizer with a tokens file."
            raise TokenizerError(msg)
        return _tokens_to_token_ids(tokens_list, self.token2id)


class EspeakTokenizer(Tokenizer):
    """A simple tokenizer with Espeak g2p function."""

    def __init__(self, token_file: str | None = None, lang: str = "en-us"):
        r"""Initialize EspeakTokenizer.

        Args:
            token_file: The file that contains information that maps tokens to ids,
              which is a text file with '{token}\t{token_id}' per line.
            lang: The language identifier, see
              https://github.com/rhasspy/espeak-ng/blob/master/docs/languages.md.
        """
        # Parse token file
        self.has_tokens = False
        self.lang = lang
        if token_file is None:
            log.debug("tokenizer_no_tokens_file", note="will fail when map to ids")
            return
        self.token2id = _load_token_file(token_file)
        self.pad_id = self.token2id["_"]  # padding
        self.vocab_size = len(self.token2id)
        self.has_tokens = True

    def g2p(self, text: str) -> list[str]:
        """Convert text to phonemes using espeak.

        Args:
            text: Input text to phonemize.

        Returns:
            List of phoneme strings. Raises TokenizerError on failure.
        """
        try:
            tokens = phonemize_espeak(text, self.lang)
            tokens = reduce(lambda x, y: x + y, tokens)
        except Exception as ex:
            log.warning("tokenization_failed", lang=self.lang, error=str(ex))
            msg = f"Tokenization failed for lang={self.lang!r}: {ex}"
            raise TokenizerError(msg) from ex
        return tokens

    def texts_to_token_ids(
        self,
        texts: list[str],
    ) -> list[list[int]]:
        """Convert texts to token id sequences.

        Args:
            texts: List of input text strings.

        Returns:
            List of token id sequences.
        """
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def texts_to_tokens(
        self,
        texts: list[str],
    ) -> list[list[str]]:
        """Convert texts to phoneme token sequences.

        Args:
            texts: List of input text strings.

        Returns:
            List of phoneme token sequences.
        """
        tokens_list = [self.g2p(texts[i]) for i in range(len(texts))]
        return tokens_list

    def tokens_to_token_ids(
        self,
        tokens_list: list[list[str]],
    ) -> list[list[int]]:
        """Convert token sequences to token id sequences.

        Args:
            tokens_list: List of token sequences.

        Returns:
            List of token id sequences.
        """
        if not self.has_tokens:
            msg = "Please initialize Tokenizer with a tokens file."
            raise TokenizerError(msg)
        return _tokens_to_token_ids(tokens_list, self.token2id)


class EmiliaTokenizer(Tokenizer):
    """Tokenizer for the Emilia multilingual (Chinese + English) dataset."""

    def __init__(self, token_file: str | None = None, token_type="phone"):  # noqa: S107
        r"""Initialize EmiliaTokenizer.

        Args:
            token_file: The file that contains information that maps tokens to ids,
              which is a text file with '{token}\t{token_id}' per line.
            token_type: Type of tokenizer; only 'phone' is currently supported.
        """
        if token_type != "phone":  # noqa: S105
            msg = f"Only support phone tokenizer for Emilia, but get {token_type}."
            raise TokenizerError(msg)

        self.english_normalizer = EnglishTextNormalizer()
        self.chinese_normalizer = ChineseTextNormalizer()

        self.has_tokens = False
        if token_file is None:
            log.debug("tokenizer_no_tokens_file", note="will fail when map to ids")
            return
        self.token2id = _load_token_file(token_file)
        self.pad_id = self.token2id["_"]  # padding

        self.vocab_size = len(self.token2id)
        self.has_tokens = True

    def texts_to_token_ids(
        self,
        texts: list[str],
    ) -> list[list[int]]:
        """Convert texts to token id sequences.

        Args:
            texts: List of input text strings.

        Returns:
            List of token id sequences.
        """
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def preprocess_text(
        self,
        text: str,
    ) -> str:
        """Preprocess text before tokenization.

        Args:
            text: Input text string.

        Returns:
            Preprocessed text.
        """
        return self.map_punctuations(text)

    def texts_to_tokens(
        self,
        texts: list[str],
    ) -> list[list[str]]:
        """Convert texts to phoneme token sequences.

        Args:
            texts: List of input text strings (modified in place).

        Returns:
            List of phoneme token sequences.
        """
        for i in range(len(texts)):
            # Text normalization
            texts[i] = self.preprocess_text(texts[i])

        phoneme_list = []
        for text in texts:
            # now only en and ch
            segments = self.get_segment(text)
            all_phoneme = []
            for index in range(len(segments)):
                seg = segments[index]
                if seg[1] == "zh":
                    phoneme = self.tokenize_ZH(seg[0])
                elif seg[1] == "en":
                    phoneme = self.tokenize_EN(seg[0])
                elif seg[1] == "pinyin":
                    phoneme = self.tokenize_pinyin(seg[0])
                elif seg[1] == "tag":
                    phoneme = [seg[0]]
                else:
                    log.warning("unknown_language_segment_skipped", segment=str(seg))
                    continue
                all_phoneme += phoneme
            phoneme_list.append(all_phoneme)
        return phoneme_list

    def tokens_to_token_ids(
        self,
        tokens_list: list[list[str]],
    ) -> list[list[int]]:
        """Convert token sequences to token id sequences.

        Args:
            tokens_list: List of token sequences.

        Returns:
            List of token id sequences.
        """
        if not self.has_tokens:
            msg = "Please initialize Tokenizer with a tokens file."
            raise TokenizerError(msg)
        return _tokens_to_token_ids(tokens_list, self.token2id)

    def tokenize_ZH(self, text: str) -> list[str]:
        """Tokenize Chinese text to phoneme tokens.

        Args:
            text: Chinese input text.

        Returns:
            List of phoneme tokens. Raises TokenizerError on failure.
        """
        try:
            text = self.chinese_normalizer.normalize(text)
            segs = list(jieba.cut(text))
            full = lazy_pinyin(
                segs,
                style=Style.TONE3,
                tone_sandhi=True,
                neutral_tone_with_five=True,
            )
            phones = []
            for x in full:
                # valid pinyin (in tone3 style) is alphabet + 1 number in [1-5].
                if not (x[0:-1].isalpha() and x[-1] in ("1", "2", "3", "4", "5")):
                    phones.append(x)
                    continue
                phones.extend(self.seperate_pinyin(x))
        except Exception as ex:
            log.warning("tokenization_failed", lang="zh", error=str(ex))
            msg = f"Chinese tokenization failed: {ex}"
            raise TokenizerError(msg) from ex
        return phones

    def tokenize_EN(self, text: str) -> list[str]:
        """Tokenize English text to phoneme tokens.

        Args:
            text: English input text.

        Returns:
            List of phoneme tokens. Raises TokenizerError on failure.
        """
        try:
            text = self.english_normalizer.normalize(text)
            tokens = phonemize_espeak(text, "en-us")
            tokens = reduce(lambda x, y: x + y, tokens)
        except Exception as ex:
            log.warning("tokenization_failed", lang="en", error=str(ex))
            msg = f"English tokenization failed: {ex}"
            raise TokenizerError(msg) from ex
        return tokens

    def tokenize_pinyin(self, text: str) -> list[str]:
        """Tokenize a pinyin string enclosed in angle brackets.

        Args:
            text: A pinyin string like '<le5>'.

        Returns:
            List of phoneme tokens. Raises TokenizerError on failure.
        """
        if not (text.startswith("<") and text.endswith(">")):
            msg = f"Expected pinyin enclosed in '<>', got: {text!r}"
            raise TokenizerError(msg)
        try:
            text = text.lstrip("<").rstrip(">")
            # valid pinyin (in tone3 style) is alphabet + 1 number in [1-5].
            if not (text[0:-1].isalpha() and text[-1] in ("1", "2", "3", "4", "5")):
                log.warning("invalid_pinyin_skipped", text=text)
                return []
            return self.seperate_pinyin(text)
        except Exception as ex:
            log.warning("tokenize_pinyin_failed", error=str(ex))
            msg = f"Pinyin tokenization failed: {ex}"
            raise TokenizerError(msg) from ex

    def seperate_pinyin(self, text: str) -> list[str]:
        """Separate pinyin into initial and final."""
        pinyins = []
        initial = to_initials(text, strict=False)
        # don't want to share tokens with espeak tokens,
        # so use tone3 style
        final = to_finals_tone3(
            text,
            strict=False,
            neutral_tone_with_five=True,
        )
        if initial != "":
            # don't want to share tokens with espeak tokens,
            # so add a '0' after each initial
            pinyins.append(initial + "0")
        if final != "":
            pinyins.append(final)
        return pinyins

    def map_punctuations(self, text):
        """Map Chinese/full-width punctuation to ASCII equivalents.

        Args:
            text: Input text string.

        Returns:
            Text with mapped punctuation characters.
        """
        text = text.replace("，", ",")
        text = text.replace("。", ".")
        text = text.replace("！", "!")
        text = text.replace("？", "?")
        text = text.replace("；", ";")
        text = text.replace("：", ":")
        text = text.replace("、", ",")
        text = text.replace("'", "'")
        text = text.replace(
            """, '"')
        text = text.replace(""",
            '"',
        )
        text = text.replace("'", "'")
        text = text.replace("⋯", "…")
        text = text.replace("···", "…")
        text = text.replace("・・・", "…")
        text = text.replace("...", "…")
        return text

    def get_segment(self, text: str) -> list[str]:
        """Split a text into segments based on language types.

        Handles Chinese, English, Pinyin, tags, etc.

        Args:
            text (str): Input text to be segmented

        Returns:
            list[str]: Segmented text parts with their language types

        Example:
            Input: 我们是小米人,是吗? Yes I think so!霍...啦啦啦
            Output: [('我们是小米人,是吗? ', 'zh'),
                ('Yes I think so!', 'en'), ('霍...啦啦啦', 'zh')]
        """
        # Stores the final segmented parts and their language types
        segments = []
        # Stores the language type of each character in the input text
        types = []
        temp_seg = ""
        temp_lang = ""

        # Each part is a character, or a special string enclosed in <> and []
        # <> denotes pinyin string, [] denotes other special strings.
        _part_pattern = re.compile(r"[<[].*?[>\]]|.")
        text = _part_pattern.findall(text)

        for part in text:
            if self.is_chinese(part) or self.is_pinyin(part):
                types.append("zh")
            elif self.is_alphabet(part):
                types.append("en")
            else:
                types.append("other")

        if len(types) != len(text):
            msg = f"types length {len(types)} does not match text length {len(text)}"
            raise RuntimeError(msg)

        for i in range(len(types)):
            # find the first char of the seg
            if i == 0:
                temp_seg += text[i]
                temp_lang = types[i]
            else:
                if temp_lang == "other":
                    temp_seg += text[i]
                    temp_lang = types[i]
                else:
                    if types[i] in [temp_lang, "other"]:
                        temp_seg += text[i]
                    else:
                        segments.append((temp_seg, temp_lang))
                        temp_seg = text[i]
                        temp_lang = types[i]

        segments.append((temp_seg, temp_lang))

        # Handle "pinyin" and "tag" types
        segments = self.split_segments(segments)
        return segments

    def split_segments(self, segments):
        """Split segments into smaller parts if special strings enclosed by [] or <> are found.

        Where <> denotes pinyin strings, [] denotes other special strings.

        Args:
            segments (list): A list of tuples where each tuple contains:
                - temp_seg (str): The text segment to be split.
                - temp_lang (str): The language code associated with the segment.

        Returns:
            list: A list of smaller segments.
        """
        result = []
        for temp_seg, temp_lang in segments:
            parts = re.split(r"([<[].*?[>\]])", temp_seg)
            for part in parts:
                if not part:
                    continue
                if self.is_pinyin(part):
                    result.append((part, "pinyin"))
                elif self.is_tag(part):
                    result.append((part, "tag"))
                else:
                    result.append((part, temp_lang))
        return result

    def is_chinese(self, char: str) -> bool:
        """Return True if char is a Chinese character.

        Args:
            char: Single character string to check.

        Returns:
            True if char is in the CJK Unified Ideographs range.
        """
        return bool(char >= "一" and char <= "龥")

    def is_alphabet(self, char: str) -> bool:
        """Return True if char is an ASCII letter.

        Args:
            char: Single character string to check.

        Returns:
            True if char is an ASCII letter.
        """
        return bool(char >= "A" and char <= "Z" or char >= "a" and char <= "z")

    def is_pinyin(self, part: str) -> bool:
        """Return True if part is a pinyin tag enclosed in angle brackets.

        Args:
            part: String to check.

        Returns:
            True if part starts with '<' and ends with '>'.
        """
        return bool(part.startswith("<") and part.endswith(">"))

    def is_tag(self, part: str) -> bool:
        """Return True if part is a special tag enclosed in square brackets.

        Args:
            part: String to check.

        Returns:
            True if part starts with '[' and ends with ']'.
        """
        return bool(part.startswith("[") and part.endswith("]"))


class DialogTokenizer(EmiliaTokenizer):
    """Tokenizer for two-speaker dialog (Emilia-based) with speaker-turn tags."""

    def __init__(self, token_file: str | None = None, token_type="phone"):  # noqa: S107
        """Initialize DialogTokenizer.

        Args:
            token_file: Path to the token file mapping tokens to ids.
            token_type: Token type; only 'phone' is supported.
        """
        super().__init__(token_file=token_file, token_type=token_type)
        if token_file:
            self.spk_a_id = self.token2id["[S1]"]
            self.spk_b_id = self.token2id["[S2]"]

    def preprocess_text(
        self,
        text: str,
    ) -> str:
        """Strip whitespace around speaker tags and map punctuation.

        Args:
            text: Input dialog text.

        Returns:
            Preprocessed text.
        """
        text = re.sub(r"\s*(\[S[12]\])\s*", r"\1", text)
        text = self.map_punctuations(text)
        return text


class LibriTTSTokenizer(Tokenizer):
    """Tokenizer for LibriTTS, supporting bpe, char, and phone token types."""

    def __init__(self, token_file: str | None = None, token_type="char"):  # noqa: S107
        r"""Initialize LibriTTSTokenizer.

        Args:
            token_file: The file that maps tokens to ids (text file with
              '{token}\t{token_id}' per line for char/phone, or a BPE model file).
            token_type: One of 'bpe', 'char', or 'phone'.
        """
        self.type = token_type
        if token_type not in ["bpe", "char", "phone"]:
            msg = f"Unsupported token_type {token_type!r}; expected one of: bpe, char, phone"
            raise TokenizerError(msg)
        try:
            import tacotron_cleaner.cleaners
        except Exception as ex:
            msg = f"{ex}\nPlease run\npip install espnet_tts_frontend"
            raise TokenizerError(msg) from ex

        self.normalize = tacotron_cleaner.cleaners.custom_english_cleaners

        self.has_tokens = False
        if token_file is None:
            log.debug("tokenizer_no_tokens_file", note="will fail when map to ids")
            return
        if token_type == "bpe":  # noqa: S105
            import sentencepiece as spm

            self.sp = spm.SentencePieceProcessor()
            self.sp.load(token_file)
            self.pad_id = self.sp.piece_to_id("<pad>")
            self.vocab_size = self.sp.get_piece_size()
        else:
            self.token2id = _load_token_file(token_file)
            self.pad_id = self.token2id["_"]  # padding
            self.vocab_size = len(self.token2id)
        self.has_tokens = True

    def texts_to_token_ids(
        self,
        texts: list[str],
    ) -> list[list[int]]:
        """Convert texts to token id sequences.

        Args:
            texts: List of input text strings (modified in place).

        Returns:
            List of token id sequences.
        """
        if self.type == "bpe":
            for i in range(len(texts)):
                texts[i] = self.normalize(texts[i])
            return self.sp.encode(texts)
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def texts_to_tokens(
        self,
        texts: list[str],
    ) -> list[list[str]]:
        """Convert texts to token sequences.

        Args:
            texts: List of input text strings (modified in place).

        Returns:
            List of token sequences.
        """
        for i in range(len(texts)):
            texts[i] = self.normalize(texts[i])

        if self.type == "char":
            tokens_list = [list(texts[i]) for i in range(len(texts))]
        elif self.type == "phone":
            tokens_list = [phonemize_espeak(texts[i].lower(), "en-us") for i in range(len(texts))]
        elif self.type == "bpe":
            tokens_list = self.sp.encode(texts, out_type=str)

        return tokens_list

    def tokens_to_token_ids(
        self,
        tokens_list: list[list[str]],
    ) -> list[list[int]]:
        """Convert token sequences to token id sequences.

        Args:
            tokens_list: List of token sequences.

        Returns:
            List of token id sequences.
        """
        if not self.has_tokens:
            msg = "Please initialize Tokenizer with a tokens file."
            raise TokenizerError(msg)
        # BPE tokenizer uses sp.encode() directly; this path is char/phone only.
        if self.type == "bpe":
            msg = "BPE tokenizer does not support this function."
            raise TokenizerError(msg)
        return _tokens_to_token_ids(tokens_list, self.token2id)


def add_tokens(cut_set: CutSet, tokenizer: str, lang: str):
    """Add phoneme tokens to all cuts in a CutSet.

    Args:
        cut_set: The input CutSet to annotate.
        tokenizer: Name of the tokenizer to use (one of 'emilia', 'espeak',
            'dialog', 'libritts', 'simple').
        lang: Language code passed to espeak-based tokenizers.

    Returns:
        CutSet with tokens added to each supervision.
    """
    if tokenizer == "emilia":
        tokenizer = EmiliaTokenizer()
    elif tokenizer == "espeak":
        tokenizer = EspeakTokenizer(lang=lang)
    elif tokenizer == "dialog":
        tokenizer = DialogTokenizer()
    elif tokenizer == "libritts":
        tokenizer = LibriTTSTokenizer()
    elif tokenizer == "simple":
        tokenizer = SimpleTokenizer()
    else:
        msg = f"Unsupported tokenizer: {tokenizer}."
        raise ValueError(msg)

    def _prepare_cut(cut):
        # Each cut only contains one supervision
        if len(cut.supervisions) != 1:
            msg = f"Expected exactly 1 supervision per cut, got {len(cut.supervisions)}: {cut}"
            raise ValueError(msg)
        text = cut.supervisions[0].text
        tokens = tokenizer.texts_to_tokens([text])[0]
        cut.supervisions[0].tokens = tokens
        return cut

    cut_set = cut_set.map(_prepare_cut)
    return cut_set


if __name__ == "__main__":
    text = (
        "我们是5年小米人,是吗? Yes I think so! mr king, 5 years, from 2019 to 2024.霍...啦啦啦超过90%的人<le5>...?!9204"
    )
    tokenizer = EmiliaTokenizer()
    tokens = tokenizer.texts_to_tokens([text])
    log.debug("tokens", tokens="|".join(tokens[0]))
# end zipvoice/tokenizer/tokenizer.py
