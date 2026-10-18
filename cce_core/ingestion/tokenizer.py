"""
CCE Tokenizer
Token counter for context budgets. It works offline and never downloads anything.

  1. If `tiktoken` is installed (extra: `pip install context-compression-engine[tokens]`) and its
     cl100k_base encoding is already in the local cache, counts are exact for that encoding.
  2. Otherwise a character-aware estimate is used. It takes the larger of
       - a word estimate (1 token ~ 0.75 words), right for English prose,
       - a character estimate: about 1.1 tokens per CJK character, about 1 token per 2.2
         characters for symbol-heavy text (code, JSON), about 1 per 4.5 characters otherwise.
     The estimate leans high, so a hard token cap built on it is not overshot.

Other models tokenize differently from cl100k_base; treat every count as an estimate.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from functools import lru_cache

# Hiragana/Katakana, CJK ext A, CJK unified, Hangul syllables, CJK compatibility, fullwidth forms.
_CJK = re.compile("[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿＀-￯]")

_CJK_TOKENS_PER_CHAR = 1.1
_SYMBOL_CHARS_PER_TOKEN = 2.2
_TEXT_CHARS_PER_TOKEN = 4.5
_SYMBOL_HEAVY_RATIO = 0.25   # share of non-space characters that are punctuation, symbols or digits

_LAYOUT = re.compile(r"\n|\t| {2,}")   # newlines, tabs and indentation cost tokens of their own

_ENCODING_NAME = "cl100k_base"
_ENCODING_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"


@lru_cache(maxsize=1)
def _encoder():
    """The tiktoken encoder if it can be built without network access, else None."""
    try:
        import tiktoken
    except ImportError:
        return None
    # tiktoken downloads the vocabulary on first use unless it is in its cache dir. Look first.
    cache_dir = (
        os.environ.get("TIKTOKEN_CACHE_DIR")
        or os.environ.get("DATA_GYM_CACHE_DIR")
        or os.path.join(tempfile.gettempdir(), "data-gym-cache")
    )
    cached = os.path.join(cache_dir, hashlib.sha1(_ENCODING_URL.encode()).hexdigest())
    if not os.path.isfile(cached):
        return None
    try:
        return tiktoken.get_encoding(_ENCODING_NAME)
    except Exception:
        return None


def estimate(text: str) -> int:
    """Character-aware token estimate (no tiktoken). See the module docstring."""
    if not text or not text.strip():
        return 0
    cjk = len(_CJK.findall(text))
    rest = _CJK.sub(" ", text)
    cjk_tokens = cjk * _CJK_TOKENS_PER_CHAR

    by_words = len(rest.split()) / 0.75 + cjk_tokens

    non_space = "".join(rest.split())
    by_chars = cjk_tokens
    if non_space:
        symbols = sum(1 for c in non_space if not c.isalpha())
        heavy = symbols / len(non_space) >= _SYMBOL_HEAVY_RATIO
        by_chars += len(non_space) / (_SYMBOL_CHARS_PER_TOKEN if heavy else _TEXT_CHARS_PER_TOKEN)
        by_chars += len(_LAYOUT.findall(rest))
    return max(1, round(max(by_words, by_chars)))


def count(text: str) -> int:
    """Token count for a piece of text: exact via tiktoken when available offline, else estimated."""
    if not text or not text.strip():
        return 0
    enc = _encoder()
    if enc is not None:
        return max(1, len(enc.encode(text, disallowed_special=())))
    return estimate(text)


def count_messages(messages: list[dict]) -> int:
    """
    Count total tokens across a list of {"role": ..., "content": ...} dicts.
    Adds 4 tokens per message for role/formatting overhead (OpenAI-style).
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        total += count(content) + 4  # role overhead
    return total


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Cut text so that it counts at most max_tokens (including the trailing " ...").
    Cuts by characters, preferring a word boundary, so CJK text and long unbroken strings work too.
    """
    if max_tokens <= 0:
        return ""
    if count(text) <= max_tokens:
        return text

    text = text.strip()
    lo, hi = 0, len(text)   # largest prefix length whose text + " ..." fits
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(text[:mid].rstrip() + " ...") <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    cut = text[:lo]
    boundary = max(cut.rfind(" "), cut.rfind("\n"))
    if boundary > lo * 0.6:   # back up to a word boundary unless that throws away too much
        cut = cut[:boundary]
    cut = cut.rstrip()
    return (cut + " ...") if cut else ""


def fits_in_budget(text: str, budget: int) -> bool:
    return count(text) <= budget
