"""
CCE Tokenizer
Lightweight token counter — no heavy tokenizer dependency.
Uses a word-based approximation (1 token ≈ 0.75 words) which is accurate
enough for context budget decisions. Swap out count() for a tiktoken-based
implementation if you need exact counts later.
"""

import re
from typing import Union


_WHITESPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def count(text: str) -> int:
    """
    Estimate token count for a piece of text.
    Formula: (word_count / 0.75) rounds to nearest int.
    Handles empty strings and strips excessive whitespace.
    """
    if not text or not text.strip():
        return 0
    # Normalize whitespace, split into words
    words = _WHITESPACE.split(text.strip())
    word_count = len(words)
    return max(1, round(word_count / 0.75))


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
    Hard-truncate text to approximately max_tokens.
    Truncates at word boundary to avoid mid-word cuts.
    """
    if count(text) <= max_tokens:
        return text

    words = _WHITESPACE.split(text.strip())
    target_words = max(1, round(max_tokens * 0.75))
    return " ".join(words[:target_words]) + " ..."


def fits_in_budget(text: str, budget: int) -> bool:
    return count(text) <= budget