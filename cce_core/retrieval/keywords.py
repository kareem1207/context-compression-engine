"""Keyword extraction shared by the retriever and the warm-tier index."""

from __future__ import annotations

import re

WORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")

STOPWORDS = {
    "the", "a", "an", "is", "it", "to", "do", "of", "and", "or",
    "in", "on", "at", "for", "with", "that", "this", "was", "are",
    "be", "have", "has", "had", "you", "your", "we", "can", "will",
    "would", "could", "should", "what", "how", "why", "when", "where",
    "about", "just", "so", "but", "if", "not", "no", "i", "me", "my",
    "tell", "explain", "describe", "give", "show", "help",
}


def keyword_set(text: str) -> frozenset[str]:
    """Lower-cased content words (3+ letters, no stopwords)."""
    return frozenset(w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS)
