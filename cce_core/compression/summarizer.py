"""
CCE Hierarchical Summarizer
Produces three levels of summary from conversation turns and chunks:

  micro  — one sentence per turn  (most compressed, loses detail)
  meso   — 2-4 sentences per chunk (topic-level summary)
  macro  — one paragraph per session (the full story so far)

Two backends:
  "extractive" — pure Python, no LLM needed, uses sentence scoring
  "llm"        — calls local LLM (Gemma 4 via llama.cpp OpenAI-compat API)

The extractive backend is the default and runs fully offline.
Switch to "llm" in config for higher-quality summaries when the model is up.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

import httpx

from cce_core.config import CCEConfig, DEFAULT_CONFIG
from cce_core.ingestion.segmenter import Turn
from cce_core.ingestion import tokenizer

if TYPE_CHECKING:
    from cce_core.compression.chunker import Chunk


# ── Extractive helpers ────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")

_STOPWORDS = {
    "the", "a", "an", "is", "it", "to", "do", "of", "and", "or",
    "in", "on", "at", "for", "with", "that", "this", "was", "are",
    "be", "have", "has", "had", "you", "your", "we", "can", "will",
    "would", "could", "should", "what", "how", "why", "when", "where",
    "about", "just", "so", "but", "if", "not", "no", "i", "me", "my",
}


def _score_sentences(sentences: list[str], top_words: set[str]) -> list[tuple[float, str]]:
    """Score sentences by keyword overlap with the top content words."""
    scored = []
    for sent in sentences:
        words = set(_WORD_RE.findall(sent.lower())) - _STOPWORDS
        if not words:
            scored.append((0.0, sent))
            continue
        overlap = len(words & top_words) / len(words)
        # Prefer moderate-length sentences (not too short, not too long)
        length_bonus = min(len(sent.split()) / 20, 1.0) * 0.2
        scored.append((overlap + length_bonus, sent))
    return scored


def _extractive_summarize(text: str, max_tokens: int) -> str:
    """
    Extract the most representative sentences from text.
    Falls back to first-N words if text is too short to split.
    """
    sentences = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    if not sentences:
        return tokenizer.truncate_to_tokens(text, max_tokens)

    if len(sentences) == 1:
        return tokenizer.truncate_to_tokens(sentences[0], max_tokens)

    # Build word frequency map
    all_words = _WORD_RE.findall(text.lower())
    freq = Counter(w for w in all_words if w not in _STOPWORDS)
    top_words = {w for w, _ in freq.most_common(20)}

    scored = _score_sentences(sentences, top_words)
    scored.sort(key=lambda x: -x[0])

    # Greedily pick top sentences until token budget is filled
    selected: list[str] = []
    used_tokens = 0
    for _, sent in scored:
        sent_tokens = tokenizer.count(sent)
        if used_tokens + sent_tokens > max_tokens:
            break
        selected.append(sent)
        used_tokens += sent_tokens

    if not selected:
        return tokenizer.truncate_to_tokens(scored[0][1], max_tokens)

    # Re-order by original position
    order = {s: i for i, s in enumerate(sentences)}
    selected.sort(key=lambda s: order.get(s, 9999))
    return " ".join(selected)


# ── LLM backend ───────────────────────────────────────────────────────────────

def _llm_summarize(text: str, max_tokens: int, config: CCEConfig) -> str:
    """
    Call local LLM (llama.cpp OpenAI-compat endpoint) for higher-quality
    abstractive summarization. Falls back to extractive on any error.
    """
    prompt = (
        f"Summarize the following conversation excerpt in at most "
        f"{max_tokens} tokens. Be concise and factual. "
        f"Do not add information not present in the text.\n\n"
        f"Text:\n{text}\n\nSummary:"
    )
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{config.llm_endpoint}/chat/completions",
                json={
                    "model": config.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        # Graceful degradation — never hard-fail
        print(f"[CCE summarizer] LLM call failed ({exc}), using extractive fallback.")
        return _extractive_summarize(text, max_tokens)


# ── Public Summarizer class ───────────────────────────────────────────────────

class Summarizer:
    """
    Hierarchical summarizer. Attach to turns or chunks.

    Usage:
        s = Summarizer()
        micro = s.micro(turn)
        meso  = s.meso(chunk)
        macro = s.macro(all_chunks)
    """

    def __init__(self, config: CCEConfig = DEFAULT_CONFIG):
        self.config = config

    def _summarize(self, text: str, max_tokens: int) -> str:
        if self.config.summarizer_mode == "llm":
            return _llm_summarize(text, max_tokens, self.config)
        return _extractive_summarize(text, max_tokens)

    # ── Micro (per-turn) ──────────────────────────────────────────────────────

    def micro(self, turn: Turn) -> str:
        """
        Condense a single turn to one sentence (~60 tokens).
        If the turn is already short, return it as-is.
        """
        if turn.token_count <= self.config.micro_max_tokens:
            return turn.content
        return self._summarize(turn.content, self.config.micro_max_tokens)

    def micro_batch(self, turns: list[Turn]) -> list[str]:
        """Micro-summarize a list of turns. Returns parallel list of strings."""
        return [self.micro(t) for t in turns]

    # ── Meso (per-chunk) ──────────────────────────────────────────────────────

    def meso(self, chunk: "Chunk") -> str:
        """
        Condense an entire topic chunk to 2-4 sentences (~200 tokens).
        Uses the chunk's full text (all turns concatenated).
        """
        if chunk.token_count <= self.config.meso_max_tokens:
            return chunk.text
        return self._summarize(chunk.text, self.config.meso_max_tokens)

    # ── Macro (per-session) ───────────────────────────────────────────────────

    def macro(self, chunks: list["Chunk"]) -> str:
        """
        Condense the entire session to one paragraph (~400 tokens).
        Operates on meso summaries if available, else on chunk texts.
        Falls back to extractive over all meso summaries concatenated.
        """
        if not chunks:
            return ""

        parts: list[str] = []
        for chunk in chunks:
            if chunk.meso_summary:
                parts.append(chunk.meso_summary)
            else:
                parts.append(chunk.text)

        combined = "\n\n".join(parts)

        if tokenizer.count(combined) <= self.config.macro_max_tokens:
            return combined

        return self._summarize(combined, self.config.macro_max_tokens)

    # ── Convenience: annotate chunk in-place ─────────────────────────────────

    def annotate_chunk(self, chunk: "Chunk") -> "Chunk":
        """
        Fill chunk.micro_summaries and chunk.meso_summary in-place.
        Returns the same chunk (mutated) for easy chaining.
        """
        chunk.micro_summaries = self.micro_batch(chunk.turns)
        chunk.meso_summary = self.meso(chunk)
        return chunk

    def annotate_chunks(self, chunks: list["Chunk"]) -> list["Chunk"]:
        """Annotate a list of chunks. Returns them for convenience."""
        for chunk in chunks:
            self.annotate_chunk(chunk)
        return chunks