"""
CCE Semantic Chunker
Groups conversation turns into topically coherent chunks using
sentence-transformers (all-MiniLM-L6-v2) + cosine similarity sliding window.

Algorithm:
  1. Embed each turn's content using SBERT.
  2. Compute cosine similarity between every adjacent pair of turns.
  3. A topic boundary is declared when similarity drops below the threshold
     OR the window's rolling average drops significantly.
  4. Hard splits enforce min/max turn counts per chunk.

A Chunk is the core unit stored in the warm memory tier.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

from cce_core.config import CCEConfig, DEFAULT_CONFIG
from cce_core.ingestion.segmenter import Turn

if TYPE_CHECKING:
    pass


@dataclass
class Chunk:
    """A topically coherent group of turns."""
    chunk_id: str
    turns: list[Turn]
    topic_label: str              # short auto-generated label
    embedding: np.ndarray         # centroid of all turn embeddings
    created_at: datetime
    session_id: str | None = None
    metadata: dict = field(default_factory=dict)

    # filled by summarizer later
    micro_summaries: list[str] = field(default_factory=list)
    meso_summary: str = ""

    @property
    def start_index(self) -> int:
        return self.turns[0].index if self.turns else 0

    @property
    def end_index(self) -> int:
        return self.turns[-1].index if self.turns else 0

    @property
    def token_count(self) -> int:
        return sum(t.token_count for t in self.turns)

    @property
    def text(self) -> str:
        """Full text of the chunk — role: content per turn."""
        return "\n".join(f"{t.role}: {t.content}" for t in self.turns)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "turns": [t.to_dict() for t in self.turns],
            "topic_label": self.topic_label,
            "embedding": self.embedding.tolist(),
            "created_at": self.created_at.isoformat(),
            "session_id": self.session_id,
            "metadata": self.metadata,
            "micro_summaries": self.micro_summaries,
            "meso_summary": self.meso_summary,
        }

    def __repr__(self):
        return (
            f"Chunk(id={self.chunk_id[:8]}, turns={len(self.turns)}, "
            f"tokens={self.token_count}, topic={self.topic_label!r})"
        )


class SemanticChunker:
    """
    Splits a list of Turn objects into semantically coherent Chunks.

    Usage:
        chunker = SemanticChunker()
        chunks = chunker.chunk(turns)
    """

    def __init__(self, config: CCEConfig = DEFAULT_CONFIG):
        self.config = config
        self._model: SentenceTransformer | None = None  # lazy load

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.config.embedding_model)
        return self._model

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk(
        self,
        turns: list[Turn],
        session_id: str | None = None,
    ) -> list[Chunk]:
        """
        Main entry point. Takes a list of turns, returns a list of Chunks.
        Handles edge cases: empty input, single turn, fewer turns than window.
        """
        if not turns:
            return []

        if len(turns) == 1:
            return [self._make_chunk([turns[0]], session_id)]

        embeddings = self._embed_turns(turns)
        boundaries = self._detect_boundaries(embeddings)
        groups = self._split_at_boundaries(turns, boundaries)
        return [self._make_chunk(group, session_id) for group in groups]

    def embed_text(self, text: str) -> np.ndarray:
        """Embed arbitrary text — used at retrieval time for query embedding."""
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _embed_turns(self, turns: list[Turn]) -> np.ndarray:
        """
        Embed all turns in one batched call.
        Returns shape (N, embedding_dim) float32 array.
        Normalized so cosine similarity = dot product.
        """
        texts = [t.content for t in turns]
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def _detect_boundaries(self, embeddings: np.ndarray) -> set[int]:
        """
        Identify turn indices where a topic boundary should be placed.
        A boundary before index i means: start a new chunk at i.

        Strategy:
          - Compute pairwise cosine similarity between adjacent windows.
          - If similarity between window ending at i and window starting at i+1
            drops below threshold → boundary at i+1.
          - Also enforce max_turns hard cap.
        """
        n = len(embeddings)
        if n < 2:
            return set()

        boundaries: set[int] = set()
        w = self.config.chunk_window_size
        threshold = self.config.chunk_similarity_threshold

        # Compute smoothed similarity: average embedding of window vs next window
        similarities: list[float] = []
        for i in range(n - 1):
            left_start = max(0, i - w + 1)
            right_end = min(n, i + w + 1)
            left_vec = embeddings[left_start : i + 1].mean(axis=0)
            right_vec = embeddings[i + 1 : right_end].mean(axis=0)
            # cosine similarity (embeddings already normalized)
            sim = float(np.dot(left_vec, right_vec) / (
                np.linalg.norm(left_vec) * np.linalg.norm(right_vec) + 1e-8
            ))
            similarities.append(sim)

        # Boundary where similarity drops below threshold
        for i, sim in enumerate(similarities):
            if sim < threshold:
                boundaries.add(i + 1)  # boundary before the next turn

        # Enforce max_turns hard cap
        max_t = self.config.chunk_max_turns
        chunk_start = 0
        for i in range(n):
            if i - chunk_start >= max_t:
                boundaries.add(i)
                chunk_start = i
            if i in boundaries:
                chunk_start = i

        # Remove boundaries that would create chunks smaller than min_turns
        min_t = self.config.chunk_min_turns
        clean: set[int] = set()
        sorted_b = sorted(boundaries)
        prev = 0
        for b in sorted_b:
            if b - prev >= min_t:
                clean.add(b)
                prev = b
        return clean

    def _split_at_boundaries(
        self,
        turns: list[Turn],
        boundaries: set[int],
    ) -> list[list[Turn]]:
        """Split turns into groups at the detected boundary indices."""
        groups: list[list[Turn]] = []
        current: list[Turn] = []
        for i, turn in enumerate(turns):
            if i in boundaries and current:
                groups.append(current)
                current = []
            current.append(turn)
        if current:
            groups.append(current)
        return groups

    def _make_chunk(
        self,
        turns: list[Turn],
        session_id: str | None,
    ) -> Chunk:
        """Build a Chunk from a group of turns, computing its centroid embedding."""
        texts = [t.content for t in turns]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # Centroid = mean of all turn embeddings (re-normalized)
        centroid = embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-8:
            centroid = centroid / norm

        topic_label = _infer_topic_label(turns)

        return Chunk(
            chunk_id=str(uuid.uuid4()),
            turns=turns,
            topic_label=topic_label,
            embedding=centroid,
            created_at=datetime.now(timezone.utc),
            session_id=session_id,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "i", "me", "my", "the", "a", "an", "is", "it", "to", "do", "of",
    "and", "or", "in", "on", "at", "for", "with", "that", "this",
    "was", "are", "be", "have", "has", "had", "you", "your", "we",
    "can", "will", "would", "could", "should", "what", "how", "why",
    "when", "where", "about", "just", "so", "but", "if", "not", "no",
}


def _infer_topic_label(turns: list[Turn]) -> str:
    """
    Generate a short topic label from the most frequent non-stopword
    content words across all turns in the chunk.
    Falls back to 'turn <start>-<end>' if nothing useful found.
    """
    from collections import Counter
    import re

    word_re = re.compile(r"\b[a-zA-Z]{3,}\b")
    counter: Counter = Counter()

    for turn in turns:
        words = word_re.findall(turn.content.lower())
        counter.update(w for w in words if w not in _STOPWORDS)

    if not counter:
        start = turns[0].index
        end = turns[-1].index
        return f"turns {start}-{end}"

    top = counter.most_common(3)
    return " · ".join(w for w, _ in top)