"""
CCE Merger
Takes the output of the chunker (Chunk objects with embeddings) and the
summarizer (micro/meso summaries) and fuses them into MemoryNode objects
ready to be written into the memory store.

A MemoryNode is the final compressed artifact — it contains:
  - The chunk's centroid embedding (for ANN retrieval)
  - The meso summary (what this topic was about)
  - Lightweight metadata (turn range, token count, session, timestamp)
  - The micro summaries (one per turn, for fine-grained reconstruction)

The raw turn text is NOT stored in the MemoryNode — it lives in the hot tier.
This keeps the warm tier lean.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from cce_core.compression.chunker import Chunk
from cce_core.config import CCEConfig, DEFAULT_CONFIG


@dataclass
class MemoryNode:
    """
    The atomic unit of the CCE warm memory tier.
    Produced by the Merger, consumed by the MemoryStore and Retriever.
    """
    node_id: str
    chunk_id: str                  # back-reference to source Chunk
    session_id: str | None

    # Retrieval surface
    embedding: np.ndarray          # centroid of chunk (float32, normalized)
    topic_label: str               # human-readable topic

    # Summaries
    meso_summary: str              # 2-4 sentence chunk summary
    micro_summaries: list[str]     # one sentence per turn

    # Positional metadata
    turn_start: int                # index of first turn in chunk
    turn_end: int                  # index of last turn in chunk
    token_count: int               # raw token count of original turns

    created_at: datetime
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "chunk_id": self.chunk_id,
            "session_id": self.session_id,
            "embedding": self.embedding.tolist(),
            "topic_label": self.topic_label,
            "meso_summary": self.meso_summary,
            "micro_summaries": self.micro_summaries,
            "turn_start": self.turn_start,
            "turn_end": self.turn_end,
            "token_count": self.token_count,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryNode":
        return cls(
            node_id=d["node_id"],
            chunk_id=d["chunk_id"],
            session_id=d.get("session_id"),
            embedding=np.array(d["embedding"], dtype=np.float32),
            topic_label=d["topic_label"],
            meso_summary=d["meso_summary"],
            micro_summaries=d.get("micro_summaries", []),
            turn_start=d["turn_start"],
            turn_end=d["turn_end"],
            token_count=d["token_count"],
            created_at=datetime.fromisoformat(d["created_at"]),
            metadata=d.get("metadata", {}),
        )

    @property
    def compressed_text(self) -> str:
        """
        The text representation injected into LLM context.
        Format: topic label + meso summary.
        Keeps it readable and scannable for the model.
        """
        return f"[Topic: {self.topic_label}]\n{self.meso_summary}"

    def __repr__(self):
        return (
            f"MemoryNode(id={self.node_id[:8]}, "
            f"topic={self.topic_label!r}, "
            f"turns={self.turn_start}-{self.turn_end}, "
            f"tokens={self.token_count})"
        )


class Merger:
    """
    Merges annotated Chunk objects into MemoryNode objects.

    Expects chunks to already have micro_summaries and meso_summary
    filled in (i.e. Summarizer.annotate_chunk() has been called).

    Usage:
        merger = Merger()
        nodes = merger.merge(chunks)
    """

    def __init__(self, config: CCEConfig = DEFAULT_CONFIG):
        self.config = config

    def merge(self, chunks: list[Chunk]) -> list[MemoryNode]:
        """Convert a list of annotated Chunks to MemoryNodes."""
        return [self._chunk_to_node(chunk) for chunk in chunks]

    def merge_one(self, chunk: Chunk) -> MemoryNode:
        """Convert a single annotated Chunk to a MemoryNode."""
        return self._chunk_to_node(chunk)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _chunk_to_node(self, chunk: Chunk) -> MemoryNode:
        # If meso_summary wasn't filled, use the topic label as fallback
        meso = chunk.meso_summary or f"Conversation about: {chunk.topic_label}"

        # If micro_summaries weren't filled, use raw turn content
        micros = chunk.micro_summaries or [t.content for t in chunk.turns]

        # Ensure embedding is float32 and normalized
        emb = chunk.embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm

        return MemoryNode(
            node_id=str(uuid.uuid4()),
            chunk_id=chunk.chunk_id,
            session_id=chunk.session_id,
            embedding=emb,
            topic_label=chunk.topic_label,
            meso_summary=meso,
            micro_summaries=micros,
            turn_start=chunk.start_index,
            turn_end=chunk.end_index,
            token_count=chunk.token_count,
            created_at=datetime.now(timezone.utc),
            metadata=chunk.metadata.copy(),
        )