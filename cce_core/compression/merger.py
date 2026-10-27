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
import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from cce_core.compression.chunker import Chunk
from cce_core.config import CCEConfig, DEFAULT_CONFIG
from cce_core.text import term_counts


def encode_turn_vecs(vecs: np.ndarray) -> dict:
    """Per-turn embeddings as compact JSON-safe data (float16, base64)."""
    a = np.asarray(vecs, dtype=np.float16)
    return {"dim": int(a.shape[1]), "data": base64.b64encode(a.tobytes()).decode("ascii")}


def decode_turn_vecs(metadata: dict) -> np.ndarray | None:
    """Inverse of encode_turn_vecs; None for nodes created before turn vectors existed."""
    tv = (metadata or {}).get("turn_vecs")
    if not tv:
        return None
    return np.frombuffer(base64.b64decode(tv["data"]), dtype=np.float16).astype(np.float32).reshape(-1, tv["dim"])


def decode_sentence_vecs(metadata: dict):
    """(vectors, [[turn position, sentence number], ...]) of a note's sentence index, or None."""
    meta = metadata or {}
    if not meta.get("sent_vecs") or not meta.get("sent_map"):
        return None
    vecs = np.frombuffer(base64.b64decode(meta["sent_vecs"]["data"]), dtype=np.float16).astype(np.float32)
    vecs = vecs.reshape(-1, meta["sent_vecs"]["dim"])
    return (vecs, meta["sent_map"]) if len(vecs) == len(meta["sent_map"]) else None


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

        # Text the lexical (BM25) search runs on: the original messages when they are kept, else the summaries.
        if self.config.keep_raw_turns:
            body = " ".join(t.content for t in chunk.turns)
        else:
            body = " ".join(micros)
        tf = term_counts(f"{chunk.topic_label} {meso} {body}")
        roles = "".join(t.role[0] for t in chunk.turns)      # u(ser) / a(ssistant) / s(ystem) / t(ool), per turn
        sentence_meta = {}
        if chunk.sentence_embeddings is not None and len(chunk.sentence_map):
            sentence_meta = {"sent_vecs": encode_turn_vecs(chunk.sentence_embeddings), "sent_map": chunk.sentence_map}

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
            metadata={
                **chunk.metadata,
                "tf": tf,
                "roles": roles,
                **sentence_meta,
                **({"turn_vecs": encode_turn_vecs(chunk.turn_embeddings)} if chunk.turn_embeddings is not None else {}),
            },
        )


def merge_nodes(a: MemoryNode, b: MemoryNode, config: CCEConfig = DEFAULT_CONFIG) -> MemoryNode:
    """
    Fuse two adjacent notes (`a` is the earlier one) into one coarser note: a shorter combined
    summary, the union of their turns, a weighted centroid. The per-turn summaries and vectors are
    kept, so single facts inside the merged range can still be found.
    """
    from cce_core.compression.summarizer import _extractive_summarize

    na = max(a.turn_end - a.turn_start + 1, 1)
    nb = max(b.turn_end - b.turn_start + 1, 1)
    emb = a.embedding * na + b.embedding * nb
    norm = np.linalg.norm(emb)
    emb = (emb / norm if norm > 1e-8 else emb).astype(np.float32)

    label = a.topic_label if a.topic_label == b.topic_label else f"{a.topic_label}; {b.topic_label}"
    if len(label) > 80:
        label = label[:77].rstrip() + "..."

    metadata = {k: v for k, v in {**a.metadata, **b.metadata}.items() if k != "turn_vecs"}
    va, vb = decode_turn_vecs(a.metadata), decode_turn_vecs(b.metadata)
    if va is not None and vb is not None and va.shape[1] == vb.shape[1]:
        metadata["turn_vecs"] = encode_turn_vecs(np.concatenate([va, vb], axis=0))
    metadata.pop("sent_vecs", None)
    metadata.pop("sent_map", None)
    sa, sb = decode_sentence_vecs(a.metadata), decode_sentence_vecs(b.metadata)
    shift = len(a.micro_summaries)
    if sa is not None and sb is not None and sa[0].shape[1] == sb[0].shape[1]:
        metadata["sent_vecs"] = encode_turn_vecs(np.concatenate([sa[0], sb[0]], axis=0))
        metadata["sent_map"] = [*sa[1], *([i + shift, k] for i, k in sb[1])]
        metadata["sent_map"] = [list(m) for m in metadata["sent_map"]]
    elif sa is not None and sb is None:
        metadata["sent_vecs"], metadata["sent_map"] = a.metadata["sent_vecs"], a.metadata["sent_map"]
    elif sb is not None and sa is None:
        metadata["sent_vecs"] = b.metadata["sent_vecs"]
        metadata["sent_map"] = [[i + shift, k] for i, k in sb[1]]
    tf = dict(a.metadata.get("tf") or {})
    for term, n in (b.metadata.get("tf") or {}).items():
        tf[term] = tf.get(term, 0) + n
    if tf:
        metadata["tf"] = dict(sorted(tf.items(), key=lambda kv: -kv[1])[:600])
    if a.metadata.get("roles") is not None and b.metadata.get("roles") is not None:
        metadata["roles"] = a.metadata["roles"] + b.metadata["roles"]
    else:
        metadata.pop("roles", None)
    metadata["merged_notes"] = a.metadata.get("merged_notes", 1) + b.metadata.get("merged_notes", 1)

    return MemoryNode(
        node_id=str(uuid.uuid4()),
        chunk_id=a.chunk_id,
        session_id=a.session_id,
        embedding=emb,
        topic_label=label,
        meso_summary=_extractive_summarize(f"{a.meso_summary} {b.meso_summary}", config.meso_max_tokens),
        micro_summaries=[*a.micro_summaries, *b.micro_summaries],
        turn_start=min(a.turn_start, b.turn_start),
        turn_end=max(a.turn_end, b.turn_end),
        token_count=a.token_count + b.token_count,
        created_at=datetime.now(timezone.utc),
        metadata=metadata,
    )
