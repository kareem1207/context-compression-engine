"""
CCE Stateless Mode
On-demand compression and retrieval with zero persistence.
Every call is self-contained — no DB, no disk writes.

Use this when:
  - You want CCE as a one-shot transformer (compress this conversation now)
  - You're building a plugin that can't maintain state between calls
  - You want to test compression quality without spinning up a full store

The StatelessProcessor takes a full conversation, compresses it in-memory,
and returns a ready-to-use context payload in a single call.

No MemoryStore, no SQLite, no disk. Pure in-memory pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cce_core.config import CCEConfig, DEFAULT_CONFIG
from cce_core.ingestion.segmenter import segment, Turn
from cce_core.compression.merger import MemoryNode
from cce_core.retrieval.context_builder import ContextBuilder, ContextPayload

if TYPE_CHECKING:
    from cce_core.engine import CCEEngine


@dataclass
class StatelessResult:
    """Output of a stateless compression + retrieval call."""
    turns: list[Turn]
    nodes: list[MemoryNode]
    payload: ContextPayload
    stats: dict

    def to_messages(self) -> list[dict]:
        return self.payload.to_messages()

    def to_string(self) -> str:
        return self.payload.to_string()

    def to_dict(self) -> dict:
        return {
            "payload": self.payload.to_dict(),
            "stats": self.stats,
        }


class StatelessProcessor:
    """
    Stateless CCE processor. Zero persistence, pure in-memory.

    Usage:
        processor = StatelessProcessor(engine)

        # One shot: full conversation → compressed context
        result = processor.process(
            messages=[...],
            query="what did we discuss about Python?"
        )
        messages_for_llm = result.to_messages()

        # Or just get compression stats
        stats = processor.stats(messages)
    """

    def __init__(self, engine: "CCEEngine"):
        self.engine = engine
        self.config = engine.config
        self.builder = ContextBuilder(engine.config)

    def process(
        self,
        messages: list[dict] | str,
        query: str,
        top_k: int | None = None,
        include_micro: bool = False,
        session_id: str = "stateless",
    ) -> StatelessResult:
        """
        Full stateless pipeline: ingest → compress → retrieve → build context.

        Args:
            messages: Full conversation history.
            query: The current user query (used for retrieval).
            top_k: Number of memory nodes to retrieve.
            include_micro: Include per-turn micro summaries in past blocks.
            session_id: Label for the session (doesn't persist).

        Returns:
            StatelessResult with payload ready for LLM injection.
        """
        top_k = top_k or self.config.retrieval_top_k

        # Ingest + compress
        turns = self.engine.ingest(messages, session_id=session_id)
        nodes = self.engine.compress(turns, session_id=session_id)

        # Split into hot (recent) and warm (past) in-memory
        n_hot = self.config.hot_tier_max_turns
        hot_turns = turns[-n_hot:] if len(turns) > n_hot else turns
        past_nodes = nodes  # all nodes go through retrieval

        # In-memory retrieval (no DB — use pure vector math)
        query_emb = self.engine.chunker.embed_text(query)
        scored = self._score_nodes(query_emb, past_nodes)
        top_results = scored[:top_k]

        # Build context payload
        from cce_core.retrieval.retriever import RetrievalResult
        retrieval_results = [
            RetrievalResult(
                node=node,
                semantic_score=float(score),
                recency_score=0.0,
                keyword_score=0.0,
                composite_score=float(score),
            )
            for node, score in top_results
        ]

        if not retrieval_results:
            payload = self.builder.build_empty(hot_turns)
        else:
            payload = self.builder.build(
                results=retrieval_results,
                hot_turns=hot_turns,
                include_micro=include_micro,
            )

        stats = self.engine.compression_stats(turns, nodes)
        stats["hot_turns"] = len(hot_turns)
        stats["retrieved_nodes"] = len(retrieval_results)

        return StatelessResult(
            turns=turns,
            nodes=nodes,
            payload=payload,
            stats=stats,
        )

    def compress_only(
        self,
        messages: list[dict] | str,
        session_id: str = "stateless",
    ) -> tuple[list[Turn], list[MemoryNode]]:
        """
        Just compress — no retrieval, no context building.
        Useful for batch processing or pre-compression jobs.
        """
        turns = self.engine.ingest(messages, session_id=session_id)
        nodes = self.engine.compress(turns, session_id=session_id)
        return turns, nodes

    def stats(self, messages: list[dict] | str) -> dict:
        """
        Return compression stats without building a full context payload.
        Fast diagnostic call.
        """
        turns, nodes = self.compress_only(messages)
        return self.engine.compression_stats(turns, nodes)

    def _score_nodes(
        self,
        query_emb,
        nodes: list[MemoryNode],
    ) -> list[tuple[MemoryNode, float]]:
        """
        Pure in-memory cosine similarity scoring.
        Returns sorted (node, score) pairs, best first.
        """
        import numpy as np

        if not nodes:
            return []

        matrix = np.stack([n.embedding for n in nodes], axis=0)
        scores = matrix @ query_emb.astype("float32")
        order = scores.argsort()[::-1]
        return [(nodes[int(i)], float(scores[i])) for i in order]