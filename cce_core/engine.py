"""
CCE Engine — Main Orchestrator
The single entry point for all compression operations.
Phases are wired in progressively:
  Phase 1 — ingestion + compression (this file, partial)
  Phase 2 — memory store (added when memory/ is complete)
  Phase 3 — retrieval (added when retrieval/ is complete)
  Phase 4 — session management
"""

from __future__ import annotations

from cce_core.config import CCEConfig, DEFAULT_CONFIG
from cce_core.ingestion.segmenter import Turn, segment, segment_incremental
from cce_core.compression.chunker import Chunk, SemanticChunker
from cce_core.compression.summarizer import Summarizer
from cce_core.compression.merger import Merger, MemoryNode


class CCEEngine:
    """
    High-level API for the Context Compression Engine.

    Phase 1 operations (available now):
        engine.ingest(messages)          → list[Turn]
        engine.compress(turns)           → list[MemoryNode]
        engine.ingest_and_compress(msgs) → list[MemoryNode]

    Phases 2-4 operations will be added as memory/ and retrieval/ are built.
    """

    def __init__(self, config: CCEConfig = DEFAULT_CONFIG):
        self.config = config
        self.chunker = SemanticChunker(config)
        self.summarizer = Summarizer(config)
        self.merger = Merger(config)

    # ── Phase 1: Ingest ───────────────────────────────────────────────────────

    def ingest(
        self,
        messages: list[dict] | str,
        session_id: str | None = None,
    ) -> list[Turn]:
        """
        Convert raw messages into Turn objects.

        Args:
            messages: List of {"role", "content"} dicts or plain string.
            session_id: Optional session identifier.

        Returns:
            Ordered list of Turn objects.
        """
        return segment(messages, session_id=session_id)

    def ingest_one(
        self,
        message: dict,
        existing_turns: list[Turn],
        session_id: str | None = None,
    ) -> Turn:
        """Add a single new message to an existing turns list (stateful mode)."""
        return segment_incremental(message, existing_turns, session_id)

    # ── Phase 1: Compress ─────────────────────────────────────────────────────

    def compress(
        self,
        turns: list[Turn],
        session_id: str | None = None,
    ) -> list[MemoryNode]:
        """
        Full compression pipeline: chunk → summarize → merge.

        Args:
            turns: List of Turn objects from ingest().
            session_id: Optional session identifier propagated to nodes.

        Returns:
            List of MemoryNode objects ready for storage or direct use.
        """
        # 1. Semantic chunking
        chunks: list[Chunk] = self.chunker.chunk(turns, session_id=session_id)

        # 2. Hierarchical summarization (fills micro + meso on each chunk)
        self.summarizer.annotate_chunks(chunks)

        # 3. Merge into MemoryNodes
        nodes: list[MemoryNode] = self.merger.merge(chunks)

        return nodes

    def compress_incremental(
        self,
        new_turns: list[Turn],
        session_id: str | None = None,
    ) -> list[MemoryNode]:
        """
        Compress only new turns (e.g. since last checkpoint).
        Same pipeline as compress() — use this in stateful mode
        to avoid re-processing the entire history on every message.
        """
        return self.compress(new_turns, session_id=session_id)

    # ── Phase 1: Convenience ──────────────────────────────────────────────────

    def ingest_and_compress(
        self,
        messages: list[dict] | str,
        session_id: str | None = None,
    ) -> tuple[list[Turn], list[MemoryNode]]:
        """
        One-shot: ingest + compress.
        Returns both turns and nodes so the caller has full visibility.
        """
        turns = self.ingest(messages, session_id=session_id)
        nodes = self.compress(turns, session_id=session_id)
        return turns, nodes

    def macro_summary(self, nodes: list[MemoryNode]) -> str:
        """
        Produce a session-level macro summary from a list of MemoryNodes.
        Useful for generating a 'story so far' paragraph.
        """
        # Build fake chunks from nodes for the summarizer's macro() method
        # We operate directly on meso summaries here for efficiency
        if not nodes:
            return ""
        combined = "\n\n".join(n.meso_summary for n in nodes if n.meso_summary)
        from cce_core.ingestion import tokenizer
        if tokenizer.count(combined) <= self.config.macro_max_tokens:
            return combined
        from cce_core.compression.summarizer import _extractive_summarize
        return _extractive_summarize(combined, self.config.macro_max_tokens)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def compression_stats(
        self,
        turns: list[Turn],
        nodes: list[MemoryNode],
    ) -> dict:
        """
        Return compression metrics for logging / evaluation.
        """
        raw_tokens = sum(t.token_count for t in turns)
        compressed_tokens = sum(
            len(n.meso_summary.split()) for n in nodes
        )
        ratio = raw_tokens / max(compressed_tokens, 1)

        return {
            "total_turns": len(turns),
            "total_chunks": len(nodes),
            "raw_tokens": raw_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": round(ratio, 2),
            "topics": [n.topic_label for n in nodes],
        }