"""
CCE Engine — Main Orchestrator
The single entry point for all compression operations.
Phases are wired in progressively:
  Phase 1 — ingestion + compression ✓
  Phase 2 — memory store ✓
  Phase 3 — retrieval (added when retrieval/ is complete)
  Phase 4 — session management
"""

from __future__ import annotations

from cce_core.config import CCEConfig, DEFAULT_CONFIG
from cce_core.ingestion.segmenter import Turn, segment, segment_incremental
from cce_core.compression.chunker import Chunk, SemanticChunker
from cce_core.compression.summarizer import Summarizer
from cce_core.compression.merger import Merger, MemoryNode
from cce_core.memory.store import MemoryStore
from cce_core.retrieval.retriever import Retriever, RetrievalResult
from cce_core.retrieval.context_builder import ContextBuilder, ContextPayload


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
        self.retriever = Retriever(embed_fn=self.chunker.embed_text, config=config)
        self.context_builder = ContextBuilder(config)

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

    # ── Phase 2: Memory store ─────────────────────────────────────────────────

    def open_session(self, session_id: str) -> MemoryStore:
        """
        Open a stateful MemoryStore for a session.
        The compress_fn is injected so the store can auto-compress evictions.

        Usage:
            store = engine.open_session("session-abc")
            store.ingest_turns(turns)
            results = store.search_warm(query_emb)
            store.checkpoint(macro_text)
            store.close()
        """
        return MemoryStore(
            session_id=session_id,
            config=self.config,
            compress_fn=self.compress_incremental,
        )

    def full_pipeline(
        self,
        messages: list[dict] | str,
        session_id: str,
    ) -> tuple[MemoryStore, list[Turn], list[MemoryNode]]:
        """
        Full Phase 1+2 pipeline in one call:
          ingest → compress → store in warm tier

        Returns (store, turns, nodes). Call store.close() when done.
        """
        turns = self.ingest(messages, session_id=session_id)
        nodes = self.compress(turns, session_id=session_id)

        store = self.open_session(session_id)
        store.ingest_nodes(nodes, turns=turns)

        # Push latest turns into hot tier for verbatim recent context
        recent = turns[-self.config.hot_tier_max_turns:]
        for turn in recent:
            store.hot.push(turn)

        return store, turns, nodes

    # ── Phase 3: Retrieval + context assembly ─────────────────────────────────

    def retrieve(
        self,
        store: MemoryStore,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve the most relevant memory nodes for a query.

        Args:
            store: The active MemoryStore for this session.
            query: Natural language query (usually the user's latest message).
            top_k: Number of nodes to retrieve.

        Returns:
            Ranked list of RetrievalResult objects.
        """
        return self.retriever.retrieve(store, query, top_k=top_k)

    def build_context(
        self,
        store: MemoryStore,
        query: str,
        top_k: int | None = None,
        fmt: str = "messages",
        include_micro: bool = False,
    ) -> "ContextPayload":
        """
        Full Phase 3 pipeline: retrieve → build context payload.
        This is what you call right before every LLM inference.

        Args:
            store: The active MemoryStore for this session.
            query: The user's latest message / question.
            top_k: Number of memory nodes to retrieve.
            fmt: Output format — "messages", "string", or "dict".
            include_micro: Include per-turn micro summaries in past blocks.

        Returns:
            ContextPayload — call .to_messages(), .to_string(), or .to_dict().

        Example:
            payload = engine.build_context(store, query="what is gradient descent?")
            messages = payload.to_messages()
            # → inject messages into llama.cpp / OpenAI API call
        """
        results = self.retriever.retrieve(store, query, top_k=top_k)
        hot_turns = store.get_hot_turns()

        if not results:
            return self.context_builder.build_empty(hot_turns)

        return self.context_builder.build(
            results=results,
            hot_turns=hot_turns,
            include_micro=include_micro,
        )

    def query(
        self,
        store: MemoryStore,
        user_message: str,
        fmt: str = "messages",
    ) -> tuple[list[dict] | str | dict, "ContextPayload"]:
        """
        One-shot: ingest a new user message, build context, return payload.
        Designed for the stateful real-time use case:

            for user_msg in conversation:
                context, payload = engine.query(store, user_msg)
                response = call_llm(context)
                engine.ingest_one({"role": "assistant", "content": response}, store._turns)

        Returns (exported_context, payload) — use exported_context for LLM call,
        payload for diagnostics.
        """
        payload = self.build_context(store, user_message, fmt=fmt)
        return payload.export(fmt), payload

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