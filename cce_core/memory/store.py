"""
CCE Memory Store
Unified interface over all three memory tiers.
This is the only class the rest of the system talks to for storage.
It hides the hot/warm/cold split entirely.

Responsibilities:
  1. Route new turns → hot tier
  2. Detect hot tier evictions → trigger compression → push to warm tier
  3. Provide unified search across warm tier (hot tier is always included verbatim)
  4. Manage session lifecycle: open, checkpoint, close (cold tier write)
  5. Expose a clean retrieve(query) API that returns a ready-to-use context payload

The MemoryStore is stateful per session.
For multi-session use, instantiate one MemoryStore per session,
or use the SessionManager (Phase 4) which handles this automatically.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

from cce_core.config import CCEConfig, DEFAULT_CONFIG
from cce_core.ingestion.segmenter import Turn
from cce_core.compression.merger import MemoryNode
from cce_core.memory.hot_tier import HotTier
from cce_core.memory.warm_tier import WarmTier
from cce_core.memory.cold_tier import ColdTier, MacroSummary

if TYPE_CHECKING:
    pass


class MemoryStore:
    """
    The unified memory interface for one session.

    Typical lifecycle:
        store = MemoryStore(session_id="abc")
        store.ingest_turn(turn)               # called per message
        nodes = store.search(query_emb)       # called at retrieval time
        store.checkpoint()                    # saves cold tier summary
        store.close()                         # flushes and closes DB

    The store auto-compresses evicted hot tier turns into warm tier nodes
    using the provided compress_fn callback (injected by CCEEngine).
    """

    def __init__(
        self,
        session_id: str,
        config: CCEConfig = DEFAULT_CONFIG,
        compress_fn=None,   # Callable[[list[Turn]], list[MemoryNode]]
    ):
        self.session_id = session_id
        self.config = config
        self.compress_fn = compress_fn  # injected from CCEEngine

        self.hot = HotTier(config)
        self.warm = WarmTier(config)
        self.cold = ColdTier(config)

        self._turn_count = 0
        self._raw_token_count = 0
        self._opened_at = datetime.now(timezone.utc)

    # ── Ingest ────────────────────────────────────────────────────────────────

    def ingest_turn(self, turn: Turn) -> list[MemoryNode]:
        """
        Add a new turn to the store.
        If the hot tier evicts a turn, compress it and push to warm tier.
        Returns any new MemoryNodes created (empty list if no eviction).
        """
        self._turn_count += 1
        self._raw_token_count += turn.token_count

        evicted = self.hot.push(turn)
        new_nodes: list[MemoryNode] = []

        if evicted and self.compress_fn:
            # Collect all pending evictions (could be more if push_many was called)
            pending = self.hot.drain_evicted()
            if pending:
                nodes = self.compress_fn(pending, self.session_id)
                for node in nodes:
                    self.warm.upsert(node)
                new_nodes.extend(nodes)

        return new_nodes

    def ingest_turns(self, turns: list[Turn]) -> list[MemoryNode]:
        """
        Bulk ingest a list of turns.
        Evictions are batched and compressed together for efficiency.
        """
        self._turn_count += len(turns)
        self._raw_token_count += sum(t.token_count for t in turns)

        evicted_batch = self.hot.push_many(turns)
        new_nodes: list[MemoryNode] = []

        if evicted_batch and self.compress_fn:
            nodes = self.compress_fn(evicted_batch, self.session_id)
            self.warm.upsert_many(nodes)
            new_nodes.extend(nodes)

        return new_nodes

    def ingest_nodes(self, nodes: list[MemoryNode], turns: list | None = None) -> None:
        """
        Directly write pre-compressed MemoryNodes to warm tier.
        Optionally pass original turns to track raw token count correctly.
        """
        if turns:
            self._turn_count += len(turns)
            self._raw_token_count += sum(t.token_count for t in turns)
        else:
            # Best-effort: reconstruct counts from node metadata
            self._raw_token_count += sum(n.token_count for n in nodes)
            self._turn_count += sum(n.turn_end - n.turn_start + 1 for n in nodes)
        self.warm.upsert_many(nodes)

    # ── Search ────────────────────────────────────────────────────────────────

    def search_warm(
        self,
        query_embedding: np.ndarray,
        top_k: int | None = None,
        recency_boost: float | None = None,
    ) -> list[tuple[MemoryNode, float]]:
        """
        Search the warm tier for relevant memory nodes.
        Returns scored (node, similarity) pairs, sorted descending.
        """
        top_k = top_k or self.config.retrieval_top_k
        boost = recency_boost if recency_boost is not None else self.config.retrieval_recency_boost
        return self.warm.search(
            query_embedding,
            top_k=top_k,
            session_id=self.session_id,
            recency_boost=boost,
        )

    def get_hot_turns(self) -> list[Turn]:
        """Return the current hot tier turns (verbatim recent context)."""
        return self.hot.get_all()

    def get_hot_messages(self) -> list[dict]:
        """Return hot tier as OpenAI-style message list."""
        return self.hot.to_messages()

    # ── Session management ────────────────────────────────────────────────────

    def checkpoint(self, macro_text: str | None = None) -> MacroSummary | None:
        """
        Write a macro summary of this session to the cold tier.
        Call this when a session ends or at periodic intervals.

        Args:
            macro_text: pre-generated macro summary string.
                        If None, skips cold tier write.

        Returns:
            MacroSummary if written, None if skipped.
        """
        if not macro_text:
            return None

        nodes = self.warm.get_by_session(self.session_id)
        summary = self.cold.write_from_nodes(
            session_id=self.session_id,
            nodes=nodes,
            macro_text=macro_text,
            turn_count=self._turn_count,
            token_count_original=self._raw_token_count,
        )
        return summary

    def get_cold_summary(self) -> MacroSummary | None:
        """Retrieve this session's cold tier macro summary if it exists."""
        return self.cold.get(self.session_id)

    def flush_hot_to_warm(self) -> list[MemoryNode]:
        """
        Force-compress all remaining hot tier turns into warm tier.
        Call before checkpoint() to ensure nothing is left uncompressed.
        """
        remaining = self.hot.get_all()
        if not remaining or not self.compress_fn:
            return []
        nodes = self.compress_fn(remaining, self.session_id)
        self.warm.upsert_many(nodes)
        self.hot.clear()
        return nodes

    def close(self):
        """Clean up DB connection."""
        self.warm.close()

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        warm_count = self.warm.count(self.session_id)
        hot_count = self.hot.size()
        cold_exists = self.cold.exists(self.session_id)
        warm_tokens = sum(
            n.token_count for n in self.warm.get_by_session(self.session_id)
        )
        return {
            "session_id": self.session_id,
            "turns_ingested": self._turn_count,
            "raw_tokens": self._raw_token_count,
            "hot_turns": hot_count,
            "warm_nodes": warm_count,
            "warm_tokens_compressed": warm_tokens,
            "cold_summary_exists": cold_exists,
            "compression_ratio": round(
                self._raw_token_count / max(warm_tokens + hot_count * 50, 1), 2
            ),
            "opened_at": self._opened_at.isoformat(),
        }

    def __repr__(self):
        return (
            f"MemoryStore(session={self.session_id!r}, "
            f"hot={self.hot.size()}, warm={self.warm.count(self.session_id)})"
        )