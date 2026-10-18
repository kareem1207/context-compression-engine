"""
CCE Memory Store
Unified interface over all three memory tiers.
This is the only class the rest of the system talks to for storage.
It hides the hot/warm/cold split entirely.

Responsibilities:
  1. Route new turns → hot tier
  2. Evicted turns wait in a pending buffer → topic-sized compression → warm tier
  3. Provide unified search across warm tier (hot tier is always included verbatim)
  4. Manage session lifecycle: open, checkpoint, close (cold tier write)
  5. Expose a clean retrieve(query) API that returns a ready-to-use context payload

The MemoryStore is stateful per session.
For multi-session use, instantiate one MemoryStore per session,
or use the SessionManager (Phase 4) which handles this automatically.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import numpy as np

from cce_core.config import CCEConfig, DEFAULT_CONFIG
from cce_core.ids import validate_session_id
from cce_core.ingestion.segmenter import Turn
from cce_core.compression.merger import MemoryNode
from cce_core.memory.hot_tier import HotTier
from cce_core.memory.warm_tier import WarmTier
from cce_core.memory.cold_tier import ColdTier, MacroSummary


class MemoryStore:
    """
    The unified memory interface for one session.

    Typical lifecycle:
        store = MemoryStore(session_id="abc")
        store.ingest_turn(turn)               # called per message
        nodes = store.search(query_emb)       # called at retrieval time
        store.checkpoint()                    # saves cold tier summary
        store.close()                         # flushes and closes DB

    Turns that fall out of the hot tier go to a pending buffer. When the store has a
    `live_compress_fn` (the engine provides one) they are compressed in topic-sized
    batches once `config.live_batch_turns` are waiting; the last, possibly still open,
    topic stays pending until it is closed, grows past a size limit, or the session is
    flushed. With only a `compress_fn` every eviction is compressed at once.

    The hot tier, the pending buffer and the counters are saved in the warm tier's
    SQLite file as they change, so a restarted or crashed process picks up where it
    stopped and turn indexes keep growing.
    """

    def __init__(
        self,
        session_id: str,
        config: CCEConfig = DEFAULT_CONFIG,
        compress_fn=None,        # Callable[[list[Turn], str], list[MemoryNode]]
        live_compress_fn=None,   # Callable[[list[Turn], str, bool], tuple[list[MemoryNode], int]]
    ):
        self.session_id = validate_session_id(session_id)
        self.config = config
        self.compress_fn = compress_fn  # injected from CCEEngine
        self.live_compress_fn = live_compress_fn

        self.hot = HotTier(config)
        self.warm = WarmTier(config)
        self.cold = ColdTier(config)

        self._lock = threading.RLock()
        self._log = self.warm.turn_log
        self._pending: list[Turn] = []
        self._next_index = 0
        self._turn_count = 0
        self._raw_token_count = 0
        self._retry_at = 0   # pending size at which a topic-batch attempt is worth repeating
        self._opened_at = datetime.now(timezone.utc)
        self._restore()

    # ── Saved state ───────────────────────────────────────────────────────────

    def _restore(self) -> None:
        """Rebuild the hot tier, pending buffer and counters from the warm tier's database."""
        sid = self.session_id
        meta = self._log.load_meta(sid)
        if meta:
            self._next_index = meta["next_index"]
            self._turn_count = meta["turn_count"]
            self._raw_token_count = meta["raw_tokens"]
        hot = self._log.load(sid, "hot")
        pending = self._log.load(sid, "pending")
        cap = self.config.hot_tier_max_turns
        if len(hot) > cap:   # the hot tier was made smaller since the last run
            overflow, hot = hot[:-cap], hot[-cap:]
            with self.warm.transaction():
                self._log.set_state(sid, [t.index for t in overflow], "pending")
            pending = sorted(pending + overflow, key=lambda t: t.index)
        self.hot.restore(hot)
        self._pending = pending
        stored = [t.index + 1 for t in hot + pending]
        if stored:
            self._next_index = max(self._next_index, *stored)

    @property
    def next_index(self) -> int:
        """Index the next turn of this session should get."""
        return self._next_index

    def get_pending_turns(self) -> list[Turn]:
        """Turns that left the hot tier but are not in a memory note yet (oldest first)."""
        with self._lock:
            return list(self._pending)

    # ── Ingest ────────────────────────────────────────────────────────────────

    def ingest_turn(self, turn: Turn, compress: bool = True) -> list[MemoryNode]:
        """
        Add a new turn to the store.
        A turn pushed out of the hot tier joins the pending buffer; once enough are pending
        they are compressed into warm tier nodes (see the class docstring).
        Returns any new MemoryNodes created (empty list if none).
        """
        return self.ingest_turns([turn], compress=compress)

    def ingest_turns(self, turns: list[Turn], compress: bool = True) -> list[MemoryNode]:
        """Add several turns in one transaction. Compression runs once, after all of them."""
        if not turns:
            return []
        with self._lock:
            self._push(turns, count=True)
            return self.compress_pending() if compress else []

    def seed_hot(self, turns: list[Turn]) -> None:
        """Put already-counted turns into the hot tier (used after a one-shot pipeline run)."""
        with self._lock:
            self._push(turns, count=False)

    def _push(self, turns: list[Turn], count: bool) -> None:
        sid = self.session_id
        with self.warm.transaction():
            self._log.put(sid, turns, "hot")
            for turn in turns:
                self.hot.push(turn)
            evicted = self.hot.drain_evicted()
            self._log.set_state(sid, [t.index for t in evicted], "pending")
            nxt = max(t.index for t in turns) + 1
            self._log.bump_meta(
                sid,
                turns=len(turns) if count else 0,
                tokens=sum(t.token_count for t in turns) if count else 0,
                next_index=nxt,
            )
        self._pending.extend(evicted)
        self._next_index = max(self._next_index, nxt)
        if count:
            self._turn_count += len(turns)
            self._raw_token_count += sum(t.token_count for t in turns)

    # ── Compression of pending turns ──────────────────────────────────────────

    def compress_pending(self, force: bool = False) -> list[MemoryNode]:
        """
        Compress pending turns if a batch is due (or `force`). Returns the new nodes.
        Turns that are not compressed stay pending and keep being served verbatim.
        """
        cfg = self.config
        with self._lock:
            pending = self._pending
            if not pending or not (self.compress_fn or self.live_compress_fn):
                return []
            if self.live_compress_fn is None:
                nodes, consumed = self.compress_fn(list(pending), self.session_id), len(pending)
            else:
                size = len(pending)
                tokens = sum(t.token_count for t in pending)
                must_flush = (
                    force
                    or cfg.live_batch_turns <= 1
                    or size >= cfg.live_pending_max_turns
                    or tokens >= cfg.live_pending_max_tokens
                )
                if not must_flush and (size < cfg.live_batch_turns or size < self._retry_at):
                    return []
                nodes, consumed = self.live_compress_fn(list(pending), self.session_id, must_flush)
                if consumed == 0:
                    # One topic still going: look again after a few more turns instead of every turn.
                    self._retry_at = size + max(2, cfg.live_batch_turns // 2)
                    return []
            self._retry_at = 0
            self.store_nodes(nodes, pending[:consumed])
            return nodes

    def store_nodes(self, nodes: list[MemoryNode], consumed: list[Turn]) -> None:
        """Write nodes to the warm tier and drop the pending turns they replace, in one commit."""
        with self._lock:
            done = {t.index for t in consumed}
            with self.warm.transaction():
                self.warm.upsert_many(nodes)
                self._log.delete(self.session_id, sorted(done))
            self._pending = [t for t in self._pending if t.index not in done]

    def ingest_nodes(self, nodes: list[MemoryNode], turns: list | None = None) -> None:
        """
        Directly write pre-compressed MemoryNodes to warm tier.
        Optionally pass original turns to track raw token count correctly.
        """
        with self._lock:
            if turns:
                n_turns = len(turns)
                n_tokens = sum(t.token_count for t in turns)
                nxt = max(t.index for t in turns) + 1
            else:
                # Best-effort: reconstruct counts from node metadata
                n_tokens = sum(n.token_count for n in nodes)
                n_turns = sum(n.turn_end - n.turn_start + 1 for n in nodes)
                nxt = max((n.turn_end + 1 for n in nodes), default=0)
            with self.warm.transaction():
                self.warm.upsert_many(nodes)
                self._log.bump_meta(self.session_id, turns=n_turns, tokens=n_tokens, next_index=nxt)
            self._turn_count += n_turns
            self._raw_token_count += n_tokens
            self._next_index = max(self._next_index, nxt)

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
        Force-compress everything still uncompressed (pending turns, then the hot tier) into
        warm tier nodes and empty both. Call before checkpoint() so nothing is left behind.
        """
        with self._lock:
            remaining = self._pending + self.hot.get_all()
            if not remaining or not (self.compress_fn or self.live_compress_fn):
                return []
            if self.live_compress_fn is not None:
                nodes, _ = self.live_compress_fn(remaining, self.session_id, True)
            else:
                nodes = self.compress_fn(remaining, self.session_id)
            with self.warm.transaction():
                self.warm.upsert_many(nodes)
                self._log.delete(self.session_id, [t.index for t in remaining])
            self.hot.clear()
            self._pending = []
            self._retry_at = 0
            return nodes

    def close(self):
        """Clean up DB connection."""
        self.warm.close()

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        warm_count = self.warm.count(self.session_id)
        hot_count = self.hot.size()
        cold_exists = self.cold.exists(self.session_id)
        warm_tokens = self.warm.token_sum(self.session_id)
        return {
            "session_id": self.session_id,
            "turns_ingested": self._turn_count,
            "raw_tokens": self._raw_token_count,
            "hot_turns": hot_count,
            "pending_turns": len(self._pending),
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
            f"hot={self.hot.size()}, pending={len(self._pending)}, "
            f"warm={self.warm.count(self.session_id)})"
        )
