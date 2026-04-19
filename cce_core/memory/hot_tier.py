"""
CCE Hot Tier
In-RAM circular buffer of the most recent N turns.
This is what gets injected verbatim into every LLM call —
no retrieval needed, always fresh, always exact.

Design:
  - Fixed-size deque (maxlen = hot_tier_max_turns from config)
  - O(1) append, O(1) pop from left when full
  - Thread-safe via a simple lock (single-process assumption)
  - Serializable to dict for session persistence
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone

from cce_core.config import CCEConfig, DEFAULT_CONFIG
from cce_core.ingestion.segmenter import Turn


class HotTier:
    """
    Maintains the last N turns in RAM.
    Always contains the most recent conversation context —
    injected as-is into every LLM prompt.

    Usage:
        hot = HotTier()
        hot.push(turn)
        recent = hot.get_all()   # ordered oldest→newest
    """

    def __init__(self, config: CCEConfig = DEFAULT_CONFIG):
        self.config = config
        self._buffer: deque[Turn] = deque(maxlen=config.hot_tier_max_turns)
        self._lock = threading.Lock()
        self._evicted: list[Turn] = []   # turns pushed out when buffer is full

    # ── Write ─────────────────────────────────────────────────────────────────

    def push(self, turn: Turn) -> Turn | None:
        """
        Add a turn to the hot tier.
        If the buffer is full, the oldest turn is evicted and returned
        so the caller can compress it into warm/cold tier.
        Returns None if no eviction occurred.
        """
        with self._lock:
            evicted: Turn | None = None
            if len(self._buffer) == self._buffer.maxlen:
                evicted = self._buffer[0]  # leftmost = oldest
            self._buffer.append(turn)
            if evicted:
                self._evicted.append(evicted)
            return evicted

    def push_many(self, turns: list[Turn]) -> list[Turn]:
        """
        Push multiple turns. Returns list of all evicted turns in order.
        Used when loading a conversation for the first time.
        """
        evicted_batch: list[Turn] = []
        for turn in turns:
            ev = self.push(turn)
            if ev is not None:
                evicted_batch.append(ev)
        return evicted_batch

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_all(self) -> list[Turn]:
        """Return all turns in the buffer, oldest first."""
        with self._lock:
            return list(self._buffer)

    def get_latest(self, n: int) -> list[Turn]:
        """Return the last n turns (most recent), oldest first within that slice."""
        with self._lock:
            buf = list(self._buffer)
            return buf[-n:] if n < len(buf) else buf

    def peek_oldest(self) -> Turn | None:
        """Peek at the oldest turn without removing it."""
        with self._lock:
            return self._buffer[0] if self._buffer else None

    def peek_newest(self) -> Turn | None:
        """Peek at the newest turn without removing it."""
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    # ── Eviction queue ────────────────────────────────────────────────────────

    def drain_evicted(self) -> list[Turn]:
        """
        Return and clear the list of evicted turns.
        Call this periodically to compress evicted turns into warm tier.
        """
        with self._lock:
            evicted = self._evicted.copy()
            self._evicted.clear()
            return evicted

    def has_evictions(self) -> bool:
        with self._lock:
            return len(self._evicted) > 0

    # ── State ─────────────────────────────────────────────────────────────────

    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def is_full(self) -> bool:
        with self._lock:
            return len(self._buffer) == self._buffer.maxlen

    def clear(self):
        with self._lock:
            self._buffer.clear()
            self._evicted.clear()

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "turns": [t.to_dict() for t in self._buffer],
                "evicted": [t.to_dict() for t in self._evicted],
                "snapshot_at": datetime.now(timezone.utc).isoformat(),
            }

    @classmethod
    def from_dict(cls, d: dict, config: CCEConfig = DEFAULT_CONFIG) -> "HotTier":
        tier = cls(config)
        for td in d.get("turns", []):
            tier._buffer.append(Turn.from_dict(td))
        for td in d.get("evicted", []):
            tier._evicted.append(Turn.from_dict(td))
        return tier

    def to_messages(self) -> list[dict]:
        """
        Export hot tier as OpenAI-style message list.
        Ready to prepend to an LLM prompt.
        """
        return [{"role": t.role, "content": t.content} for t in self.get_all()]

    def __repr__(self):
        return f"HotTier(size={self.size()}/{self._buffer.maxlen}, evicted_pending={len(self._evicted)})"