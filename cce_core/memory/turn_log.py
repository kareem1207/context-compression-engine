"""
CCE Turn Log
Durable per-session state for live sessions, stored in the warm tier's SQLite file.

  turns         the turns that are not yet part of a memory note
                  state 'hot'     = the last N turns, injected verbatim
                  state 'pending' = evicted from the hot tier, waiting for topic-sized compression
                  state 'done'    = compressed into a note; the original text is kept (config.keep_raw_turns)
                                    so retrieval can quote the exact sentence. Deleted with the session.
  session_meta  next turn index and counters per session

Writes join the caller's transaction (WarmTier.transaction) so a turn and its counters, or a note
and the turns it replaces, are committed together. After a restart or a crash the store rebuilds the
hot tier, the pending buffer and the counters from here.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

from cce_core.ingestion.segmenter import Turn

if TYPE_CHECKING:
    from cce_core.memory.warm_tier import WarmTier


class TurnLog:
    def __init__(self, warm: "WarmTier"):
        self._warm = warm

    @property
    def _conn(self):
        return self._warm._conn

    # ── Counters ──────────────────────────────────────────────────────────────

    def load_meta(self, session_id: str) -> dict | None:
        """{'next_index', 'turn_count', 'raw_tokens'}, or None for a session with no state yet."""
        with self._warm._lock:
            row = self._conn.execute(
                "SELECT next_index, turn_count, raw_tokens FROM session_meta WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                # Notes written without live state (older versions, direct warm-tier writes).
                row = self._conn.execute(
                    "SELECT MAX(turn_end) + 1, SUM(turn_end - turn_start + 1), SUM(token_count) "
                    "FROM memory_nodes WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if row is None or row[0] is None:
                    return None
        return {"next_index": row[0], "turn_count": row[1], "raw_tokens": row[2]}

    def bump_meta(self, session_id: str, *, turns: int, tokens: int, next_index: int) -> None:
        """Add to the counters and move next_index forward (never back)."""
        with self._warm._lock:
            self._conn.execute(
                """
                INSERT INTO session_meta (session_id, next_index, turn_count, raw_tokens)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    next_index = MAX(next_index, excluded.next_index),
                    turn_count = turn_count + excluded.turn_count,
                    raw_tokens = raw_tokens + excluded.raw_tokens
                """,
                (session_id, next_index, turns, tokens),
            )
            self._warm._commit()

    # ── Turns ─────────────────────────────────────────────────────────────────

    def put(self, session_id: str, turns: list[Turn], state: str) -> None:
        if not turns:
            return
        rows = [
            (
                session_id, t.index, t.turn_id, t.role, t.content, t.token_count,
                t.timestamp.isoformat(), state, json.dumps(t.metadata),
            )
            for t in turns
        ]
        with self._warm._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO turns "
                "(session_id, idx, turn_id, role, content, token_count, timestamp, state, metadata) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )
            self._warm._commit()

    def set_state(self, session_id: str, indexes: list[int], state: str) -> None:
        if not indexes:
            return
        with self._warm._lock:
            self._conn.executemany(
                "UPDATE turns SET state = ? WHERE session_id = ? AND idx = ?",
                [(state, session_id, i) for i in indexes],
            )
            self._warm._commit()

    def delete(self, session_id: str, indexes: list[int]) -> None:
        if not indexes:
            return
        with self._warm._lock:
            self._conn.executemany(
                "DELETE FROM turns WHERE session_id = ? AND idx = ?",
                [(session_id, i) for i in indexes],
            )
            self._warm._commit()

    def retire(self, session_id: str, indexes: list[int], keep: bool) -> None:
        """Turns that are now part of a note: keep their text as 'done' rows, or drop them."""
        if keep:
            self.set_state(session_id, indexes, "done")
        else:
            self.delete(session_id, indexes)

    def drop_done(self, session_id: str, start: int, end: int) -> None:
        """Delete kept text of compressed turns in [start, end] (their note is gone)."""
        with self._warm._lock:
            self._conn.execute(
                "DELETE FROM turns WHERE session_id = ? AND state = 'done' AND idx BETWEEN ? AND ?",
                (session_id, start, end),
            )
            self._warm._commit()

    def texts(self, session_id: str, indexes: list[int]) -> dict[int, str]:
        """Original text of compressed (or any stored) turns, by turn index."""
        if not indexes:
            return {}
        with self._warm._lock:
            rows = self._conn.execute(
                f"SELECT idx, content FROM turns WHERE session_id = ? AND idx IN ({','.join('?' * len(indexes))})",
                [session_id, *indexes],
            ).fetchall()
        return {r["idx"]: r["content"] for r in rows}

    def count(self, session_id: str, state: str) -> int:
        with self._warm._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM turns WHERE session_id = ? AND state = ?", (session_id, state)
            ).fetchone()[0]

    def load(self, session_id: str, state: str) -> list[Turn]:
        with self._warm._lock:
            rows = self._conn.execute(
                "SELECT idx, turn_id, role, content, token_count, timestamp, metadata "
                "FROM turns WHERE session_id = ? AND state = ? ORDER BY idx",
                (session_id, state),
            ).fetchall()
        return [
            Turn(
                turn_id=r["turn_id"], role=r["role"], content=r["content"],
                token_count=r["token_count"], timestamp=datetime.fromisoformat(r["timestamp"]),
                index=r["idx"], metadata=json.loads(r["metadata"]),
            )
            for r in rows
        ]
