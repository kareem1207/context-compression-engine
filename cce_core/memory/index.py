"""
CCE Warm-Tier Index
One in-memory vector index per data dir, shared by every WarmTier in the process and keyed by session.

  - Per session, note centroids and per-message vectors sit in contiguous float32 matrices, so a
    query is two matrix products and no SQL. turn_start/turn_end and the note's keywords are held next
    to the vectors.
  - Appends are amortised (the arrays double), not np.vstack.
  - It reads the database through its own connection and stays current with
    PRAGMA data_version (changes when any other connection commits, in this process or another) and
    a change log (node_changes, filled by triggers): new notes are appended, anything else reloads.
  - All state is guarded by one lock. Search is brute-force cosine similarity.

Only committed rows are visible to the index.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cce_core.compression.merger import decode_turn_vecs
from cce_core.retrieval.keywords import keyword_set

_DIM_ERROR = (
    "Stored embeddings have {stored} dimensions but the active embedder produces {given}. "
    "Use the same embedder (CCE_EMBEDDER) that created these sessions."
)


def _grow(arr: np.ndarray, used: int, need: int) -> np.ndarray:
    if need <= len(arr):
        return arr
    new = np.empty((max(need, 2 * len(arr)),) + arr.shape[1:], dtype=arr.dtype)
    new[:used] = arr[:used]
    return new


class _Block:
    """All notes of one session."""

    def __init__(self, dim: int):
        self.dim = dim
        self.n = 0
        self.vecs = np.empty((16, dim), dtype=np.float32)
        self.turn_start = np.empty(16, dtype=np.int64)
        self.turn_end = np.empty(16, dtype=np.int64)
        self.t_off = np.empty(16, dtype=np.int64)      # where each note's message vectors start in tvecs
        self.t_len = np.empty(16, dtype=np.int64)      # how many it has (0 for notes saved without them)
        self.node_ids: list[str] = []
        self.kw: list[frozenset[str]] = []
        self.tn = 0
        self.tvecs = np.empty((16, dim), dtype=np.float32)

    def append(self, node_id, emb, turn_start, turn_end, kw, tv):
        i = self.n
        self.vecs = _grow(self.vecs, i, i + 1)
        self.turn_start = _grow(self.turn_start, i, i + 1)
        self.turn_end = _grow(self.turn_end, i, i + 1)
        self.t_off = _grow(self.t_off, i, i + 1)
        self.t_len = _grow(self.t_len, i, i + 1)
        self.vecs[i] = emb
        self.turn_start[i] = turn_start
        self.turn_end[i] = turn_end
        if tv is not None and tv.shape[1] == self.dim and len(tv):
            self.tvecs = _grow(self.tvecs, self.tn, self.tn + len(tv))
            self.tvecs[self.tn:self.tn + len(tv)] = tv
            self.t_off[i], self.t_len[i] = self.tn, len(tv)
            self.tn += len(tv)
        else:
            self.t_off[i], self.t_len[i] = self.tn, 0
        self.node_ids.append(node_id)
        self.kw.append(kw)
        self.n += 1

    def similarity(self, q: np.ndarray) -> np.ndarray:
        """Per note: the best of its centroid and its best-matching message."""
        n = self.n
        sem = self.vecs[:n] @ q
        if self.tn:
            has = self.t_len[:n] > 0
            if has.any():
                msg = self.tvecs[:self.tn] @ q
                best = np.maximum.reduceat(msg, self.t_off[:n][has])
                sem[has] = np.maximum(sem[has], best)
        return sem


@dataclass
class SessionScores:
    """Query scores for every note of one session (rows are in no particular order)."""
    node_ids: list[str]
    turn_start: np.ndarray
    turn_end: np.ndarray
    semantic: np.ndarray
    keywords: list[frozenset[str]]


class SharedIndex:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._blocks: dict[str, _Block] = {}
        self._where: dict[str, str] = {}   # node_id -> session key
        self._seq = 0
        self._version: int | None = None
        self._refs = 0

    # ── Keeping current ───────────────────────────────────────────────────────

    def sync(self) -> None:
        with self._lock:
            version = self._conn.execute("PRAGMA data_version").fetchone()[0]
            if version == self._version:
                return
            self._refresh()
            self._version = version

    def _refresh(self) -> None:
        if self._version is None:
            return self._reload()
        top = self._conn.execute("SELECT MIN(seq), MAX(seq) FROM node_changes").fetchone()
        if top[1] is None:
            if self._where:
                return self._reload()
            return
        if top[0] > self._seq + 1 and self._seq > 0:   # the log was pruned past what we have seen
            return self._reload()
        changes = self._conn.execute(
            "SELECT seq, op, node_id FROM node_changes WHERE seq > ? ORDER BY seq", (self._seq,)
        ).fetchall()
        if not changes:
            return
        ids, seen = [], set()
        for _, op, node_id in changes:
            if op != "i" or node_id in self._where or node_id in seen:
                return self._reload()      # a delete or a replaced note: not worth patching
            ids.append(node_id)
            seen.add(node_id)
        for start in range(0, len(ids), 400):
            chunk = ids[start:start + 400]
            rows = self._conn.execute(
                _SELECT + f" WHERE node_id IN ({','.join('?' * len(chunk))})", chunk
            ).fetchall()
            by_id = {r[0]: r for r in rows}
            for node_id in chunk:
                if node_id in by_id:           # may be gone again already; the next change says so
                    self._add(by_id[node_id])
        self._seq = changes[-1][0]

    def _reload(self) -> None:
        self._blocks, self._where = {}, {}
        self._conn.execute("BEGIN")
        try:
            self._seq = self._conn.execute("SELECT COALESCE(MAX(seq), 0) FROM node_changes").fetchone()[0]
            for row in self._conn.execute(_SELECT + " ORDER BY turn_start"):
                self._add(row)
        finally:
            self._conn.rollback()

    def _add(self, row) -> None:
        node_id, session_id, turn_start, turn_end, blob, metadata, label, summary = row
        emb = np.frombuffer(blob, dtype=np.float32)
        key = session_id or ""
        block = self._blocks.get(key)
        if block is None:
            block = self._blocks[key] = _Block(len(emb))
        elif block.dim != len(emb):
            return   # a session never mixes embedders; ignore a stray note rather than fail every query
        block.append(
            node_id, emb, turn_start, turn_end,
            keyword_set(f"{label} {summary}"), decode_turn_vecs(json.loads(metadata)),
        )
        self._where[node_id] = key

    # ── Queries ───────────────────────────────────────────────────────────────

    def session_scores(self, session_id: str, qvec: np.ndarray) -> SessionScores | None:
        with self._lock:
            self.sync()
            block = self._blocks.get(session_id or "")
            if block is None or block.n == 0:
                return None
            if block.dim != qvec.shape[0]:
                raise ValueError(_DIM_ERROR.format(stored=block.dim, given=qvec.shape[0]))
            n = block.n
            return SessionScores(
                node_ids=list(block.node_ids),
                turn_start=block.turn_start[:n].copy(),
                turn_end=block.turn_end[:n].copy(),
                semantic=block.similarity(qvec.astype(np.float32)),
                keywords=list(block.kw),
            )

    def search(self, qvec: np.ndarray, top_k: int, session_id: str | None, recency_boost: float):
        """Top (node_id, score) by centroid similarity (+ recency), within a session or across all."""
        with self._lock:
            self.sync()
            if session_id:
                blocks = [b for b in [self._blocks.get(session_id)] if b is not None and b.n]
            else:
                blocks = [b for b in self._blocks.values() if b.n]
            for b in blocks:
                if b.dim != qvec.shape[0]:
                    raise ValueError(_DIM_ERROR.format(stored=b.dim, given=qvec.shape[0]))
            if not blocks:
                return []
            q = qvec.astype(np.float32)
            scores = np.concatenate([b.vecs[:b.n] @ q for b in blocks])
            ends = np.concatenate([b.turn_end[:b.n] for b in blocks]).astype(np.float32)
            ids = [i for b in blocks for i in b.node_ids]
        if recency_boost > 0.0 and len(scores) > 1:
            top_end = ends.max()
            if top_end > 0:
                scores = scores + (ends / top_end) * recency_boost
        top_k = min(top_k, len(scores))
        pick = np.argpartition(scores, -top_k)[-top_k:]
        pick = pick[np.argsort(scores[pick])[::-1]]
        return [(ids[i], float(scores[i])) for i in pick]

    def size(self) -> int:
        with self._lock:
            self.sync()
            return len(self._where)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_SELECT = (
    "SELECT node_id, session_id, turn_start, turn_end, embedding, metadata, topic_label, meso_summary "
    "FROM memory_nodes"
)

_registry: dict[str, SharedIndex] = {}
_registry_lock = threading.Lock()


def acquire(db_path: str | Path) -> SharedIndex:
    """The shared index for this database file (created on first use)."""
    key = str(Path(db_path).resolve())
    with _registry_lock:
        index = _registry.get(key)
        if index is None:
            index = _registry[key] = SharedIndex(key)
        index._refs += 1
        return index


def release(index: SharedIndex) -> None:
    with _registry_lock:
        index._refs -= 1
        if index._refs <= 0:
            _registry.pop(index.db_path, None)
            index.close()
