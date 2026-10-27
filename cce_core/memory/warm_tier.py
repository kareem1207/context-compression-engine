"""
CCE Warm Tier
SQLite-backed store for MemoryNode objects (compressed topic chunks).
Embeddings are stored as BLOB. Search runs over a shared in-memory index
(cce_core.memory.index): one per data dir, keyed by session, kept current from
SQLite (PRAGMA data_version + a change log), so several WarmTier objects, or
several processes, see each other's writes.

Schema:
    memory_nodes table — one row per MemoryNode
    Each row stores: metadata columns + embedding BLOB + summary text

Search strategy:
    Brute-force cosine similarity over NumPy matrices. For hundreds of notes
    this beats building an ANN index and keeps the dependency footprint small
    (no FAISS needed).

Thread safety:
    SQLite with check_same_thread=False + a module-level lock.
    Single-writer, multi-reader safe for local use.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cce_core.config import CCEConfig, DEFAULT_CONFIG
from cce_core.compression.merger import MemoryNode, merge_nodes
from cce_core.memory import index as index_module


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS memory_nodes (
    node_id         TEXT PRIMARY KEY,
    chunk_id        TEXT NOT NULL,
    session_id      TEXT,
    topic_label     TEXT NOT NULL,
    meso_summary    TEXT NOT NULL,
    micro_summaries TEXT NOT NULL,   -- JSON array of strings
    turn_start      INTEGER NOT NULL,
    turn_end        INTEGER NOT NULL,
    token_count     INTEGER NOT NULL,
    embedding       BLOB NOT NULL,   -- float32 numpy array, raw bytes
    created_at      TEXT NOT NULL,
    metadata        TEXT NOT NULL    -- JSON object
);
"""

_INSERT_NODE = """
INSERT OR REPLACE INTO memory_nodes
(node_id, chunk_id, session_id, topic_label, meso_summary,
 micro_summaries, turn_start, turn_end, token_count,
 embedding, created_at, metadata)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
"""

_CREATE_IDX_SESSION = """
CREATE INDEX IF NOT EXISTS idx_session ON memory_nodes (session_id);
"""

_CREATE_IDX_TURN = """
CREATE INDEX IF NOT EXISTS idx_turn_start ON memory_nodes (turn_start);
"""


# Database schema version, kept in PRAGMA user_version.
#   0 = a 0.1.x database (memory_nodes only) or a brand-new file
#   1 = adds the `turns` and `session_meta` tables (live-session state survives restarts)
#   2 = adds memory_nodes.hits (how often a note was retrieved; decides what gets merged first)
#   3 = adds node_changes + triggers (the change log the shared search index follows)
#   4 = adds meta (key/value; holds the embedder the notes were made with)
#   5 = turns.state 'done' rows (original text of compressed turns) + an index on (session_id, state)
SCHEMA_VERSION = 5

_CREATE_TURNS = """
CREATE TABLE IF NOT EXISTS turns (
    session_id   TEXT NOT NULL,
    idx          INTEGER NOT NULL,
    turn_id      TEXT NOT NULL,
    role         TEXT NOT NULL,
    content      TEXT NOT NULL,
    token_count  INTEGER NOT NULL,
    timestamp    TEXT NOT NULL,
    state        TEXT NOT NULL,      -- 'hot' (recent, verbatim) | 'pending' (evicted, not yet compressed)
    metadata     TEXT NOT NULL,      -- JSON object
    PRIMARY KEY (session_id, idx)
);
"""

_CREATE_SESSION_META = """
CREATE TABLE IF NOT EXISTS session_meta (
    session_id   TEXT PRIMARY KEY,
    next_index   INTEGER NOT NULL,   -- index the next turn gets; only ever grows
    turn_count   INTEGER NOT NULL,
    raw_tokens   INTEGER NOT NULL
);
"""

_CREATE_CHANGES = """
CREATE TABLE IF NOT EXISTS node_changes (
    seq      INTEGER PRIMARY KEY AUTOINCREMENT,
    op       TEXT NOT NULL,          -- 'i' = note inserted (or replaced), 'd' = note deleted
    node_id  TEXT NOT NULL
);
"""

_CREATE_TRIGGERS = (
    """CREATE TRIGGER IF NOT EXISTS memory_nodes_ai AFTER INSERT ON memory_nodes
       BEGIN INSERT INTO node_changes (op, node_id) VALUES ('i', NEW.node_id); END""",
    """CREATE TRIGGER IF NOT EXISTS memory_nodes_ad AFTER DELETE ON memory_nodes
       BEGIN INSERT INTO node_changes (op, node_id) VALUES ('d', OLD.node_id); END""",
)

_CREATE_META = """
CREATE TABLE IF NOT EXISTS meta (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
"""

_KEEP_CHANGES = 20000   # change-log rows kept; an index that fell further behind reloads instead

logger = logging.getLogger(__name__)


def _emb_to_blob(emb: np.ndarray) -> bytes:
    return emb.astype(np.float32).tobytes()


def _blob_to_emb(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class WarmTier:
    """
    Persistent store for compressed MemoryNode objects.

    Usage:
        warm = WarmTier()
        warm.upsert(node)
        results = warm.search(query_embedding, top_k=5)
        node = warm.get(node_id)
    """

    def __init__(self, config: CCEConfig = DEFAULT_CONFIG):
        self.config = config
        config.ensure_dirs()
        self._db_path = str(config.db_path)
        self._lock = threading.RLock()
        self._tx_depth = 0
        self._conn = self._connect()
        self._init_schema()
        from cce_core.memory.turn_log import TurnLog
        self.turn_log = TurnLog(self)
        self.index = index_module.acquire(self._db_path)   # shared by every WarmTier on this data dir

    # ── Connection ────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads during writes
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self):
        with self._lock:
            self._conn.execute(_CREATE_TABLE)
            self._conn.execute(_CREATE_IDX_SESSION)
            self._conn.execute(_CREATE_IDX_TURN)
            self._conn.commit()
            self._migrate()
            self._conn.execute(
                "DELETE FROM node_changes WHERE seq <= (SELECT MAX(seq) FROM node_changes) - ?", (_KEEP_CHANGES,)
            )
            self._conn.commit()

    def _migrate(self):
        """Bring the database up to SCHEMA_VERSION in place. Existing notes are kept."""
        conn = self._conn
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version == SCHEMA_VERSION:
            return
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"{self._db_path} has schema version {version}, newer than this CCE ({SCHEMA_VERSION}). "
                "Upgrade context-compression-engine to open it."
            )
        conn.execute("BEGIN IMMEDIATE")   # another process may be migrating at the same time
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version < 1:
                conn.execute(_CREATE_TURNS)
                conn.execute(_CREATE_SESSION_META)
                # 0.1.x kept no per-session counters: continue numbering after the last note.
                conn.execute(
                    """
                    INSERT OR IGNORE INTO session_meta (session_id, next_index, turn_count, raw_tokens)
                    SELECT session_id, MAX(turn_end) + 1, SUM(turn_end - turn_start + 1), SUM(token_count)
                    FROM memory_nodes WHERE session_id IS NOT NULL GROUP BY session_id
                    """
                )
                migrated = conn.execute("SELECT COUNT(*) FROM session_meta").fetchone()[0]
                if migrated:
                    logger.info("Migrated %s: %d session(s) upgraded to schema 1", self._db_path, migrated)
            if version < 2:
                conn.execute("ALTER TABLE memory_nodes ADD COLUMN hits INTEGER NOT NULL DEFAULT 0")
            if version < 3:
                conn.execute(_CREATE_CHANGES)
                for trigger in _CREATE_TRIGGERS:
                    conn.execute(trigger)
            if version < 4:
                conn.execute(_CREATE_META)
            if version < 5:
                # Compressed turns may now stay in `turns` (state 'done'); restoring a session reads
                # only its 'hot' and 'pending' rows, so look them up by state, not by scanning.
                conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_state ON turns (session_id, state)")
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
        except BaseException:
            conn.rollback()
            raise

    @contextmanager
    def transaction(self):
        """Group writes (from this class and TurnLog) into one commit; rolled back on error."""
        with self._lock:
            self._tx_depth += 1
            try:
                yield
            except BaseException:
                if self._tx_depth == 1:
                    self._conn.rollback()
                raise
            else:
                if self._tx_depth == 1:
                    self._conn.commit()
            finally:
                self._tx_depth -= 1

    def _commit(self):
        """Commit now, unless inside transaction()."""
        if self._tx_depth == 0:
            self._conn.commit()

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(self, node: MemoryNode) -> None:
        """Insert or replace a MemoryNode. Updates in-memory matrix."""
        with self._lock:
            self._conn.execute(_INSERT_NODE, self._node_row(node))
            self._enforce_limit(node.session_id)
            self._commit()

    def upsert_many(self, nodes: list[MemoryNode]) -> None:
        """Batch upsert — much faster than calling upsert() in a loop."""
        if not nodes:
            return
        with self._lock:
            self._conn.executemany(_INSERT_NODE, [self._node_row(n) for n in nodes])
            for sid in {n.session_id for n in nodes}:
                self._enforce_limit(sid)
            self._commit()

    @staticmethod
    def _node_row(n: MemoryNode) -> tuple:
        return (
            n.node_id, n.chunk_id, n.session_id, n.topic_label,
            n.meso_summary, json.dumps(n.micro_summaries),
            n.turn_start, n.turn_end, n.token_count,
            _emb_to_blob(n.embedding), n.created_at.isoformat(),
            json.dumps(n.metadata),
        )

    # ── Limit ─────────────────────────────────────────────────────────────────

    def _enforce_limit(self, session_id: str | None) -> bool:
        """
        Keep a session at or under config.warm_tier_max_chunks (0 or None = unlimited).
        Nothing is dropped: the least-retrieved pair of neighbouring notes (oldest first on ties)
        is fused into one coarser note, until the session fits. The newest notes are left alone.
        Must be called with the lock held. Returns True if notes were merged.
        """
        limit = self.config.warm_tier_max_chunks
        if not limit or limit < 1 or not session_id:
            return False
        rows = self._conn.execute(
            "SELECT *, hits FROM memory_nodes WHERE session_id = ? ORDER BY turn_start", (session_id,)
        ).fetchall()
        if len(rows) <= limit:
            return False
        nodes = [self._row_to_node(r) for r in rows]
        hits = [r["hits"] for r in rows]
        removed: list[str] = []
        merges = 0
        while len(nodes) > limit:
            n = len(nodes)
            protect = max(0, min(n - 2, max(1, limit // 4)))   # newest notes stay as they are
            candidates = range(n - 1 - protect)
            i = min(
                candidates,
                key=lambda i: (hits[i] + hits[i + 1], nodes[i + 1].turn_end - nodes[i].turn_start, i),
            )
            fused = merge_nodes(nodes[i], nodes[i + 1], self.config)
            removed += [nodes[i].node_id, nodes[i + 1].node_id]
            nodes[i:i + 2] = [fused]
            hits[i:i + 2] = [hits[i] + hits[i + 1]]
            merges += 1
        fused_nodes = [(nd, h) for nd, h in zip(nodes, hits) if nd.metadata.get("merged_notes", 1) > 1]
        self._conn.executemany("DELETE FROM memory_nodes WHERE node_id = ?", [(i,) for i in removed])
        for nd, h in fused_nodes:
            # a note that was fused earlier and again now is written once, under its latest id
            self._conn.execute(_INSERT_NODE, self._node_row(nd))
            self._conn.execute("UPDATE memory_nodes SET hits = ? WHERE node_id = ?", (h, nd.node_id))
        logger.info(
            "Session %r exceeded warm_tier_max_chunks=%d: merged %d pair(s) of old notes, %d notes left",
            session_id, limit, merges, len(nodes),
        )
        return True

    def record_hits(self, node_ids: list[str]) -> None:
        """Count a retrieval for each note (the least-retrieved are merged first when over the limit)."""
        if not node_ids:
            return
        with self._lock:
            self._conn.executemany(
                "UPDATE memory_nodes SET hits = hits + 1 WHERE node_id = ?", [(i,) for i in node_ids]
            )
            self._commit()

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, node_id: str) -> MemoryNode | None:
        """Fetch a single node by ID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memory_nodes WHERE node_id = ?", (node_id,)
            ).fetchone()
        return self._row_to_node(row) if row else None

    def get_by_session(self, session_id: str) -> list[MemoryNode]:
        """Fetch all nodes for a given session, ordered by turn position."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memory_nodes WHERE session_id = ? ORDER BY turn_start",
                (session_id,),
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_by_turn_range(self, start: int, end: int) -> list[MemoryNode]:
        """Fetch all nodes whose turn range overlaps [start, end]."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM memory_nodes
                   WHERE turn_start <= ? AND turn_end >= ?
                   ORDER BY turn_start""",
                (end, start),
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_all(self, session_id: str | None = None) -> list[MemoryNode]:
        """Fetch all nodes, optionally filtered by session."""
        with self._lock:
            if session_id:
                rows = self._conn.execute(
                    "SELECT * FROM memory_nodes WHERE session_id = ? ORDER BY turn_start",
                    (session_id,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM memory_nodes ORDER BY turn_start"
                ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_many(self, node_ids: list[str]) -> list[MemoryNode]:
        """Fetch nodes by id in one query, in the order given (missing ids are skipped)."""
        if not node_ids:
            return []
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM memory_nodes WHERE node_id IN ({','.join('?' * len(node_ids))})", node_ids
            ).fetchall()
        by_id = {r["node_id"]: self._row_to_node(r) for r in rows}
        return [by_id[i] for i in node_ids if i in by_id]

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int | None = None,
        session_id: str | None = None,
        recency_boost: float = 0.0,
    ) -> list[tuple[MemoryNode, float]]:
        """
        Find the top_k most semantically similar nodes to query_embedding.

        Args:
            query_embedding: float32 normalized vector (384-dim for MiniLM)
            top_k: number of results to return (defaults to config.retrieval_top_k)
            session_id: if set, restrict search to this session
            recency_boost: added to scores for more recent nodes (turn_end closer to max)

        Returns:
            List of (MemoryNode, score) tuples, sorted by score descending.
        """
        top_k = top_k or self.config.retrieval_top_k
        hits = self.index.search(query_embedding, top_k, session_id, recency_boost)
        nodes = {n.node_id: n for n in self.get_many([h[0] for h in hits])}
        return [(nodes[i], s) for i, s in hits if i in nodes]

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete(self, node_id: str) -> bool:
        """Delete a note together with the kept original text of the turns it covered."""
        with self._lock:
            row = self._conn.execute(
                "SELECT session_id, turn_start, turn_end FROM memory_nodes WHERE node_id = ?", (node_id,)
            ).fetchone()
            cur = self._conn.execute(
                "DELETE FROM memory_nodes WHERE node_id = ?", (node_id,)
            )
            if row is not None and row["session_id"]:
                self.turn_log.drop_done(row["session_id"], row["turn_start"], row["turn_end"])
            self._commit()
        return cur.rowcount > 0

    def delete_session(self, session_id: str) -> int:
        """Delete all nodes for a session. Returns count deleted."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM memory_nodes WHERE session_id = ?", (session_id,)
            )
            self._conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM session_meta WHERE session_id = ?", (session_id,))
            self._commit()
        return cur.rowcount

    # ── Stats ─────────────────────────────────────────────────────────────────

    def count(self, session_id: str | None = None) -> int:
        with self._lock:
            if session_id:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM memory_nodes WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM memory_nodes"
                ).fetchone()
        return row[0]

    # ── Meta ──────────────────────────────────────────────────────────────────

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta_if_absent(self, key: str, value: str) -> str:
        """Store value unless the key exists already (also across processes). Returns what is stored."""
        with self._lock:
            self._conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)", (key, value))
            self._commit()
        return self.get_meta(key)

    def stored_embedding_dim(self) -> int | None:
        """Dimension of the vectors already in the notes table (None when it is empty)."""
        with self._lock:
            row = self._conn.execute("SELECT length(embedding) / 4 FROM memory_nodes LIMIT 1").fetchone()
        return int(row[0]) if row else None

    def note_texts(self, session_id: str) -> list[tuple[str, int, int, str, int]]:
        """(topic_label, turn_start, turn_end, meso_summary, original token_count) of a session's notes."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT topic_label, turn_start, turn_end, meso_summary, token_count FROM memory_nodes "
                "WHERE session_id = ? ORDER BY turn_start", (session_id,)
            ).fetchall()
        return [tuple(r) for r in rows]

    def token_sum(self, session_id: str) -> int:
        """Total token_count of a session's notes, without loading them."""
        with self._lock:
            return self._conn.execute(
                "SELECT COALESCE(SUM(token_count), 0) FROM memory_nodes WHERE session_id = ?", (session_id,)
            ).fetchone()[0]

    def close(self):
        if self.index is not None:
            index_module.release(self.index)
            self.index = None
        self._conn.close()

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> MemoryNode:
        return MemoryNode(
            node_id=row["node_id"],
            chunk_id=row["chunk_id"],
            session_id=row["session_id"],
            topic_label=row["topic_label"],
            meso_summary=row["meso_summary"],
            micro_summaries=json.loads(row["micro_summaries"]),
            turn_start=row["turn_start"],
            turn_end=row["turn_end"],
            token_count=row["token_count"],
            embedding=_blob_to_emb(row["embedding"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=json.loads(row["metadata"]),
        )

    def __repr__(self):
        return f"WarmTier(db={self._db_path!r}, nodes={self.count()})"