"""
CCE Warm Tier
SQLite-backed store for MemoryNode objects (compressed topic chunks).
Embeddings are stored as BLOB and loaded into a NumPy matrix for fast
ANN (approximate nearest neighbour) search via cosine similarity.

Schema:
    memory_nodes table — one row per MemoryNode
    Each row stores: metadata columns + embedding BLOB + summary text

ANN Strategy:
    For up to ~500 nodes (warm_tier_max_chunks default), brute-force
    cosine similarity over a NumPy matrix is faster than building an
    index (FAISS/HNSWlib have high startup cost for small corpora).
    When node count exceeds ~1000, we switch to a chunked matrix search.
    This keeps the dependency footprint minimal (no FAISS needed).

Thread safety:
    SQLite with check_same_thread=False + a module-level lock.
    Single-writer, multi-reader safe for local use.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cce_core.config import CCEConfig, DEFAULT_CONFIG
from cce_core.compression.merger import MemoryNode


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

_CREATE_IDX_SESSION = """
CREATE INDEX IF NOT EXISTS idx_session ON memory_nodes (session_id);
"""

_CREATE_IDX_TURN = """
CREATE INDEX IF NOT EXISTS idx_turn_start ON memory_nodes (turn_start);
"""


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
        self._lock = threading.Lock()
        self._conn = self._connect()
        self._init_schema()

        # In-memory embedding matrix for fast ANN — rebuilt on load, kept in sync
        self._node_ids: list[str] = []
        self._matrix: np.ndarray | None = None   # shape (N, dim)
        self._matrix_dirty = True
        self._load_matrix()

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

    # ── Matrix management ─────────────────────────────────────────────────────

    def _load_matrix(self):
        """Load all embeddings into RAM as a NumPy matrix for ANN search."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT node_id, embedding FROM memory_nodes ORDER BY turn_start"
            ).fetchall()

        if not rows:
            self._node_ids = []
            self._matrix = None
            self._matrix_dirty = False
            return

        self._node_ids = [r["node_id"] for r in rows]
        embeddings = [_blob_to_emb(r["embedding"]) for r in rows]
        self._matrix = np.stack(embeddings, axis=0)  # (N, dim)
        self._matrix_dirty = False

    def _rebuild_matrix_if_needed(self):
        if self._matrix_dirty:
            self._load_matrix()

    def _append_to_matrix(self, node_id: str, embedding: np.ndarray):
        """Fast path: append one embedding without full reload."""
        self._node_ids.append(node_id)
        emb = embedding.reshape(1, -1).astype(np.float32)
        if self._matrix is None:
            self._matrix = emb
        else:
            self._matrix = np.vstack([self._matrix, emb])

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(self, node: MemoryNode) -> None:
        """Insert or replace a MemoryNode. Updates in-memory matrix."""
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO memory_nodes
                (node_id, chunk_id, session_id, topic_label, meso_summary,
                 micro_summaries, turn_start, turn_end, token_count,
                 embedding, created_at, metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    node.node_id,
                    node.chunk_id,
                    node.session_id,
                    node.topic_label,
                    node.meso_summary,
                    json.dumps(node.micro_summaries),
                    node.turn_start,
                    node.turn_end,
                    node.token_count,
                    _emb_to_blob(node.embedding),
                    node.created_at.isoformat(),
                    json.dumps(node.metadata),
                ),
            )
            self._conn.commit()

        # Keep matrix in sync without full reload
        if node.node_id not in self._node_ids:
            self._append_to_matrix(node.node_id, node.embedding)
        else:
            # Existing node updated — mark dirty for full rebuild
            self._matrix_dirty = True

    def upsert_many(self, nodes: list[MemoryNode]) -> None:
        """Batch upsert — much faster than calling upsert() in a loop."""
        if not nodes:
            return
        rows = [
            (
                n.node_id, n.chunk_id, n.session_id, n.topic_label,
                n.meso_summary, json.dumps(n.micro_summaries),
                n.turn_start, n.turn_end, n.token_count,
                _emb_to_blob(n.embedding), n.created_at.isoformat(),
                json.dumps(n.metadata),
            )
            for n in nodes
        ]
        with self._lock:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO memory_nodes
                (node_id, chunk_id, session_id, topic_label, meso_summary,
                 micro_summaries, turn_start, turn_end, token_count,
                 embedding, created_at, metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
            self._conn.commit()
        self._matrix_dirty = True  # Rebuild on next search

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

    # ── ANN Search ────────────────────────────────────────────────────────────

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
        self._rebuild_matrix_if_needed()

        if self._matrix is None or len(self._node_ids) == 0:
            return []

        qvec = query_embedding.astype(np.float32)

        # If session filter, operate on subset
        if session_id:
            return self._search_filtered(qvec, top_k, session_id, recency_boost)

        return self._search_matrix(qvec, top_k, self._node_ids, self._matrix, recency_boost)

    def _search_matrix(
        self,
        qvec: np.ndarray,
        top_k: int,
        node_ids: list[str],
        matrix: np.ndarray,
        recency_boost: float,
    ) -> list[tuple[MemoryNode, float]]:
        """Brute-force cosine similarity over the full embedding matrix."""
        # Cosine similarity: since embeddings are normalized, sim = dot product
        scores = matrix @ qvec  # shape (N,)

        if recency_boost > 0.0 and len(scores) > 1:
            # Normalize turn_end across nodes → [0, 1], multiply by boost
            turn_ends = np.array([
                self._get_turn_end(nid) for nid in node_ids
            ], dtype=np.float32)
            max_te = turn_ends.max()
            if max_te > 0:
                recency = turn_ends / max_te
                scores = scores + recency * recency_boost

        top_k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            node = self.get(node_ids[idx])
            if node:
                results.append((node, float(scores[idx])))
        return results

    def _search_filtered(
        self,
        qvec: np.ndarray,
        top_k: int,
        session_id: str,
        recency_boost: float,
    ) -> list[tuple[MemoryNode, float]]:
        """Search within a specific session — load subset matrix."""
        nodes = self.get_by_session(session_id)
        if not nodes:
            return []
        matrix = np.stack([n.embedding for n in nodes], axis=0)
        scores = matrix @ qvec
        if recency_boost > 0.0 and len(scores) > 1:
            turn_ends = np.array([n.turn_end for n in nodes], dtype=np.float32)
            max_te = turn_ends.max()
            if max_te > 0:
                recency = turn_ends / max_te
                scores = scores + recency * recency_boost
        top_k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [(nodes[i], float(scores[i])) for i in top_indices]

    def _get_turn_end(self, node_id: str) -> int:
        """Lightweight fetch of turn_end for recency boost calculation."""
        with self._lock:
            row = self._conn.execute(
                "SELECT turn_end FROM memory_nodes WHERE node_id = ?", (node_id,)
            ).fetchone()
        return row["turn_end"] if row else 0

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete(self, node_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM memory_nodes WHERE node_id = ?", (node_id,)
            )
            self._conn.commit()
        if cur.rowcount > 0:
            self._matrix_dirty = True
            return True
        return False

    def delete_session(self, session_id: str) -> int:
        """Delete all nodes for a session. Returns count deleted."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM memory_nodes WHERE session_id = ?", (session_id,)
            )
            self._conn.commit()
        if cur.rowcount > 0:
            self._matrix_dirty = True
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

    def close(self):
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