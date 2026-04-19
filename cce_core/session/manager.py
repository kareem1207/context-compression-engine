"""
CCE Session Manager
Manages the lifecycle of multiple concurrent sessions.
Each session gets its own MemoryStore. The manager handles:

  - Opening / resuming sessions
  - Routing messages to the correct store
  - Periodic checkpointing (cold tier write)
  - Graceful session closure
  - Session registry (in-memory + disk index)

This is what the MCP server and REST API talk to.
One SessionManager instance per CCEEngine process.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from cce_core.config import CCEConfig, DEFAULT_CONFIG
from cce_core.memory.store import MemoryStore
from cce_core.ingestion.segmenter import Turn, segment, segment_incremental

if TYPE_CHECKING:
    from cce_core.engine import CCEEngine


_REGISTRY_FILE = "sessions.json"


class SessionManager:
    """
    Multi-session lifecycle manager.

    Usage:
        manager = SessionManager(engine)

        # Open or resume a session
        store = manager.open("session-abc")

        # Ingest a new turn
        manager.add_message("session-abc", {"role": "user", "content": "hello"})

        # Build context for LLM
        payload = manager.build_context("session-abc", "hello")

        # Close and checkpoint
        manager.close("session-abc")
    """

    def __init__(self, engine: "CCEEngine"):
        self.engine = engine
        self.config = engine.config
        self._lock = threading.Lock()

        # Active sessions: session_id → MemoryStore
        self._sessions: dict[str, MemoryStore] = {}

        # Turn buffers: session_id → list[Turn] (for incremental processing)
        self._turn_buffers: dict[str, list[Turn]] = {}

        # Session metadata registry
        self._registry: dict[str, dict] = self._load_registry()

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def open(self, session_id: str) -> MemoryStore:
        """
        Open or resume a session.
        If the session has warm tier data on disk, it's already accessible
        via the WarmTier (SQLite persists between runs).
        Returns the MemoryStore for this session.
        """
        with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]

            store = self.engine.open_session(session_id)
            self._sessions[session_id] = store
            self._turn_buffers[session_id] = []

            # Register if new
            if session_id not in self._registry:
                self._registry[session_id] = {
                    "session_id": session_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "last_active": datetime.now(timezone.utc).isoformat(),
                    "turn_count": 0,
                    "status": "active",
                }
                self._save_registry()

            return store

    def get(self, session_id: str) -> MemoryStore | None:
        """Get an already-open session store. Returns None if not open."""
        with self._lock:
            return self._sessions.get(session_id)

    def get_or_open(self, session_id: str) -> MemoryStore:
        """Get session if open, otherwise open it."""
        store = self.get(session_id)
        return store if store is not None else self.open(session_id)

    def close(self, session_id: str, checkpoint: bool = True) -> bool:
        """
        Close a session.
        If checkpoint=True, writes macro summary to cold tier first.
        Returns True if session was open, False if not found.
        """
        with self._lock:
            store = self._sessions.get(session_id)
            if not store:
                return False

            if checkpoint:
                self._do_checkpoint(session_id, store)

            store.close()
            del self._sessions[session_id]
            del self._turn_buffers[session_id]

            if session_id in self._registry:
                self._registry[session_id]["status"] = "closed"
                self._registry[session_id]["last_active"] = datetime.now(timezone.utc).isoformat()
                self._save_registry()

        return True

    def close_all(self, checkpoint: bool = True):
        """Close all active sessions."""
        session_ids = list(self._sessions.keys())
        for sid in session_ids:
            self.close(sid, checkpoint=checkpoint)

    # ── Message handling ──────────────────────────────────────────────────────

    def add_message(
        self,
        session_id: str,
        message: dict,
    ) -> Turn:
        """
        Add a single message to a session.
        Auto-opens the session if not already open.
        Returns the Turn object created.

        Args:
            session_id: Target session.
            message: {"role": "user"|"assistant", "content": "..."}
        """
        store = self.get_or_open(session_id)
        buf = self._turn_buffers[session_id]

        turn = segment_incremental(message, buf, session_id=session_id)
        store.ingest_turn(turn)

        with self._lock:
            if session_id in self._registry:
                self._registry[session_id]["turn_count"] += 1
                self._registry[session_id]["last_active"] = datetime.now(timezone.utc).isoformat()

        return turn

    def add_messages(
        self,
        session_id: str,
        messages: list[dict],
    ) -> list[Turn]:
        """
        Bulk-add messages to a session. Efficient for loading history.
        """
        store = self.get_or_open(session_id)
        buf = self._turn_buffers[session_id]

        turns = segment(messages, session_id=session_id)
        # Adjust indices to continue from existing buffer
        offset = len(buf)
        for t in turns:
            t.index += offset
        buf.extend(turns)

        store.ingest_turns(turns)

        with self._lock:
            if session_id in self._registry:
                self._registry[session_id]["turn_count"] += len(turns)
                self._registry[session_id]["last_active"] = datetime.now(timezone.utc).isoformat()

        return turns

    # ── Context retrieval ─────────────────────────────────────────────────────

    def build_context(
        self,
        session_id: str,
        query: str,
        fmt: str = "messages",
        top_k: int | None = None,
    ):
        """
        Build compressed context for an LLM call.
        Auto-opens session if needed.

        Returns the exported context (list[dict], str, or dict based on fmt).
        """
        store = self.get_or_open(session_id)
        payload = self.engine.build_context(store, query, top_k=top_k, fmt=fmt)
        return payload.export(fmt)

    def build_context_payload(
        self,
        session_id: str,
        query: str,
        top_k: int | None = None,
    ):
        """Same as build_context but returns the full ContextPayload object."""
        store = self.get_or_open(session_id)
        return self.engine.build_context(store, query, top_k=top_k)

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def checkpoint(self, session_id: str) -> bool:
        """
        Manually trigger a checkpoint for a session.
        Flushes hot tier to warm, writes macro summary to cold.
        Returns True if successful.
        """
        with self._lock:
            store = self._sessions.get(session_id)
        if not store:
            return False
        self._do_checkpoint(session_id, store)
        return True

    def _do_checkpoint(self, session_id: str, store: MemoryStore):
        """Internal checkpoint logic. Called within close() or manually."""
        # Flush any remaining hot tier turns to warm
        store.flush_hot_to_warm()

        # Generate macro summary from all warm nodes
        nodes = store.warm.get_by_session(session_id)
        if nodes:
            macro = self.engine.macro_summary(nodes)
            store.checkpoint(macro)

    # ── Session info ──────────────────────────────────────────────────────────

    def list_sessions(self) -> list[dict]:
        """List all known sessions (active + closed) with metadata."""
        with self._lock:
            result = list(self._registry.values())

        # Annotate active ones
        active = set(self._sessions.keys())
        for entry in result:
            entry["is_active"] = entry["session_id"] in active

        return result

    def list_active(self) -> list[str]:
        """Return session IDs of currently open sessions."""
        with self._lock:
            return list(self._sessions.keys())

    def session_stats(self, session_id: str) -> dict | None:
        """Return detailed stats for a session."""
        store = self.get_or_open(session_id)
        if not store:
            return None
        return store.stats()

    def is_active(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def exists(self, session_id: str) -> bool:
        """True if session has ever been opened (in registry)."""
        with self._lock:
            return session_id in self._registry

    def delete(self, session_id: str) -> bool:
        """
        Permanently delete a session — closes it, removes warm + cold data.
        WARNING: irreversible.
        """
        self.close(session_id, checkpoint=False)
        store = MemoryStore(session_id, self.config)
        store.warm.delete_session(session_id)
        store.cold.delete(session_id)
        store.close()
        with self._lock:
            self._registry.pop(session_id, None)
            self._save_registry()
        return True

    # ── Registry persistence ──────────────────────────────────────────────────

    def _registry_path(self) -> Path:
        self.config.ensure_dirs()
        return self.config.base_dir / _REGISTRY_FILE

    def _load_registry(self) -> dict[str, dict]:
        path = self._registry_path()
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_registry(self):
        """Must be called within self._lock."""
        path = self._registry_path()
        path.write_text(
            json.dumps(self._registry, indent=2), encoding="utf-8"
        )

    def __repr__(self):
        return (
            f"SessionManager(active={len(self._sessions)}, "
            f"total_known={len(self._registry)})"
        )