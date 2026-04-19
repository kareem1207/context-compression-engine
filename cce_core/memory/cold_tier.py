"""
CCE Cold Tier
Disk-based store for macro (session-level) summaries.
Each session gets one JSON file containing its macro summary +
a lightweight index (session_id → file path + metadata).

Cold tier is write-once per session — once a session is macro-summarized
it doesn't change. This makes it append-only and trivially safe.

Structure on disk:
    .cce_data/cold/
        index.json              ← maps session_id to metadata
        <session_id>.json       ← full macro summary record

A MacroSummary contains:
  - session_id
  - macro_text: the one-paragraph session summary
  - node_count: how many warm nodes were summarized
  - turn_count: total turns in the session
  - token_count_original: tokens before compression
  - token_count_compressed: tokens in macro_text
  - topics: list of topic labels from all chunks
  - created_at
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from cce_core.config import CCEConfig, DEFAULT_CONFIG
from cce_core.ingestion import tokenizer


_INDEX_FILE = "index.json"


@dataclass
class MacroSummary:
    session_id: str
    macro_text: str
    node_count: int
    turn_count: int
    token_count_original: int
    token_count_compressed: int
    topics: list[str]
    created_at: datetime
    metadata: dict = field(default_factory=dict)

    @property
    def compression_ratio(self) -> float:
        return round(
            self.token_count_original / max(self.token_count_compressed, 1), 2
        )

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "macro_text": self.macro_text,
            "node_count": self.node_count,
            "turn_count": self.turn_count,
            "token_count_original": self.token_count_original,
            "token_count_compressed": self.token_count_compressed,
            "topics": self.topics,
            "compression_ratio": self.compression_ratio,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MacroSummary":
        return cls(
            session_id=d["session_id"],
            macro_text=d["macro_text"],
            node_count=d["node_count"],
            turn_count=d["turn_count"],
            token_count_original=d["token_count_original"],
            token_count_compressed=d["token_count_compressed"],
            topics=d.get("topics", []),
            created_at=datetime.fromisoformat(d["created_at"]),
            metadata=d.get("metadata", {}),
        )

    def __repr__(self):
        return (
            f"MacroSummary(session={self.session_id!r}, "
            f"ratio={self.compression_ratio}x, topics={self.topics})"
        )


class ColdTier:
    """
    Append-only disk store for session-level macro summaries.

    Usage:
        cold = ColdTier()
        cold.write(summary)
        s = cold.get("session-id")
        all_summaries = cold.list_all()
    """

    def __init__(self, config: CCEConfig = DEFAULT_CONFIG):
        self.config = config
        config.ensure_dirs()
        self._cold_dir = config.cold_dir
        self._index_path = self._cold_dir / _INDEX_FILE
        self._lock = threading.Lock()
        self._index: dict[str, dict] = self._load_index()

    # ── Write ─────────────────────────────────────────────────────────────────

    def write(self, summary: MacroSummary) -> Path:
        """
        Persist a MacroSummary to disk.
        Overwrites if session_id already exists.
        Returns the path to the written file.
        """
        file_path = self._cold_dir / f"{summary.session_id}.json"
        data = summary.to_dict()

        with self._lock:
            file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            self._index[summary.session_id] = {
                "file": file_path.name,
                "created_at": summary.created_at.isoformat(),
                "node_count": summary.node_count,
                "turn_count": summary.turn_count,
                "compression_ratio": summary.compression_ratio,
                "topics": summary.topics,
            }
            self._save_index()

        return file_path

    def write_from_nodes(
        self,
        session_id: str,
        nodes: list,         # list[MemoryNode] — avoid circular import
        macro_text: str,
        turn_count: int,
        token_count_original: int,
    ) -> MacroSummary:
        """
        Convenience: build a MacroSummary from MemoryNodes and write it.
        """
        summary = MacroSummary(
            session_id=session_id,
            macro_text=macro_text,
            node_count=len(nodes),
            turn_count=turn_count,
            token_count_original=token_count_original,
            token_count_compressed=tokenizer.count(macro_text),
            topics=[n.topic_label for n in nodes],
            created_at=datetime.now(timezone.utc),
        )
        self.write(summary)
        return summary

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, session_id: str) -> MacroSummary | None:
        """Load a macro summary by session_id. Returns None if not found."""
        with self._lock:
            entry = self._index.get(session_id)
        if not entry:
            return None
        file_path = self._cold_dir / entry["file"]
        if not file_path.exists():
            return None
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return MacroSummary.from_dict(data)

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._index

    def list_all(self) -> list[dict]:
        """
        Return index metadata for all stored sessions.
        Lightweight — doesn't load full macro texts.
        """
        with self._lock:
            return [
                {"session_id": sid, **meta}
                for sid, meta in self._index.items()
            ]

    def list_sessions(self) -> list[str]:
        with self._lock:
            return list(self._index.keys())

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete(self, session_id: str) -> bool:
        with self._lock:
            if session_id not in self._index:
                return False
            entry = self._index.pop(session_id)
            self._save_index()

        file_path = self._cold_dir / entry["file"]
        if file_path.exists():
            file_path.unlink()
        return True

    # ── Stats ─────────────────────────────────────────────────────────────────

    def count(self) -> int:
        with self._lock:
            return len(self._index)

    def total_sessions_token_saved(self) -> int:
        """Rough estimate of tokens saved across all sessions."""
        total = 0
        with self._lock:
            entries = list(self._index.values())
        for e in entries:
            ratio = e.get("compression_ratio", 1.0)
            node_count = e.get("node_count", 0)
            total += int(node_count * 200 * (ratio - 1))  # 200 tokens avg per node
        return total

    # ── Index management ──────────────────────────────────────────────────────

    def _load_index(self) -> dict[str, dict]:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_index(self):
        """Must be called within self._lock."""
        self._index_path.write_text(
            json.dumps(self._index, indent=2), encoding="utf-8"
        )

    def __repr__(self):
        return f"ColdTier(dir={self._cold_dir!r}, sessions={self.count()})" 