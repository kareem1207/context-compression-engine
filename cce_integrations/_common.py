"""Shared by the LangChain and LlamaIndex classes."""

from __future__ import annotations

from pathlib import Path

from cce_core.config import CCEConfig
from cce_core.engine import CCEEngine
from cce_core.session.manager import SessionManager


def make_manager(manager: SessionManager | None, data_dir: str | Path | None, config: dict | None) -> SessionManager:
    """The given manager, or a new one on `data_dir` (default: CCE_DATA_DIR / .cce_data) with CCEConfig options."""
    if manager is not None:
        return manager
    options = dict(config or {})
    if data_dir is not None:
        options["base_dir"] = Path(data_dir)
    return CCEEngine(CCEConfig(**options)).create_session_manager()


def context_pairs(manager: SessionManager, session_id: str, query: str | None) -> list[tuple[str, str]]:
    """(role, text) pairs: the CCE context for `query`, or the plain recent turns when there is none yet."""
    store = manager.get_or_open(session_id)
    if not query:
        recent = store.get_pending_turns() + store.get_hot_turns()
        return [(t.role, t.content) for t in recent]
    payload = manager.build_context_payload(session_id, query)
    return [(m["role"], m["content"]) for m in payload.to_messages()]


def last_user_text(manager: SessionManager, session_id: str) -> str | None:
    store = manager.get_or_open(session_id)
    for t in reversed(store.get_pending_turns() + store.get_hot_turns()):
        if t.role == "user":
            return t.content
    return None
