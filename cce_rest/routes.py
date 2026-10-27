"""CCE REST routes (all under /v1, plus /health)."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from cce_core.auth import allows, configured_keys, scope_for
from cce_core.ids import is_valid_session_id


def require_key(request: Request, authorization: Optional[str] = Header(default=None), x_api_key: Optional[str] = Header(default=None)):
    """
    When CCE_API_KEY or CCE_API_KEYS is set, every /v1 route needs a key (Bearer token or X-API-Key header).
    The session patterns of the key are kept for the route (request.state.scope; None = no authentication).
    """
    request.state.scope = None
    keys = configured_keys()
    if not keys:
        return
    given = x_api_key or (authorization[7:] if authorization and authorization.lower().startswith("bearer ") else "")
    scope = scope_for(given, keys)
    if scope is None:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
    request.state.scope = scope


router = APIRouter()
secured = APIRouter(dependencies=[Depends(require_key)])

Fmt = Literal["messages", "string", "dict"]


class Message(BaseModel):
    role: str = Field(..., description="user | assistant | system | tool")
    content: str = Field(..., max_length=200_000)


class AddMessagesIn(BaseModel):
    messages: list[Message] = Field(..., min_length=1, max_length=5000)


class ContextIn(BaseModel):
    explain: bool = False
    since: Optional[datetime] = Field(default=None, description="only notes and recent turns from this moment on (ISO 8601)")
    until: Optional[datetime] = Field(default=None, description="only notes and recent turns up to this moment")
    last_turns: Optional[int] = Field(default=None, ge=1, description="only the last N turns of the session and the notes that reach into them")
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    fmt: Fmt = "messages"


class CompressIn(BaseModel):
    messages: list[Message] = Field(..., min_length=1, max_length=5000)
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    fmt: Fmt = "messages"


def _manager(request: Request):
    return request.app.state.manager


def _check_id(session_id: str) -> str:
    if not is_valid_session_id(session_id):
        raise HTTPException(status_code=422, detail="session_id must be 1-64 characters: letters, digits, _ . - (and not '.', '..' or 'index')")
    return session_id


def _allow(request: Request, session_id: str) -> None:
    scope = getattr(request.state, "scope", None)
    if scope is not None and not allows(scope, session_id):
        raise HTTPException(status_code=403, detail="This API key may not use this session")


def _require(request: Request, session_id: str):
    _check_id(session_id)
    _allow(request, session_id)
    if not _manager(request).exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")


@router.get("/health", tags=["meta"])
def health(request: Request):
    return {"status": "ok", "active_sessions": len(_manager(request).list_active())}


@secured.post("/v1/compress", tags=["stateless"])
def compress(body: CompressIn, request: Request):
    """One-shot: compress a conversation and return the context for a query. Nothing is stored."""
    result = request.app.state.processor.process(
        [m.model_dump() for m in body.messages], query=body.query, top_k=body.top_k
    )
    return {"context": result.payload.export(body.fmt), "stats": result.stats}


@secured.get("/v1/sessions", tags=["sessions"])
def list_sessions(request: Request):
    scope = getattr(request.state, "scope", None)
    sessions = _manager(request).list_sessions()
    return {"sessions": [s for s in sessions if scope is None or allows(scope, s["session_id"])]}


@secured.post("/v1/sessions/{session_id}/messages", tags=["sessions"], status_code=201)
def add_messages(session_id: str, body: AddMessagesIn, request: Request):
    _check_id(session_id)
    _allow(request, session_id)
    turns = _manager(request).add_messages(session_id, [m.model_dump() for m in body.messages])
    return {"session_id": session_id, "turns_added": len(turns)}


@secured.post("/v1/sessions/{session_id}/context", tags=["sessions"])
def get_context(session_id: str, body: ContextIn, request: Request, explain: bool = False):
    """Context for a query. `?explain=1` (or `"explain": true`) adds which notes and sentences were chosen,
    their scores (dense, recency, BM25), the token budget split and what was dropped."""
    _require(request, session_id)
    payload = _manager(request).build_context_payload(
        session_id, body.query, top_k=body.top_k, since=body.since, until=body.until, last_turns=body.last_turns)
    out = {
        "context": payload.export(body.fmt),
        "token_count": payload.token_count,
        "was_truncated": payload.was_truncated,
        "retrieved_node_ids": payload.retrieved_node_ids,
    }
    if explain or body.explain:
        out["explain"] = payload.explain
    return out


class PinIn(BaseModel):
    text: Optional[str] = Field(default=None, max_length=4000, description="text to pin")
    turn: Optional[int] = Field(default=None, ge=0, description="or the index of a stored turn to pin")


class PinTextIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


class ForgetIn(BaseModel):
    node_id: Optional[str] = Field(default=None, description="forget this note and the text it covered")
    start: Optional[int] = Field(default=None, ge=0)
    end: Optional[int] = Field(default=None, ge=0)


class NoteEditIn(BaseModel):
    topic_label: Optional[str] = Field(default=None, max_length=200)
    meso_summary: Optional[str] = Field(default=None, max_length=8000)


class TurnEditIn(BaseModel):
    content: str = Field(..., min_length=1, max_length=200_000)


def _guard(call):
    """Turn the store's ValueError / KeyError into 4xx answers."""
    try:
        return call()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0]))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@secured.get("/v1/sessions/{session_id}/export", tags=["memory"])
def export_session(session_id: str, request: Request):
    """Everything stored for the session as JSON: turns, notes, pins, session summary, stats (no vectors)."""
    _require(request, session_id)
    return _manager(request).export_session(session_id)


@secured.get("/v1/sessions/{session_id}/pins", tags=["memory"])
def list_pins(session_id: str, request: Request):
    _require(request, session_id)
    return {"pins": _manager(request).list_pins(session_id)}


@secured.post("/v1/sessions/{session_id}/pins", tags=["memory"], status_code=201)
def add_pin(session_id: str, body: PinIn, request: Request):
    """Pin a text (or a stored turn): it is sent with every context for this session."""
    _require(request, session_id)
    return {"pin_id": _guard(lambda: _manager(request).pin(session_id, body.text, body.turn))}


@secured.put("/v1/sessions/{session_id}/pins/{pin_id}", tags=["memory"])
def edit_pin(session_id: str, pin_id: str, body: PinTextIn, request: Request):
    _require(request, session_id)
    if not _manager(request).edit_pin(session_id, pin_id, body.text):
        raise HTTPException(status_code=404, detail="pin not found")
    return {"pin_id": pin_id, "updated": True}


@secured.delete("/v1/sessions/{session_id}/pins/{pin_id}", tags=["memory"])
def remove_pin(session_id: str, pin_id: str, request: Request):
    _require(request, session_id)
    if not _manager(request).unpin(session_id, pin_id):
        raise HTTPException(status_code=404, detail="pin not found")
    return {"pin_id": pin_id, "removed": True}


@secured.post("/v1/sessions/{session_id}/forget", tags=["memory"])
def forget(session_id: str, body: ForgetIn, request: Request):
    """Forget a note, or turns [start, end]. Every note that overlaps is deleted whole, with the text it covered."""
    _require(request, session_id)
    return _guard(lambda: _manager(request).forget(session_id, body.node_id, body.start, body.end))


@secured.patch("/v1/sessions/{session_id}/notes/{node_id}", tags=["memory"])
def edit_note(session_id: str, node_id: str, body: NoteEditIn, request: Request):
    _require(request, session_id)
    if body.topic_label is None and body.meso_summary is None:
        raise HTTPException(status_code=422, detail="give topic_label and/or meso_summary")
    if not _manager(request).edit_note(session_id, node_id, body.topic_label, body.meso_summary):
        raise HTTPException(status_code=404, detail="note not found")
    return {"node_id": node_id, "updated": True}


@secured.patch("/v1/sessions/{session_id}/turns/{index}", tags=["memory"])
def edit_turn(session_id: str, index: int, body: TurnEditIn, request: Request):
    _require(request, session_id)
    if not _manager(request).edit_turn(session_id, index, body.content):
        raise HTTPException(status_code=404, detail="turn not found")
    return {"index": index, "updated": True}


@secured.get("/v1/sessions/{session_id}/summary", tags=["sessions"])
def summary(session_id: str, request: Request):
    _require(request, session_id)
    manager = _manager(request)
    store = manager.get_or_open(session_id)
    nodes = store.warm.get_by_session(session_id)
    if not nodes:
        return {"session_id": session_id, "macro_summary": "", "node_count": 0, "topics": []}
    return {
        "session_id": session_id,
        "macro_summary": request.app.state.engine.macro_summary(nodes),
        "node_count": len(nodes),
        "topics": [n.topic_label for n in nodes],
    }


@secured.get("/v1/sessions/{session_id}/stats", tags=["sessions"])
def stats(session_id: str, request: Request):
    _require(request, session_id)
    return _manager(request).session_stats(session_id)


@secured.post("/v1/sessions/{session_id}/close", tags=["sessions"])
def close(session_id: str, request: Request, checkpoint: bool = True):
    _require(request, session_id)
    return {"session_id": session_id, "closed": _manager(request).close(session_id, checkpoint=checkpoint)}


@secured.delete("/v1/sessions/{session_id}", tags=["sessions"])
def delete(session_id: str, request: Request):
    _require(request, session_id)
    return {"session_id": session_id, "deleted": _manager(request).delete(session_id)}

router.include_router(secured)
