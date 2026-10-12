"""CCE REST routes (all under /v1, plus /health)."""

from __future__ import annotations

import hmac
import os
import re
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

def require_key(authorization: Optional[str] = Header(default=None), x_api_key: Optional[str] = Header(default=None)):
    """When CCE_API_KEY is set, every /v1 route needs it (Bearer token or X-API-Key header)."""
    expected = os.getenv("CCE_API_KEY")
    if not expected:
        return
    given = x_api_key or (authorization[7:] if authorization and authorization.lower().startswith("bearer ") else "")
    if not given or not hmac.compare_digest(given, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


router = APIRouter()
secured = APIRouter(dependencies=[Depends(require_key)])

_SESSION_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
Fmt = Literal["messages", "string", "dict"]


class Message(BaseModel):
    role: str = Field(..., description="user | assistant | system | tool")
    content: str = Field(..., max_length=200_000)


class AddMessagesIn(BaseModel):
    messages: list[Message] = Field(..., min_length=1, max_length=5000)


class ContextIn(BaseModel):
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
    if not _SESSION_RE.match(session_id):
        raise HTTPException(status_code=422, detail="session_id must be 1-64 characters: letters, digits, _ . -")
    return session_id


def _require(request: Request, session_id: str):
    _check_id(session_id)
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
    return {"sessions": _manager(request).list_sessions()}


@secured.post("/v1/sessions/{session_id}/messages", tags=["sessions"], status_code=201)
def add_messages(session_id: str, body: AddMessagesIn, request: Request):
    _check_id(session_id)
    turns = _manager(request).add_messages(session_id, [m.model_dump() for m in body.messages])
    return {"session_id": session_id, "turns_added": len(turns)}


@secured.post("/v1/sessions/{session_id}/context", tags=["sessions"])
def get_context(session_id: str, body: ContextIn, request: Request):
    _require(request, session_id)
    payload = _manager(request).build_context_payload(session_id, body.query, top_k=body.top_k)
    return {
        "context": payload.export(body.fmt),
        "token_count": payload.token_count,
        "was_truncated": payload.was_truncated,
        "retrieved_node_ids": payload.retrieved_node_ids,
    }


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
