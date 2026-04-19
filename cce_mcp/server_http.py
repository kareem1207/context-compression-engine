"""
CCE REST Bridge for Demo
Thin FastAPI wrapper around the CCE engine so the demo client
can call CCE tools over HTTP without needing a full MCP client.

Run: uv run python -m cce_mcp.server_http
Listens on: http://localhost:9000
"""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from cce_core.config import CCEConfig
from cce_core.engine import CCEEngine
from cce_core.session.manager import SessionManager

config  = CCEConfig(summarizer_mode="extractive", hot_tier_max_turns=4, retrieval_top_k=3)
engine  = CCEEngine(config)
manager = SessionManager(engine)

app = FastAPI(title="CCE Demo Bridge")


class HistoryIn(BaseModel):
    session_id: str
    messages: list[dict]

class ContextIn(BaseModel):
    session_id: str
    query: str
    fmt: str = "messages"
    top_k: Optional[int] = None

class CloseIn(BaseModel):
    session_id: str
    checkpoint: bool = True


@app.post("/ingest_history")
def ingest_history(body: HistoryIn):
    turns = manager.add_messages(body.session_id, body.messages)
    store = manager.get(body.session_id)
    stats = store.stats()
    return {
        "status": "ok",
        "turns_loaded": len(turns),
        "warm_nodes": stats.get("warm_nodes", 0),
        "hot_turns": stats.get("hot_turns", 0),
    }


@app.post("/retrieve_context")
def retrieve_context(body: ContextIn):
    payload = manager.build_context_payload(body.session_id, body.query, top_k=body.top_k)
    return {
        "status": "ok",
        "context": payload.export(body.fmt),
        "past_blocks": len(payload.past_blocks),
        "recent_turns": len(payload.recent_turns),
        "token_count": payload.token_count,
        "was_truncated": payload.was_truncated,
    }


@app.post("/close_session")
def close_session(body: CloseIn):
    manager.close(body.session_id, checkpoint=body.checkpoint)
    return {"status": "ok", "session_id": body.session_id}


@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(manager.list_active())}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="warning")