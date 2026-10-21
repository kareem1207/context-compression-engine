"""MCP tool functions, called the way the server calls them (no transport)."""

import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")

from cce_core.engine import CCEEngine
from cce_mcp.schema import CloseSessionInput, IngestTurnInput, RetrieveContextInput, SessionStatsInput
from cce_mcp import server

from tests.test_live_sessions import PLANTED, make_config, make_messages


class FakeCtx:
    """The part of mcp's Context that the tools use."""

    def __init__(self, engine, manager):
        self.request_context = SimpleNamespace(lifespan_context={"engine": engine, "manager": manager})

    async def report_progress(self, *args, **kwargs):
        return None


def call(coro):
    return json.loads(asyncio.run(coro))


def test_live_session_through_the_mcp_tools(tmp_path):
    cfg = make_config(tmp_path)
    engine = CCEEngine(cfg)
    ctx = FakeCtx(engine, engine.create_session_manager())
    msgs = make_messages()

    for m in msgs:
        out = call(server.cce_ingest_turn(IngestTurnInput(session_id="mcp", **m), ctx))
        assert out["status"] == "ok"
    assert out["index"] == len(msgs) - 1

    stats = call(server.cce_session_stats(SessionStatsInput(session_id="mcp"), ctx))
    assert stats["status"] == "ok"
    covered_by_notes = len(msgs) - stats["hot_turns"] - stats["pending_turns"]
    assert 0 < stats["warm_nodes"] < covered_by_notes

    out = call(server.cce_retrieve_context(
        RetrieveContextInput(session_id="mcp", query="How many invoice retry attempts are allowed?", fmt="string"), ctx))
    assert out["status"] == "ok" and PLANTED in out["context"]

    assert call(server.cce_close_session(CloseSessionInput(session_id="mcp"), ctx))["status"] == "ok"
