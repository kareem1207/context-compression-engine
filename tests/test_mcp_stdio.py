"""
The MCP server as a client sees it: started as a subprocess on stdio, every tool called through the protocol.
"""

import asyncio
import json
import os
import sys

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

from tests.test_live_sessions import PLANTED, make_messages  # noqa: E402

EXPECTED_TOOLS = {
    "cce_ingest_turn", "cce_ingest_history", "cce_retrieve_context", "cce_summarize_session",
    "cce_session_stats", "cce_close_session", "cce_list_sessions", "cce_stateless_compress",
    "cce_pin_memory", "cce_forget_memory", "cce_export_session",
}


def parse(result) -> dict:
    assert not result.isError, result
    return json.loads(result.content[0].text)


async def scenario(data_dir):
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "cce_mcp.server"],
        env={**os.environ, "CCE_DATA_DIR": str(data_dir), "CCE_EMBEDDER": "hash", "CCE_HOT_TIER_TURNS": "4",
             "CCE_TOP_K": "3", "CCE_MAX_TOKENS": "450", "PYTHONIOENCODING": "utf-8"},
    )
    seen = {}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {t.name: t for t in (await session.list_tools()).tools}
            seen["tools"] = set(tools)

            async def call(name, **args):
                return parse(await session.call_tool(name, {"params": args} if args else {}))

            msgs = make_messages()
            seen["history"] = await call("cce_ingest_history", session_id="e2e", messages=msgs[:16])
            for m in msgs[16:]:
                seen["turn"] = await call("cce_ingest_turn", session_id="e2e", role=m["role"], content=m["content"])
            seen["stats"] = await call("cce_session_stats", session_id="e2e")
            seen["context"] = await call("cce_retrieve_context", session_id="e2e",
                                         query="How many invoice retry attempts are allowed?", fmt="string", explain=True)
            seen["summary"] = await call("cce_summarize_session", session_id="e2e")
            seen["list"] = await call("cce_list_sessions")
            seen["stateless"] = await call("cce_stateless_compress", messages=msgs,
                                           query="How many invoice retry attempts are allowed?", fmt="string")
            seen["pin"] = await call("cce_pin_memory", session_id="e2e", action="add", text="the codeword is Marigold")
            seen["pins"] = await call("cce_pin_memory", session_id="e2e", action="list")
            seen["pinned_ctx"] = await call("cce_retrieve_context", session_id="e2e", query="unrelated hiking question", fmt="string",
                                            last_turns=6)
            seen["export"] = await call("cce_export_session", session_id="e2e")
            seen["forget_refused"] = await call("cce_forget_memory", session_id="e2e", start=0, end=1)
            seen["forget"] = await call("cce_forget_memory", session_id="e2e", start=0, end=1, confirm=True)
            seen["close"] = await call("cce_close_session", session_id="e2e", checkpoint=True)
            seen["bad_id"] = await session.call_tool(
                "cce_ingest_turn", {"params": {"session_id": "../../x", "role": "user", "content": "hi"}})
    return seen


def test_every_tool_works_over_stdio(tmp_path):
    seen = asyncio.run(asyncio.wait_for(scenario(tmp_path / "data"), timeout=240))

    assert seen["tools"] == EXPECTED_TOOLS          # a new tool without a test here fails this line
    assert seen["history"]["turns_loaded"] == 16 and seen["turn"]["index"] == len(make_messages()) - 1
    assert seen["stats"]["turns_ingested"] == len(make_messages()) and seen["stats"]["warm_nodes"] >= 1
    ctx = seen["context"]
    assert PLANTED in ctx["context"] and ctx["explain"]["notes"] and ctx["token_count"] <= 450
    assert seen["summary"]["node_count"] >= 1 and seen["summary"]["macro_summary"]
    assert any(s["session_id"] == "e2e" for s in seen["list"]["sessions"])
    assert PLANTED in seen["stateless"]["context"]
    assert seen["close"]["checkpointed"] is True
    assert seen["pin"]["pin_id"] and seen["pins"]["pins"][0]["text"] == "the codeword is Marigold"
    assert "Marigold" in seen["pinned_ctx"]["context"]                       # pinned text rides along with an unrelated question
    assert len(seen["export"]["turns"]) == len(make_messages()) and seen["export"]["pins"]
    assert seen["forget_refused"]["status"] == "error" and seen["forget"]["status"] == "ok"

    bad = seen["bad_id"]                             # a path-traversal id is refused by the schema
    assert bad.isError or "error" in bad.content[0].text.lower()

    # the data written by the subprocess is there after it exited
    assert (tmp_path / "data" / "warm.db").exists() and (tmp_path / "data" / "cold" / "e2e.json").exists()
    assert not (tmp_path / "x.json").exists()
