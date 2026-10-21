"""Explain output: which notes and sentences were chosen, scores, the token budget split, what was dropped."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from cce_core.cli import format_explain, main as cce_main
from cce_core.engine import CCEEngine
from cce_mcp import schema, server
from cce_rest.app import create_app

from tests.test_live_sessions import PLANTED, make_config, make_messages

QUESTION = "How many invoice retry attempts are allowed?"


def live_manager(tmp_path, **kw):
    cfg = make_config(tmp_path, retrieval_top_k=3, **kw)
    mgr = CCEEngine(cfg).create_session_manager()
    for m in make_messages():
        mgr.add_message("s", m)
    mgr.close("s")
    return cfg, CCEEngine(cfg).create_session_manager()


def test_payload_explains_scores_budget_and_choices(tmp_path):
    _, mgr = live_manager(tmp_path)
    ex = mgr.build_context_payload("s", QUESTION).explain

    q = ex["query"]
    assert q["text"] == QUESTION and q["mode"] == "search" and q["notes_in_session"] == 3 and q["embedder"]["name"] == "hash"
    b = ex["budget"]
    assert b["total"] == b["system_note"] + b["overhead"] + b["recent_turns"] + b["pending_turns"] + b["past_notes"]
    assert b["total"] <= b["max_tokens"]

    notes = ex["notes"]
    assert len(notes) == 3 and all(n["included"] for n in notes) and ex["dropped_for_budget"] == []
    best = notes[0]
    assert best["topic"].startswith("billing") and set(best["scores"]) == {"dense", "recency", "bm25", "fused"}
    assert any(PLANTED in quote for quote in best["quotes"])
    assert [n["scores"]["fused"] for n in notes] == sorted((n["scores"]["fused"] for n in notes), reverse=True)


def test_overview_and_lexical_terms_are_reported(tmp_path):
    _, mgr = live_manager(tmp_path)
    over = mgr.build_context_payload("s", "What did we talk about in this conversation?").explain
    assert over["query"]["mode"] == "overview"
    assert all(n["overview_lines"] for n in over["notes"])
    lex = mgr.build_context_payload("s", "What does ERR-4021-X mean?").explain
    assert "err-4021-x" in lex["query"]["bm25_terms"]


def test_notes_that_do_not_fit_are_listed_as_dropped(tmp_path):
    _, mgr = live_manager(tmp_path, context_max_tokens=300)
    ex = mgr.build_context_payload("s", QUESTION).explain
    assert ex["dropped_for_budget"], "with a 300-token cap not every note can be sent"
    dropped = {n["node_id"] for n in ex["notes"] if not n["included"]}
    assert dropped == set(ex["dropped_for_budget"]) and all(n["tokens"] == 0 for n in ex["notes"] if not n["included"])
    assert ex["budget"]["total"] <= 300


def test_rest_returns_explain_only_when_asked(tmp_path):
    cfg = make_config(tmp_path, retrieval_top_k=3)
    with TestClient(create_app(cfg)) as c:
        c.post("/v1/sessions/r/messages", json={"messages": make_messages()})
        plain = c.post("/v1/sessions/r/context", json={"query": QUESTION}).json()
        assert "explain" not in plain
        asked = c.post("/v1/sessions/r/context?explain=1", json={"query": QUESTION}).json()
        assert asked["explain"]["notes"] and asked["explain"]["budget"]["total"] == asked["token_count"]
        body = c.post("/v1/sessions/r/context", json={"query": QUESTION, "explain": True}).json()
        assert "explain" in body


def test_mcp_tool_returns_explain(tmp_path):
    cfg = make_config(tmp_path, retrieval_top_k=3)
    engine = CCEEngine(cfg)
    ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context={"engine": engine, "manager": engine.create_session_manager()}))

    async def progress(*a, **k):
        return None

    ctx.report_progress = progress
    mgr = ctx.request_context.lifespan_context["manager"]
    for m in make_messages():
        mgr.add_message("m", m)
    with_it = json.loads(asyncio.run(server.cce_retrieve_context(
        schema.RetrieveContextInput(session_id="m", query=QUESTION, explain=True), ctx)))
    without = json.loads(asyncio.run(server.cce_retrieve_context(
        schema.RetrieveContextInput(session_id="m", query=QUESTION), ctx)))
    assert with_it["explain"]["notes"] and "explain" not in without


# ── cce inspect ───────────────────────────────────────────────────────────────

def run_cli(capsys, tmp_path, *args):
    code = None
    try:
        cce_main(["inspect", *args, "--data-dir", str(tmp_path / "data"), "--embedder", "hash"])
    except SystemExit as e:
        code = e.code
    return code, capsys.readouterr()


def test_cce_inspect_lists_sessions_notes_and_explains(tmp_path, capsys):
    live_manager(tmp_path)
    code, out = run_cli(capsys, tmp_path)
    assert code == 0 and "s " in out.out and "closed" in out.out

    code, out = run_cli(capsys, tmp_path, "s")
    assert code == 0 and "3 notes" in out.out and out.out.count("turns") >= 4

    code, out = run_cli(capsys, tmp_path, "s", "--query", QUESTION)
    assert code == 0
    for needle in ("Token budget:", "Notes, best first:", "dense", "bm25", "fused", "sent", PLANTED[:30]):
        assert needle in out.out, needle

    code, out = run_cli(capsys, tmp_path, "s", "--query", QUESTION, "--json")
    assert json.loads(out.out)["query"]["text"] == QUESTION

    code, out = run_cli(capsys, tmp_path, "s", "--json")
    assert json.loads(out.out)["stats"]["warm_nodes"] == 3


def test_cce_inspect_errors_are_clear(tmp_path, capsys):
    live_manager(tmp_path)
    code, out = run_cli(capsys, tmp_path, "nope")
    assert code == 1 and "Unknown session" in out.err
    with pytest.raises(SystemExit, match="No CCE data"):
        cce_main(["inspect", "--data-dir", str(tmp_path / "empty"), "--embedder", "hash"])
    # a different embedder than the folder was made with is explained, not a traceback
    code, out = run_cli(capsys, tmp_path, "s")
    assert code == 0
    code = None
    try:
        cce_main(["inspect", "s", "--data-dir", str(tmp_path / "data"), "--embedder", "onnx"])
    except SystemExit as e:
        code = e.code
    assert code in (0, 2)           # 2 when the ONNX extra/model is present (mismatch), ImportError message otherwise


def test_format_explain_marks_dropped_notes():
    text = format_explain({
        "query": {"text": "q", "mode": "search", "top_k": 2, "embedder": {"name": "hash"}, "bm25_terms": [], "notes_in_session": 2, "notes_retrieved": 2},
        "budget": {"total": 10, "max_tokens": 20, "usable": 18, "system_note": 1, "recent_turns": 2, "pending_turns": 0, "past_notes": 3, "overhead": 4},
        "recent": {"turns": 1, "shortened_or_dropped": False}, "pending": {"turns_available": 0},
        "notes": [{"node_id": "a", "topic": "alpha", "turns": [0, 3], "included": True, "tokens": 5,
                   "scores": {"dense": 0.5, "recency": 0.1, "bm25": 0.0, "fused": 0.02}, "quotes": ["a quote"], "overview_lines": []},
                  {"node_id": "b", "topic": "beta", "turns": [4, 7], "included": False, "tokens": 0,
                   "scores": {"dense": 0.1, "recency": 0.2, "bm25": 0.0, "fused": 0.01}, "quotes": [], "overview_lines": []}],
        "dropped_for_budget": ["b"],
    })
    assert "[sent   ] turns 0-3  alpha" in text and "[DROPPED] turns 4-7  beta" in text and "1 retrieved note(s) did not fit" in text
