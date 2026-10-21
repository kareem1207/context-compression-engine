"""Managing what is remembered: pin, edit, forget, export, and time-aware queries."""

import json
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from cce_core.config import CCEConfig
from cce_core.engine import CCEEngine
from cce_rest.app import create_app

from tests.test_live_sessions import PLANTED, make_config, make_messages

QUESTION = "How many invoice retry attempts are allowed?"


def cfg_for(tmp_path, **kw) -> CCEConfig:
    return make_config(tmp_path, retrieval_top_k=3, **kw)


def session(tmp_path, sid="s", close=True, **kw):
    cfg = cfg_for(tmp_path, **kw)
    mgr = CCEEngine(cfg).create_session_manager()
    for m in make_messages():
        mgr.add_message(sid, m)
    if close:
        mgr.close(sid)
        mgr = CCEEngine(cfg).create_session_manager()
    return cfg, mgr


def text_of(mgr, sid="s", query=QUESTION, **kw) -> str:
    return mgr.build_context_payload(sid, query, **kw).to_string()


# ── Pins ──────────────────────────────────────────────────────────────────────

def test_a_pin_is_sent_with_every_context_and_survives_a_restart(tmp_path):
    cfg, mgr = session(tmp_path)
    pin_id = mgr.pin("s", "The customer's name is Gertrude Hollowell.")
    for query in (QUESTION, "what is the weather like on hiking trails", "What did we talk about in this conversation?"):
        payload = mgr.build_context_payload("s", query)
        assert "Gertrude Hollowell" in payload.to_string() and payload.pinned == ["The customer's name is Gertrude Hollowell."]
        assert "=== PINNED (always included) ===" in payload.to_messages()[0]["content"]
        assert payload.explain["budget"]["pinned"] > 0 and payload.token_count <= cfg.context_max_tokens
    mgr.close_all()
    again = CCEEngine(cfg).create_session_manager()
    assert again.list_pins("s")[0]["pin_id"] == pin_id and "Gertrude" in text_of(again)


def test_pins_can_be_edited_and_removed_and_leave_with_the_session(tmp_path):
    cfg, mgr = session(tmp_path)
    pin_id = mgr.pin("s", "old text")
    assert mgr.edit_pin("s", pin_id, "new text") and "new text" in text_of(mgr) and "old text" not in text_of(mgr)
    assert mgr.unpin("s", pin_id) and mgr.list_pins("s") == [] and "new text" not in text_of(mgr)
    assert not mgr.unpin("s", pin_id) and not mgr.edit_pin("s", "nope", "x")
    mgr.pin("s", "gone with the session")
    mgr.delete("s")
    fresh = CCEEngine(cfg).create_session_manager()
    assert fresh.list_pins("s") == []


def test_a_turn_can_be_pinned(tmp_path):
    _, mgr = session(tmp_path)                       # closed: every turn is in a note and its text is kept
    mgr.pin("s", turn=1)
    assert mgr.list_pins("s")[0]["source"] == "turn:1" and "answer 0 about billing" in mgr.list_pins("s")[0]["text"]
    with pytest.raises(KeyError):
        mgr.pin("s", turn=999)
    with pytest.raises(ValueError):
        mgr.pin("s")
    with pytest.raises(ValueError):
        mgr.pin("s", "text", 1)


def test_pins_never_take_more_than_a_quarter_of_the_budget(tmp_path):
    cfg, mgr = session(tmp_path, context_max_tokens=500)
    for i in range(30):
        mgr.pin("s", f"pinned fact number {i}: " + "detail " * 30)
    payload = mgr.build_context_payload("s", QUESTION)
    assert 0 < len(payload.pinned) < 30 and payload.explain["budget"]["pinned"] <= 500 // 4 + 20
    assert payload.token_count <= 500


# ── Editing ───────────────────────────────────────────────────────────────────

def test_editing_a_note_changes_what_is_sent_and_what_is_found(tmp_path):
    _, mgr = session(tmp_path)
    store = mgr.get_or_open("s")
    node = store.warm.get_by_session("s")[0]
    mgr.build_context_payload("s", QUESTION)                      # the search index has loaded the old note
    assert mgr.edit_note("s", node.node_id, topic_label="billing corrections", meso_summary="Corrected: the retry limit is nine attempts, ticket QX-7731.")
    ctx = text_of(mgr, query="What is ticket QX-7731 about?")
    assert "retry limit is nine attempts" in ctx and "billing corrections" in ctx
    assert store.warm.get(node.node_id).metadata["edited"] is True
    assert not mgr.edit_note("s", "no-such-note", topic_label="x")


def test_editing_a_turn(tmp_path):
    _, mgr = session(tmp_path, close=False)
    store = mgr.get("s")
    recent = store.hot.get_all()[-1]
    assert mgr.edit_turn("s", recent.index, "A corrected last message about hiking.")
    assert "A corrected last message" in text_of(mgr) and store.hot.get_all()[-1].content.startswith("A corrected")
    old = 1                                                       # already in a note
    assert mgr.edit_turn("s", old, "The retry limit is eleven attempts.")
    assert store.raw_texts(store.warm.get_by_session("s")[0], [1 - store.warm.get_by_session("s")[0].turn_start])[1] == "The retry limit is eleven attempts."
    assert not mgr.edit_turn("s", 999, "x")
    with pytest.raises(ValueError):
        mgr.edit_turn("s", old, "   ")


# ── Forgetting ────────────────────────────────────────────────────────────────

def raw_rows(cfg, sid="s"):
    import sqlite3
    con = sqlite3.connect(cfg.db_path)
    try:
        return [r[0] for r in con.execute("SELECT idx FROM turns WHERE session_id = ? ORDER BY idx", (sid,))]
    finally:
        con.close()


def test_forgetting_a_note_removes_it_and_the_text_it_covered(tmp_path):
    cfg, mgr = session(tmp_path)
    notes = mgr.get_or_open("s").warm.get_by_session("s")
    victim = notes[0]
    report = mgr.forget("s", node_id=victim.node_id)
    assert [n["node_id"] for n in report["notes_removed"]] == [victim.node_id]
    assert PLANTED not in text_of(mgr)                            # the fact lived in that note
    rows = raw_rows(cfg)
    assert all(not victim.turn_start <= i <= victim.turn_end for i in rows) and rows
    assert len(mgr.get_or_open("s").warm.get_by_session("s")) == len(notes) - 1
    with pytest.raises(KeyError):
        mgr.forget("s", node_id="no-such-note")


def test_forgetting_a_turn_range_deletes_every_note_that_overlaps_it(tmp_path):
    cfg, mgr = session(tmp_path)
    notes = mgr.get_or_open("s").warm.get_by_session("s")
    first, second = notes[0], notes[1]
    report = mgr.forget("s", start=first.turn_end, end=second.turn_start)       # touches exactly these two notes
    assert {n["node_id"] for n in report["notes_removed"]} == {first.node_id, second.node_id}
    assert [n.node_id for n in mgr.get_or_open("s").warm.get_by_session("s")] == [notes[2].node_id]
    assert raw_rows(cfg) == list(range(notes[2].turn_start, notes[2].turn_end + 1))
    with pytest.raises(ValueError):
        mgr.forget("s")
    with pytest.raises(ValueError):
        mgr.forget("s", start=5, end=2)


def test_forgetting_recent_and_pending_turns(tmp_path):
    cfg, mgr = session(tmp_path, close=False, hot_tier_max_turns=10)       # 24 messages: 10 recent, some pending
    store = mgr.get("s")
    recent = [t.index for t in store.hot.get_all()]
    pending = [t.index for t in store.get_pending_turns()]
    assert recent and pending
    mgr.forget("s", start=min(pending), end=max(recent))
    assert store.hot.get_all() == [] and store.get_pending_turns() == []
    mgr.close("s", checkpoint=False)
    assert raw_rows(cfg) == [i for i in raw_rows(cfg) if i < min(pending)]
    again = CCEEngine(cfg).create_session_manager().get_or_open("s")
    assert again.hot.size() == 0 and again.get_pending_turns() == []
    assert again.next_index == 24                                           # numbering never goes back


def test_forget_leaves_other_sessions_alone(tmp_path):
    cfg, mgr = session(tmp_path)
    for m in make_messages():
        mgr.add_message("other", m)
    mgr.close("other")
    mgr = CCEEngine(cfg).create_session_manager()
    mgr.forget("s", start=0, end=23)
    assert PLANTED in text_of(mgr, "other") and mgr.get_or_open("s").warm.count("s") == 0


# ── Export ────────────────────────────────────────────────────────────────────

def test_export_holds_everything_stored(tmp_path):
    _, mgr = session(tmp_path)
    mgr.pin("s", "pinned")
    dump = mgr.export_session("s")
    json.dumps(dump)                                              # plain data
    assert dump["format"] == "cce-session-export/1" and dump["session_id"] == "s"
    assert len(dump["turns"]) == 24 and {t["state"] for t in dump["turns"]} == {"done"}
    assert dump["turns"][1]["content"].startswith("answer 0 about billing") and dump["turns"][0]["role"] == "user"
    assert len(dump["notes"]) == 3 and dump["notes"][0]["summary"] and dump["notes"][0]["time"]
    assert dump["pins"][0]["text"] == "pinned" and dump["session_summary"]["macro_text"]
    assert dump["stats"]["turns_ingested"] == 24
    assert "embedding" not in json.dumps(dump["notes"][0]).lower() or "vec" not in json.dumps(dump["notes"][0]).lower()


# ── Time-aware queries ────────────────────────────────────────────────────────

def two_periods(tmp_path):
    cfg = cfg_for(tmp_path, hot_tier_max_turns=4)
    mgr = CCEEngine(cfg).create_session_manager()
    msgs = make_messages()
    for m in msgs[:8]:                                            # billing
        mgr.add_message("t", m)
    time.sleep(0.05)
    boundary = datetime.now(timezone.utc)
    time.sleep(0.05)
    for m in msgs[8:]:                                            # hiking, then python
        mgr.add_message("t", m)
    mgr.close("t")
    return boundary, CCEEngine(cfg).create_session_manager()


def topics(payload) -> str:
    return " ".join(payload.past_blocks).lower()


def test_since_and_until_limit_the_notes(tmp_path):
    boundary, mgr = two_periods(tmp_path)
    everything = topics(mgr.build_context_payload("t", "billing invoices stripe webhook plans hiking trails"))
    assert "billing" in everything and "hiking" in everything
    before = topics(mgr.build_context_payload("t", "billing invoices stripe webhook plans hiking trails", until=boundary))
    after = topics(mgr.build_context_payload("t", "billing invoices stripe webhook plans hiking trails", since=boundary))
    assert "billing" in before and "hiking" not in before
    assert "hiking" in after and "billing" not in after
    nothing = mgr.build_context_payload("t", QUESTION, since=boundary + timedelta(days=1))
    assert nothing.past_blocks == [] and nothing.recent_turns == []                      # a window with no data


def test_last_turns_limits_the_notes_and_recent_turns(tmp_path):
    _, mgr = two_periods(tmp_path)
    payload = mgr.build_context_payload("t", "billing invoices stripe webhook plans python pandas", last_turns=8)
    assert "billing" not in topics(payload) and "python" in topics(payload)
    assert payload.explain["query"]["filters"] == {"last_turns": 8}
    older = mgr.build_context_payload("t", "billing invoices stripe", last_turns=1000)
    assert "billing" in topics(older)


def test_time_filters_apply_to_recent_turns_too(tmp_path):
    cfg = cfg_for(tmp_path)
    mgr = CCEEngine(cfg).create_session_manager()
    for m in make_messages()[:6]:
        mgr.add_message("t", m)
    store = mgr.get("t")
    assert len(mgr.build_context_payload("t", "billing").recent_turns) == 6
    assert len(mgr.build_context_payload("t", "billing", last_turns=2).recent_turns) == 2
    assert mgr.build_context_payload("t", "billing", since=datetime.now(timezone.utc) + timedelta(hours=1)).recent_turns == []
    assert store.next_index == 6


# ── REST ──────────────────────────────────────────────────────────────────────

def test_rest_endpoints(tmp_path):
    cfg = cfg_for(tmp_path)
    with TestClient(create_app(cfg)) as c:
        assert c.post("/v1/sessions/r/messages", json={"messages": make_messages()}).status_code == 201
        pin = c.post("/v1/sessions/r/pins", json={"text": "always remember the blue folder"}).json()["pin_id"]
        assert c.get("/v1/sessions/r/pins").json()["pins"][0]["pin_id"] == pin
        assert "blue folder" in str(c.post("/v1/sessions/r/context", json={"query": "weather"}).json()["context"])
        assert c.put(f"/v1/sessions/r/pins/{pin}", json={"text": "the red folder"}).status_code == 200
        assert c.post("/v1/sessions/r/pins", json={}).status_code == 422
        assert c.post("/v1/sessions/r/pins", json={"turn": 9999}).status_code == 404
        assert c.delete(f"/v1/sessions/r/pins/{pin}").status_code == 200 and c.delete(f"/v1/sessions/r/pins/{pin}").status_code == 404

        assert c.patch("/v1/sessions/r/turns/0", json={"content": "edited first message"}).status_code in (200, 404)
        exported = c.get("/v1/sessions/r/export").json()
        assert exported["format"].startswith("cce-session-export") and len(exported["turns"]) == 24

        since = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        empty = c.post("/v1/sessions/r/context", json={"query": QUESTION, "since": since}).json()
        assert empty["retrieved_node_ids"] == []
        assert c.post("/v1/sessions/r/context", json={"query": QUESTION, "last_turns": 0}).status_code == 422

        c.post("/v1/sessions/r/close")
        note = c.get("/v1/sessions/r/export").json()["notes"][0]
        assert c.patch(f"/v1/sessions/r/notes/{note['node_id']}", json={"meso_summary": "rewritten"}).status_code == 200
        assert c.patch(f"/v1/sessions/r/notes/{note['node_id']}", json={}).status_code == 422
        assert c.patch("/v1/sessions/r/notes/nope", json={"meso_summary": "x"}).status_code == 404

        assert c.post("/v1/sessions/r/forget", json={}).status_code == 422
        assert c.post("/v1/sessions/r/forget", json={"node_id": "nope"}).status_code == 404
        gone = c.post("/v1/sessions/r/forget", json={"node_id": note["node_id"]}).json()
        assert gone["notes_removed"][0]["node_id"] == note["node_id"]
        assert c.get("/v1/sessions/ghost/export").status_code == 404


def test_rest_cors_allows_the_new_methods(tmp_path):
    with TestClient(create_app(cfg_for(tmp_path))) as c:
        r = c.options("/v1/sessions/x/pins/1", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "PUT"})
        assert r.status_code == 200 and "PUT" in r.headers["access-control-allow-methods"]


# ── MCP tools and the cce command ─────────────────────────────────────────────

def test_mcp_tools(tmp_path):
    import asyncio
    from types import SimpleNamespace

    from cce_mcp import schema, server

    cfg = cfg_for(tmp_path)
    engine = CCEEngine(cfg)
    mgr = engine.create_session_manager()
    for m in make_messages():
        mgr.add_message("m", m)
    mgr.close("m")
    ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context={"engine": engine, "manager": mgr}))

    async def progress(*args, **kwargs):
        return None

    ctx.report_progress = progress

    def call(fn, model, **kw):
        return json.loads(asyncio.run(fn(model(**kw), ctx)))

    added = call(server.cce_pin_memory, schema.PinMemoryInput, session_id="m", action="add", text="pinned via mcp")
    assert added["status"] == "ok"
    listed = call(server.cce_pin_memory, schema.PinMemoryInput, session_id="m", action="list")
    assert listed["pins"][0]["text"] == "pinned via mcp"
    assert call(server.cce_pin_memory, schema.PinMemoryInput, session_id="m", action="edit", pin_id=added["pin_id"], text="changed")["updated"]
    assert call(server.cce_pin_memory, schema.PinMemoryInput, session_id="m", action="remove")["status"] == "error"      # needs pin_id
    assert call(server.cce_pin_memory, schema.PinMemoryInput, session_id="m", action="remove", pin_id=added["pin_id"])["removed"]

    refused = call(server.cce_forget_memory, schema.ForgetMemoryInput, session_id="m", start=0, end=3)
    assert refused["status"] == "error" and "confirm" in refused["message"]
    done = call(server.cce_forget_memory, schema.ForgetMemoryInput, session_id="m", start=0, end=3, confirm=True)
    assert done["status"] == "ok" and done["notes_removed"]

    dump = call(server.cce_export_session, schema.ExportSessionInput, session_id="m")
    assert dump["status"] == "ok" and dump["format"].startswith("cce-session-export")
    assert call(server.cce_export_session, schema.ExportSessionInput, session_id="nope")["status"] == "error"

    out = call(server.cce_retrieve_context, schema.RetrieveContextInput, session_id="m", query=QUESTION, last_turns=5)
    assert out["status"] == "ok"


def test_cce_command_manages_a_data_folder(tmp_path, capsys):
    from cce_core.cli import main as cce

    cfg, mgr = session(tmp_path)
    mgr.close_all()
    base = ["--data-dir", str(cfg.base_dir), "--embedder", "hash"]

    def run(*args):
        try:
            cce([*args, *base])
        except SystemExit as e:
            return e.code, capsys.readouterr()
        raise AssertionError("cce always exits")

    code, out = run("pin", "s", "--text", "remember the blue folder")
    pin_id = out.out.strip()
    assert code == 0 and len(pin_id) == 12
    code, out = run("pins", "s")
    assert code == 0 and "blue folder" in out.out
    code, out = run("export", "s", "--out", str(tmp_path / "dump.json"))
    assert code == 0 and json.loads((tmp_path / "dump.json").read_text(encoding="utf-8"))["pins"][0]["pin_id"] == pin_id
    code, out = run("unpin", "s", pin_id)
    assert code == 0 and "removed" in out.out
    code, out = run("unpin", "s", pin_id)
    assert code == 1

    code, out = run("forget", "s", "--turns", "0-3")
    assert code == 1 and "--yes" in out.err                      # nothing is forgotten without confirmation
    assert CCEEngine(cfg).create_session_manager().get_or_open("s").warm.count("s") == 3
    code, out = run("forget", "s", "--turns", "0-3", "--yes")
    assert code == 0 and "Forgot" in out.out
    code, out = run("export", "ghost")
    assert code == 1
