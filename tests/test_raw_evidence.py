"""
Details buried in longer messages survive compression: the original text of compressed turns is kept in
SQLite and a retrieved topic quotes the sentence that answers the question.
"""

import shutil
import sqlite3
from pathlib import Path

import pytest

from cce_core.config import CCEConfig
from cce_core.engine import CCEEngine
from cce_core.ingestion import tokenizer
from cce_core.memory.warm_tier import SCHEMA_VERSION

from tests.test_live_sessions import make_config

FIXTURE_10 = Path(__file__).parent / "fixtures" / "cce_1_0_data"

FACT = "Her sourdough starter is named Gertrude and is fed twice a day."
TEXT = "sourdough bread starter, hydration and baking temperature"
QUERY = "what is the sourdough starter called?"
# Every answer carries numbers: the extractive summaries prefer those sentences and drop the plain fact.
DISTRACTOR = " Use {a} percent hydration and bake at {b} degrees for {c} minutes."


def embedders():
    params = ["hash"]
    try:
        import sentence_transformers  # noqa: F401
        from cce_core.embedding import load_embedder
        load_embedder(CCEConfig(embedder="neural"))
        params.append("neural")
    except Exception:
        params.append(pytest.param("neural", marks=pytest.mark.skip(reason="MiniLM model not available")))
    return params


def conversation(position: str, distractors: bool = True) -> list[dict]:
    msgs = []
    for r in range(8):
        msgs.append({"role": "user", "content": f"Question {r} about {TEXT}. Tell me more about {TEXT}."})
        body = f"Answer {r}: here is more about {TEXT}, with details and examples."
        if distractors:
            body += DISTRACTOR.format(a=70 + r, b=200 + r, c=30 + r)
        if r == 3:
            body = {
                "first": f"{FACT} {body}",
                "middle": f"{body} {FACT} It rises well.",
                "last": f"{body} {FACT}",
            }[position]
        msgs.append({"role": "assistant", "content": body})
    return msgs + [{"role": "user", "content": "filler a"}, {"role": "assistant", "content": "filler b"}] * 6


def cfg_for(tmp_path, embedder="hash", **kw) -> CCEConfig:
    opts = dict(embedder=embedder, hot_tier_max_turns=10, retrieval_top_k=3)
    opts.update(kw)
    return make_config(tmp_path, **opts)


def done_rows(db: Path, session: str | None = None) -> int:
    con = sqlite3.connect(db)
    try:
        q, args = "SELECT COUNT(*) FROM turns WHERE state = 'done'", ()
        if session:
            q, args = q + " AND session_id = ?", (session,)
        return con.execute(q, args).fetchone()[0]
    finally:
        con.close()


# ── One-shot mode ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("embedder", embedders())
@pytest.mark.parametrize("position", ["first", "middle", "last"])
@pytest.mark.parametrize("distractors", [True, False])
def test_oneshot_finds_a_fact_buried_in_a_long_message(tmp_path, embedder, position, distractors):
    engine = CCEEngine(cfg_for(tmp_path, embedder))
    res = engine.create_stateless_processor().process(conversation(position, distractors), query=QUERY)
    assert "gertrude" in res.payload.to_string().lower()


def test_oneshot_with_the_raw_text_switched_off_loses_it(tmp_path):
    """The opt-out really does go back to quoting the one-line summaries."""
    engine = CCEEngine(cfg_for(tmp_path, keep_raw_turns=False))
    res = engine.create_stateless_processor().process(conversation("middle"), query=QUERY)
    assert "gertrude" not in res.payload.to_string().lower()


def test_two_facts_in_one_message_are_both_quoted(tmp_path):
    msgs = conversation("middle")
    msgs[7]["content"] += " The oven is preheated to 230 degrees with a pan of steam."
    engine = CCEEngine(cfg_for(tmp_path))
    res = engine.create_stateless_processor().process(
        msgs, query="what is the sourdough starter called and what is the oven preheated to with steam?")
    text = res.payload.to_string().lower()
    assert "gertrude" in text and "230 degrees" in text


def test_a_short_sentence_brings_its_neighbour(tmp_path):
    msgs = conversation("middle", distractors=True)
    msgs[7]["content"] = msgs[7]["content"].replace(FACT, "The starter is named Gertrude. She is fed twice a day at noon.")
    engine = CCEEngine(cfg_for(tmp_path))
    text = engine.create_stateless_processor().process(msgs, query="starter named Gertrude").payload.to_string()
    assert "named Gertrude" in text and "fed twice a day" in text


# ── Live sessions ─────────────────────────────────────────────────────────────

def stream(cfg, msgs, sid="s"):
    mgr = CCEEngine(cfg).create_session_manager()
    for m in msgs:
        mgr.add_message(sid, m)
    return mgr


@pytest.mark.parametrize("embedder", embedders())
def test_live_fact_survives_close_and_reopen_with_a_new_engine(tmp_path, embedder):
    cfg = cfg_for(tmp_path, embedder)
    mgr = stream(cfg, conversation("middle"))
    mgr.close("s")

    fresh = CCEEngine(cfg).create_session_manager()
    payload = fresh.build_context_payload("s", QUERY)
    assert "gertrude" in payload.to_string().lower()
    fresh.close_all(checkpoint=False)


def test_live_fact_survives_a_crash(tmp_path):
    cfg = cfg_for(tmp_path, live_batch_turns=4, live_pending_max_turns=10)
    mgr = stream(cfg, conversation("middle"))
    store = mgr.get("s")
    # the message with the fact (turn 7) has been compressed, so only the kept raw text can bring it back
    assert 7 not in {t.index for t in store.get_pending_turns()} | {t.index for t in store.hot.get_all()}
    assert done_rows(cfg.db_path, "s") > 0
    # no close(): the process "dies"

    fresh = CCEEngine(cfg).create_session_manager()
    assert "gertrude" in fresh.build_context_payload("s", QUERY).to_string().lower()
    fresh.close_all(checkpoint=False)


def test_live_with_the_opt_out_stores_no_raw_text(tmp_path):
    cfg = cfg_for(tmp_path, keep_raw_turns=False)
    mgr = stream(cfg, conversation("middle"))
    mgr.close("s")
    assert done_rows(cfg.db_path) == 0


def test_env_switch(monkeypatch):
    monkeypatch.setenv("CCE_KEEP_RAW_TURNS", "0")
    assert CCEConfig().keep_raw_turns is False
    monkeypatch.setenv("CCE_KEEP_RAW_TURNS", "1")
    assert CCEConfig().keep_raw_turns is True
    monkeypatch.delenv("CCE_KEEP_RAW_TURNS")
    assert CCEConfig().keep_raw_turns is True


# ── Cleanup: no orphans ───────────────────────────────────────────────────────

def test_delete_session_removes_the_raw_text(tmp_path):
    cfg = cfg_for(tmp_path)
    mgr = stream(cfg, conversation("middle"))
    mgr.close("s")
    other = CCEEngine(cfg).create_session_manager()
    for m in conversation("first")[:12]:
        other.add_message("keep", m)
    other.close("keep")
    assert done_rows(cfg.db_path, "s") > 0 and done_rows(cfg.db_path, "keep") > 0

    CCEEngine(cfg).create_session_manager().delete("s")
    assert done_rows(cfg.db_path, "s") == 0
    assert done_rows(cfg.db_path, "keep") > 0        # other sessions are untouched


def test_deleting_a_note_removes_the_text_it_covered(tmp_path):
    cfg = cfg_for(tmp_path)
    mgr = stream(cfg, conversation("middle"))
    mgr.close("s")
    store = CCEEngine(cfg).open_session("s")
    first, rest = store.warm.get_by_session("s")[0], store.warm.get_by_session("s")[1:]
    before = done_rows(cfg.db_path, "s")
    assert store.warm.delete(first.node_id)
    assert done_rows(cfg.db_path, "s") == before - (first.turn_end - first.turn_start + 1)
    assert all(store.raw_texts(n, [0]) for n in rest)   # the other notes keep theirs
    store.close()


def test_warm_limit_merging_leaves_no_orphan_raw_rows(tmp_path):
    cfg = cfg_for(tmp_path, warm_tier_max_chunks=3, live_batch_turns=4)
    msgs = []
    for t in range(10):
        words = f"subject{t}alpha subject{t}beta subject{t}gamma"
        for r in range(3):
            msgs += [{"role": "user", "content": f"question {r} about {words} and more {words}."},
                     {"role": "assistant", "content": f"answer {r} about {words}. Detail {t}.{r} is {t * 7 + r} units."}]
    mgr = stream(cfg, msgs)
    mgr.close("s")

    con = sqlite3.connect(cfg.db_path)
    nodes = con.execute("SELECT turn_start, turn_end FROM memory_nodes WHERE session_id = 's'").fetchall()
    rows = [r[0] for r in con.execute("SELECT idx FROM turns WHERE session_id = 's' AND state = 'done'")]
    con.close()
    assert 0 < len(nodes) <= 3
    covered = {i for a, b in nodes for i in range(a, b + 1)}
    assert set(rows) == covered == set(range(len(msgs)))   # every kept row has a note, every note its rows


# ── Old databases ─────────────────────────────────────────────────────────────

def test_1_0_database_migrates_and_still_answers(tmp_path):
    dst = tmp_path / "v10"
    shutil.copytree(FIXTURE_10, dst)
    db = dst / "warm.db"
    con = sqlite3.connect(db)
    assert con.execute("PRAGMA user_version").fetchone()[0] == 4      # what CCE 1.0 wrote
    con.close()

    cfg = CCEConfig(embedder="hash", hot_tier_max_turns=6, base_dir=dst)
    mgr = CCEEngine(cfg).create_session_manager()
    store = mgr.open("v10")
    con = sqlite3.connect(db)
    assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    con.close()

    # notes without raw text keep the old behaviour (quote the one-line summary of the message)
    assert store.stats()["raw_turns_stored"] == 0
    assert "seven attempts" in mgr.build_context("v10", "invoice retry limit attempts", fmt="string")

    # and new turns of the same session keep raw text
    for i in range(14):
        mgr.add_message("v10", {"role": "user", "content": f"hiking boots trails mountains weather {i}"})
    mgr.close("v10")
    assert done_rows(db, "v10") > 0


# ── Token cap ─────────────────────────────────────────────────────────────────

def test_token_cap_holds_with_long_raw_turns(tmp_path):
    cap = 700
    cfg = cfg_for(tmp_path, context_max_tokens=cap)
    long_answer = " ".join(f"Sentence {i} explains sourdough starter feeding schedule number {i} in detail." for i in range(120))
    msgs = []
    for r in range(8):
        msgs += [{"role": "user", "content": f"Question {r} about {TEXT}."},
                 {"role": "assistant", "content": long_answer if r != 3 else long_answer + " " + FACT}]
    msgs += [{"role": "user", "content": "filler a"}, {"role": "assistant", "content": "filler b"}] * 6

    res = CCEEngine(cfg).create_stateless_processor().process(msgs, query=QUERY)
    assert res.payload.token_count <= cap
    assert tokenizer.count_messages(res.payload.to_messages()) <= cap
    assert tokenizer.count(res.payload.to_string()) <= cap

    mgr = stream(cfg, msgs)
    payload = mgr.build_context_payload("s", QUERY)
    assert tokenizer.count_messages(payload.to_messages()) <= cap
    mgr.close_all(checkpoint=False)


# ── Session stats ─────────────────────────────────────────────────────────────

def test_short_session_does_not_report_a_ratio_below_one(tmp_path):
    mgr = stream(cfg_for(tmp_path), conversation("middle")[:10])
    stats = mgr.get("s").stats()
    assert stats["hot_turns"] == 10 and stats["warm_nodes"] == 0
    assert stats["tokens_sent"] == stats["raw_tokens"]       # everything is still verbatim
    assert stats["compression_ratio"] is None and stats["tokens_saved"] == 0
    mgr.close_all(checkpoint=False)


def test_stats_use_the_real_size_of_what_is_sent(tmp_path):
    cfg = cfg_for(tmp_path, hot_tier_max_turns=6)
    mgr = stream(cfg, conversation("middle") * 2)
    store = mgr.get("s")
    stats = store.stats()

    notes = store.warm.get_by_session("s")
    expected = (
        sum(tokenizer.count(f"[Topic: {n.topic_label}] (turns {n.turn_start}-{n.turn_end})\n{n.meso_summary}") for n in notes)
        + sum(t.token_count for t in store.hot.get_all())
        + sum(t.token_count for t in store.get_pending_turns())
    )
    assert stats["tokens_sent"] == expected
    assert stats["warm_tokens_compressed"] == expected - sum(t.token_count for t in store.hot.get_all()) \
        - sum(t.token_count for t in store.get_pending_turns())
    assert stats["warm_tokens_original"] == sum(n.token_count for n in notes)
    assert stats["tokens_saved"] == stats["raw_tokens"] - expected > 0
    assert stats["compression_ratio"] == round(stats["raw_tokens"] / expected, 2) > 1
    mgr.close_all(checkpoint=False)


def test_rest_stats_expose_the_new_fields(tmp_path):
    from fastapi.testclient import TestClient
    from cce_rest.app import create_app

    with TestClient(create_app(cfg_for(tmp_path))) as c:
        c.post("/v1/sessions/r/messages", json={"messages": conversation("middle")[:4]})
        s = c.get("/v1/sessions/r/stats").json()
        assert s["compression_ratio"] is None and s["tokens_saved"] == 0 and "tokens_sent" in s
