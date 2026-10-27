"""
Live-session state survives a restart or a crash, turn indexes keep growing, and a database made
by CCE 0.1.x is upgraded in place.
"""

import shutil
import sqlite3
from pathlib import Path

import pytest

from cce_core.config import CCEConfig
from cce_core.engine import CCEEngine
from cce_core.memory.warm_tier import SCHEMA_VERSION

from tests.test_live_sessions import PLANTED, make_config, make_messages

FIXTURE = Path(__file__).parent / "fixtures" / "cce_0_1_data"


def fresh(cfg: CCEConfig):
    """A new engine and manager on the same data dir, as after a process restart."""
    engine = CCEEngine(cfg)
    return engine, engine.create_session_manager()


def test_restart_restores_hot_tier_pending_and_counters(tmp_path):
    cfg = make_config(tmp_path, hot_tier_max_turns=10)
    _, mgr = fresh(cfg)
    msgs = make_messages(rounds_per_topic=2)   # 12 messages: 10 hot, 2 pending
    for m in msgs:
        mgr.add_message("s1", m)
    store = mgr.get("s1")
    hot_before = [(t.index, t.content) for t in store.hot.get_all()]
    stats_before = store.stats()
    mgr.close("s1", checkpoint=False)   # leave the buffers as they are

    _, mgr2 = fresh(cfg)
    store2 = mgr2.open("s1")
    assert [(t.index, t.content) for t in store2.hot.get_all()] == hot_before
    assert [t.index for t in store2.get_pending_turns()] == [0, 1]
    stats = store2.stats()
    assert stats["turns_ingested"] == stats_before["turns_ingested"] == 12
    assert stats["raw_tokens"] == stats_before["raw_tokens"]
    assert store2.next_index == 12

    turn = mgr2.add_message("s1", {"role": "user", "content": "after the restart"})
    assert turn.index == 12
    mgr2.close("s1")


def test_crash_without_close_loses_nothing(tmp_path):
    cfg = make_config(tmp_path)
    _, mgr = fresh(cfg)
    msgs = make_messages()
    for m in msgs:
        mgr.add_message("s1", m)
    # no close(): the process "dies" here, with the first manager's connection still open

    _, mgr2 = fresh(cfg)
    store = mgr2.open("s1")
    covered = {t.index for t in store.hot.get_all()} | {t.index for t in store.get_pending_turns()}
    for n in store.warm.get_by_session("s1"):
        covered.update(range(n.turn_start, n.turn_end + 1))
    assert covered == set(range(len(msgs)))
    assert store.next_index == len(msgs)
    assert PLANTED in mgr2.build_context("s1", "How many invoice retry attempts?", fmt="string")


def test_turn_indexes_grow_across_restarts_and_notes_stay_ordered(tmp_path):
    cfg = make_config(tmp_path)
    msgs = make_messages()
    first, second = msgs[:12], msgs[12:]
    _, mgr = fresh(cfg)
    for m in first:
        mgr.add_message("s1", m)
    mgr.close("s1")

    _, mgr2 = fresh(cfg)
    indexes = [mgr2.add_message("s1", m).index for m in second]
    assert indexes == list(range(12, 12 + len(second)))
    mgr2.close("s1")

    engine, _ = fresh(cfg)
    store = engine.open_session("s1")
    nodes = store.warm.get_by_session("s1")
    assert [n.turn_start for n in nodes] == sorted(n.turn_start for n in nodes)
    ranges = sorted((n.turn_start, n.turn_end) for n in nodes)
    assert all(a_end < b_start for (_, a_end), (b_start, _) in zip(ranges, ranges[1:])), "notes overlap"
    store.close()


def test_add_messages_in_one_call_continue_the_numbering(tmp_path):
    cfg = make_config(tmp_path)
    _, mgr = fresh(cfg)
    mgr.add_messages("s1", make_messages()[:5])
    mgr.close("s1", checkpoint=False)
    _, mgr2 = fresh(cfg)
    turns = mgr2.add_messages("s1", make_messages()[5:9])
    assert [t.index for t in turns] == [5, 6, 7, 8]
    mgr2.close("s1")


def test_deleting_a_session_removes_its_saved_turns(tmp_path):
    cfg = make_config(tmp_path)
    _, mgr = fresh(cfg)
    mgr.add_messages("s1", make_messages()[:8])
    mgr.delete("s1")
    _, mgr2 = fresh(cfg)
    assert mgr2.open("s1").next_index == 0
    mgr2.close("s1", checkpoint=False)


# ── Upgrade of a database made by CCE 0.1.x ───────────────────────────────────

@pytest.fixture
def legacy_dir(tmp_path):
    dst = tmp_path / "legacy"
    shutil.copytree(FIXTURE, dst)
    return dst


def user_version(db: Path) -> int:
    con = sqlite3.connect(db)
    try:
        return con.execute("PRAGMA user_version").fetchone()[0]
    finally:
        con.close()


def test_legacy_database_is_migrated_in_place(legacy_dir):
    db = legacy_dir / "warm.db"
    assert user_version(db) == 0                       # what 0.1.x wrote

    cfg = CCEConfig(embedder="hash", hot_tier_max_turns=6, base_dir=legacy_dir)
    engine, mgr = fresh(cfg)
    store = mgr.open("legacy")
    assert user_version(db) == SCHEMA_VERSION

    # The old notes are still there and still answer questions.
    assert store.warm.count("legacy") == 2
    assert "seven attempts" in mgr.build_context("legacy", "invoice retry limit attempts", fmt="string")

    # Numbering continues after the last note (it ended at turn 15), counters were derived from the notes.
    assert store.next_index == 16
    assert store.stats()["turns_ingested"] == 16
    assert mgr.add_message("legacy", {"role": "user", "content": "back again"}).index == 16
    mgr.close("legacy")

    # Opening again does not migrate twice or lose anything.
    _, mgr2 = fresh(cfg)
    assert mgr2.open("legacy").warm.count("legacy") >= 2
    mgr2.close("legacy", checkpoint=False)


def test_database_from_a_newer_cce_is_refused_with_a_clear_error(tmp_path):
    cfg = make_config(tmp_path)
    cfg.ensure_dirs()
    con = sqlite3.connect(cfg.db_path)
    con.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")
    con.close()
    with pytest.raises(RuntimeError, match="newer than this CCE"):
        CCEEngine(cfg).open_session("s1")
