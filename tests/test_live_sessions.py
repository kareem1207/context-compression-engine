"""
Live sessions (MCP / REST): evicted turns are compressed in topic-sized batches, and a turn that
has left the hot tier but is not in a note yet is still served to retrieval.
"""

import pytest

from cce_core.config import CCEConfig
from cce_core.engine import CCEEngine

TOPICS = [
    "billing invoices stripe webhook plans",
    "hiking trails mountains boots weather",
    "python pandas dataframe merge groupby",
]
PLANTED = "The invoice retry limit is seven attempts."


def make_messages(rounds_per_topic: int = 4) -> list[dict]:
    """Three topics, one after another, with one fact planted in the very first user message."""
    msgs = []
    for r in range(rounds_per_topic * len(TOPICS)):
        t = TOPICS[r // rounds_per_topic]
        extra = f" {PLANTED}" if r == 0 else ""
        msgs += [
            {"role": "user", "content": f"question {r} about {t} and more {t}.{extra}"},
            {"role": "assistant", "content": f"answer {r} about {t} with details {t} {t}"},
        ]
    return msgs


def make_config(tmp_path, **kw) -> CCEConfig:
    base = dict(embedder="hash", hot_tier_max_turns=6, retrieval_top_k=3, base_dir=tmp_path / "data")
    base.update(kw)
    return CCEConfig(**base)


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path)


def covered_and_pending(store) -> set[int]:
    """Every turn index that is in a note or waiting verbatim in the pending buffer."""
    idx = {t.index for t in store.get_pending_turns()}
    for n in store.warm.get_by_session(store.session_id):
        idx.update(range(n.turn_start, n.turn_end + 1))
    return idx


def test_live_path_groups_turns_by_topic(cfg):
    mgr = CCEEngine(cfg).create_session_manager()
    msgs = make_messages()
    for m in msgs:
        mgr.add_message("s1", m)
    store = mgr.get("s1")

    evicted = len(msgs) - store.hot.size()
    nodes = store.warm.get_by_session("s1")
    assert nodes, "a batch should have been compressed by now"
    assert len(nodes) < evicted - len(store.get_pending_turns())   # fewer notes than turns they cover
    assert any(n.turn_end > n.turn_start for n in nodes)           # at least one multi-turn note
    mgr.close("s1")


def test_no_gap_between_notes_pending_and_hot(cfg):
    """After every single message, each turn is in a note, in the pending buffer, or in the hot tier."""
    mgr = CCEEngine(cfg).create_session_manager()
    for i, m in enumerate(make_messages(), start=1):
        mgr.add_message("s1", m)
        store = mgr.get("s1")
        hot = {t.index for t in store.hot.get_all()}
        assert covered_and_pending(store) | hot == set(range(i)), f"gap after message {i}"
    mgr.close("s1")


def test_pending_turn_is_retrievable_before_it_is_compressed(cfg):
    """12 messages with a hot tier of 10 leave turn 0 and 1 pending: no note yet, but no gap."""
    cfg = make_config(cfg.base_dir.parent, hot_tier_max_turns=10)
    mgr = CCEEngine(cfg).create_session_manager()
    for m in make_messages(rounds_per_topic=2):
        mgr.add_message("s1", m)
    store = mgr.get("s1")
    assert store.warm.count("s1") == 0
    assert [t.index for t in store.get_pending_turns()] == [0, 1]

    text = mgr.build_context("s1", "How many invoice retry attempts are allowed?", fmt="string")
    assert PLANTED in text
    mgr.close("s1")


def test_flush_on_close_leaves_nothing_uncompressed(cfg):
    engine = CCEEngine(cfg)
    mgr = engine.create_session_manager()
    msgs = make_messages()
    for m in msgs:
        mgr.add_message("s1", m)
    mgr.close("s1")

    store = engine.open_session("s1")
    assert store.get_pending_turns() == [] and store.hot.size() == 0
    assert covered_and_pending(store) == set(range(len(msgs)))
    store.close()


def test_checkpoint_flushes_pending(cfg):
    mgr = CCEEngine(cfg).create_session_manager()
    for m in make_messages():
        mgr.add_message("s1", m)
    assert mgr.get("s1").get_pending_turns() or mgr.get("s1").hot.size()   # something was still uncompressed
    assert mgr.checkpoint("s1")
    assert mgr.get("s1").get_pending_turns() == []
    mgr.close("s1")


def test_pending_buffer_is_flushed_when_it_gets_too_big(tmp_path):
    """One topic that never changes: the open topic cannot stay pending for ever."""
    cfg = make_config(tmp_path, hot_tier_max_turns=4, live_pending_max_turns=10)
    mgr = CCEEngine(cfg).create_session_manager()
    for r in range(40):
        mgr.add_message("s1", {"role": "user", "content": f"invoice {r} stripe webhook plans billing invoice"})
    store = mgr.get("s1")
    assert len(store.get_pending_turns()) < 10
    assert store.warm.count("s1") >= 1
    mgr.close("s1")


def test_batch_size_is_configurable(tmp_path):
    msgs = make_messages()

    def node_count(batch: int) -> int:
        cfg = make_config(tmp_path / f"b{batch}", live_batch_turns=batch)
        mgr = CCEEngine(cfg).create_session_manager()
        for m in msgs:
            mgr.add_message("s", m)
        n = mgr.get("s").warm.count("s")
        mgr.close("s", checkpoint=False)
        return n

    assert node_count(1) > node_count(8)   # 1 = compress every eviction on its own (the old behaviour)


def test_live_and_one_shot_both_retrieve_the_planted_fact(tmp_path):
    msgs = make_messages()
    query = "How many invoice retry attempts are allowed?"

    live_cfg = make_config(tmp_path / "live")
    live = CCEEngine(live_cfg).create_session_manager()
    for m in msgs:
        live.add_message("s", m)
    live_ctx = live.build_context("s", query, fmt="string")

    shot_cfg = make_config(tmp_path / "shot")
    engine = CCEEngine(shot_cfg)
    store, _, _ = engine.full_pipeline(msgs, "s")
    shot_ctx = engine.build_context(store, query).export("string")

    assert PLANTED in live_ctx
    assert PLANTED in shot_ctx
    live.close("s")
    store.close()


def test_question_about_the_oldest_topic_after_many_turns(cfg):
    mgr = CCEEngine(cfg).create_session_manager()
    for m in make_messages():
        mgr.add_message("s1", m)
    text = mgr.build_context("s1", "What was the invoice retry limit?", fmt="string")
    assert "seven attempts" in text
    mgr.close("s1")
