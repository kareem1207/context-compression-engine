"""warm_tier_max_chunks and cold_tier_max_summaries are enforced (0 or None = unlimited)."""

import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from cce_core.compression.merger import MemoryNode
from cce_core.config import CCEConfig
from cce_core.engine import CCEEngine
from cce_core.memory.cold_tier import ColdTier, MacroSummary
from cce_core.memory.warm_tier import WarmTier

from tests.test_live_sessions import PLANTED, make_config


def long_chat(topics: int = 12, rounds: int = 3) -> list[dict]:
    """Many distinct topics; the invoice fact sits in the very first message."""
    msgs = []
    for t in range(topics):
        words = f"subject{t}alpha subject{t}beta subject{t}gamma subject{t}delta"
        for r in range(rounds):
            extra = f" {PLANTED}" if (t, r) == (0, 0) else ""
            msgs += [
                {"role": "user", "content": f"question {r} about {words} and more {words}.{extra}"},
                {"role": "assistant", "content": f"answer {r} about {words} with details {words}"},
            ]
    return msgs


def covered(store) -> set[int]:
    idx = {t.index for t in store.get_pending_turns()} | {t.index for t in store.hot.get_all()}
    for n in store.warm.get_by_session(store.session_id):
        idx.update(range(n.turn_start, n.turn_end + 1))
    return idx


# ── Warm tier ─────────────────────────────────────────────────────────────────

def test_warm_tier_limit_merges_old_notes_and_keeps_everything_findable(tmp_path):
    cfg = make_config(tmp_path, warm_tier_max_chunks=4, live_batch_turns=4)
    mgr = CCEEngine(cfg).create_session_manager()
    msgs = long_chat()
    for m in msgs:
        mgr.add_message("s", m)
    store = mgr.get("s")

    nodes = store.warm.get_by_session("s")
    assert 0 < len(nodes) <= 4
    assert covered(store) == set(range(len(msgs))), "merging must not lose turn ranges"

    newest = max(nodes, key=lambda n: n.turn_end)
    assert newest.turn_end >= len(msgs) - cfg.hot_tier_max_turns - 8   # recent data was not merged away
    assert any(n.metadata.get("merged_notes", 1) > 1 for n in nodes)   # old ones became coarser notes

    # the fact from the oldest, merged-away part is still found
    assert PLANTED in mgr.build_context("s", "How many invoice retry attempts are allowed?", fmt="string")
    mgr.close("s")


def test_without_a_limit_nothing_is_merged(tmp_path):
    for limit in (0, None):
        cfg = make_config(tmp_path / f"l{limit}", warm_tier_max_chunks=limit, live_batch_turns=4)
        mgr = CCEEngine(cfg).create_session_manager()
        for m in long_chat():
            mgr.add_message("s", m)
        nodes = mgr.get("s").warm.get_by_session("s")
        assert len(nodes) > 4
        assert not any(n.metadata.get("merged_notes", 1) > 1 for n in nodes)
        mgr.close("s", checkpoint=False)


def test_limit_is_per_session(tmp_path):
    cfg = make_config(tmp_path, warm_tier_max_chunks=3, live_batch_turns=4)
    mgr = CCEEngine(cfg).create_session_manager()
    for sid in ("a", "b"):
        for m in long_chat(topics=8):
            mgr.add_message(sid, m)
    assert mgr.get("a").warm.count("a") <= 3
    assert mgr.get("b").warm.count("b") <= 3
    mgr.close_all(checkpoint=False)


def _node(i: int, session: str = "s") -> MemoryNode:
    emb = np.zeros(8, dtype=np.float32)
    emb[i % 8] = 1.0
    return MemoryNode(
        node_id=f"n{i}", chunk_id=f"c{i}", session_id=session, embedding=emb, topic_label=f"topic {i}",
        meso_summary=f"Summary of topic {i}.", micro_summaries=[f"micro {i}"], turn_start=i * 2, turn_end=i * 2 + 1,
        token_count=10, created_at=datetime.now(timezone.utc),
    )


def test_least_retrieved_notes_are_merged_first(tmp_path):
    cfg = CCEConfig(embedder="hash", warm_tier_max_chunks=3, base_dir=tmp_path)
    warm = WarmTier(cfg)
    warm.upsert_many([_node(i) for i in range(3)])
    warm.record_hits(["n0"] * 5)         # note 0 is useful, so it should stay a note of its own
    warm.upsert(_node(3))                # 4 notes, limit 3: one pair is fused
    nodes = warm.get_by_session("s")
    assert len(nodes) == 3
    assert any(n.node_id == "n0" for n in nodes), "the often-retrieved note was merged"
    fused = [n for n in nodes if n.metadata.get("merged_notes", 1) > 1]
    assert len(fused) == 1 and (fused[0].turn_start, fused[0].turn_end) == (2, 5)   # notes 1 and 2
    assert any(n.node_id == "n3" for n in nodes), "the newest note was merged"
    warm.close()


# ── Cold tier ─────────────────────────────────────────────────────────────────

def _summary(i: int) -> MacroSummary:
    return MacroSummary(
        session_id=f"sess{i}", macro_text=f"summary {i}", node_count=1, turn_count=2, token_count_original=10,
        token_count_compressed=5, topics=["t"], created_at=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=i),
    )


def test_cold_tier_drops_the_oldest_summaries_beyond_the_limit(tmp_path):
    cfg = CCEConfig(cold_tier_max_summaries=3, base_dir=tmp_path)
    cold = ColdTier(cfg)
    for i in range(6):
        cold.write(_summary(i))
    assert cold.count() == 3
    assert sorted(cold.list_sessions()) == ["sess3", "sess4", "sess5"]
    assert sorted(p.stem for p in cfg.cold_dir.glob("sess*.json")) == ["sess3", "sess4", "sess5"]
    assert cold.get("sess5").macro_text == "summary 5" and cold.get("sess0") is None


def test_cold_tier_without_a_limit_keeps_everything(tmp_path):
    for limit in (0, None):
        cold = ColdTier(CCEConfig(cold_tier_max_summaries=limit, base_dir=tmp_path / f"c{limit}"))
        for i in range(6):
            cold.write(_summary(i))
        assert cold.count() == 6
