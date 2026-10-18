"""The warm-tier search index is shared per data dir, kept current from SQLite, and does no per-row SQL."""

import sqlite3
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from cce_core.compression.merger import MemoryNode, encode_turn_vecs
from cce_core.config import CCEConfig
from cce_core.engine import CCEEngine
from cce_core.memory.warm_tier import WarmTier
from cce_core.retrieval.retriever import node_similarity, is_overview_query

from tests.test_live_sessions import make_config, make_messages

DIM = 32
ROOT = Path(__file__).resolve().parent.parent


def unit(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i % DIM] = 1.0
    return v


def node(i: int, session: str = "s", turns: int = 2) -> MemoryNode:
    return MemoryNode(
        node_id=f"{session}-n{i}", chunk_id=f"c{i}", session_id=session, embedding=unit(i), topic_label=f"topic {i}",
        meso_summary=f"Summary of topic {i}.", micro_summaries=[f"micro {i}.{t}" for t in range(turns)],
        turn_start=i * turns, turn_end=i * turns + turns - 1, token_count=10, created_at=datetime.now(timezone.utc),
        metadata={"turn_vecs": encode_turn_vecs(np.stack([unit(i + t) for t in range(turns)]))},
    )


@pytest.fixture
def cfg(tmp_path):
    return CCEConfig(embedder="hash", base_dir=tmp_path / "data", warm_tier_max_chunks=0)


def test_one_index_per_data_dir_whatever_the_number_of_sessions(cfg):
    tiers = [WarmTier(cfg) for _ in range(4)]
    assert all(t.index is tiers[0].index for t in tiers)

    rows = 0
    for s in range(50):
        batch = [node(i, f"sess{s}") for i in range(6)]
        tiers[s % 4].upsert_many(batch)
        rows += len(batch)
    index = tiers[0].index
    index.sync()

    assert len(index._blocks) == 50
    assert sum(b.n for b in index._blocks.values()) == rows
    # allocated vector rows are about the number of notes, not sessions x notes (the old per-store copy)
    assert sum(len(b.vecs) for b in index._blocks.values()) <= 3 * rows
    # a session query only looks at its own notes
    scores = index.session_scores("sess7", unit(0))
    assert len(scores.node_ids) == 6 and all(i.startswith("sess7-") for i in scores.node_ids)
    for t in tiers:
        t.close()


def test_a_second_warm_tier_sees_rows_written_by_the_first(cfg):
    first, second = WarmTier(cfg), WarmTier(cfg)
    assert second.search(unit(3), top_k=3, session_id="s") == []
    first.upsert_many([node(i) for i in range(5)])
    found = second.search(unit(3), top_k=1, session_id="s")
    assert [n.node_id for n, _ in found] == ["s-n3"]          # no restart, no reload call

    first.upsert(node(9))
    assert second.search(unit(9), top_k=1, session_id="s")[0][0].node_id == "s-n9"
    first.delete("s-n3")
    assert "s-n3" not in [n.node_id for n, _ in second.search(unit(3), top_k=5, session_id="s")]
    first.close(); second.close()


def test_rows_written_by_another_process_are_seen(cfg):
    tier = WarmTier(cfg)
    tier.upsert(node(0))
    assert len(tier.search(unit(0), top_k=5, session_id="s")) == 1

    code = (
        "import sys, numpy as np\n"
        "from datetime import datetime, timezone\n"
        "from cce_core.config import CCEConfig\n"
        "from cce_core.compression.merger import MemoryNode\n"
        "from cce_core.memory.warm_tier import WarmTier\n"
        "from pathlib import Path\n"
        "w = WarmTier(CCEConfig(embedder='hash', base_dir=Path(sys.argv[1])))\n"
        "e = np.zeros(32, dtype=np.float32); e[5] = 1\n"
        "w.upsert(MemoryNode('other-proc', 'c', 's', e, 't', 'written by a second process', ['m'], 50, 51, 5,\n"
        "                    datetime.now(timezone.utc)))\n"
        "w.close()\n"
    )
    subprocess.run([sys.executable, "-c", code, str(cfg.base_dir)], cwd=ROOT, check=True, timeout=120)

    found = tier.search(unit(5), top_k=1, session_id="s")
    assert found and found[0][0].node_id == "other-proc"
    tier.close()


def test_rows_inserted_by_a_plain_sqlite_connection_are_seen(cfg):
    """The change log is filled by triggers, so any writer is tracked, not just this library."""
    tier = WarmTier(cfg)
    tier.upsert(node(0))
    tier.search(unit(0), top_k=1, session_id="s")
    other = sqlite3.connect(cfg.db_path)
    other.execute(
        "INSERT INTO memory_nodes (node_id, chunk_id, session_id, topic_label, meso_summary, micro_summaries, "
        "turn_start, turn_end, token_count, embedding, created_at, metadata) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("raw", "c", "s", "t", "raw insert", "[]", 90, 91, 5, unit(11).tobytes(), "2026-01-01T00:00:00+00:00", "{}"),
    )
    other.commit()
    other.close()
    assert tier.search(unit(11), top_k=1, session_id="s")[0][0].node_id == "raw"
    tier.close()


@pytest.mark.parametrize("session", ["s", None])
def test_search_issues_no_per_row_sql(cfg, session):
    counts = {}
    for rows in (10, 300):
        c = CCEConfig(embedder="hash", base_dir=cfg.base_dir / f"r{rows}", warm_tier_max_chunks=0)
        tier = WarmTier(c)
        tier.upsert_many([node(i) for i in range(rows)])
        tier.search(unit(1), top_k=3, session_id=session, recency_boost=0.2)   # first query loads the index
        seen = []
        tier._conn.set_trace_callback(seen.append)
        tier.index._conn.set_trace_callback(seen.append)
        tier.search(unit(1), top_k=3, session_id=session, recency_boost=0.2)
        counts[rows] = len(seen)
        tier.close()
    assert counts[10] == counts[300] <= 3, counts


def test_appends_are_amortised(cfg):
    tier = WarmTier(cfg)
    tier.upsert(node(0))
    tier.index.sync()
    block = tier.index._blocks["s"]
    reallocations, last = 0, id(block.vecs)
    for i in range(1, 600):
        tier.upsert(node(i, turns=1))
        tier.index.sync()
        if id(block.vecs) != last:
            reallocations, last = reallocations + 1, id(block.vecs)
    assert block.n == 600
    assert reallocations <= 8, reallocations   # doubling: about log2(600), not one copy per note
    tier.close()


def test_concurrent_writers_and_readers(cfg):
    tier = WarmTier(cfg)
    errors = []

    def writer(k):
        try:
            for i in range(30):
                tier.upsert(node(k * 100 + i, f"w{k}"))
        except Exception as e:   # pragma: no cover
            errors.append(e)

    def reader():
        try:
            for _ in range(60):
                tier.search(unit(2), top_k=3, session_id=None, recency_boost=0.1)
                tier.index.session_scores("w0", unit(2))
        except Exception as e:   # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(k,)) for k in range(3)] + [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert tier.index.size() == 90
    tier.close()


# ── Retrieval results are the same as before the index existed ────────────────

def _reference_ranking(store, retriever, query, top_k, boost=0.2, weight=0.15):
    """The pre-index algorithm: load every note, score each in Python."""
    qvec = retriever.embed_fn(query).astype(np.float32)
    kws = retriever._extract_keywords(query)
    nodes = store.warm.get_all(session_id=store.session_id)
    rows = [(n, *node_similarity(n, qvec)) for n in nodes]
    top_end = max(n.turn_end for n, _, _ in rows)
    out = []
    for n, sem, text in rows:
        rec = (n.turn_end / max(top_end, 1)) * boost
        kw = retriever._keyword_score(n, kws) * weight
        out.append((n.node_id, sem + rec + kw, text))
    if is_overview_query(query):
        return [(i, s, t) for (i, s, t), n in sorted(zip(out, [r[0] for r in rows]), key=lambda p: p[1].turn_start)]
    out.sort(key=lambda r: -r[1])
    return out[:top_k]


@pytest.mark.parametrize("query", [
    "How many invoice retry attempts are allowed?",
    "hiking boots weather trails",
    "python pandas merge",
    "what did we talk about so far?",
    "something unrelated entirely",
])
def test_index_ranking_matches_the_per_note_reference(tmp_path, query):
    cfg = make_config(tmp_path, retrieval_top_k=4)
    engine = CCEEngine(cfg)
    store, _, nodes = engine.full_pipeline(make_messages(), "s")
    assert len(nodes) >= 2

    got = engine.retriever.retrieve(store, query, top_k=4)
    want = _reference_ranking(store, engine.retriever, query, 4)
    assert [r.node.node_id for r in got] == [w[0] for w in want][: len(got)]
    for r, (_, score, text) in zip(got, want):
        assert r.composite_score == pytest.approx(score, abs=1e-5)
        assert r.best_turn == text
    store.close()
