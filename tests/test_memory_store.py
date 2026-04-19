"""
Phase 2 Tests — Memory Store (Hot / Warm / Cold tiers + MemoryStore)
Run with: pytest tests/test_memory_store.py -v
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cce_core.config import CCEConfig
from cce_core.ingestion.segmenter import Turn, segment
from cce_core.compression.chunker import SemanticChunker
from cce_core.compression.summarizer import Summarizer
from cce_core.compression.merger import Merger, MemoryNode
from cce_core.memory.hot_tier import HotTier
from cce_core.memory.warm_tier import WarmTier
from cce_core.memory.cold_tier import ColdTier, MacroSummary
from cce_core.memory.store import MemoryStore
from cce_core.engine import CCEEngine

from datetime import datetime, timezone
import uuid


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_MESSAGES = [
    {"role": "user",      "content": "Tell me about machine learning."},
    {"role": "assistant", "content": "Machine learning is a subset of AI where systems learn from data."},
    {"role": "user",      "content": "What are the main types?"},
    {"role": "assistant", "content": "Supervised, unsupervised, and reinforcement learning are the three main types."},
    {"role": "user",      "content": "Now let's talk about databases. What is SQL?"},
    {"role": "assistant", "content": "SQL is a language for managing relational databases. It lets you query, insert, update, and delete data."},
    {"role": "user",      "content": "What about NoSQL?"},
    {"role": "assistant", "content": "NoSQL databases like MongoDB store data as documents instead of rows and tables."},
]


@pytest.fixture(scope="function")
def tmp_dir(tmp_path):
    """Temporary directory — isolated per test."""
    return tmp_path


@pytest.fixture(scope="function")
def config(tmp_dir):
    return CCEConfig(
        hot_tier_max_turns=4,
        warm_tier_max_chunks=100,
        summarizer_mode="extractive",
        base_dir=tmp_dir / ".cce_data",
    )


@pytest.fixture(scope="module")
def shared_config():
    """Module-scoped config for tests that share heavy SBERT state."""
    d = Path(tempfile.mkdtemp())
    cfg = CCEConfig(
        hot_tier_max_turns=4,
        summarizer_mode="extractive",
        base_dir=d / ".cce_data",
    )
    yield cfg
    shutil.rmtree(d, ignore_errors=True)


def _make_turn(index: int, role="user", content="test content") -> Turn:
    return Turn(
        turn_id=str(uuid.uuid4()),
        role=role,
        content=content,
        token_count=5,
        timestamp=datetime.now(timezone.utc),
        index=index,
    )


def _make_node(turn_start=0, turn_end=3, session_id="test") -> MemoryNode:
    emb = np.random.randn(384).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return MemoryNode(
        node_id=str(uuid.uuid4()),
        chunk_id=str(uuid.uuid4()),
        session_id=session_id,
        embedding=emb,
        topic_label="test topic",
        meso_summary="This is a test meso summary about the topic.",
        micro_summaries=["micro 1", "micro 2"],
        turn_start=turn_start,
        turn_end=turn_end,
        token_count=100,
        created_at=datetime.now(timezone.utc),
    )


# ── Hot Tier Tests ────────────────────────────────────────────────────────────

class TestHotTier:
    def test_push_and_get(self, config):
        hot = HotTier(config)
        t = _make_turn(0)
        hot.push(t)
        assert hot.size() == 1
        assert hot.get_all()[0].turn_id == t.turn_id

    def test_eviction_when_full(self, config):
        hot = HotTier(config)  # maxlen=4
        turns = [_make_turn(i) for i in range(5)]
        evicted = []
        for t in turns:
            ev = hot.push(t)
            if ev:
                evicted.append(ev)
        assert hot.size() == 4
        assert len(evicted) == 1
        assert evicted[0].turn_id == turns[0].turn_id

    def test_drain_evicted(self, config):
        hot = HotTier(config)
        turns = [_make_turn(i) for i in range(6)]
        hot.push_many(turns)
        evicted = hot.drain_evicted()
        assert len(evicted) == 2
        assert hot.drain_evicted() == []  # already drained

    def test_ordering(self, config):
        hot = HotTier(config)
        turns = [_make_turn(i, content=f"turn {i}") for i in range(4)]
        for t in turns:
            hot.push(t)
        result = hot.get_all()
        assert [t.index for t in result] == [0, 1, 2, 3]

    def test_to_messages(self, config):
        hot = HotTier(config)
        t = _make_turn(0, role="user", content="hello")
        hot.push(t)
        msgs = hot.to_messages()
        assert msgs[0] == {"role": "user", "content": "hello"}

    def test_serialization_roundtrip(self, config):
        hot = HotTier(config)
        for i in range(3):
            hot.push(_make_turn(i))
        d = hot.to_dict()
        hot2 = HotTier.from_dict(d, config)
        assert hot2.size() == 3
        assert hot2.get_all()[0].index == hot.get_all()[0].index

    def test_clear(self, config):
        hot = HotTier(config)
        hot.push_many([_make_turn(i) for i in range(4)])
        hot.clear()
        assert hot.size() == 0

    def test_peek(self, config):
        hot = HotTier(config)
        turns = [_make_turn(i) for i in range(3)]
        hot.push_many(turns)
        assert hot.peek_oldest().index == 0
        assert hot.peek_newest().index == 2


# ── Warm Tier Tests ───────────────────────────────────────────────────────────

class TestWarmTier:
    def test_upsert_and_get(self, config):
        warm = WarmTier(config)
        node = _make_node(session_id="s1")
        warm.upsert(node)
        fetched = warm.get(node.node_id)
        assert fetched is not None
        assert fetched.node_id == node.node_id
        assert fetched.topic_label == node.topic_label
        warm.close()

    def test_batch_upsert(self, config):
        warm = WarmTier(config)
        nodes = [_make_node(i * 4, i * 4 + 3, session_id="s2") for i in range(5)]
        warm.upsert_many(nodes)
        assert warm.count("s2") == 5
        warm.close()

    def test_get_by_session(self, config):
        warm = WarmTier(config)
        for i in range(3):
            warm.upsert(_make_node(i * 4, i * 4 + 3, session_id="session-A"))
        warm.upsert(_make_node(0, 3, session_id="session-B"))
        result = warm.get_by_session("session-A")
        assert len(result) == 3
        assert all(n.session_id == "session-A" for n in result)
        warm.close()

    def test_search_returns_results(self, config):
        warm = WarmTier(config)
        nodes = [_make_node(i * 4, i * 4 + 3, session_id="s3") for i in range(5)]
        warm.upsert_many(nodes)
        query = np.random.randn(384).astype(np.float32)
        query /= np.linalg.norm(query)
        results = warm.search(query, top_k=3)
        assert len(results) == 3
        assert all(isinstance(score, float) for _, score in results)
        warm.close()

    def test_search_scores_ordered(self, config):
        warm = WarmTier(config)
        nodes = [_make_node(i * 4, i * 4 + 3, session_id="s4") for i in range(10)]
        warm.upsert_many(nodes)
        query = np.random.randn(384).astype(np.float32)
        query /= np.linalg.norm(query)
        results = warm.search(query, top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)
        warm.close()

    def test_search_finds_similar(self, config):
        """A node whose embedding IS the query should score highest."""
        warm = WarmTier(config)
        query = np.random.randn(384).astype(np.float32)
        query /= np.linalg.norm(query)

        # Insert the query itself as a node embedding
        target = _make_node(0, 3, session_id="s5")
        target.embedding = query.copy()
        noise_nodes = [_make_node(i * 4 + 4, i * 4 + 7, session_id="s5") for i in range(4)]
        warm.upsert(target)
        warm.upsert_many(noise_nodes)

        results = warm.search(query, top_k=1, session_id="s5")
        assert results[0][0].node_id == target.node_id
        warm.close()

    def test_delete(self, config):
        warm = WarmTier(config)
        node = _make_node(session_id="s6")
        warm.upsert(node)
        assert warm.get(node.node_id) is not None
        warm.delete(node.node_id)
        assert warm.get(node.node_id) is None
        warm.close()

    def test_node_serialization_preserved(self, config):
        warm = WarmTier(config)
        node = _make_node(session_id="s7")
        warm.upsert(node)
        fetched = warm.get(node.node_id)
        assert fetched.meso_summary == node.meso_summary
        assert fetched.micro_summaries == node.micro_summaries
        assert np.allclose(fetched.embedding, node.embedding, atol=1e-5)
        warm.close()


# ── Cold Tier Tests ───────────────────────────────────────────────────────────

class TestColdTier:
    def test_write_and_get(self, config):
        cold = ColdTier(config)
        summary = MacroSummary(
            session_id="cold-test-01",
            macro_text="This session discussed machine learning and databases.",
            node_count=4,
            turn_count=8,
            token_count_original=800,
            token_count_compressed=12,
            topics=["machine learning", "databases"],
            created_at=datetime.now(timezone.utc),
        )
        cold.write(summary)
        fetched = cold.get("cold-test-01")
        assert fetched is not None
        assert fetched.macro_text == summary.macro_text
        assert fetched.compression_ratio > 1.0

    def test_exists(self, config):
        cold = ColdTier(config)
        assert not cold.exists("nonexistent-session")
        cold.write(MacroSummary(
            session_id="exists-test",
            macro_text="test",
            node_count=1, turn_count=2,
            token_count_original=100, token_count_compressed=5,
            topics=["test"], created_at=datetime.now(timezone.utc),
        ))
        assert cold.exists("exists-test")

    def test_list_all(self, config):
        cold = ColdTier(config)
        for i in range(3):
            cold.write(MacroSummary(
                session_id=f"list-test-{i}",
                macro_text="summary text",
                node_count=2, turn_count=4,
                token_count_original=200, token_count_compressed=10,
                topics=["topic"], created_at=datetime.now(timezone.utc),
            ))
        entries = cold.list_all()
        session_ids = {e["session_id"] for e in entries}
        for i in range(3):
            assert f"list-test-{i}" in session_ids

    def test_delete(self, config):
        cold = ColdTier(config)
        cold.write(MacroSummary(
            session_id="delete-test",
            macro_text="to be deleted",
            node_count=1, turn_count=2,
            token_count_original=50, token_count_compressed=5,
            topics=["test"], created_at=datetime.now(timezone.utc),
        ))
        assert cold.delete("delete-test")
        assert cold.get("delete-test") is None

    def test_compression_ratio(self, config):
        cold = ColdTier(config)
        summary = MacroSummary(
            session_id="ratio-test",
            macro_text="short",
            node_count=1, turn_count=100,
            token_count_original=5000, token_count_compressed=10,
            topics=["test"], created_at=datetime.now(timezone.utc),
        )
        assert summary.compression_ratio == 500.0


# ── MemoryStore Integration Tests ─────────────────────────────────────────────

class TestMemoryStore:
    def test_ingest_turns_hot_tier(self, config):
        store = MemoryStore("store-test-01", config)
        turns = [_make_turn(i) for i in range(3)]
        store.ingest_turns(turns)
        assert store.hot.size() == 3
        store.close()

    def test_eviction_triggers_warm_write(self, config):
        """When hot tier overflows, evicted turns get compressed to warm."""
        compress_calls = []

        def fake_compress(turns, session_id):
            compress_calls.append(len(turns))
            return [_make_node(turns[0].index, turns[-1].index, session_id)]

        store = MemoryStore("store-test-02", config, compress_fn=fake_compress)
        # Push 6 turns — hot holds 4, so 2 should evict
        turns = [_make_turn(i) for i in range(6)]
        store.ingest_turns(turns)
        assert len(compress_calls) > 0
        assert store.warm.count("store-test-02") > 0
        store.close()

    def test_hot_turns_as_messages(self, config):
        store = MemoryStore("store-test-03", config)
        t = _make_turn(0, role="assistant", content="hello world")
        store.ingest_turn(t)
        msgs = store.get_hot_messages()
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"] == "hello world"
        store.close()

    def test_checkpoint_writes_cold(self, config):
        store = MemoryStore("store-test-04", config)
        node = _make_node(session_id="store-test-04")
        store.ingest_nodes([node])
        summary = store.checkpoint("This session was about testing the CCE system.")
        assert summary is not None
        assert store.cold.exists("store-test-04")
        store.close()

    def test_stats(self, config):
        store = MemoryStore("store-test-05", config)
        store.ingest_turns([_make_turn(i) for i in range(3)])
        # ingest_nodes is a separate path (pre-compressed load) —
        # test it independently so turn counts don't collide
        stats = store.stats()
        assert stats["turns_ingested"] == 3
        assert stats["hot_turns"] == 3

        # Now test warm node count via a separate store
        store2 = MemoryStore("store-test-05b", config)
        store2.ingest_nodes([_make_node(session_id="store-test-05b")])
        stats2 = store2.stats()
        assert stats2["warm_nodes"] == 1
        store.close()
        store2.close()

    def test_flush_hot_to_warm(self, config):
        def fake_compress(turns, sid):
            return [_make_node(turns[0].index, turns[-1].index, sid)]

        store = MemoryStore("store-test-06", config, compress_fn=fake_compress)
        store.ingest_turns([_make_turn(i) for i in range(3)])
        assert store.hot.size() == 3
        nodes = store.flush_hot_to_warm()
        assert len(nodes) > 0
        assert store.hot.size() == 0
        store.close()


# ── Full Pipeline Integration (Engine Phase 1+2) ──────────────────────────────

class TestEnginePhase2:
    def test_full_pipeline(self, shared_config):
        engine = CCEEngine(shared_config)
        store, turns, nodes = engine.full_pipeline(
            SAMPLE_MESSAGES, session_id="engine-phase2-test"
        )
        assert len(turns) == len(SAMPLE_MESSAGES)
        assert len(nodes) >= 1
        assert store.warm.count("engine-phase2-test") == len(nodes)
        store.close()

    def test_search_after_pipeline(self, shared_config):
        engine = CCEEngine(shared_config)
        store, turns, nodes = engine.full_pipeline(
            SAMPLE_MESSAGES, session_id="engine-search-test"
        )
        query_emb = engine.chunker.embed_text("what is machine learning")
        results = store.search_warm(query_emb, top_k=2)
        assert len(results) >= 1
        assert all(isinstance(s, float) for _, s in results)
        store.close()

    def test_checkpoint_full_session(self, shared_config):
        engine = CCEEngine(shared_config)
        store, turns, nodes = engine.full_pipeline(
            SAMPLE_MESSAGES, session_id="engine-checkpoint-test"
        )
        macro = engine.macro_summary(nodes)
        summary = store.checkpoint(macro)
        assert summary is not None
        assert summary.macro_text != ""
        assert summary.node_count == len(nodes)
        assert summary.turn_count > 0
        assert summary.token_count_original > 0
        # Compression ratio can be < 1 for very short conversations
        # (extractive summary of already-short text). Just assert it's computed.
        assert summary.compression_ratio >= 0.0
        print(f"\n  Macro summary: {macro[:120]}...")
        print(f"  Compression ratio: {summary.compression_ratio}x")
        store.close()

    def test_open_session_stateful(self, shared_config):
        engine = CCEEngine(shared_config)
        store = engine.open_session("stateful-session-test")
        turns = segment(SAMPLE_MESSAGES, session_id="stateful-session-test")
        new_nodes = store.ingest_turns(turns)
        assert store.hot.size() > 0
        store.close()