"""
Phase 3 Tests — Retriever + Context Builder
Run with: pytest tests/test_retriever.py -v
"""

import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import uuid

import numpy as np
import pytest

from cce_core.config import CCEConfig
from cce_core.compression.merger import MemoryNode
from cce_core.ingestion.segmenter import Turn
from cce_core.retrieval.retriever import Retriever, RetrievalResult
from cce_core.retrieval.context_builder import ContextBuilder, ContextPayload
from cce_core.memory.store import MemoryStore
from cce_core.engine import CCEEngine


# ── Fixtures ──────────────────────────────────────────────────────────────────

LONG_CONVERSATION = [
    {"role": "user",      "content": "Let's talk about Python. What makes it popular?"},
    {"role": "assistant", "content": "Python is popular because of its simple syntax, large ecosystem, and versatility. It's used in web development, data science, AI, and automation."},
    {"role": "user",      "content": "What are the best Python frameworks for web development?"},
    {"role": "assistant", "content": "Django is the most complete framework with batteries included. FastAPI is excellent for APIs due to its speed and auto-documentation. Flask is minimal and flexible."},
    {"role": "user",      "content": "Now tell me about SQL databases. What is normalization?"},
    {"role": "assistant", "content": "Normalization is the process of organizing a database to reduce redundancy. It involves dividing large tables into smaller ones and defining relationships between them."},
    {"role": "user",      "content": "What are the normal forms?"},
    {"role": "assistant", "content": "First Normal Form requires atomic values. Second Normal Form eliminates partial dependencies. Third Normal Form removes transitive dependencies."},
    {"role": "user",      "content": "Can we switch topics to machine learning? Explain overfitting."},
    {"role": "assistant", "content": "Overfitting happens when a model learns the training data too well, including its noise, and fails to generalize to new data. It's like memorizing answers instead of understanding concepts."},
    {"role": "user",      "content": "How do you prevent overfitting?"},
    {"role": "assistant", "content": "Common techniques include regularization (L1/L2), dropout in neural networks, cross-validation, early stopping, and getting more training data."},
    {"role": "user",      "content": "Back to Python — what are decorators?"},
    {"role": "assistant", "content": "Decorators are functions that modify the behavior of other functions. They use the @syntax and are widely used for logging, authentication, and caching."},
]


@pytest.fixture(scope="module")
def shared_config():
    d = Path(tempfile.mkdtemp())
    cfg = CCEConfig(
        hot_tier_max_turns=6,
        retrieval_top_k=3,
        summarizer_mode="extractive",
        base_dir=d / ".cce_data",
    )
    yield cfg
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def engine(shared_config):
    return CCEEngine(shared_config)


@pytest.fixture(scope="module")
def populated_store(engine, shared_config):
    """Full pipeline store loaded with the long conversation."""
    store, turns, nodes = engine.full_pipeline(
        LONG_CONVERSATION, session_id="phase3-test-session"
    )
    yield store, turns, nodes
    store.close()


def _make_turn(index: int, role="user", content="test content") -> Turn:
    return Turn(
        turn_id=str(uuid.uuid4()), role=role, content=content,
        token_count=10, timestamp=datetime.now(timezone.utc), index=index,
    )


def _make_node(topic: str, summary: str, turn_start=0, turn_end=3, session_id="test") -> MemoryNode:
    emb = np.random.randn(384).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return MemoryNode(
        node_id=str(uuid.uuid4()), chunk_id=str(uuid.uuid4()),
        session_id=session_id, embedding=emb, topic_label=topic,
        meso_summary=summary, micro_summaries=["micro 1", "micro 2"],
        turn_start=turn_start, turn_end=turn_end, token_count=150,
        created_at=datetime.now(timezone.utc),
    )


# ── Retriever Tests ───────────────────────────────────────────────────────────

class TestRetriever:
    def test_retrieve_returns_results(self, engine, populated_store):
        store, turns, nodes = populated_store
        results = engine.retriever.retrieve(store, "what is Python?", top_k=3)
        assert len(results) >= 1
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_results_sorted_by_composite_score(self, engine, populated_store):
        store, turns, nodes = populated_store
        results = engine.retriever.retrieve(store, "machine learning overfitting", top_k=5)
        scores = [r.composite_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_semantic_score_in_range(self, engine, populated_store):
        store, turns, nodes = populated_store
        results = engine.retriever.retrieve(store, "database normalization SQL", top_k=3)
        for r in results:
            assert -1.0 <= r.semantic_score <= 1.0

    def test_keyword_score_non_negative(self, engine, populated_store):
        store, turns, nodes = populated_store
        results = engine.retriever.retrieve(store, "Python decorators syntax", top_k=3)
        for r in results:
            assert r.keyword_score >= 0.0

    def test_relevant_topic_ranked_higher(self, engine, populated_store):
        """Querying 'SQL normalization' should rank SQL nodes above Python nodes."""
        store, turns, nodes = populated_store
        results = engine.retriever.retrieve(store, "SQL normalization database", top_k=3)
        assert len(results) >= 1
        top_text = results[0].node.topic_label.lower() + results[0].node.meso_summary.lower()
        # Top result should mention SQL/database/normalization somewhere
        relevant_keywords = {"sql", "database", "normalization", "normal", "redundancy", "tables"}
        found = any(kw in top_text for kw in relevant_keywords)
        assert found, f"Expected SQL topic at top, got: {results[0].node.topic_label!r}"

    def test_retrieve_by_turn(self, engine, populated_store):
        store, turns, nodes = populated_store
        results = engine.retriever.retrieve_by_turn(store, turn_index=0)
        # turn 0 must be covered by some node
        assert len(results) >= 1
        assert all(r.node.turn_start <= 0 <= r.node.turn_end for r in results)

    def test_retrieve_all(self, engine, populated_store):
        store, turns, nodes = populated_store
        results = engine.retriever.retrieve_all(store)
        assert len(results) == len(nodes)

    def test_result_to_dict(self, engine, populated_store):
        store, turns, nodes = populated_store
        results = engine.retriever.retrieve(store, "Python frameworks", top_k=1)
        d = results[0].to_dict()
        assert "node_id" in d
        assert "composite_score" in d
        assert "semantic_score" in d
        assert "meso_summary" in d

    def test_empty_store_returns_empty(self, shared_config):
        store = MemoryStore("empty-retrieval-session", shared_config)
        embed_fn = lambda t: np.random.randn(384).astype(np.float32)
        retriever = Retriever(embed_fn=embed_fn, config=shared_config)
        results = retriever.retrieve(store, "anything", top_k=3)
        assert results == []
        store.close()


# ── Context Builder Tests ─────────────────────────────────────────────────────

class TestContextBuilder:
    def test_build_produces_payload(self, engine, populated_store):
        store, turns, nodes = populated_store
        results = engine.retriever.retrieve(store, "Python", top_k=3)
        hot_turns = store.get_hot_turns()
        builder = ContextBuilder(engine.config)
        payload = builder.build(results, hot_turns)
        assert isinstance(payload, ContextPayload)

    def test_payload_has_past_blocks(self, engine, populated_store):
        store, turns, nodes = populated_store
        results = engine.retriever.retrieve(store, "machine learning", top_k=3)
        hot_turns = store.get_hot_turns()
        payload = ContextBuilder(engine.config).build(results, hot_turns)
        assert len(payload.past_blocks) >= 1

    def test_payload_has_recent_turns(self, engine, populated_store):
        store, turns, nodes = populated_store
        results = engine.retriever.retrieve(store, "decorators", top_k=2)
        hot_turns = store.get_hot_turns()
        payload = ContextBuilder(engine.config).build(results, hot_turns)
        assert len(payload.recent_turns) == len(hot_turns)

    def test_to_messages_format(self, engine, populated_store):
        store, turns, nodes = populated_store
        results = engine.retriever.retrieve(store, "SQL", top_k=2)
        hot_turns = store.get_hot_turns()
        payload = ContextBuilder(engine.config).build(results, hot_turns)
        messages = payload.to_messages()
        assert isinstance(messages, list)
        assert messages[0]["role"] == "system"
        assert "Context Compression Engine" in messages[0]["content"]
        # All remaining messages should be user/assistant
        for msg in messages[1:]:
            assert msg["role"] in ("user", "assistant", "tool")

    def test_to_string_format(self, engine, populated_store):
        store, turns, nodes = populated_store
        results = engine.retriever.retrieve(store, "overfitting", top_k=2)
        hot_turns = store.get_hot_turns()
        payload = ContextBuilder(engine.config).build(results, hot_turns)
        s = payload.to_string()
        assert isinstance(s, str)
        assert "PAST CONTEXT" in s
        assert "RECENT CONTEXT" in s

    def test_to_dict_format(self, engine, populated_store):
        store, turns, nodes = populated_store
        results = engine.retriever.retrieve(store, "Python", top_k=2)
        hot_turns = store.get_hot_turns()
        payload = ContextBuilder(engine.config).build(results, hot_turns)
        d = payload.to_dict()
        assert "past_blocks" in d
        assert "recent_turns" in d
        assert "token_count" in d
        assert "was_truncated" in d

    def test_token_budget_respected(self, engine, populated_store):
        store, turns, nodes = populated_store
        results = engine.retriever.retrieve(store, "Python SQL machine learning", top_k=5)
        hot_turns = store.get_hot_turns()
        tight_budget = 300
        payload = ContextBuilder(engine.config).build(
            results, hot_turns, max_tokens=tight_budget
        )
        # Token count should be at or near budget (not wildly over)
        assert payload.token_count <= tight_budget + 100  # allow 100 token tolerance

    def test_build_empty_no_results(self, engine):
        hot_turns = [_make_turn(i) for i in range(3)]
        builder = ContextBuilder(engine.config)
        payload = builder.build_empty(hot_turns)
        assert payload.past_blocks == []
        assert len(payload.recent_turns) == 3
        assert payload.retrieved_node_ids == []

    def test_include_micro_summaries(self, engine, populated_store):
        store, turns, nodes = populated_store
        results = engine.retriever.retrieve(store, "Python", top_k=2)
        hot_turns = store.get_hot_turns()
        payload = ContextBuilder(engine.config).build(
            results, hot_turns, include_micro=True
        )
        # With include_micro, blocks should contain "  - " bullet lines
        combined = "\n".join(payload.past_blocks)
        assert "  - " in combined

    def test_node_ids_tracked(self, engine, populated_store):
        store, turns, nodes = populated_store
        results = engine.retriever.retrieve(store, "database", top_k=3)
        hot_turns = store.get_hot_turns()
        payload = ContextBuilder(engine.config).build(results, hot_turns)
        assert len(payload.retrieved_node_ids) >= 1
        for nid in payload.retrieved_node_ids:
            assert isinstance(nid, str)


# ── Engine Phase 3 Integration ────────────────────────────────────────────────

class TestEnginePhase3:
    def test_build_context_returns_payload(self, engine, populated_store):
        store, turns, nodes = populated_store
        payload = engine.build_context(store, "what is Python?")
        assert isinstance(payload, ContextPayload)
        assert payload.token_count > 0

    def test_build_context_messages_format(self, engine, populated_store):
        store, turns, nodes = populated_store
        payload = engine.build_context(store, "explain overfitting")
        messages = payload.to_messages()
        assert messages[0]["role"] == "system"
        assert len(messages) >= 2

    def test_build_context_empty_warm(self, engine, shared_config):
        """When warm tier is empty, should still return a valid payload."""
        store = MemoryStore("phase3-empty-warm", shared_config)
        hot_turns = [_make_turn(0, content="hello")]
        for t in hot_turns:
            store.hot.push(t)
        payload = engine.build_context(store, "hello")
        assert isinstance(payload, ContextPayload)
        assert payload.past_blocks == []
        store.close()

    def test_query_returns_context_and_payload(self, engine, populated_store):
        store, turns, nodes = populated_store
        context, payload = engine.query(store, "what did we say about decorators?")
        assert isinstance(context, list)  # default fmt="messages"
        assert isinstance(payload, ContextPayload)
        assert context[0]["role"] == "system"

    def test_query_string_format(self, engine, populated_store):
        store, turns, nodes = populated_store
        context, payload = engine.query(store, "SQL normalization", fmt="string")
        assert isinstance(context, str)
        assert len(context) > 0

    def test_full_end_to_end_flow(self, engine, shared_config):
        """
        Simulate a real conversation:
        1. Load history into store
        2. New user message arrives
        3. Build context
        4. Verify context contains relevant past + recent turns
        """
        store, turns, nodes = engine.full_pipeline(
            LONG_CONVERSATION, session_id="e2e-phase3-test"
        )

        new_query = "What overfitting prevention techniques did you mention?"
        payload = engine.build_context(store, new_query)
        context_str = payload.to_string()

        # The payload should reference overfitting/regularization/dropout somewhere
        relevant = any(
            kw in context_str.lower()
            for kw in ["overfitting", "regularization", "dropout", "cross-validation"]
        )
        assert relevant, f"Expected overfitting context, got:\n{context_str[:500]}"

        print(f"\n  Context tokens: {payload.token_count}")
        print(f"  Past blocks: {len(payload.past_blocks)}")
        print(f"  Recent turns: {len(payload.recent_turns)}")
        print(f"  Was truncated: {payload.was_truncated}")
        store.close()