"""
Phase 4 Tests — Session Manager + Stateless Processor
Run with: pytest tests/test_session.py -v
"""

import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import uuid

import pytest

from cce_core.config import CCEConfig
from cce_core.engine import CCEEngine
from cce_core.session.manager import SessionManager
from cce_core.session.stateless import StatelessProcessor, StatelessResult
from cce_core.retrieval.context_builder import ContextPayload


CONVERSATION_A = [
    {"role": "user",      "content": "What is recursion in programming?"},
    {"role": "assistant", "content": "Recursion is when a function calls itself. It needs a base case to stop, otherwise it runs forever."},
    {"role": "user",      "content": "Can you give me a simple example?"},
    {"role": "assistant", "content": "Sure! A factorial function: factorial(n) = n * factorial(n-1), with factorial(0) = 1 as the base case."},
    {"role": "user",      "content": "Now tell me about Docker. What is a container?"},
    {"role": "assistant", "content": "A container packages code and its dependencies together so the application runs reliably across different environments."},
    {"role": "user",      "content": "How is a container different from a virtual machine?"},
    {"role": "assistant", "content": "VMs include a full OS, so they're heavy. Containers share the host OS kernel, making them lightweight and fast to start."},
]

CONVERSATION_B = [
    {"role": "user",      "content": "Explain REST APIs."},
    {"role": "assistant", "content": "REST is an architectural style for APIs. It uses HTTP methods like GET, POST, PUT, DELETE to perform operations on resources."},
    {"role": "user",      "content": "What's the difference between PUT and PATCH?"},
    {"role": "assistant", "content": "PUT replaces the entire resource. PATCH updates only the fields you specify, making it more efficient for partial updates."},
]


@pytest.fixture(scope="module")
def shared_config():
    d = Path(tempfile.mkdtemp())
    cfg = CCEConfig(
        hot_tier_max_turns=4,
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
def manager(engine):
    m = engine.create_session_manager()
    yield m
    m.close_all(checkpoint=False)


# ── Session Manager Tests ─────────────────────────────────────────────────────

class TestSessionManager:
    def test_open_creates_session(self, manager):
        store = manager.open("sm-test-open")
        assert store is not None
        assert manager.is_active("sm-test-open")

    def test_open_idempotent(self, manager):
        store1 = manager.open("sm-test-idem")
        store2 = manager.open("sm-test-idem")
        assert store1 is store2

    def test_get_or_open(self, manager):
        store = manager.get_or_open("sm-test-get-or-open")
        assert store is not None

    def test_add_message(self, manager):
        manager.open("sm-test-msg")
        turn = manager.add_message("sm-test-msg", {
            "role": "user", "content": "What is Python?"
        })
        assert turn.role == "user"
        assert turn.content == "What is Python?"

    def test_add_messages_bulk(self, manager):
        manager.open("sm-test-bulk")
        turns = manager.add_messages("sm-test-bulk", CONVERSATION_A)
        assert len(turns) == len(CONVERSATION_A)

    def test_session_registry_updated(self, manager):
        manager.open("sm-test-registry")
        manager.add_message("sm-test-registry", {"role": "user", "content": "hello"})
        sessions = manager.list_sessions()
        ids = {s["session_id"] for s in sessions}
        assert "sm-test-registry" in ids

    def test_list_active(self, manager):
        manager.open("sm-test-active-1")
        manager.open("sm-test-active-2")
        active = manager.list_active()
        assert "sm-test-active-1" in active
        assert "sm-test-active-2" in active

    def test_build_context_after_messages(self, manager, engine):
        manager.open("sm-test-ctx")
        manager.add_messages("sm-test-ctx", CONVERSATION_A)
        context = manager.build_context("sm-test-ctx", "what is Docker?")
        assert isinstance(context, list)
        assert context[0]["role"] == "system"

    def test_build_context_payload(self, manager):
        manager.open("sm-test-payload")
        manager.add_messages("sm-test-payload", CONVERSATION_A)
        payload = manager.build_context_payload("sm-test-payload", "recursion example")
        assert isinstance(payload, ContextPayload)

    def test_close_session(self, manager):
        manager.open("sm-test-close")
        manager.add_messages("sm-test-close", CONVERSATION_B)
        result = manager.close("sm-test-close", checkpoint=False)
        assert result is True
        assert not manager.is_active("sm-test-close")

    def test_close_nonexistent(self, manager):
        result = manager.close("nonexistent-xyz", checkpoint=False)
        assert result is False

    def test_checkpoint_writes_cold(self, manager, engine):
        sid = "sm-test-checkpoint"
        manager.open(sid)
        manager.add_messages(sid, CONVERSATION_A)
        # Force compress everything into warm
        store = manager.get(sid)
        store.flush_hot_to_warm()
        result = manager.checkpoint(sid)
        assert result is True
        # Cold tier should now have an entry
        assert store.cold.exists(sid)

    def test_multi_session_isolation(self, manager):
        """Messages to session A should not appear in session B."""
        manager.open("sm-iso-A")
        manager.open("sm-iso-B")
        manager.add_message("sm-iso-A", {"role": "user", "content": "Session A content about Python"})
        manager.add_message("sm-iso-B", {"role": "user", "content": "Session B content about Docker"})

        store_a = manager.get("sm-iso-A")
        store_b = manager.get("sm-iso-B")

        hot_a = store_a.get_hot_turns()
        hot_b = store_b.get_hot_turns()

        contents_a = {t.content for t in hot_a}
        contents_b = {t.content for t in hot_b}

        assert "Session A content about Python" in contents_a
        assert "Session A content about Python" not in contents_b
        assert "Session B content about Docker" in contents_b
        assert "Session B content about Docker" not in contents_a

    def test_session_stats(self, manager):
        sid = "sm-test-stats"
        manager.open(sid)
        manager.add_messages(sid, CONVERSATION_B)
        stats = manager.session_stats(sid)
        assert stats is not None
        assert stats["turns_ingested"] == len(CONVERSATION_B)

    def test_delete_session(self, manager):
        sid = "sm-test-delete"
        manager.open(sid)
        manager.add_messages(sid, CONVERSATION_B)
        manager.delete(sid)
        assert not manager.is_active(sid)
        assert not manager.exists(sid)

    def test_repr(self, manager):
        r = repr(manager)
        assert "SessionManager" in r


# ── Stateless Processor Tests ─────────────────────────────────────────────────

class TestStatelessProcessor:
    def test_process_returns_result(self, engine):
        proc = engine.create_stateless_processor()
        result = proc.process(CONVERSATION_A, query="what is Docker?")
        assert isinstance(result, StatelessResult)

    def test_result_has_turns(self, engine):
        proc = engine.create_stateless_processor()
        result = proc.process(CONVERSATION_A, query="recursion")
        assert len(result.turns) == len(CONVERSATION_A)

    def test_result_has_nodes(self, engine):
        proc = engine.create_stateless_processor()
        result = proc.process(CONVERSATION_A, query="Docker container")
        assert len(result.nodes) >= 1

    def test_result_has_payload(self, engine):
        proc = engine.create_stateless_processor()
        result = proc.process(CONVERSATION_A, query="factorial recursion")
        assert isinstance(result.payload, ContextPayload)

    def test_to_messages_format(self, engine):
        proc = engine.create_stateless_processor()
        result = proc.process(CONVERSATION_A, query="Docker vs VM")
        messages = result.to_messages()
        assert isinstance(messages, list)
        assert messages[0]["role"] == "system"

    def test_to_string_format(self, engine):
        proc = engine.create_stateless_processor()
        result = proc.process(CONVERSATION_A, query="recursion base case")
        s = result.to_string()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_to_dict_format(self, engine):
        proc = engine.create_stateless_processor()
        result = proc.process(CONVERSATION_A, query="container")
        d = result.to_dict()
        assert "payload" in d
        assert "stats" in d

    def test_stats_contains_compression_ratio(self, engine):
        proc = engine.create_stateless_processor()
        result = proc.process(CONVERSATION_A, query="anything")
        assert "compression_ratio" in result.stats
        assert "total_turns" in result.stats
        assert result.stats["total_turns"] == len(CONVERSATION_A)

    def test_compress_only(self, engine):
        proc = engine.create_stateless_processor()
        turns, nodes = proc.compress_only(CONVERSATION_A)
        assert len(turns) == len(CONVERSATION_A)
        assert len(nodes) >= 1

    def test_stats_method(self, engine):
        proc = engine.create_stateless_processor()
        stats = proc.stats(CONVERSATION_A)
        assert "compression_ratio" in stats
        assert "total_turns" in stats

    def test_relevant_content_in_payload(self, engine):
        """Querying about Docker should surface Docker content in past blocks."""
        proc = engine.create_stateless_processor()
        result = proc.process(CONVERSATION_A, query="What is a Docker container?")
        combined = " ".join(result.payload.past_blocks).lower()
        assert any(kw in combined for kw in ["container", "docker", "vm", "virtual"]), \
            f"Expected Docker content in past blocks. Got: {combined[:300]}"

    def test_empty_conversation(self, engine):
        proc = engine.create_stateless_processor()
        result = proc.process(
            [{"role": "user", "content": "hello"}],
            query="hello"
        )
        assert isinstance(result, StatelessResult)
        assert len(result.turns) == 1

    def test_no_persistence(self, engine, shared_config):
        """Stateless processor should not write to DB."""
        from cce_core.memory.warm_tier import WarmTier
        warm = WarmTier(shared_config)
        count_before = warm.count(session_id="stateless")
        warm.close()

        proc = engine.create_stateless_processor()
        proc.process(CONVERSATION_A, query="test", session_id="stateless")

        warm2 = WarmTier(shared_config)
        count_after = warm2.count(session_id="stateless")
        warm2.close()

        assert count_after == count_before  # no DB writes


# ── Engine Phase 4 Integration ────────────────────────────────────────────────

class TestEnginePhase4:
    def test_create_session_manager(self, engine):
        manager = engine.create_session_manager()
        assert isinstance(manager, SessionManager)

    def test_create_stateless_processor(self, engine):
        proc = engine.create_stateless_processor()
        assert isinstance(proc, StatelessProcessor)

    def test_stateful_vs_stateless_same_content(self, engine, shared_config):
        """
        Both modes should produce a valid context payload for the same conversation.
        Not a diff test — just validates both paths are functional end to end.
        """
        # Stateful
        manager = engine.create_session_manager()
        manager.add_messages("phase4-stateful", CONVERSATION_A)
        stateful_ctx = manager.build_context("phase4-stateful", "Docker container")
        manager.close("phase4-stateful", checkpoint=False)

        # Stateless
        proc = engine.create_stateless_processor()
        stateless_result = proc.process(CONVERSATION_A, query="Docker container")
        stateless_ctx = stateless_result.to_messages()

        # Both should produce valid message lists with a system message
        assert stateful_ctx[0]["role"] == "system"
        assert stateless_ctx[0]["role"] == "system"
        assert len(stateful_ctx) >= 2
        assert len(stateless_ctx) >= 2

    def test_full_session_lifecycle(self, engine):
        """Open → add messages → query → checkpoint → close."""
        manager = engine.create_session_manager()
        sid = "phase4-lifecycle"

        # Open
        manager.open(sid)
        assert manager.is_active(sid)

        # Add messages
        turns = manager.add_messages(sid, CONVERSATION_A)
        assert len(turns) == len(CONVERSATION_A)

        # Query
        payload = manager.build_context_payload(sid, "what is recursion?")
        assert isinstance(payload, ContextPayload)

        # Checkpoint
        store = manager.get(sid)
        store.flush_hot_to_warm()
        manager.checkpoint(sid)

        # Close
        manager.close(sid, checkpoint=False)
        assert not manager.is_active(sid)

        # Registry still knows about it
        assert manager.exists(sid)