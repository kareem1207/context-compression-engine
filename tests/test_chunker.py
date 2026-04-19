"""
Phase 1 Tests — Ingestion + Compression
Run with: pytest tests/test_chunker.py -v
"""

import pytest
import numpy as np

from cce_core.config import CCEConfig
from cce_core.ingestion.tokenizer import count, truncate_to_tokens
from cce_core.ingestion.segmenter import segment, Turn
from cce_core.compression.chunker import SemanticChunker
from cce_core.compression.summarizer import Summarizer
from cce_core.compression.merger import Merger
from cce_core.engine import CCEEngine


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_CONVERSATION = [
    {"role": "user",      "content": "Hey, can you help me understand how neural networks work?"},
    {"role": "assistant", "content": "Sure! Neural networks are inspired by the human brain. They consist of layers of nodes called neurons that process information."},
    {"role": "user",      "content": "What are the different types of layers?"},
    {"role": "assistant", "content": "The main types are input layers, hidden layers, and output layers. Hidden layers do the heavy lifting of feature extraction."},
    {"role": "user",      "content": "Got it. Now can we talk about something different — what is gradient descent?"},
    {"role": "assistant", "content": "Gradient descent is an optimization algorithm used to minimize the loss function by iteratively moving in the direction of steepest descent."},
    {"role": "user",      "content": "How does learning rate affect it?"},
    {"role": "assistant", "content": "The learning rate controls the step size. Too high and you overshoot the minimum, too low and training takes forever."},
    {"role": "user",      "content": "Completely different topic — can you recommend a Python library for data visualization?"},
    {"role": "assistant", "content": "Matplotlib is the classic choice. Seaborn is built on top of it and gives you nicer defaults. Plotly is great for interactive charts."},
    {"role": "user",      "content": "Which one is best for beginners?"},
    {"role": "assistant", "content": "Seaborn is probably the best starting point. The defaults are beautiful and it integrates well with pandas DataFrames."},
]


@pytest.fixture(scope="module")
def config():
    return CCEConfig(
        chunk_similarity_threshold=0.45,
        chunk_min_turns=2,
        chunk_max_turns=8,
        summarizer_mode="extractive",
    )


@pytest.fixture(scope="module")
def turns(config):
    return segment(SAMPLE_CONVERSATION, session_id="test-session-001")


@pytest.fixture(scope="module")
def chunker(config):
    return SemanticChunker(config)


@pytest.fixture(scope="module")
def chunks(chunker, turns):
    return chunker.chunk(turns, session_id="test-session-001")


# ── Tokenizer tests ───────────────────────────────────────────────────────────

class TestTokenizer:
    def test_empty_string(self):
        assert count("") == 0

    def test_single_word(self):
        assert count("hello") >= 1

    def test_token_count_reasonable(self):
        text = "The quick brown fox jumps over the lazy dog"
        tokens = count(text)
        assert 8 <= tokens <= 16  # 9 words → ~12 tokens

    def test_truncate(self):
        text = " ".join(["word"] * 100)
        truncated = truncate_to_tokens(text, 20)
        assert count(truncated) <= 25  # allow small overshoot from "..."


# ── Segmenter tests ───────────────────────────────────────────────────────────

class TestSegmenter:
    def test_turn_count(self, turns):
        assert len(turns) == len(SAMPLE_CONVERSATION)

    def test_roles_preserved(self, turns):
        assert turns[0].role == "user"
        assert turns[1].role == "assistant"

    def test_indices_sequential(self, turns):
        for i, t in enumerate(turns):
            assert t.index == i

    def test_token_counts_positive(self, turns):
        for t in turns:
            assert t.token_count > 0

    def test_session_id_attached(self, turns):
        for t in turns:
            assert t.metadata.get("session_id") == "test-session-001"

    def test_serialization_roundtrip(self, turns):
        d = turns[0].to_dict()
        t2 = Turn.from_dict(d)
        assert t2.content == turns[0].content
        assert t2.role == turns[0].role

    def test_plain_text_input(self):
        plain = "User: Hello there\nAssistant: Hi! How can I help?\nUser: What is Python?"
        turns = segment(plain)
        assert len(turns) == 3
        assert turns[0].role == "user"
        assert turns[1].role == "assistant"


# ── Chunker tests ─────────────────────────────────────────────────────────────

class TestChunker:
    def test_produces_chunks(self, chunks):
        assert len(chunks) >= 1

    def test_all_turns_covered(self, chunks, turns):
        all_turn_ids = {t.turn_id for chunk in chunks for t in chunk.turns}
        original_ids = {t.turn_id for t in turns}
        assert all_turn_ids == original_ids

    def test_no_turn_duplication(self, chunks):
        all_ids = [t.turn_id for chunk in chunks for t in chunk.turns]
        assert len(all_ids) == len(set(all_ids))

    def test_chunk_has_embedding(self, chunks):
        for chunk in chunks:
            assert isinstance(chunk.embedding, np.ndarray)
            assert chunk.embedding.shape == (384,)  # all-MiniLM-L6-v2 dim

    def test_embedding_normalized(self, chunks):
        for chunk in chunks:
            norm = np.linalg.norm(chunk.embedding)
            assert abs(norm - 1.0) < 1e-5

    def test_topic_labels_non_empty(self, chunks):
        for chunk in chunks:
            assert chunk.topic_label.strip() != ""

    def test_min_turns_respected(self, chunks, config):
        for chunk in chunks:
            assert len(chunk.turns) >= config.chunk_min_turns

    def test_max_turns_respected(self, chunks, config):
        for chunk in chunks:
            assert len(chunk.turns) <= config.chunk_max_turns

    def test_semantic_separation(self, chunks):
        # The sample conversation has 3 clearly different topics:
        # neural networks, gradient descent, data visualization
        # We expect at least 2 chunks (probably 3)
        assert len(chunks) >= 2

    def test_embed_text_query(self, chunker):
        emb = chunker.embed_text("what is a neural network")
        assert emb.shape == (384,)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-5


# ── Summarizer tests ──────────────────────────────────────────────────────────

class TestSummarizer:
    def test_micro_short_turn(self, turns, config):
        s = Summarizer(config)
        short_turn = turns[0]
        result = s.micro(short_turn)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_micro_long_turn(self, config):
        s = Summarizer(config)
        long_content = " ".join(["This is a test sentence with meaningful content."] * 20)
        from cce_core.ingestion.segmenter import Turn
        from datetime import datetime, timezone
        t = Turn(
            turn_id="test", role="user", content=long_content,
            token_count=count(long_content),
            timestamp=datetime.now(timezone.utc), index=0
        )
        result = s.micro(t)
        assert count(result) <= config.micro_max_tokens + 10  # small tolerance

    def test_meso_summary(self, chunks, config):
        s = Summarizer(config)
        for chunk in chunks:
            result = s.meso(chunk)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_annotate_chunk(self, chunks, config):
        s = Summarizer(config)
        chunk = chunks[0]
        s.annotate_chunk(chunk)
        assert len(chunk.micro_summaries) == len(chunk.turns)
        assert chunk.meso_summary != ""

    def test_macro_summary(self, chunks, config):
        s = Summarizer(config)
        s.annotate_chunks(chunks)
        result = s.macro(chunks)
        assert isinstance(result, str)
        assert len(result) > 0


# ── Merger tests ──────────────────────────────────────────────────────────────

class TestMerger:
    def test_merge_produces_nodes(self, chunks, config):
        s = Summarizer(config)
        s.annotate_chunks(chunks)
        m = Merger(config)
        nodes = m.merge(chunks)
        assert len(nodes) == len(chunks)

    def test_node_embedding_shape(self, chunks, config):
        s = Summarizer(config)
        s.annotate_chunks(chunks)
        m = Merger(config)
        nodes = m.merge(chunks)
        for node in nodes:
            assert node.embedding.shape == (384,)

    def test_node_serialization_roundtrip(self, chunks, config):
        from cce_core.compression.merger import MemoryNode
        s = Summarizer(config)
        s.annotate_chunks(chunks)
        m = Merger(config)
        nodes = m.merge(chunks)
        d = nodes[0].to_dict()
        node2 = MemoryNode.from_dict(d)
        assert node2.node_id == nodes[0].node_id
        assert node2.topic_label == nodes[0].topic_label
        assert np.allclose(node2.embedding, nodes[0].embedding)


# ── Engine integration test ───────────────────────────────────────────────────

class TestCCEEngine:
    def test_ingest_and_compress(self, config):
        engine = CCEEngine(config)
        turns, nodes = engine.ingest_and_compress(
            SAMPLE_CONVERSATION, session_id="engine-test"
        )
        assert len(turns) == len(SAMPLE_CONVERSATION)
        assert len(nodes) >= 1

    def test_compression_stats(self, config):
        engine = CCEEngine(config)
        turns, nodes = engine.ingest_and_compress(SAMPLE_CONVERSATION)
        stats = engine.compression_stats(turns, nodes)
        assert stats["compression_ratio"] > 1.0
        assert stats["total_turns"] == len(SAMPLE_CONVERSATION)
        print(f"\n  Compression ratio: {stats['compression_ratio']}x")
        print(f"  Topics: {stats['topics']}")

    def test_macro_summary(self, config):
        engine = CCEEngine(config)
        _, nodes = engine.ingest_and_compress(SAMPLE_CONVERSATION)
        macro = engine.macro_summary(nodes)
        assert isinstance(macro, str)
        assert len(macro) > 0