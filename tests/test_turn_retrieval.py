"""Turn-level retrieval: one specific fact inside a long, mixed topic must still be found."""

import numpy as np
import pytest

from cce_core.compression.merger import decode_turn_vecs, encode_turn_vecs
from cce_core.config import CCEConfig
from cce_core.engine import CCEEngine
from cce_core.retrieval.retriever import node_similarity

FILLER = [
    ("Tell me about the history of bread making.", "Bread making goes back thousands of years, from flat griddle breads to sourdough starters kept alive for generations. Bakers learned that fermentation improves flavour and keeps loaves fresh for longer."),
    ("How do bakeries manage their ovens?", "Commercial ovens are preheated before dawn, loaded in batches and steamed at the start of each bake so the crust expands properly before it sets."),
    ("What flour is best for pizza dough?", "A high protein flour such as bread flour gives a chewy base, while finely milled tipo 00 gives a softer, blistered crust at high temperature."),
    ("Why does dough need to rest?", "Resting lets gluten relax and enzymes break down starch, which makes the dough easier to shape and improves the final texture."),
]
FACT = ("By the way, our staging database password rotates every 45 days.", "Got it, I will keep that in mind.")


def convo():
    msgs = []
    for i, (u, a) in enumerate(FILLER * 2):
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        if i == 3:
            msgs += [{"role": "user", "content": FACT[0]}, {"role": "assistant", "content": FACT[1]}]
    return msgs


def test_turn_vectors_round_trip():
    v = np.random.default_rng(0).normal(size=(5, 384)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    out = decode_turn_vecs({"turn_vecs": encode_turn_vecs(v)})
    assert out.shape == (5, 384) and np.allclose(out, v, atol=1e-2)
    assert decode_turn_vecs({}) is None


@pytest.mark.parametrize("embedder", ["neural", "hash"])
def test_needle_in_a_mixed_topic_is_retrieved(tmp_path, embedder):
    engine = CCEEngine(CCEConfig(embedder=embedder, base_dir=tmp_path, hot_tier_max_turns=2, retrieval_top_k=2,
                                 chunk_max_turns=40, hash_chunk_similarity_threshold=0.0, chunk_similarity_threshold=0.0))
    res = engine.create_stateless_processor().process(convo(), "how many days until the staging database password rotates?")
    context = "\n".join(m["content"] for m in res.payload.to_messages())
    assert "45 days" in context


def test_node_similarity_prefers_the_matching_message(tmp_path):
    engine = CCEEngine(CCEConfig(embedder="hash", base_dir=tmp_path, hash_chunk_similarity_threshold=0.0, chunk_max_turns=40))
    turns, nodes = engine.ingest_and_compress(convo())
    q = engine.chunker.embed_text("staging database password rotates")
    scored = [(node_similarity(n, q), n) for n in nodes]
    (score, text), node = max(scored, key=lambda x: x[0][0])
    assert "45 days" in text
    assert score > 0.3


def test_turn_vectors_survive_the_warm_tier(tmp_path):
    engine = CCEEngine(CCEConfig(embedder="hash", base_dir=tmp_path, hot_tier_max_turns=2))
    store, _, nodes = engine.full_pipeline(convo(), "persist")
    store.close()
    reopened = engine.open_session("persist")
    stored = reopened.warm.get_all(session_id="persist")
    assert stored and all(decode_turn_vecs(n.metadata) is not None for n in stored)
    reopened.close()


def test_list_spread_over_several_messages_is_all_present(tmp_path):
    """Answers built up turn by turn: every item must reach the context, not just one quote per topic."""
    items = ["register the domain name", "publish the privacy policy", "run the payment test", "prepare the press kit"]
    msgs = []
    for i, (u, a) in enumerate(FILLER * 2):
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        if i < len(items):
            msgs += [{"role": "user", "content": f"For the launch checklist, we also need to {items[i]}."},
                     {"role": "assistant", "content": "Got it, I will keep that in mind."}]
    engine = CCEEngine(CCEConfig(embedder="hash", base_dir=tmp_path, hot_tier_max_turns=2, retrieval_top_k=5,
                                 meso_max_tokens=60, hash_chunk_similarity_threshold=0.0, chunk_max_turns=40))
    res = engine.create_stateless_processor().process(msgs, "What is everything on our launch checklist?")
    context = "\n".join(m["content"] for m in res.payload.to_messages()).lower()
    for word in ("domain", "privacy", "payment", "press"):
        assert word in context, word
