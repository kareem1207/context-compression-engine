"""The hashing embedder and the choice between embedders."""

import numpy as np
import pytest

from cce_core.config import CCEConfig
from cce_core.embedding import HashingEmbedder, load_embedder
from cce_core.engine import CCEEngine

CONVO = (
    [{"role": "user" if i % 2 == 0 else "assistant", "content": c} for i, c in enumerate([
        "How do I verify webhook signatures from the payment provider?",
        "Verify the HMAC signature header against the raw webhook body before parsing anything.",
        "And what about replay attacks on those webhooks?",
        "Reject events with an old timestamp and store each event id so a webhook is never processed twice.",
        "I want a quick pasta dinner tonight with garlic.",
        "Cook spaghetti, warm sliced garlic in olive oil with chili, and toss with a splash of pasta water.",
        "Why does my pasta sauce taste bland?",
        "Salt the pasta water well, finish the sauce with starchy water and add lemon at the end.",
    ])]
)


def test_hash_embedder_is_normalised_and_deterministic():
    e = HashingEmbedder(256)
    a, b = e.encode("verify the webhook signature"), e.encode("verify the webhook signature")
    assert a.shape == (256,) and np.allclose(a, b)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-5
    assert e.encode(["one", "two words"]).shape == (2, 256)
    assert e.encode([]).shape == (0, 256)
    assert float(np.linalg.norm(e.encode(""))) == 0.0


def test_hash_embedder_prefers_related_text():
    e = HashingEmbedder()
    q = e.encode("webhook signature verification")
    assert float(q @ e.encode("verify the signature on each webhook")) > float(q @ e.encode("garlic pasta with olive oil"))


def test_explicit_modes():
    assert load_embedder(CCEConfig(embedder="hash")).name == "hash"
    assert load_embedder(CCEConfig(embedder="neural")).name == "neural"      # sentence-transformers is a dev dependency
    with pytest.raises(ValueError):
        load_embedder(CCEConfig(embedder="nope"))


def test_auto_falls_back_to_hash_without_sentence_transformers(monkeypatch):
    import cce_core.embedding as emb
    monkeypatch.setattr(emb, "neural_available", lambda: False)
    assert emb.load_embedder(CCEConfig(embedder="auto")).name == "hash"
    with pytest.raises(ImportError):
        emb.load_embedder(CCEConfig(embedder="neural"))


def test_whole_pipeline_runs_on_the_hash_embedder(tmp_path):
    engine = CCEEngine(CCEConfig(embedder="hash", base_dir=tmp_path, hot_tier_max_turns=2, retrieval_top_k=2))
    res = engine.create_stateless_processor().process(CONVO, "how do I verify webhook signatures?")
    assert res.stats["total_turns"] == len(CONVO)
    assert any("HMAC" in b for b in res.payload.past_blocks)       # the right topic was retrieved
    assert engine.chunker.model.name == "hash"


def test_mixing_embedders_gives_a_clear_error(tmp_path):
    cfg_hash = CCEConfig(embedder="hash", base_dir=tmp_path)
    engine = CCEEngine(cfg_hash)
    store, _, _ = engine.full_pipeline(CONVO, "mix")
    store.close()
    other = CCEEngine(CCEConfig(embedder="neural", base_dir=tmp_path))
    with pytest.raises(ValueError, match="dimensions"):     # refused when the session is opened
        other.open_session("mix")
