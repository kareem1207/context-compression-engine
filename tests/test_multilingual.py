"""The ONNX multilingual embedder (no PyTorch): loading, choice of embedder, and search in es / hi / zh."""

import sys
from pathlib import Path

import numpy as np
import pytest

import cce_core.embedding as emb
from cce_core.config import CCEConfig
from cce_core.engine import CCEEngine, EmbedderMismatchError
from cce_core.text import length_in_words, split_sentences

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
import dataset_multilingual as ds  # noqa: E402


@pytest.fixture(scope="module")
def onnx_embedder():
    if not emb.onnx_available():
        pytest.skip("the onnx extra is not installed")
    try:
        return emb.load_embedder(CCEConfig(embedder="onnx"))
    except Exception as exc:   # model not cached and no network
        pytest.skip(f"ONNX model not available: {exc}")


def present(text: str, key: str) -> bool:
    return key.lower() in text.lower()


# ── Data and text helpers (no model needed) ───────────────────────────────────

@pytest.mark.parametrize("lang", list(ds.LANGS))
def test_every_key_is_in_its_own_fact(lang):
    for statement, _question, key in ds.LANGS[lang]["facts"]:
        assert key.lower() in statement.lower()


@pytest.mark.parametrize("lang", list(ds.LANGS))
def test_conversations_hold_every_fact_once(lang):
    data = ds.build(lang, 3)
    users = " ".join(m["content"] for m in data["messages"] if m["role"] == "user").lower()
    assert all(q["keys"][0].lower() in users for q in data["questions"])
    assert ds.build(lang, 3) == data and ds.build(lang, 4) != data        # deterministic per seed


def test_sentences_split_after_cjk_and_devanagari_punctuation():
    assert split_sentences("我姐姐的婚礼在里斯本举行。她很高兴！你呢？ Next one. Done") == \
        ["我姐姐的婚礼在里斯本举行。", "她很高兴！", "你呢？", "Next one.", "Done"]
    assert split_sentences("मेरी बहन की शादी लिस्बन में है। वह खुश है।") == ["मेरी बहन की शादी लिस्बन में है।", "वह खुश है।"]


def test_chinese_text_counts_as_long_by_characters():
    assert length_in_words("我姐姐的婚礼下个月在里斯本举行。" * 8) > 40
    assert length_in_words("short text") == 2


# ── Choosing an embedder ──────────────────────────────────────────────────────

class _Stub:
    def __init__(self, name):
        self.name = name


def test_auto_prefers_neural_then_onnx_then_hash(monkeypatch):
    monkeypatch.setattr(emb, "NeuralEmbedder", lambda model: _Stub("neural"))
    monkeypatch.setattr(emb, "OnnxEmbedder", lambda repo, max_length: _Stub("onnx"))
    cfg = CCEConfig(embedder="auto")
    for neural, onnx, expected in ((True, True, "neural"), (False, True, "onnx"), (False, False, "hash"), (True, False, "neural")):
        monkeypatch.setattr(emb, "neural_available", lambda v=neural: v)
        monkeypatch.setattr(emb, "onnx_available", lambda v=onnx: v)
        assert emb.load_embedder(cfg).name == expected


def test_explicit_onnx_without_the_extra_explains_how_to_install(monkeypatch):
    monkeypatch.setattr(emb, "onnx_available", lambda: False)
    with pytest.raises(ImportError, match=r"context-compression-engine\[onnx\]"):
        emb.load_embedder(CCEConfig(embedder="onnx"))
    with pytest.raises(ValueError, match="onnx"):
        emb.load_embedder(CCEConfig(embedder="nope"))


# ── The embedder ──────────────────────────────────────────────────────────────

def test_onnx_embedder_gives_unit_vectors_and_matches_across_languages(onnx_embedder):
    assert onnx_embedder.name == "onnx" and onnx_embedder.dim == 384
    v = onnx_embedder.encode(["My sister's wedding is in Lisbon.", "我姐姐的婚礼在里斯本举行。", "मेरी बहन की शादी लिस्बन में है।",
                              "El perro come croquetas todos los días."])
    assert v.shape == (4, 384) and np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)
    assert v[0] @ v[1] > v[0] @ v[3] and v[0] @ v[2] > v[0] @ v[3]
    single = onnx_embedder.encode("hola")
    assert single.shape == (384,) and onnx_embedder.encode([]).shape == (0, 384)
    assert np.allclose(onnx_embedder.encode(["x" * 5000])[0], onnx_embedder.encode(["x" * 5000])[0])     # truncates, deterministic


def test_blank_text_does_not_crash(onnx_embedder):
    assert onnx_embedder.encode(["", "   "]).shape == (2, 384)


# ── Search in other languages ─────────────────────────────────────────────────

@pytest.mark.parametrize("lang", ["es", "hi", "zh"])
def test_oneshot_finds_the_facts_in_each_language(tmp_path, onnx_embedder, lang):
    engine = CCEEngine(CCEConfig(embedder="onnx", base_dir=tmp_path, hot_tier_max_turns=4, retrieval_top_k=3))
    proc = engine.create_stateless_processor()
    found = total = 0
    for seed in range(3):
        data = ds.build(lang, seed)
        for q in data["questions"]:
            ctx = proc.process(data["messages"], q["q"]).payload.to_string()
            found += present(ctx, q["keys"][0])
            total += 1
    assert found / total >= 0.9, f"{lang}: {found}/{total}"


@pytest.mark.parametrize("lang", ["hi", "zh"])
def test_live_session_finds_the_facts_after_reopen(tmp_path, onnx_embedder, lang):
    cfg = CCEConfig(embedder="onnx", base_dir=tmp_path, hot_tier_max_turns=4, retrieval_top_k=3)
    data = ds.build(lang, 1)
    mgr = CCEEngine(cfg).create_session_manager()
    for m in data["messages"]:
        mgr.add_message("s", m)
    mgr.close("s")
    fresh = CCEEngine(cfg).create_session_manager()
    found = sum(present(fresh.build_context_payload("s", q["q"]).to_string(), q["keys"][0]) for q in data["questions"])
    assert found >= 3, f"{lang}: {found}/4"
    fresh.close_all(checkpoint=False)


def test_embedder_is_recorded_and_mixing_is_refused(tmp_path, onnx_embedder):
    cfg = CCEConfig(embedder="onnx", base_dir=tmp_path)
    engine = CCEEngine(cfg)
    engine.open_session("a").close()
    assert engine.embedder_identity() == {"name": "onnx", "dim": 384, "model": cfg.onnx_model}
    with pytest.raises(EmbedderMismatchError):
        CCEEngine(CCEConfig(embedder="hash", base_dir=tmp_path)).open_session("a")
