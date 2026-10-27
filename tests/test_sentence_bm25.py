"""
G2: one vector per sentence for long messages (found and quoted on their own).
G3: BM25 over the notes' text, fused with the dense ranking, for identifiers, codes and names.
"""

import uuid
import zlib
from datetime import datetime, timezone

import numpy as np
import pytest

from cce_core.compression.merger import MemoryNode, decode_sentence_vecs, encode_turn_vecs, merge_nodes
from cce_core.config import CCEConfig
from cce_core.engine import CCEEngine
from cce_core.retrieval.bm25 import Bm25, fuse
from cce_core.retrieval.retriever import node_evidence, query_terms, sentence_similarity
from cce_core.text import distinctive_terms, term_counts, terms

from tests.test_live_sessions import make_config


class ScrambleEmbedder:
    """Vectors that carry no meaning: the same text always gives the same random vector. Dense search is blind."""

    name = "hash"
    dim = 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.random.default_rng(zlib.crc32(text.encode())).normal(size=self.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def encode(self, texts, **kw):
        if isinstance(texts, str):
            return self._vec(texts)
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)


def scrambled_engine(tmp_path, **kw) -> CCEEngine:
    cfg = CCEConfig(embedder="hash", hash_dim=64, base_dir=tmp_path / "data", hot_tier_max_turns=4,
                    retrieval_top_k=2, **kw)
    engine = CCEEngine(cfg)
    engine.chunker._model = ScrambleEmbedder()
    return engine


TOPIC_WORDS = [
    "billing invoices stripe webhook plans", "hiking trails mountains boots weather",
    "python pandas dataframe merge groupby", "kitchen oven sourdough baking temperature",
    "garden tomatoes soil watering compost", "guitar chords scales practice tempo",
]


def chat_with(fact_topic: int, fact: str, question_filler: str = "") -> list[dict]:
    msgs = []
    for t, words in enumerate(TOPIC_WORDS):
        for r in range(3):
            extra = f" {fact}" if (t, r) == (fact_topic, 1) else ""
            msgs += [
                {"role": "user", "content": f"question {r} about {words} and more {words}.{extra}"},
                {"role": "assistant", "content": f"answer {r} about {words} with details {words}"},
            ]
    return msgs


# ── Terms ─────────────────────────────────────────────────────────────────────

def test_terms_keep_whole_identifiers_and_their_pieces():
    t = terms("Error ERR-4021-X on host db_primary 3f2a9c1e-8b7d-4e1a, the retry")
    assert "err-4021-x" in t and "4021" in t and "db_primary" in t and "3f2a9c1e-8b7d-4e1a" in t
    assert "the" not in t and "x" not in t        # stopwords and single characters are dropped


@pytest.mark.parametrize("question, expected", [
    ("What does ERR-4021-X mean?", ["err-4021-x", "4021"]),
    ("Which host is 3f2a9c1e-8b7d-4e1a on?", ["3f2a9c1e-8b7d-4e1a", "3f2a9c1e", "8b7d", "4e1a"]),
    ("What did Gertrude say about the starter?", ["gertrude"]),
    ("Is the SLA 99.9 percent?", ["sla", "99.9"]),
    ("Which person should be paged first if production breaks?", []),     # ordinary words: dense search only
    ("What do we call the upcoming application inside the company?", []),
])
def test_distinctive_terms(question, expected):
    got = distinctive_terms(question)
    assert all(e in got for e in expected)
    if not expected:
        assert got == []


def test_term_counts_are_capped():
    text = " ".join(f"word{i} " * (i % 5 + 1) for i in range(2000))
    assert len(term_counts(text, max_terms=100)) == 100


# ── BM25 and fusion ───────────────────────────────────────────────────────────

def test_bm25_ranks_the_document_with_the_rare_term_first():
    bm = Bm25()
    for text in ["common words everywhere", "common words and err-4021-x here", "common words again common"]:
        bm.add(term_counts(text))
    s = bm.score(["err-4021-x"])
    assert int(np.argmax(s)) == 1 and (s[[0, 2]] == 0).all()
    assert bm.score(["absent-term"]).sum() == 0


def test_fusion_promotes_a_lexical_hit_and_leaves_other_notes_alone():
    dense = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float32)
    lexical = np.array([0, 0, 0, 0, 0, 7.0], dtype=np.float32)
    fused = fuse(dense, (lexical, lexical > 0, 1.0))
    assert int(np.argmax(fused)) == 5                         # last by dense, first by dense + exact match
    none = fuse(dense, (np.zeros(6, dtype=np.float32), np.zeros(6, dtype=bool), 1.0))
    assert list(np.argsort(-none)) == [0, 1, 2, 3, 4, 5]      # no lexical hit: the dense order is untouched


# ── G3 end to end: rare tokens that dense search misses ───────────────────────

@pytest.mark.parametrize("fact, question, needle", [
    ("The failing job id is 3f2a9c1e-8b7d-4e1a-9c2d-5a6b7c8d9e0f.", "Which job is 3f2a9c1e-8b7d-4e1a-9c2d-5a6b7c8d9e0f?", "3f2a9c1e-8b7d-4e1a-9c2d-5a6b7c8d9e0f"),
    ("The checkout service fails with ERR-4021-X every night.", "What does ERR-4021-X mean?", "ERR-4021-X"),
    ("The new maintainer is Gertrude Hollowell from Tallinn.", "What did Hollowell say?", "Hollowell"),
])
def test_oneshot_finds_identifiers_that_dense_search_cannot(tmp_path, fact, question, needle):
    msgs = chat_with(3, fact)
    blind = scrambled_engine(tmp_path / "off", lexical_weight=0.0).create_stateless_processor()
    assert needle not in blind.process(msgs, question).payload.to_string()      # dense alone: wrong topic

    with_bm25 = scrambled_engine(tmp_path / "on").create_stateless_processor()
    assert needle in with_bm25.process(msgs, question).payload.to_string()


def test_live_session_finds_identifiers_that_dense_search_cannot(tmp_path):
    fact = "The checkout service fails with ERR-4021-X every night."
    msgs = chat_with(3, fact)
    for weight, should_find in ((0.0, False), (1.0, True)):
        engine = scrambled_engine(tmp_path / f"w{weight}", lexical_weight=weight)
        mgr = engine.create_session_manager()
        for m in msgs:
            mgr.add_message("s", m)
        mgr.close("s")                                    # everything is in notes, nothing verbatim
        payload = CCEEngine.__new__(CCEEngine)            # reopen with a fresh engine, same scrambled embedder
        fresh = scrambled_engine(tmp_path / f"w{weight}", lexical_weight=weight)
        text = fresh.create_session_manager().build_context_payload("s", "What does ERR-4021-X mean?").to_string()
        assert ("ERR-4021-X" in text) is should_find


def test_paraphrased_questions_do_not_use_bm25():
    cfg = CCEConfig(embedder="hash")
    assert query_terms("What do we call the upcoming application inside the company?", cfg) == []
    assert query_terms("What does ERR-4021-X mean?", cfg) == ["err-4021-x", "4021"]
    assert query_terms("anything", CCEConfig(embedder="hash", lexical_weight=0.0)) == []
    assert "application" in query_terms("upcoming application", CCEConfig(embedder="hash", lexical_terms="all"))


# ── G2: sentence vectors ──────────────────────────────────────────────────────

FILLER = "The weather this season has been quite unpredictable across many regions of the country. "
BURIED = "The vault code word is zephyrquill and nobody else knows it."


def long_message(fact: str, at: int, sentences: int = 8) -> str:
    parts = [f"{FILLER.strip()} Day {i} brought another round of changes." for i in range(sentences)]
    parts.insert(at, fact)
    return " ".join(parts)


def test_long_messages_get_sentence_vectors_and_short_ones_do_not(tmp_path):
    engine = CCEEngine(make_config(tmp_path))
    msgs = [
        {"role": "user", "content": "short question about the vault"},
        {"role": "assistant", "content": long_message(BURIED, 3)},
        {"role": "user", "content": "another short one"},
        {"role": "assistant", "content": "also short"},
    ]
    nodes = engine.compress(engine.ingest(msgs))
    stored = decode_sentence_vecs(nodes[0].metadata)
    assert stored is not None
    vecs, smap = stored
    assert vecs.shape[0] == len(smap) >= 8 and set(i for i, _ in smap) == {1}      # only the long message (position 1)
    assert [k for _, k in smap] == list(range(len(smap)))
    assert nodes[0].metadata["sent_vecs"]["dim"] == vecs.shape[1]

    off = CCEEngine(make_config(tmp_path / "off", sentence_index_min_words=0))
    assert decode_sentence_vecs(off.compress(off.ingest(msgs))[0].metadata) is None


def test_a_buried_sentence_outscores_the_message_it_sits_in(tmp_path):
    engine = CCEEngine(make_config(tmp_path))
    msgs = [{"role": "user", "content": "tell me"}, {"role": "assistant", "content": long_message(BURIED, 4)}] * 2
    node = engine.compress(engine.ingest(msgs))[0]
    q = engine.chunker.embed_text("what is the vault code word")
    from cce_core.compression.merger import decode_turn_vecs
    assert sentence_similarity(node, q) > float((decode_turn_vecs(node.metadata) @ q).max())


def test_evidence_comes_from_the_stored_sentences_without_embedding_at_query_time(tmp_path):
    engine = CCEEngine(make_config(tmp_path))
    msgs = [{"role": "user", "content": "tell me"}, {"role": "assistant", "content": long_message(BURIED, 5)}]
    turns = engine.ingest(msgs)
    node = engine.compress(turns)[0]
    by_index = {t.index: t.content for t in turns}
    calls = []

    def embed_many(texts):
        calls.append(list(texts))
        return engine.chunker.embed_texts(texts)

    q = engine.chunker.embed_text("what is the vault code word")
    quotes = node_evidence(node, q, raw_fn=lambda n, pos: {i: by_index[n.turn_start + i] for i in pos},
                           embed_many=embed_many)
    assert any("zephyrquill" in quote for quote in quotes)
    assert not any(len(c) > 3 for c in calls)        # the long message's sentences were not embedded again


@pytest.mark.parametrize("position", [0, 4, 9])
def test_fact_buried_anywhere_in_a_long_message_reaches_the_context(tmp_path, position):
    msgs = []
    for t, words in enumerate(TOPIC_WORDS):
        for r in range(3):
            body = f"answer {r} about {words} with details {words}"
            if (t, r) == (2, 1):
                body = long_message(BURIED, position) + f" Also {words}."
            msgs += [{"role": "user", "content": f"question {r} about {words} and more {words}"},
                     {"role": "assistant", "content": body}]
    proc = CCEEngine(make_config(tmp_path, retrieval_top_k=3)).create_stateless_processor()
    assert "zephyrquill" in proc.process(msgs, "what is the vault code word").payload.to_string()


def test_merging_notes_keeps_sentence_vectors_and_terms_aligned(tmp_path):
    engine = CCEEngine(make_config(tmp_path))
    nodes = []
    for k, fact in enumerate(["alpha", "omega"]):
        msgs = [{"role": "user", "content": f"short {fact}"}, {"role": "assistant", "content": long_message(f"The {fact} marker is set.", 2 + k)}]
        turns = engine.ingest(msgs)
        for t in turns:
            t.index += 2 * k
        node = engine.compress(turns)[0]
        nodes.append(node)
    a, b = nodes
    merged = merge_nodes(a, b, engine.config)
    vecs, smap = decode_sentence_vecs(merged.metadata)
    na = len(decode_sentence_vecs(a.metadata)[1])
    assert len(vecs) == len(smap) == na + len(decode_sentence_vecs(b.metadata)[1])
    assert all(p >= len(a.micro_summaries) for p, _ in smap[na:])            # b's positions are shifted past a's
    assert merged.metadata["tf"].get("alpha", 0) >= 1 and merged.metadata["tf"].get("omega", 0) >= 1
