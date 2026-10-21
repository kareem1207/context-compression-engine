"""Overview questions ("what did we talk about?"): every topic, in order, with lines that name it."""

import numpy as np
import pytest

from cce_core.compression.merger import merge_nodes
from cce_core.config import CCEConfig
from cce_core.engine import CCEEngine
from cce_core.ingestion import tokenizer
from cce_core.retrieval.retriever import fill_overviews, overview_lines

from tests.test_live_sessions import make_config

TOPICS = [
    ("billing", "How do I verify the Stripe webhook signature for invoices?",
     "Verify the signature header against the raw body. Stripe retries failed deliveries for three days. "
     "Store each event identifier so a replay is ignored."),
    ("hiking", "Which trails near Chamonix suit a first alpine hike?",
     "Start with the Lac Blanc loop. Bring waterproof boots because the snow line sits low. "
     "Check the mountain weather bulletin the evening before."),
    ("pandas", "How can I merge two dataframes on a composite key in pandas?",
     "Use merge with a list of columns. Pass validate to catch duplicate keys early. "
     "A groupby after the merge keeps the totals consistent."),
    ("sourdough", "What hydration should my sourdough starter have?",
     "Seventy percent hydration is a good default. Feed the starter twice a day at room temperature. "
     "Discard half before each feeding to keep it lively."),
]


def chat(rounds: int = 3) -> list[dict]:
    msgs = []
    for _, question, answer in TOPICS:
        for r in range(rounds):
            msgs += [{"role": "user", "content": question if r == 0 else f"{question} Tell me more, round {r}."},
                     {"role": "assistant", "content": answer}]
    return msgs


QUESTION = "What did we talk about in this conversation?"
MARKERS = ["webhook", "alpine", "composite", "starter"]


def cfg(tmp_path, **kw) -> CCEConfig:
    return make_config(tmp_path, hot_tier_max_turns=4, **kw)


def test_oneshot_overview_names_every_topic_in_order(tmp_path):
    res = CCEEngine(cfg(tmp_path)).create_stateless_processor().process(chat(), QUESTION)
    text = "\n".join(res.payload.past_blocks).lower()
    positions = [text.find(m) for m in MARKERS]
    assert all(p >= 0 for p in positions), positions
    assert positions == sorted(positions)                 # conversation order


def test_live_overview_after_reopen(tmp_path):
    c = cfg(tmp_path)
    mgr = CCEEngine(c).create_session_manager()
    for m in chat():
        mgr.add_message("s", m)
    mgr.close("s")
    text = CCEEngine(c).create_session_manager().build_context_payload("s", QUESTION).to_string().lower()
    assert all(m in text for m in MARKERS)


def test_overview_stays_within_the_token_cap_with_many_topics(tmp_path):
    msgs = []
    for t in range(40):
        words = f"subject{t}alpha subject{t}beta subject{t}gamma"
        for r in range(2):
            msgs += [{"role": "user", "content": f"question {r} about {words} and more {words}."},
                     {"role": "assistant", "content": f"answer {r} about {words}. " + "Detail sentence number one is here. " * 6}]
    c = cfg(tmp_path, context_max_tokens=900)
    res = CCEEngine(c).create_stateless_processor().process(msgs, QUESTION)
    assert tokenizer.count_messages(res.payload.to_messages()) <= 900
    assert 1 < len(res.payload.past_blocks) <= c.overview_max_topics
    topics = [int(b.split("turns ")[1].split("-")[0]) for b in res.payload.past_blocks]
    assert topics == sorted(topics)


def _node(tmp_path):
    engine = CCEEngine(cfg(tmp_path))
    turns = engine.ingest(chat(rounds=2)[:6])
    return engine, turns, engine.compress(turns)[0]


def test_first_line_is_the_first_user_message_and_the_budget_is_kept(tmp_path):
    _, turns, node = _node(tmp_path)
    by = {t.index: t.content for t in turns}
    raw = lambda n, pos: {i: by[n.turn_start + i] for i in pos}
    lines = overview_lines(node, raw, budget_tokens=60, doc_freq={}, n_notes=1)
    assert lines[0].startswith("How do I verify the Stripe webhook")
    assert sum(tokenizer.count(l) for l in lines) <= 60 + 8
    assert len(set(lines)) == len(lines)


def test_later_lines_add_new_terms_rather_than_repeat(tmp_path):
    _, turns, node = _node(tmp_path)
    by = {t.index: t.content for t in turns}
    raw = lambda n, pos: {i: by[n.turn_start + i] for i in pos}
    lines = overview_lines(node, raw, budget_tokens=120, doc_freq={}, n_notes=1)
    joined = " ".join(lines).lower()
    assert "retries failed deliveries" in joined or "replay" in joined       # something beyond the question
    assert joined.count("verify the signature header") <= 1


def test_budget_is_shared_evenly_between_topics(tmp_path):
    engine = CCEEngine(cfg(tmp_path))
    res = engine.create_stateless_processor().process(chat(), QUESTION)
    results = [type("R", (), {"node": n, "overview_lines": []})() for n in res.nodes]
    fill_overviews(results, None, engine.config)
    sizes = [sum(tokenizer.count(l) for l in r.overview_lines) for r in results]
    share = engine.config.context_max_tokens * engine.config.overview_budget_share / len(results)
    assert all(s <= min(max(share, 30), 160) + 10 for s in sizes)


def test_notes_from_before_roles_and_term_counts_still_work(tmp_path):
    engine, turns, node = _node(tmp_path)
    for key in ("roles", "tf"):
        node.metadata.pop(key, None)
    lines = overview_lines(node, None, budget_tokens=80, doc_freq={}, n_notes=1)
    assert lines and all(isinstance(l, str) and l for l in lines)


def test_merged_notes_keep_roles(tmp_path):
    engine = CCEEngine(cfg(tmp_path))
    turns = engine.ingest(chat(rounds=2))
    a, b = engine.compress(turns[:4])[0], engine.compress(turns[4:8])[0]
    merged = merge_nodes(a, b, engine.config)
    assert merged.metadata["roles"] == a.metadata["roles"] + b.metadata["roles"]
    assert len(merged.metadata["roles"]) == len(merged.micro_summaries)
