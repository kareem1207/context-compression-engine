"""Token counting: CJK text, code and JSON, offline behaviour, and the hard context cap."""

import json
import uuid
from datetime import datetime, timezone

import numpy as np
import pytest

from cce_core.compression.merger import MemoryNode
from cce_core.config import CCEConfig
from cce_core.ingestion import tokenizer
from cce_core.ingestion.segmenter import Turn
from cce_core.retrieval.context_builder import ContextBuilder
from cce_core.retrieval.retriever import RetrievalResult

CJK = "机器学习是人工智能的一个分支，它使计算机能够从数据中学习，而无需明确编程。" * 6
JSON_TEXT = json.dumps(
    {"id": 12345, "name": "Alice", "tags": ["a", "b", "c"], "nested": {"x": 1.5, "y": None, "z": [1, 2, 3, 4]}, "ok": True},
    separators=(",", ":"),
) * 4
CODE = (
    "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n\n"
    "for i in range(10):\n    print(f\"fib({i}) = {fib(i)}\")\n"
)
PROSE = (
    "The quick brown fox jumps over the lazy dog while the sun sets slowly behind the distant "
    "mountains, and everyone watching agrees that it was a remarkably peaceful evening."
)

# Token counts of the same strings with tiktoken's cl100k_base (hardcoded: CI does not need tiktoken).
CL100K = {"cjk": 240, "json": 180, "code": 51, "prose": 31}
SAMPLES = {"cjk": CJK, "json": JSON_TEXT, "code": CODE, "prose": PROSE}


@pytest.fixture(autouse=True)
def no_tiktoken(monkeypatch):
    """Run every test on the offline estimate, whether or not tiktoken is installed."""
    monkeypatch.setattr(tokenizer, "_encoder", lambda: None)


@pytest.mark.parametrize("name", list(SAMPLES))
def test_estimate_is_close_to_tiktoken_and_does_not_undercount_much(name):
    est = tokenizer.estimate(SAMPLES[name])
    real = CL100K[name]
    assert 0.85 * real <= est <= 1.5 * real, f"{name}: estimated {est}, cl100k_base says {real}"


def test_cjk_and_minified_json_are_no_longer_one_token():
    # Before: words / 0.75 counted both of these as a single token.
    assert tokenizer.count("机器学习" * 100) >= 300
    assert tokenizer.count(JSON_TEXT) >= 100
    assert tokenizer.count(CODE) >= 40


def test_plain_english_is_unchanged():
    assert tokenizer.count(" ".join(["word"] * 100)) == 133
    assert tokenizer.count("") == 0 and tokenizer.count("   \n ") == 0
    assert tokenizer.count("hi") == 1


def test_count_messages_keeps_its_overhead():
    msgs = [{"role": "user", "content": CJK}, {"role": "assistant", "content": PROSE}]
    assert tokenizer.count_messages(msgs) == tokenizer.count(CJK) + tokenizer.count(PROSE) + 8


def test_tiktoken_is_only_used_when_its_vocabulary_is_already_cached(monkeypatch, tmp_path):
    """No network at import or call time: an empty cache dir means the estimate is used."""
    monkeypatch.undo()   # drop the autouse patch
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    tokenizer._encoder.cache_clear()
    try:
        assert tokenizer._encoder() is None
        assert tokenizer.count(PROSE) == tokenizer.estimate(PROSE)
    finally:
        tokenizer._encoder.cache_clear()


@pytest.mark.parametrize("name", list(SAMPLES))
def test_truncate_to_tokens_respects_the_counter(name):
    text = SAMPLES[name] * 5
    for limit in (1, 5, 40, 120):
        out = tokenizer.truncate_to_tokens(text, limit)
        assert tokenizer.count(out) <= limit, (name, limit)
        if limit >= 40:
            assert out.endswith("...") and out.rstrip(". ") in text
    assert tokenizer.truncate_to_tokens(text, 10**6) == text
    assert tokenizer.truncate_to_tokens(text, 0) == ""


# ── The hard cap in ContextBuilder ────────────────────────────────────────────

def _turn(i: int, content: str, role: str = "user") -> Turn:
    return Turn(str(uuid.uuid4()), role, content, tokenizer.count(content), datetime.now(timezone.utc), i)


def _result(summary: str, start: int = 0, end: int = 3) -> RetrievalResult:
    node = MemoryNode(
        node_id=str(uuid.uuid4()), chunk_id="c", session_id="s", embedding=np.zeros(4, dtype=np.float32),
        topic_label="topic", meso_summary=summary, micro_summaries=[], turn_start=start, turn_end=end,
        token_count=10, created_at=datetime.now(timezone.utc),
    )
    return RetrievalResult(node, 0.9, 0.0, 0.0, 0.9)


@pytest.mark.parametrize("name", list(SAMPLES))
@pytest.mark.parametrize("where", ["hot", "pending", "node", "all"])
def test_context_never_exceeds_the_cap(name, where):
    cap = 500
    cfg = CCEConfig(context_max_tokens=cap, hot_tier_max_turns=4)
    big = (SAMPLES[name] + "\n") * 40   # far over the cap on its own
    hot = [_turn(i, big if where in ("hot", "all") else "short message") for i in range(4)]
    pending = [_turn(10 + i, big if where in ("pending", "all") else "earlier message") for i in range(4)]
    results = [_result(big if where in ("node", "all") else "a summary", 0, 3), _result(big, 4, 8)]

    payload = ContextBuilder(cfg).build(results, hot, pending_turns=pending)

    assert payload.token_count <= cap
    # what the model actually receives, counted independently of payload.token_count
    assert tokenizer.count_messages(payload.to_messages()) <= cap
    assert tokenizer.count(payload.to_string()) <= cap
    assert payload.was_truncated


def test_runs_of_one_character_are_not_counted_per_character():
    """Regression for the lost recall questions: a template line of underscores counted ~1,700 tokens
    (tiktoken: ~160) and crowded the past context out of the payload."""
    template = 'Event Name: ' + "_" * 60 + " Event Date: " + "_" * 60 + " " + "-" * 80 + "\n" + "=" * 80
    assert tokenizer.estimate(template) < 40
    page = ("Catering Agreement This agreement is made on " + "_" * 400 + " between the caterer and the client. ") * 10
    assert tokenizer.estimate(page) < 1.5 * len(page.split()) / 0.75


def test_ordinary_english_chat_counts_as_it_always_did():
    chat = ("Sure! Here are a few ideas for your weekend trip: visit the old town, try the local food, "
            "and take the morning train so you arrive before the crowds. Let me know if you want more details.")
    assert tokenizer.estimate(chat) == round(len(chat.split()) / 0.75)
