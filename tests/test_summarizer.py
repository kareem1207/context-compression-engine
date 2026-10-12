

# ── Detail-preserving extractive summaries ───────────────────────────────────

def _turn(i, role, content):
    from cce_core.ingestion.segmenter import Turn
    from datetime import datetime, timezone
    from cce_core.ingestion import tokenizer
    return Turn(turn_id=str(i), role=role, content=content, token_count=tokenizer.count(content),
                timestamp=datetime.now(timezone.utc), index=i)


def test_short_facts_survive_a_tight_summary_budget():
    from cce_core.compression.summarizer import _extractive_summarize_turns
    filler = ("The general approach here is to keep things simple and consistent across the whole project. " * 4).strip()
    turns = [
        _turn(0, "user", "How should I structure the project?"),
        _turn(1, "assistant", filler),
        _turn(2, "user", "By the way, our production deploy window is Tuesday at 3 AM."),
        _turn(3, "assistant", "Got it, I will keep that in mind."),
        _turn(4, "user", "What else matters?"),
        _turn(5, "assistant", filler),
    ]
    summary = _extractive_summarize_turns(turns, max_tokens=45)
    assert "Tuesday at 3 AM" in summary
    assert "Got it" not in summary          # acknowledgements are dropped
    assert "What else matters?" not in summary


def test_summary_keeps_conversation_order_and_budget():
    from cce_core.compression.summarizer import _extractive_summarize_turns
    from cce_core.ingestion import tokenizer
    turns = [_turn(i, "assistant", f"Point number {i} has a distinct detail about item {i * 7}.") for i in range(10)]
    out = _extractive_summarize_turns(turns, max_tokens=40)
    assert tokenizer.count(out) <= 40
    nums = [int(n) for n in __import__("re").findall(r"number (\d+)", out)]
    assert nums == sorted(nums)


def test_empty_and_single_sentence_inputs():
    from cce_core.compression.summarizer import _extractive_summarize_turns
    assert _extractive_summarize_turns([], 50) == ""
    assert _extractive_summarize_turns([_turn(0, "user", "Only one sentence.")], 50) == "Only one sentence."
