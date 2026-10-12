"""
Model-free test for OVERVIEW questions: "what did we talk about?"

Nothing in such a question matches any single message, so plain search has nothing to latch onto.
CCE keeps a summary per topic, which is what it should use here.

A conversation is six real dialogues (UltraChat) back to back, i.e. six topics. Each topic gets 3 distinctive
keywords (rare words from its first message). A topic counts as covered if the context mentions any of them.
Scored: topics covered out of six, and how often all six are covered.

Compared at the same token budget:
  cce        CCE's context for the question
  cce_macro  CCE's context plus its session summary (what the summarise-session tool returns)
  rag        plain RAG over raw messages (the best-matching turns), same budget

  uv run --group bench python benchmarks/retrieval_overview.py --conversations 20
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cce_core.config import CCEConfig  # noqa: E402
from cce_core.engine import CCEEngine  # noqa: E402
from cce_core.ingestion import tokenizer  # noqa: E402
import dataset_real  # noqa: E402
from retrieval_recall import rag_context  # noqa: E402

QUESTIONS = [
    "What did we talk about in this conversation?",
    "Can you give me a quick overview of everything we covered?",
    "Remind me of the main topics we went through.",
]
WORD = re.compile(r"[a-z]{5,}")


def keywords(first_message: str, df: Counter, n: int = 3) -> list[str]:
    words = {w for w in WORD.findall(first_message.lower())}
    return sorted(words, key=lambda w: (df[w], w))[:n]


def build(rng: random.Random, dialogues, df, topics: int = 6, exchanges: int = 3):
    picked, tries = [], 0
    while len(picked) < topics and tries < 200:
        tries += 1
        d = rng.choice(dialogues)
        if len(d) >= 2 * exchanges and len(d[0]["content"].split()) >= 8:
            picked.append(d)
    messages, topic_keys = [], []
    for d in picked:
        for i in range(0, 2 * exchanges, 2):
            u = dataset_real._clip(d[i]["content"], 40)
            a = dataset_real._clip(d[i + 1]["content"], 70)
            messages += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        topic_keys.append(keywords(d[0]["content"], df))
    return messages, topic_keys


def covered(context: str, topic_keys: list[list[str]]) -> int:
    text = context.lower()
    return sum(any(k in text for k in keys) for keys in topic_keys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", type=int, default=20)
    ap.add_argument("--seed", type=int, default=900)
    ap.add_argument("--hot", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--meso", type=int, default=150)
    args = ap.parse_args()

    dialogues = dataset_real.load_dialogues()
    df = Counter(w for d in dialogues for w in set(WORD.findall(d[0]["content"].lower())))
    engine = CCEEngine(CCEConfig(base_dir=Path(tempfile.mkdtemp()), hot_tier_max_turns=args.hot,
                                 retrieval_top_k=args.top_k, meso_max_tokens=args.meso))
    proc = engine.create_stateless_processor()

    methods = ["cce", "cce_macro", "rag", "rag_macro_budget"]
    total = {m: 0 for m in methods}
    full = {m: 0 for m in methods}
    tokens = {m: 0 for m in methods}
    n = 0
    for c in range(args.conversations):
        rng = random.Random(args.seed + c)
        messages, topic_keys = build(rng, dialogues, df)
        for q in QUESTIONS:
            res = proc.process(messages, q)
            ctx = "\n".join(m["content"] for m in res.payload.to_messages())
            macro = engine.macro_summary(engine.compress(engine.ingest(messages)))
            budget = tokenizer.count(ctx) + 4 * len(res.payload.to_messages())
            ctx_macro = ctx + "\n\nSession summary:\n" + macro
            budget_macro = budget + tokenizer.count(macro) + 8
            texts = {
                "cce": ctx,
                "cce_macro": ctx_macro,
                "rag": rag_context(engine, messages, q, args.hot, budget),
                "rag_macro_budget": rag_context(engine, messages, q, args.hot, budget_macro),
            }
            n += 1
            for m, t in texts.items():
                k = covered(t, topic_keys)
                total[m] += k
                full[m] += k == len(topic_keys)
                tokens[m] += tokenizer.count(t)
        print(f"conversation {c + 1}/{args.conversations} done", flush=True)

    print(f"\nTopics mentioned in the context ({n} overview questions, 6 topics each, chats of about 6 x 3 exchanges)")
    print(f"  {'method':<18} {'topics covered':>16} {'all six':>10}   mean context tokens")
    for m in methods:
        print(f"  {m:<18} {total[m] / (6 * n):>15.0%} {full[m]:>6}/{n:<4}   {round(tokens[m] / n)}")


if __name__ == "__main__":
    main()
