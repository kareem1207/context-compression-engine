"""
Model-free test of the cases plain search should find hard: answers spread over SEVERAL messages.

Tasks (each a question whose answer needs more than one message):
  list_clustered   four checklist items stated in four consecutive exchanges (one topic)
  list_scattered   four checklist items stated at random places in the chat
  correction       a value is stated, then later corrected; the context must contain the NEW value

A task passes when the context contains every needed keyword (for corrections: the new value).
Compared at the SAME token budget: CCE vs plain RAG over raw messages (top turns by similarity + recent turns).

  uv run --group bench python benchmarks/retrieval_multi.py --conversations 12
"""
from __future__ import annotations

import argparse
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cce_core.config import CCEConfig  # noqa: E402
from cce_core.engine import CCEEngine  # noqa: E402
from cce_core.ingestion import tokenizer  # noqa: E402
import dataset_real  # noqa: E402
from retrieval_recall import present, rag_context  # noqa: E402

ITEMS = [
    ("register the domain name", "domain"), ("publish the privacy policy", "privacy"),
    ("run the payment test", "payment"), ("prepare the press kit", "press"),
    ("set up the backup plan", "backup"), ("book the launch venue", "venue"),
    ("order the printed banners", "banners"), ("train the support team", "support"),
    ("schedule the social posts", "social"), ("review the security audit", "audit"),
]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
ACK = "Got it, I will keep that in mind."


def base_pairs(seed: int, tokens: int, dialogues) -> list[list[dict]]:
    data = dataset_real.build(seed=seed, target_tokens=tokens, n_questions=1, dialogues=dialogues)
    m = data["messages"]
    # drop the single planted fact the builder adds, keep real filler as exchanges
    pairs = [[m[i], m[i + 1]] for i in range(0, len(m) - 1, 2)
             if "By the way" not in m[i]["content"] and not m[i]["content"].startswith(("Just so", "Our ", "Quick note", "Remember", "For the record", "I booked", "The on-call", "I usually", "My apartment"))]
    return pairs


def make(task: str, seed: int, tokens: int, dialogues):
    rng = random.Random(seed)
    pairs = base_pairs(seed, tokens, dialogues)
    n = len(pairs)
    if task.startswith("list"):
        chosen = rng.sample(ITEMS, 4)
        if task == "list_clustered":
            start = rng.randrange(2, int(n * 0.7))
            slots = [start + i for i in range(4)]
        else:
            slots = sorted(rng.sample(range(2, int(n * 0.8)), 4))
        for slot, (item, _) in sorted(zip(slots, chosen), reverse=True):
            pairs.insert(slot + 1, [{"role": "user", "content": f"For the launch checklist, we also need to {item}."},
                                    {"role": "assistant", "content": ACK}])
        msgs = [m for p in pairs for m in p]
        return msgs, "What is everything on our launch checklist?", [k for _, k in chosen]
    old, new = rng.sample(DAYS, 2)
    a, b = sorted(rng.sample(range(2, int(n * 0.8)), 2))
    pairs.insert(b + 1, [{"role": "user", "content": f"Update: the release window has moved to {new} at 3 AM."}, {"role": "assistant", "content": ACK}])
    pairs.insert(a + 1, [{"role": "user", "content": f"Our release window is {old} at 3 AM."}, {"role": "assistant", "content": ACK}])
    msgs = [m for p in pairs for m in p]
    return msgs, "When is our release window?", [new.lower()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", type=int, default=12)
    ap.add_argument("--tokens", type=int, default=5000)
    ap.add_argument("--seed-start", type=int, default=700)
    ap.add_argument("--hot", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--meso", type=int, default=150)
    args = ap.parse_args()

    dialogues = dataset_real.load_dialogues()
    engine = CCEEngine(CCEConfig(base_dir=Path(tempfile.mkdtemp()), hot_tier_max_turns=args.hot,
                                 retrieval_top_k=args.top_k, meso_max_tokens=args.meso))
    proc = engine.create_stateless_processor()
    tasks = ["list_clustered", "list_scattered", "correction"]
    score = {t: {"cce": 0, "rag": 0, "n": 0, "cce_tokens": 0} for t in tasks}
    for c in range(args.conversations):
        for t in tasks:
            msgs, question, keys = make(t, args.seed_start + c, args.tokens, dialogues)
            res = proc.process(msgs, question)
            ctx = "\n".join(m["content"] for m in res.payload.to_messages())
            budget = sum(tokenizer.count(m["content"]) + 4 for m in res.payload.to_messages())
            rag = rag_context(engine, msgs, question, args.hot, budget)
            s = score[t]
            s["n"] += 1
            s["cce"] += present(ctx, keys)
            s["rag"] += present(rag, keys)
            s["cce_tokens"] += budget
        print(f"conversation {c + 1}/{args.conversations} done", flush=True)

    print(f"\nAnswer fully present in the context ({args.conversations} conversations per task, chats of about {args.tokens} tokens)")
    print(f"  {'task':<16} {'CCE':>10} {'plain RAG':>12}   mean context tokens")
    for t, s in score.items():
        print(f"  {t:<16} {s['cce']:>4}/{s['n']:<5} {s['rag']:>6}/{s['n']:<5}   {round(s['cce_tokens'] / s['n'])}")


if __name__ == "__main__":
    main()
