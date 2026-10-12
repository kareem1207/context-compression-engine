"""
Benchmark v2: many conversations built from real chat text, two modes, and CCE variants.

  --mode fit       the whole conversation fits the model window; compare full / recent / CCE
  --mode overflow  the conversation is longer than the window (run the server with a small --ctx-size);
                   "window" is what most apps do: send the newest turns that fit

CCE variants (each is an extra column, same questions):
  cce         default engine (neural embedder if installed, detail-aware summaries)
  cce_hash    dependency-free hashing embedder
  cce_plain   detail-aware summaries switched off

Example:
  llama-server -m model.gguf --port 8080 --ctx-size 8192
  uv run --group bench python benchmarks/run_benchmark_v2.py --mode fit --label deepseek-fit
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cce_core.config import CCEConfig  # noqa: E402
from cce_core.engine import CCEEngine  # noqa: E402
from cce_core.ingestion import tokenizer  # noqa: E402
import dataset_real  # noqa: E402
from run_benchmark import INSTRUCTION, ask, is_correct, recent_within_budget  # noqa: E402

VARIANTS = {
    "cce": {},
    "cce_hash": {"embedder": "hash"},
    "cce_plain": {"detail_summaries": False},
}


def make_engine(overrides: dict, hot: int, top_k: int, meso: int) -> CCEEngine:
    cfg = CCEConfig(base_dir=Path(tempfile.mkdtemp(prefix="cce_bench2_")), hot_tier_max_turns=hot,
                    retrieval_top_k=top_k, meso_max_tokens=meso, **overrides)
    return CCEEngine(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8080/v1")
    ap.add_argument("--label", required=True)
    ap.add_argument("--mode", choices=["fit", "overflow"], default="fit")
    ap.add_argument("--conversations", type=int, default=3)
    ap.add_argument("--questions", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=None, help="conversation size (estimated tokens)")
    ap.add_argument("--window", type=int, default=3300, help="overflow mode: tokens the model window leaves for history")
    ap.add_argument("--variants", default="cce,cce_hash,cce_plain")
    ap.add_argument("--seed-start", type=int, default=500, help="first conversation seed (100-103 were used while developing)")
    ap.add_argument("--hot", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--meso", type=int, default=150)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()

    target = args.tokens or (5000 if args.mode == "fit" else 9000)
    variants = [v for v in args.variants.split(",") if v]
    base_name = "full" if args.mode == "fit" else "window"
    conds = [base_name, "recent"] + variants

    dialogues = dataset_real.load_dialogues()
    engines = {v: make_engine(VARIANTS[v], args.hot, args.top_k, args.meso) for v in variants}
    processors = {v: e.create_stateless_processor() for v, e in engines.items()}

    rows = []
    with httpx.Client() as client:
        ask(client, args.endpoint, [{"role": "user", "content": "Say ok."}], max_tokens=4)
        for c in range(args.conversations):
            data = dataset_real.build(seed=args.seed_start + c, target_tokens=target, n_questions=args.questions, dialogues=dialogues)
            messages = data["messages"]
            est = sum(tokenizer.count(m["content"]) + 4 for m in messages)
            print(f"conversation {c + 1}/{args.conversations}: {len(messages)} turns, about {est} tokens", flush=True)
            # Build every prompt first, then ask condition by condition. Asking one condition for all
            # questions in a row lets the server reuse its cached prompt prefix (the shared history).
            batch = []
            for q in data["questions"]:
                question = [{"role": "user", "content": q["q"] + INSTRUCTION}]
                ctxs, builds = {}, {}
                for v in variants:
                    t0 = time.perf_counter()
                    res = processors[v].process(messages, q["q"])
                    builds[v] = round(time.perf_counter() - t0, 2)
                    ctxs[v] = res.payload.to_messages()
                budget = sum(tokenizer.count(m["content"]) + 4 for m in ctxs[variants[0]])
                prompts = {
                    base_name: (messages if args.mode == "fit" else recent_within_budget(messages, args.window)) + question,
                    "recent": recent_within_budget(messages, budget) + question,
                    **{v: ctxs[v] + question for v in variants},
                }
                batch.append(({"conversation": c, "question": q["q"], "kind": q["kind"], "keys": q["keys"],
                               "turns_from_end": q["turns_from_end"], "build_seconds": builds}, prompts, q["keys"]))
            for name in conds:
                for row, prompts, keys in batch:
                    out = ask(client, args.endpoint, prompts[name])
                    out["correct"] = is_correct(out["answer"], keys)
                    row[name] = out
                done = sum(r[name]["correct"] for r, _, _ in batch)
                print(f"  conversation {c + 1} {name:<10} {done}/{len(batch)} correct", flush=True)
            rows.extend(r for r, _, _ in batch)

    def summary(name: str) -> dict:
        got = [r[name] for r in rows]
        return {"correct": sum(g["correct"] for g in got), "of": len(got),
                "accuracy": round(sum(g["correct"] for g in got) / len(got), 3),
                "mean_prompt_tokens": round(sum(g["prompt_tokens"] for g in got) / len(got)),
                "mean_prompt_eval_seconds": round(sum((g["prompt_ms"] or 0) for g in got) / len(got) / 1000, 2)}

    result = {
        "label": args.label, "mode": args.mode, "date": time.strftime("%Y-%m-%d"),
        "machine": f"{platform.processor() or platform.machine()}, {platform.system()}",
        "setup": {"conversations": args.conversations, "questions_per_conversation": args.questions,
                  "target_tokens": target, "window_tokens": args.window if args.mode == "overflow" else None,
                  "hot_tier_max_turns": args.hot, "retrieval_top_k": args.top_k, "meso_max_tokens": args.meso,
                  "dataset": "UltraChat 200k (test_sft) filler + randomised planted facts"},
        "summary": {n: summary(n) for n in conds},
        "by_question_type": {k: {n: round(sum(r[n]["correct"] for r in rows if r["kind"] == k)
                                   / max(1, sum(1 for r in rows if r["kind"] == k)), 3) for n in conds}
                             for k in ("lexical", "paraphrase")},
        "rows": rows,
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    path = Path(args.out) / f"{args.label}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nSummary")
    for n, s in result["summary"].items():
        print(f"  {n:<10} {s['correct']}/{s['of']} ({s['accuracy']:.0%})  mean prompt tokens {s['mean_prompt_tokens']}  prompt eval {s['mean_prompt_eval_seconds']}s")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
