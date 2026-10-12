"""
Benchmark: does CCE keep a local model's answers right while sending far fewer tokens?

Three ways of building the prompt for the same question about a fact mentioned earlier:
  full    the entire conversation, then the question
  recent  only the newest turns, cut to the SAME token budget CCE used (a fair, naive baseline)
  cce     CCE's compressed context (topic summaries + the last few turns) then the question

Needs a llama.cpp server (OpenAI-compatible) with a model loaded:
  llama-server -m model.gguf --port 8080 --ctx-size 8192
Run:
  uv run python benchmarks/run_benchmark.py --endpoint http://localhost:8080/v1 --label gemma4-e4b
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
import dataset  # noqa: E402

INSTRUCTION = (
    "\n\n(Answer in one short sentence using only what was said earlier in this conversation. "
    "If it was never mentioned, reply 'not mentioned'.)"
)


def ask(client: httpx.Client, endpoint: str, messages: list[dict], max_tokens: int = 200) -> dict:
    start = time.perf_counter()
    r = client.post(
        f"{endpoint}/chat/completions",
        json={
            "model": "local", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "cache_prompt": True,
            # Models with a thinking mode would spend the whole token limit "thinking" and return no answer.
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=1200,
    )
    r.raise_for_status()
    body = r.json()
    elapsed = time.perf_counter() - start
    timings = body.get("timings", {})
    return {
        "answer": (body["choices"][0]["message"]["content"] or "").strip(),
        "finish_reason": body["choices"][0].get("finish_reason"),
        "prompt_tokens": body["usage"]["prompt_tokens"],
        "cached_tokens": (body["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "prompt_ms": timings.get("prompt_ms"),
        "seconds": round(elapsed, 2),
    }


def is_correct(answer: str, keys: list[str]) -> bool:
    text = answer.lower().replace(",", "")
    return all(k.lower() in text for k in keys)


def recent_within_budget(messages: list[dict], budget_tokens: int) -> list[dict]:
    """Newest turns that fit the budget (estimated with the engine's own token counter)."""
    kept: list[dict] = []
    used = 0
    for m in reversed(messages):
        cost = tokenizer.count(m["content"]) + 4
        if used + cost > budget_tokens and kept:
            break
        kept.append(m)
        used += cost
    return list(reversed(kept))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8080/v1")
    ap.add_argument("--label", default="local-model")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hot", type=int, default=4, help="recent turns CCE keeps verbatim")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--meso", type=int, default=200, help="summary budget per topic (tokens)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()

    data = dataset.build(seed=args.seed)
    messages, questions = data["messages"], data["questions"]
    full_tokens = sum(tokenizer.count(m["content"]) + 4 for m in messages)

    cfg = CCEConfig(
        base_dir=Path(tempfile.mkdtemp(prefix="cce_bench_")),
        hot_tier_max_turns=args.hot,
        retrieval_top_k=args.top_k,
        meso_max_tokens=args.meso,
    )
    engine = CCEEngine(cfg)
    processor = engine.create_stateless_processor()

    print(f"{len(messages)} turns, about {full_tokens} tokens; {len(questions)} questions; model {args.label}")
    rows = []
    with httpx.Client() as client:
        # Warm the server so the first timing is not model-loading time.
        ask(client, args.endpoint, [{"role": "user", "content": "Say ok."}], max_tokens=4)

        # 1) CCE contexts first (cheap), so we know each question's token budget.
        cce_ctx = []
        for q in questions:
            t0 = time.perf_counter()
            res = processor.process(messages, q["q"])
            build_s = time.perf_counter() - t0
            cce_ctx.append((res.payload.to_messages(), build_s, res.payload.retrieved_node_ids, res.stats))

        for i, q in enumerate(questions):
            question = [{"role": "user", "content": q["q"] + INSTRUCTION}]
            ctx_msgs, build_s, node_ids, stats = cce_ctx[i]
            ctx_tokens = sum(tokenizer.count(m["content"]) + 4 for m in ctx_msgs)

            conditions = {
                "full": messages + question,
                "recent": recent_within_budget(messages, ctx_tokens) + question,
                "cce": ctx_msgs + question,
            }
            row = {"question": q["q"], "kind": q["kind"], "keys": q["keys"], "turns_from_end": q["turns_from_end"],
                   "cce_build_seconds": round(build_s, 2)}
            for name, msgs in conditions.items():
                out = ask(client, args.endpoint, msgs)
                out["correct"] = is_correct(out["answer"], q["keys"])
                row[name] = out
            rows.append(row)
            flags = " ".join(f"{n}={'Y' if row[n]['correct'] else 'n'}" for n in conditions)
            print(f"[{i + 1:>2}/{len(questions)}] {q['kind']:<10} {flags}  prompt tokens: "
                  + ", ".join(f"{n} {row[n]['prompt_tokens']}" for n in conditions))

    def summary(name: str) -> dict:
        got = [r[name] for r in rows]
        return {
            "accuracy": round(sum(g["correct"] for g in got) / len(got), 3),
            "correct": sum(g["correct"] for g in got),
            "mean_prompt_tokens": round(sum(g["prompt_tokens"] for g in got) / len(got)),
            "mean_prompt_eval_seconds": round(sum((g["prompt_ms"] or 0) for g in got) / len(got) / 1000, 2),
        }

    by_kind = {
        kind: {n: round(sum(r[n]["correct"] for r in rows if r["kind"] == kind) / max(1, sum(1 for r in rows if r["kind"] == kind)), 3)
               for n in ("full", "recent", "cce")}
        for kind in ("lexical", "paraphrase")
    }
    result = {
        "model": args.label,
        "date": time.strftime("%Y-%m-%d"),
        "machine": f"{platform.processor() or platform.machine()}, {platform.system()}",
        "dataset": {"seed": args.seed, "turns": len(messages), "approx_tokens": full_tokens, "questions": len(questions)},
        "cce_config": {"hot_tier_max_turns": args.hot, "retrieval_top_k": args.top_k, "meso_max_tokens": args.meso,
                       "summarizer": "extractive", "embedder": cfg.embedding_model},
        "summary": {n: summary(n) for n in ("full", "recent", "cce")},
        "by_question_type": by_kind,
        "rows": rows,
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) / f"{args.label}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nSummary")
    for n, s in result["summary"].items():
        print(f"  {n:<7} accuracy {s['correct']}/{len(rows)} ({s['accuracy']:.0%})  mean prompt tokens {s['mean_prompt_tokens']}")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
