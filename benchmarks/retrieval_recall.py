"""
Model-free check: does the context CCE builds actually contain the fact needed to answer?
Fast (no LLM), so it can run on many conversations. A fact counts as "present" when its answer
keywords appear in the context text.

  uv run --group bench python benchmarks/retrieval_recall.py --conversations 10 --mode fit
"""
from __future__ import annotations

import argparse
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
from run_benchmark_v2 import VARIANTS  # noqa: E402


def present(context: str, keys: list[str]) -> bool:
    text = context.lower().replace(",", "")
    return all(k.lower() in text for k in keys)


def parse_minimums(items: list[str]) -> dict[str, float]:
    """['cce=0.95', 'live=0.9'] -> {'cce': 0.95, 'live': 0.9}"""
    out = {}
    for item in items:
        name, _, value = item.partition("=")
        out[name] = float(value)
    return out


def check_minimums(minimums: dict[str, float], scores: dict[str, float]) -> None:
    """Exit with status 1 if any named score is below its minimum (used by the CI recall gate)."""
    failed = [f"{n}: {scores.get(n, 0.0):.0%} < {m:.0%}" for n, m in minimums.items() if scores.get(n, 0.0) < m]
    if failed:
        print("RECALL GATE FAILED: " + "; ".join(failed), flush=True)
        raise SystemExit(1)
    if minimums:
        print("recall gate passed: " + ", ".join(f"{n} >= {m:.0%}" for n, m in minimums.items()), flush=True)


def rag_context(engine, messages, query, hot, budget_tokens):
    """Plain RAG baseline: the most similar raw turns that fit the budget, plus the last `hot` turns."""
    import numpy as np
    turns = engine.ingest(messages)
    embs = engine.chunker.model.encode([t.content for t in turns], normalize_embeddings=True, show_progress_bar=False)
    q = engine.chunker.embed_text(query)
    order = np.argsort(-(embs @ q))
    keep, used = set(range(max(0, len(turns) - hot), len(turns))), 0
    used = sum(turns[i].token_count + 4 for i in keep)
    for i in order:
        cost = turns[int(i)].token_count + 4
        if int(i) in keep or used + cost > budget_tokens:
            continue
        keep.add(int(i)); used += cost
    return "\n".join(f"{turns[i].role}: {turns[i].content}" for i in sorted(keep))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", type=int, default=10)
    ap.add_argument("--questions", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=5000)
    ap.add_argument("--seed-start", type=int, default=500, help="first conversation seed (100-103 were used while developing)")
    ap.add_argument("--hot", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--meso", type=int, default=150)
    ap.add_argument("--variants", default="cce,cce_hash,cce_plain")
    ap.add_argument("--min", action="append", default=[], metavar="VARIANT=FRACTION",
                    help="exit with status 1 if the variant finds fewer facts than this fraction (CI gate); repeatable")
    args = ap.parse_args()

    dialogues = dataset_real.load_dialogues()
    variants = args.variants.split(",")
    procs = {}
    for v in variants:
        cfg = CCEConfig(base_dir=Path(tempfile.mkdtemp()), hot_tier_max_turns=args.hot, retrieval_top_k=args.top_k,
                        meso_max_tokens=args.meso, **VARIANTS[v])
        procs[v] = CCEEngine(cfg).create_stateless_processor()

    rag_engine = CCEEngine(CCEConfig(base_dir=Path(tempfile.mkdtemp())))
    rag = {"lexical": [0, 0], "paraphrase": [0, 0]}
    hits = {v: {"lexical": [0, 0], "paraphrase": [0, 0]} for v in variants}
    sizes = {v: [] for v in variants}
    recent_hits = [0, 0]
    for c in range(args.conversations):
        data = dataset_real.build(seed=args.seed_start + c, target_tokens=args.tokens, n_questions=args.questions, dialogues=dialogues)
        for q in data["questions"]:
            for v in variants:
                res = procs[v].process(data["messages"], q["q"])
                msgs = res.payload.to_messages()
                text = "\n".join(m["content"] for m in msgs)
                hits[v][q["kind"]][1] += 1
                hits[v][q["kind"]][0] += present(text, q["keys"])
                sizes[v].append(sum(tokenizer.count(m["content"]) + 4 for m in msgs))
                if v == variants[0]:
                    budget = sizes[v][-1]
                    rag[q["kind"]][1] += 1
                    rag[q["kind"]][0] += present(rag_context(rag_engine, data["messages"], q["q"], args.hot, budget), q["keys"])
        print(f"conversation {c + 1}/{args.conversations} done", flush=True)

    total_q = args.conversations * args.questions
    print(f"\nFact present in the context ({total_q} questions, conversations of about {args.tokens} tokens)")
    for v in variants:
        h = hits[v]
        tot = sum(x[0] for x in h.values())
        print(f"  {v:<10} {tot}/{total_q} ({tot / total_q:.0%})  lexical {h['lexical'][0]}/{h['lexical'][1]}  "
              f"paraphrase {h['paraphrase'][0]}/{h['paraphrase'][1]}  mean context tokens {round(sum(sizes[v]) / len(sizes[v]))}")
    scores = {v: sum(x[0] for x in hits[v].values()) / total_q for v in variants}
    rt = sum(x[0] for x in rag.values())
    print(f"  {'rag_turns':<10} {rt}/{total_q} ({rt / total_q:.0%})  lexical {rag['lexical'][0]}/{rag['lexical'][1]}  "
          f"paraphrase {rag['paraphrase'][0]}/{rag['paraphrase'][1]}  (plain RAG over raw turns, same token budget as the first variant)")
    check_minimums(parse_minimums(args.min), scores)


if __name__ == "__main__":
    main()
