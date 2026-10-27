"""
Model-free check of search in other languages: is the planted fact in the context CCE builds?

Hand-written conversations in Spanish, Hindi and Chinese (and English as a control), four facts each, asked as
paraphrases in the same language (see dataset_multilingual.py). Compares the embedders:

  onnx    multilingual int8 ONNX model (extra: onnx)
  neural  all-MiniLM-L6-v2 through sentence-transformers (English model; extra: neural)
  hash    dependency-free hashing embedder (matches shared Latin-script words only)

  uv run --group bench python benchmarks/retrieval_multilingual.py --seeds 5
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
import dataset_multilingual  # noqa: E402
from retrieval_recall import check_minimums, parse_minimums, present  # noqa: E402

LANGS = ["es", "hi", "zh", "en"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5, help="conversations per language")
    ap.add_argument("--embedders", default="onnx,neural,hash")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--hot", type=int, default=4)
    ap.add_argument("--min", action="append", default=[], metavar="EMBEDDER=FRACTION",
                    help="exit with status 1 if an embedder finds fewer facts than this fraction over es, hi and zh")
    args = ap.parse_args()

    results: dict[str, dict[str, list[int]]] = {}
    for emb in args.embedders.split(","):
        engine = CCEEngine(CCEConfig(embedder=emb, base_dir=Path(tempfile.mkdtemp()),
                                     hot_tier_max_turns=args.hot, retrieval_top_k=args.top_k))
        proc = engine.create_stateless_processor()
        results[emb] = {}
        for lang in LANGS:
            found = total = 0
            for seed in range(args.seeds):
                data = dataset_multilingual.build(lang, seed)
                for q in data["questions"]:
                    ctx = "\n".join(m["content"] for m in proc.process(data["messages"], q["q"]).payload.to_messages())
                    found += present(ctx, q["keys"])
                    total += 1
            results[emb][lang] = [found, total]
        print(f"{emb} done", flush=True)

    print(f"\nFact present in the context ({args.seeds} conversations x 4 questions per language)")
    print(f"  {'embedder':<8}" + "".join(f"{l:>10}" for l in LANGS) + f"{'es+hi+zh':>12}")
    scores = {}
    for emb, per in results.items():
        non_en = [per[l] for l in LANGS if l != "en"]
        found, total = sum(x[0] for x in non_en), sum(x[1] for x in non_en)
        scores[emb] = found / total
        print(f"  {emb:<8}" + "".join(f"{per[l][0]:>6}/{per[l][1]:<3}" for l in LANGS) + f"{found:>7}/{total:<4}")
    check_minimums(parse_minimums(args.min), scores)


if __name__ == "__main__":
    main()
