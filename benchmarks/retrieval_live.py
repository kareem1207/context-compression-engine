"""
Model-free check of the LIVE-SESSION path (what MCP and REST users run), on the same real chat text as
retrieval_recall.py. Is the fact needed to answer present in the context CCE builds?

Modes, per conversation and question:
  oneshot       stateless processor on the whole chat (the path the older benchmarks measured)
  live          messages streamed one by one through SessionManager.add_message
  live_restart  streamed, then close() and a brand-new engine + manager on the same data folder
  live_crash    streamed, then a new engine WITHOUT close() (simulated crash)
  live_buried   like live_restart, but every fact is buried inside a longer multi-sentence message
  oneshot_buried  the same buried chat, handed to the stateless processor

  uv run --group bench python benchmarks/retrieval_live.py --conversations 6
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
from retrieval_recall import present  # noqa: E402

MODES = ["oneshot", "live", "live_restart", "live_crash", "live_buried", "oneshot_buried"]


def make_cfg(base: Path, hot: int, top_k: int, meso: int, embedder: str | None) -> CCEConfig:
    extra = {"embedder": embedder} if embedder else {}
    return CCEConfig(base_dir=base, hot_tier_max_turns=hot, retrieval_top_k=top_k, meso_max_tokens=meso, **extra)


def bury(messages: list[dict], questions: list[dict], fillers: list[str]) -> list[dict]:
    """Put each fact in the middle of a longer message made of real filler sentences."""
    out, k = [], 0
    for m in messages:
        if any(present(m["content"], q["keys"]) for q in questions):
            pre, post = fillers[k % len(fillers)], fillers[(k + 1) % len(fillers)]
            k += 2
            m = {**m, "content": f"{pre} {m['content']} {post}"}
        out.append(m)
    return out


def stream(mgr, sid: str, messages: list[dict]) -> None:
    for m in messages:
        mgr.add_message(sid, m)


def ctx_text(payload) -> tuple[str, int]:
    msgs = payload.to_messages()
    return "\n".join(m["content"] for m in msgs), sum(tokenizer.count(m["content"]) + 4 for m in msgs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", type=int, default=6)
    ap.add_argument("--questions", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=5000)
    ap.add_argument("--seed-start", type=int, default=500)
    ap.add_argument("--hot", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--meso", type=int, default=150)
    ap.add_argument("--embedder", default=None, help="hash for a fast run without the neural model")
    ap.add_argument("--modes", default=",".join(MODES))
    args = ap.parse_args()
    modes = args.modes.split(",")

    dialogues = dataset_real.load_dialogues()
    hits = {m: {"lexical": [0, 0], "paraphrase": [0, 0]} for m in modes}
    sizes = {m: [] for m in modes}
    notes = {m: [] for m in modes}
    filler_pool = [d[i + 1]["content"] for d in dialogues[:60] for i in range(0, len(d) - 1, 2) if d[i + 1]["role"] == "assistant"][:80]
    fillers = [" ".join(t.split()[:25]) + "." for t in filler_pool if len(t.split()) > 25]

    for c in range(args.conversations):
        data = dataset_real.build(seed=args.seed_start + c, target_tokens=args.tokens, n_questions=args.questions, dialogues=dialogues)
        msgs, qs = data["messages"], data["questions"]
        built = {}
        for mode in modes:
            base = Path(tempfile.mkdtemp(prefix=f"cce_live_{mode}_"))
            cfg = lambda: make_cfg(base, args.hot, args.top_k, args.meso, args.embedder)  # noqa: E731
            if mode in ("oneshot", "oneshot_buried"):
                eng = CCEEngine(cfg())
                built[mode] = ("oneshot", eng.create_stateless_processor(), bury(msgs, qs, fillers) if mode == "oneshot_buried" else msgs)
                continue
            use = bury(msgs, qs, fillers) if mode == "live_buried" else msgs
            eng = CCEEngine(cfg()); mgr = eng.create_session_manager()
            stream(mgr, "s", use)
            if mode in ("live_restart", "live_buried"):
                mgr.close("s")
                mgr = CCEEngine(cfg()).create_session_manager()
            elif mode == "live_crash":
                mgr = CCEEngine(cfg()).create_session_manager()      # old manager never closed
            built[mode] = ("live", mgr, None)
            notes[mode].append(mgr.get_or_open("s").warm.count("s"))

        for q in qs:
            for mode in modes:
                kind, obj, chat = built[mode]
                if kind == "oneshot":
                    text, size = ctx_text(obj.process(chat, q["q"]).payload)
                else:
                    text, size = ctx_text(obj.build_context_payload("s", q["q"]))
                hits[mode][q["kind"]][1] += 1
                hits[mode][q["kind"]][0] += present(text, q["keys"])
                sizes[mode].append(size)
        print(f"conversation {c + 1}/{args.conversations} done", flush=True)

    total_q = args.conversations * args.questions
    print(f"\nFact present in the context ({total_q} questions, chats of about {args.tokens} tokens, embedder={args.embedder or 'neural'})")
    for mode in modes:
        h = hits[mode]
        tot = sum(x[0] for x in h.values())
        n = f"  notes/conv {sum(notes[mode]) / len(notes[mode]):.1f}" if notes[mode] else ""
        print(f"  {mode:<13} {tot}/{total_q} ({tot / total_q:.0%})  lexical {h['lexical'][0]}/{h['lexical'][1]}  "
              f"paraphrase {h['paraphrase'][0]}/{h['paraphrase'][1]}  mean context tokens {round(sum(sizes[mode]) / len(sizes[mode]))}{n}")


if __name__ == "__main__":
    main()
