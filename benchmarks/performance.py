"""
Speed and memory numbers.

  messages per second   live ingestion through SessionManager.add_message (embedding, chunking, summaries included)
  query latency         build_context_payload on a session with those messages: p50 and p95
  database growth       warm.db (+ WAL) bytes per 1,000 messages, with and without keep_raw_turns
  index memory          Python memory held by the shared search index per 10,000 synthetic notes
                        (384 dimensions, 6 message vectors, 8 sentence vectors and 80 terms per note), measured with tracemalloc

  uv run --group bench python benchmarks/performance.py --embedders hash,onnx,neural --messages 1000

Numbers depend on the machine, the embedder and the text; the script prints the machine next to them.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import dataset_real  # noqa: E402
from cce_core.compression.merger import MemoryNode, encode_turn_vecs  # noqa: E402
from cce_core.config import CCEConfig  # noqa: E402
from cce_core.engine import CCEEngine  # noqa: E402
from cce_core.memory.warm_tier import WarmTier  # noqa: E402


def conversation(n: int) -> list[dict]:
    """About n real chat messages (UltraChat), each clipped to 120 words."""
    out: list[dict] = []
    for dialogue in dataset_real.load_dialogues():
        for m in dialogue:
            out.append({"role": m["role"], "content": dataset_real._clip(m["content"], 120)})
            if len(out) >= n:
                return out
    return out


def db_bytes(base: Path) -> int:
    return sum(p.stat().st_size for p in base.glob("warm.db*"))


def measure(embedder: str, messages: list[dict], queries: int, keep_raw: bool) -> dict:
    base = Path(tempfile.mkdtemp(prefix="cce_perf_"))
    cfg = CCEConfig(embedder=embedder, base_dir=base, keep_raw_turns=keep_raw, retrieval_top_k=5)
    engine = CCEEngine(cfg)
    engine.chunker.embed_text("warm up")                      # load the model before timing
    mgr = engine.create_session_manager()
    mgr.open("perf")
    empty = db_bytes(base)

    started = time.perf_counter()
    for m in messages:
        mgr.add_message("perf", m)
    ingest_s = time.perf_counter() - started
    mgr.close("perf")
    size = db_bytes(base) - empty

    mgr = CCEEngine(cfg).create_session_manager()
    mgr.build_context_payload("perf", "warm up the index")
    asked = [m["content"] for m in messages if m["role"] == "user"]
    timings = []
    for i in range(queries):
        q = " ".join(asked[(i * 7) % len(asked)].split()[:16])
        t = time.perf_counter()
        mgr.build_context_payload("perf", q)
        timings.append((time.perf_counter() - t) * 1000)
    timings.sort()
    notes = mgr.get_or_open("perf").warm.count("perf")
    mgr.close_all(checkpoint=False)
    return {
        "embedder": embedder, "keep_raw_turns": keep_raw, "messages": len(messages), "notes": notes,
        "messages_per_second": round(len(messages) / ingest_s, 1),
        "query_ms_p50": round(statistics.median(timings), 1), "query_ms_p95": round(timings[int(0.95 * (len(timings) - 1))], 1),
        "db_kb_per_1000_messages": round(size / len(messages) * 1000 / 1024, 1),
    }


def index_memory(notes: int = 10_000, dim: int = 384) -> dict:
    rng = np.random.default_rng(0)

    def unit(n: int) -> np.ndarray:
        v = rng.normal(size=(n, dim)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    base = Path(tempfile.mkdtemp(prefix="cce_mem_"))
    tier = WarmTier(CCEConfig(embedder="hash", base_dir=base, warm_tier_max_chunks=0))
    batch = []
    for i in range(notes):
        meta = {"turn_vecs": encode_turn_vecs(unit(6)), "sent_vecs": encode_turn_vecs(unit(8)),
                "sent_map": [[k // 4, k % 4] for k in range(8)],
                "tf": {f"term{int(t)}": 1 + int(t) % 3 for t in rng.integers(0, 50_000, 80)}, "roles": "uauaua"}
        batch.append(MemoryNode(f"n{i}", "c", f"s{i % 10}", unit(1)[0], f"topic {i}", "summary text " * 10, ["m"] * 6,
                                i * 6, i * 6 + 5, 300, datetime.now(timezone.utc), meta))
        if len(batch) == 500:
            tier.upsert_many(batch)
            batch = []
    tier.close()

    tier = WarmTier(CCEConfig(embedder="hash", base_dir=base, warm_tier_max_chunks=0))
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    tier.index.sync()
    after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    arrays = sum(b.vecs.nbytes + b.tvecs.nbytes + b.svecs.nbytes for b in tier.index._blocks.values())
    tier.close()
    return {"notes": notes, "index_mb": round((after - before) / 1e6, 1), "vector_arrays_mb": round(arrays / 1e6, 1),
            "db_mb": round(db_bytes(base) / 1e6, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedders", default="hash,onnx,neural")
    ap.add_argument("--messages", type=int, default=1000)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--notes", type=int, default=10_000, help="synthetic notes for the memory number")
    ap.add_argument("--out", help="write the numbers as JSON")
    args = ap.parse_args()

    machine = {"cpu": platform.processor() or platform.machine(), "cores": os.cpu_count(), "os": platform.platform(),
               "python": platform.python_version(), "date": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
    print("machine:", machine)
    messages = conversation(args.messages)
    rows = []
    for emb in args.embedders.split(","):
        rows.append(measure(emb, messages, args.queries, keep_raw=True))
        print(rows[-1], flush=True)
    raw_off = measure(args.embedders.split(",")[0], messages, 10, keep_raw=False)
    memory = index_memory(args.notes)
    print("keep_raw_turns off:", raw_off)
    print("index memory:", memory)

    print(f"\n{len(messages)} real chat messages, {args.queries} queries")
    print(f"  {'embedder':<8} {'msg/s':>7} {'p50 ms':>8} {'p95 ms':>8} {'notes':>6} {'DB KB/1000 msg':>15}")
    for r in rows:
        print(f"  {r['embedder']:<8} {r['messages_per_second']:>7} {r['query_ms_p50']:>8} {r['query_ms_p95']:>8} {r['notes']:>6} {r['db_kb_per_1000_messages']:>15}")
    print(f"  database growth without raw text ({raw_off['embedder']}): {raw_off['db_kb_per_1000_messages']} KB per 1000 messages")
    print(f"  search index: {memory['index_mb']} MB of Python memory for {memory['notes']} notes "
          f"({memory['vector_arrays_mb']} MB of that is the vector arrays); database {memory['db_mb']} MB")
    if args.out:
        Path(args.out).write_text(json.dumps({"machine": machine, "runs": rows, "keep_raw_off": raw_off, "index_memory": memory}, indent=2),
                                  encoding="utf-8")


if __name__ == "__main__":
    main()
