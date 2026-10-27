"""
`cce` command line: look inside a CCE data folder.

  cce inspect                         sessions in the data folder
  cce inspect SESSION                 its notes: topics, turn ranges, sizes
  cce inspect SESSION --query "..."   why a question gets the context it gets (scores, budget, what was dropped)
  cce inspect SESSION --query "..." --json

Nothing is changed. The data folder is --data-dir, else CCE_DATA_DIR, else .cce_data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def format_explain(explain: dict) -> str:
    """Readable version of ContextPayload.explain."""
    q = explain.get("query", {})
    b = explain.get("budget", {})
    lines = []
    if q:
        lines.append(f"Question: {q.get('text')!r}  ({q.get('mode')} mode, top_k {q.get('top_k')})")
        lines.append(f"Embedder: {q.get('embedder', {}).get('model') or q.get('embedder', {}).get('name')}   "
                     f"BM25 terms: {q.get('bm25_terms') or 'none (no identifiers, codes or names in the question)'}")
        lines.append(f"Notes in the session: {q.get('notes_in_session')}, retrieved: {q.get('notes_retrieved')}")
    lines.append("")
    lines.append(f"Token budget: {b.get('total')} used of {b.get('max_tokens')} "
                 f"(usable {b.get('usable')}): system note {b.get('system_note')}, recent turns {b.get('recent_turns')}, "
                 f"pending turns {b.get('pending_turns')}, notes {b.get('past_notes')}, overhead {b.get('overhead')}")
    rec, pen = explain.get("recent", {}), explain.get("pending", {})
    lines.append(f"Recent turns sent: {rec.get('turns')}" + (" (some shortened or dropped to fit)" if rec.get("shortened_or_dropped") else ""))
    if pen.get("turns_available"):
        lines.append(f"Pending turns (not in a note yet): {pen.get('turns_included')} of {pen.get('turns_available')} sent verbatim"
                     + (" (cut to fit)" if pen.get("cut") else ""))
    lines.append("")
    lines.append("Notes, best first:" if q.get("mode") != "overview" else "Notes, in conversation order:")
    for n in explain.get("notes", []):
        s = n["scores"]
        mark = "sent   " if n["included"] else "DROPPED"
        lines.append(f"  [{mark}] turns {n['turns'][0]}-{n['turns'][1]}  {n['topic']}  ({n['tokens']} tokens)")
        lines.append(f"            dense {s['dense']}  recency {s['recency']}  bm25 {s['bm25']}  fused {s['fused']}")
        for quote in n.get("quotes", []):
            lines.append(f"            quote: {quote[:140]}")
        for line in n.get("overview_lines", []):
            lines.append(f"            line:  {line[:140]}")
    if explain.get("dropped_for_budget"):
        lines.append("")
        lines.append(f"{len(explain['dropped_for_budget'])} retrieved note(s) did not fit the token budget.")
    return "\n".join(lines)


def _manager(args):
    from cce_core.config import CCEConfig
    from cce_core.engine import CCEEngine

    options = {"embedder": args.embedder} if args.embedder else {}
    if args.data_dir:
        options["base_dir"] = Path(args.data_dir)
    config = CCEConfig(**options)
    if not config.db_path.exists():
        raise SystemExit(f"No CCE data in {config.base_dir.resolve()} (looked for warm.db). Use --data-dir.")
    return CCEEngine(config).create_session_manager()


def cmd_inspect(args) -> int:
    from cce_core.ingestion import tokenizer

    mgr = _manager(args)
    if not args.session:
        sessions = mgr.list_sessions()
        if not sessions:
            print("No sessions.")
            return 0
        print(f"{'session':<24} {'status':<8} {'turns':>6} {'notes':>6} {'raw tok':>8} {'sent tok':>9} {'ratio':>6}")
        for s in sorted(sessions, key=lambda x: x["session_id"]):
            st = mgr.get_or_open(s["session_id"]).stats()
            ratio = f"{st['compression_ratio']}x" if st["compression_ratio"] else "-"
            print(f"{s['session_id']:<24} {s.get('status', '?'):<8} {st['turns_ingested']:>6} {st['warm_nodes']:>6} "
                  f"{st['raw_tokens']:>8} {st['tokens_sent']:>9} {ratio:>6}")
        return 0

    if not mgr.exists(args.session):
        print(f"Unknown session {args.session!r}.", file=sys.stderr)
        return 1
    store = mgr.get_or_open(args.session)
    if args.query:
        payload = mgr.build_context_payload(args.session, args.query, top_k=args.top_k)
        if args.json:
            print(json.dumps(payload.explain, indent=2, ensure_ascii=False))
        else:
            print(format_explain(payload.explain))
        return 0

    stats = store.stats()
    if args.json:
        print(json.dumps({"stats": stats, "notes": [
            {"node_id": n.node_id, "topic": n.topic_label, "turns": [n.turn_start, n.turn_end], "tokens": n.token_count}
            for n in store.warm.get_by_session(args.session)]}, indent=2, ensure_ascii=False))
        return 0
    print(f"Session {args.session}: {stats['turns_ingested']} turns, {stats['warm_nodes']} notes, "
          f"{stats['hot_turns']} recent + {stats['pending_turns']} pending turns, "
          f"{stats['raw_tokens']} raw tokens -> {stats['tokens_sent']} sent"
          + (f" ({stats['compression_ratio']}x)" if stats["compression_ratio"] else ""))
    for n in store.warm.get_by_session(args.session):
        print(f"  turns {n.turn_start:>4}-{n.turn_end:<4} {tokenizer.count(n.meso_summary):>4} tok  {n.topic_label}")
    return 0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="cce", description="Look inside a CCE data folder.")
    sub = ap.add_subparsers(dest="command", required=True)
    ins = sub.add_parser("inspect", help="list sessions, show a session's notes, or explain what a question retrieves")
    ins.add_argument("session", nargs="?", help="session id (omit to list sessions)")
    ins.add_argument("--query", "-q", help="explain the context this question gets")
    ins.add_argument("--top-k", type=int, default=None)
    ins.add_argument("--data-dir", help="data folder (default: CCE_DATA_DIR or .cce_data)")
    ins.add_argument("--embedder", help="auto, neural, onnx or hash (must match the folder)")
    ins.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)
    try:
        code = cmd_inspect(args)
    except ValueError as exc:      # embedder mismatch, bad session id
        print(f"error: {exc}", file=sys.stderr)
        code = 2
    raise SystemExit(code)


if __name__ == "__main__":
    main()
