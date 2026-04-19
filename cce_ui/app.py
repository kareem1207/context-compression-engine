"""
CCE Visual Dashboard
- Auto-simulates a 24-turn SaaS conversation through CCE on startup
- Shows hot tier, warm tier, cold tier populating live
- Shows summarizer steps when compression fires
- You can also type your own messages after simulation
- Rerun button resets and re-simulates everything

Run: uv run python -m cce_ui.app
Open: http://localhost:7860
"""

import asyncio
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from cce_core.config import CCEConfig
from cce_core.engine import CCEEngine
from cce_core.session.manager import SessionManager

LLAMA_ENDPOINT = os.getenv("CCE_LLM_ENDPOINT", "http://localhost:8080/v1")
SESSION_ID = "demo-session"

DEMO_CONVERSATION = [
    {"role": "user",      "content": "I'm building a SaaS product from scratch. Where do I start with the backend?"},
    {"role": "assistant", "content": "Start with a monolith, not microservices. Use FastAPI for Python or Express for Node. PostgreSQL as your primary database."},
    {"role": "user",      "content": "What about authentication? Should I build it myself?"},
    {"role": "assistant", "content": "Never build auth from scratch. Use Auth0, Clerk, or Supabase Auth. They handle OAuth, MFA, and session management."},
    {"role": "user",      "content": "How should I structure my database for multi-tenant SaaS?"},
    {"role": "assistant", "content": "Shared schema with a tenant_id column is simplest. Separate schema per tenant gives better isolation but is harder to maintain."},
    {"role": "user",      "content": "What about payments and subscriptions?"},
    {"role": "assistant", "content": "Stripe is the standard. Use Stripe Billing. Listen to webhooks: invoice.paid, subscription.deleted, payment_intent.failed."},
    {"role": "user",      "content": "React or Next.js for the frontend?"},
    {"role": "assistant", "content": "Next.js — SSR for marketing pages, built-in API routes, clean auth layouts. Tailwind + shadcn/ui for components."},
    {"role": "user",      "content": "How do I handle file uploads?"},
    {"role": "assistant", "content": "Presigned URLs directly to S3 or Cloudflare R2. Never pipe files through your backend server."},
    {"role": "user",      "content": "What about background jobs?"},
    {"role": "assistant", "content": "BullMQ with Redis for Node, Celery for Python. Queues for emails, image processing, webhooks. Never block API handlers."},
    {"role": "user",      "content": "How do I monitor in production?"},
    {"role": "assistant", "content": "Sentry for errors, Datadog for metrics, Axiom for logs. Structured JSON logging with request_id and user_id from day one."},
    {"role": "user",      "content": "How do I handle database migrations safely?"},
    {"role": "assistant", "content": "Alembic (Python) or Prisma Migrate (Node). Always backward-compatible: add nullable columns, backfill data, then add constraints."},
    {"role": "user",      "content": "Security must-haves?"},
    {"role": "assistant", "content": "HTTPS everywhere, parameterized queries, CORS configured, secrets in env vars, input validation on every endpoint, 2FA on all cloud accounts."},
    {"role": "user",      "content": "How do I scale when traffic grows?"},
    {"role": "assistant", "content": "Vertical scaling first. Then horizontal with a load balancer. Read replicas before sharding. Profile first — bottlenecks are almost always N+1 queries."},
]

config = CCEConfig(summarizer_mode="extractive", hot_tier_max_turns=6, retrieval_top_k=3)
engine = CCEEngine(config)
manager = SessionManager(engine)
manager.open(SESSION_ID)

app = FastAPI()
clients: list[WebSocket] = []

STOPWORDS = {"the","a","an","is","it","to","do","of","and","or","in","on","at","for","with","that","this","was","are","be","have","has","had","you","your","we","can","will","would","could","should","what","how","why","when","where","about","just","so","but","if","not","no","i","me","my"}
SENT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")


def summarizer_steps(text: str) -> dict:
    sents = [s.strip() for s in SENT_RE.split(text) if s.strip()] or [text]
    words = WORD_RE.findall(text.lower())
    freq = Counter(w for w in words if w not in STOPWORDS)
    top = {w: c for w, c in freq.most_common(12)}
    scored = []
    for s in sents:
        ws = set(WORD_RE.findall(s.lower())) - STOPWORDS
        score = round(len(ws & set(top)) / max(len(ws), 1) + min(len(s.split()) / 20, 1) * 0.2, 3)
        scored.append({"text": s, "score": score, "hits": list(ws & set(top))})
    scored_s = sorted(scored, key=lambda x: -x["score"])
    sel, used = [], 0
    for item in scored_s:
        t = max(1, round(len(item["text"].split()) / 0.75))
        if used + t > 200: break
        sel.append(item["text"])
        used += t
    order = {s: i for i, s in enumerate(sents)}
    sel.sort(key=lambda s: order.get(s, 999))
    return {"original": text, "step1_sentences": sents, "step2_freq": sorted(top.items(), key=lambda x: -x[1])[:10], "step3_scored": scored, "step4_selected": sel, "summary": " ".join(sel)}


def get_state():
    store = manager.get(SESSION_ID)
    if not store:
        return {}
    hot = store.get_hot_turns()
    warm = store.warm.get_by_session(SESSION_ID)
    cold_exists = store.cold.exists(SESSION_ID)
    cold_text = ""
    if cold_exists:
        cs = store.cold.get(SESSION_ID)
        cold_text = cs.macro_text if cs else ""
    s = store.stats()
    return {
        "hot": [{"index": t.index, "role": t.role, "content": t.content, "tokens": t.token_count} for t in hot],
        "warm": [{"node_id": n.node_id[:8], "topic": n.topic_label, "summary": n.meso_summary, "micro": n.micro_summaries, "turns": f"{n.turn_start}–{n.turn_end}", "tokens_original": n.token_count} for n in warm],
        "cold_exists": cold_exists, "cold_text": cold_text,
        "stats": s,
    }


async def send_all(event: str, data: dict):
    msg = json.dumps({"event": event, "data": data})
    dead = []
    for ws in clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in clients: clients.remove(ws)


async def process_user_turn(text: str, turn_index: int = None, total: int = None):
    """Process one user turn through the full CCE pipeline and broadcast each step.
    
    We push directly to the store's hot tier WITHOUT the auto-compress_fn so we
    can intercept evictions ourselves and broadcast each summarizer step visually.
    """
    from cce_core.ingestion.segmenter import Turn as TurnObj
    import uuid
    from datetime import datetime, timezone

    prefix = f"[{turn_index}/{total}] " if turn_index is not None else ""
    store = manager.get(SESSION_ID)
    if not store:
        return

    # Build Turn manually (bypass manager.add_message which auto-compresses)
    buf = manager._turn_buffers.get(SESSION_ID, [])
    turn = TurnObj(
        turn_id=str(uuid.uuid4()),
        role="user",
        content=text,
        token_count=max(1, round(len(text.split()) / 0.75)),
        timestamp=datetime.now(timezone.utc),
        index=len(buf),
    )
    buf.append(turn)
    manager._turn_buffers[SESSION_ID] = buf

    await send_all("log", {"type": "info", "text": f"{prefix}Pushing to hot tier (RAM)..."})

    # Push to hot tier directly — capture evicted turn
    evicted_turn = store.hot.push(turn)
    store._turn_count += 1
    store._raw_token_count += turn.token_count

    await send_all("log", {"type": "ok", "text": f"Turn #{turn.index} · {turn.token_count} tokens · in hot tier"})
    await send_all("state", get_state())
    await asyncio.sleep(0.2)

    # If eviction happened, collect ALL pending evictions and compress
    if evicted_turn is not None:
        # Drain the evicted queue (includes the one we just got back)
        pending = store.hot.drain_evicted()
        if not pending:
            pending = [evicted_turn]

        await send_all("log", {"type": "warn", "text": f"Hot tier full — {len(pending)} turn(s) evicted → starting compression..."})
        await asyncio.sleep(0.15)
        await send_all("log", {"type": "info", "text": "Chunker: embedding with SBERT · grouping by cosine similarity..."})

        chunks = engine.chunker.chunk(pending, SESSION_ID)

        for chunk in chunks:
            await send_all("log", {"type": "info", "text": f"Summarizer → chunk: '{chunk.topic_label}'"})
            steps = summarizer_steps(chunk.text)
            await send_all("summarizer", {"topic": chunk.topic_label, "steps": steps})
            await asyncio.sleep(0.4)

        engine.summarizer.annotate_chunks(chunks)
        nodes = engine.merger.merge(chunks)
        store.warm.upsert_many(nodes)

        await send_all("log", {"type": "ok", "text": f"{len(nodes)} MemoryNode(s) → SQLite warm tier"})
        await send_all("state", get_state())
        await asyncio.sleep(0.2)


async def call_llm(messages: list[dict]) -> str:
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{LLAMA_ENDPOINT}/chat/completions",
                json={"model": "gemma4", "messages": messages, "max_tokens": 400, "temperature": 0.7},
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    except httpx.ConnectError:
        return "[llama.cpp not running on :8080 — start it first]"
    except Exception as e:
        return f"[Error: {e}]"


async def run_simulation(ws: WebSocket):
    """Feed the demo conversation through CCE turn by turn."""
    total = len(DEMO_CONVERSATION)
    await send_all("sim_start", {"total": total})
    await send_all("log", {"type": "info", "text": f"Starting simulation — {total} turns from SaaS demo conversation"})

    for i, msg in enumerate(DEMO_CONVERSATION):
        role = msg["role"]
        content = msg["content"]

        # Show turn appearing in chat
        await send_all("chat_msg", {"role": role, "content": content, "simulated": True})
        await send_all("sim_progress", {"current": i + 1, "total": total})
        await asyncio.sleep(0.3)

        if role == "user":
            await process_user_turn(content, turn_index=i + 1, total=total)
        else:
            # Push assistant turn directly to hot tier (no compression check needed)
            store_a = manager.get(SESSION_ID)
            if store_a:
                from cce_core.ingestion.segmenter import Turn as TurnObj
                import uuid
                from datetime import datetime, timezone
                buf_a = manager._turn_buffers.get(SESSION_ID, [])
                at = TurnObj(turn_id=str(uuid.uuid4()), role="assistant", content=content,
                    token_count=max(1, round(len(content.split())/0.75)),
                    timestamp=datetime.now(timezone.utc), index=len(buf_a))
                buf_a.append(at)
                manager._turn_buffers[SESSION_ID] = buf_a
                store_a.hot.push(at)
                store_a._turn_count += 1
                store_a._raw_token_count += at.token_count
            await send_all("state", get_state())
            await asyncio.sleep(0.1)

    # Checkpoint at end
    store = manager.get(SESSION_ID)
    if store:
        store.flush_hot_to_warm()
        nodes = store.warm.get_by_session(SESSION_ID)
        if nodes:
            macro = engine.macro_summary(nodes)
            store.checkpoint(macro)
            await send_all("log", {"type": "ok", "text": "Session checkpointed — macro summary written to cold tier"})
            await send_all("state", get_state())

    await send_all("sim_done", {"total": total})
    await send_all("log", {"type": "ok", "text": f"Simulation complete — {total} turns processed · all memory tiers populated · try asking a question!"})


@app.websocket("/ws")
async def ws_handler(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    await ws.send_text(json.dumps({"event": "state", "data": get_state()}))
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg["type"] == "simulate":
                # Reset + run simulation
                manager.close(SESSION_ID, checkpoint=False)
                manager.open(SESSION_ID)
                await send_all("reset_ui", {})
                await send_all("log", {"type": "info", "text": "Session reset — starting fresh simulation"})
                await run_simulation(ws)

            elif msg["type"] == "chat":
                text = msg["content"].strip()
                if not text: continue
                await send_all("log", {"type": "info", "text": f"Your message: \"{text[:60]}...\""})
                await process_user_turn(text)

                # Retrieve + call LLM
                store = manager.get(SESSION_ID)
                await send_all("log", {"type": "info", "text": "Retrieving relevant context from memory..."})
                payload = engine.build_context(store, text)
                results = engine.retriever.retrieve(store, text)
                retrieved = [{"topic": r.node.topic_label, "score": round(r.composite_score, 3)} for r in results]
                await send_all("log", {"type": "ok", "text": f"Retrieved {len(results)} node(s) · {payload.token_count} tokens total context"})
                await send_all("retrieved", {"nodes": retrieved})

                messages = payload.to_messages()
                messages.append({"role": "user", "content": text})
                await send_all("log", {"type": "info", "text": "Calling Gemma 4..."})
                response = await call_llm(messages)
                from cce_core.ingestion.segmenter import Turn as TurnObj
                import uuid
                from datetime import datetime, timezone
                storeR = manager.get(SESSION_ID)
                bufR = manager._turn_buffers.get(SESSION_ID, [])
                at2 = TurnObj(turn_id=str(uuid.uuid4()), role="assistant", content=response,
                    token_count=max(1, round(len(response.split())/0.75)),
                    timestamp=datetime.now(timezone.utc), index=len(bufR))
                bufR.append(at2)
                manager._turn_buffers[SESSION_ID] = bufR
                if storeR:
                    storeR.hot.push(at2)
                    storeR._turn_count += 1
                    storeR._raw_token_count += at2.token_count
                await send_all("log", {"type": "ok", "text": "Gemma 4 responded"})
                await send_all("state", get_state())
                await send_all("chat_msg", {"role": "assistant", "content": response, "simulated": False})

            elif msg["type"] == "checkpoint":
                store = manager.get(SESSION_ID)
                if store:
                    store.flush_hot_to_warm()
                    nodes = store.warm.get_by_session(SESSION_ID)
                    macro = engine.macro_summary(nodes)
                    store.checkpoint(macro)
                    await send_all("log", {"type": "ok", "text": "Checkpointed — macro summary saved to cold tier"})
                    await send_all("state", get_state())

    except WebSocketDisconnect:
        if ws in clients: clients.remove(ws)


@app.get("/")
async def root():
    return HTMLResponse((Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    print("\n  CCE Dashboard → http://localhost:7860\n")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning")