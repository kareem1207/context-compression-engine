"""
CCE Visual Demo
Serves a single HTML page with a landing screen + live pipeline visualizer.

Run: uv run python examples/visual_demo.py
Open: http://localhost:7861
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

LLAMA_ENDPOINT = "http://localhost:8080/v1"
CCE_ENDPOINT   = "http://localhost:9000"

CONVERSATION = [
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
    {"role": "assistant", "content": "HTTPS everywhere, parameterized queries, CORS configured, secrets in env vars, input validation on every endpoint, 2FA on cloud accounts."},
    {"role": "user",      "content": "How do I scale when traffic grows?"},
    {"role": "assistant", "content": "Vertical scaling first. Then horizontal with a load balancer. Read replicas before sharding. Profile first — N+1 queries are almost always the bottleneck."},
]

QUERY = "What did you say about authentication, payments, and security?"


app = FastAPI()


async def emit(ws: WebSocket, event: str, data: dict):
    await ws.send_text(json.dumps({"event": event, **data}))


async def call_llm(messages: list[dict]) -> tuple[str, int, int]:
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            t0 = time.time()
            resp = await client.post(
                f"{LLAMA_ENDPOINT}/chat/completions",
                json={"model": "gemma4", "messages": messages, "max_tokens": 512, "temperature": 0.7},
            )
            resp.raise_for_status()
            elapsed = time.time() - t0
            data = resp.json()
            usage = data.get("usage", {})
            content = data["choices"][0]["message"]["content"].strip()
            return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    except httpx.ConnectError:
        return "[llama.cpp not running on port 8080]", 0, 0
    except Exception as e:
        return f"[Error: {e}]", 0, 0


async def call_cce(endpoint: str, tool: str, params: dict) -> dict:
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{endpoint}/{tool}", json=params)
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        return {"error": f"CCE REST bridge not running at {endpoint}"}
    except Exception as e:
        return {"error": str(e)}


def count_tokens(messages: list[dict]) -> int:
    return sum(max(1, round(len(m.get("content", "").split()) / 0.75)) + 4 for m in messages)


async def run_without_cce(ws: WebSocket):
    await emit(ws, "section", {"title": "WITHOUT CCE", "subtitle": "Full conversation sent to Gemma 4 on every call"})
    await asyncio.sleep(0.3)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        *CONVERSATION,
        {"role": "user", "content": QUERY},
    ]
    tokens = count_tokens(messages)

    await emit(ws, "step", {"status": "info", "text": f"Conversation loaded — {len(CONVERSATION)} turns"})
    await asyncio.sleep(0.4)
    await emit(ws, "step", {"status": "info", "text": f"Query: \"{QUERY}\""})
    await asyncio.sleep(0.4)
    await emit(ws, "tokens", {"label": "Tokens sent to Gemma 4", "value": tokens, "total": tokens, "pct": 100})
    await asyncio.sleep(0.4)
    await emit(ws, "step", {"status": "loading", "text": f"Sending all {len(CONVERSATION)} turns to Gemma 4..."})

    response, prompt_tok, completion_tok = await call_llm(messages)

    await emit(ws, "step", {"status": "done", "text": f"Response received — {prompt_tok} prompt tokens, {completion_tok} completion tokens"})
    await emit(ws, "response", {"label": "Gemma 4 (without CCE)", "text": response, "tokens_sent": tokens})
    return tokens, response


async def run_with_cce_local(ws: WebSocket, without_tokens: int):
    """Run CCE locally (no MCP bridge needed)."""
    from cce_core.config import CCEConfig
    from cce_core.engine import CCEEngine

    config = CCEConfig(summarizer_mode="extractive", hot_tier_max_turns=4, retrieval_top_k=3)
    engine = CCEEngine(config)

    await emit(ws, "section", {"title": "WITH CCE (local engine)", "subtitle": "Compress → retrieve → send only what's relevant"})
    await asyncio.sleep(0.3)

    await emit(ws, "step", {"status": "loading", "text": "Ingesting conversation into CCE..."})
    store, turns, nodes = engine.full_pipeline(CONVERSATION, session_id="visual-demo")
    stats = engine.compression_stats(turns, nodes)
    await emit(ws, "step", {"status": "done", "text": f"{len(CONVERSATION)} turns → {len(nodes)} memory nodes in {stats['compression_ratio']}x compression"})
    await asyncio.sleep(0.3)

    await emit(ws, "memory", {
        "nodes": [{"topic": n.topic_label, "turns": f"{n.turn_start}–{n.turn_end}", "summary": n.meso_summary} for n in nodes]
    })
    await asyncio.sleep(0.4)

    await emit(ws, "step", {"status": "loading", "text": f"Retrieving relevant context for query..."})
    results = engine.retriever.retrieve(store, QUERY)
    payload = engine.build_context(store, QUERY)
    await emit(ws, "step", {"status": "done", "text": f"Retrieved {len(results)} relevant node(s)"})
    await emit(ws, "retrieved", {
        "nodes": [{"topic": r.node.topic_label, "score": round(r.composite_score, 3), "summary": r.node.meso_summary[:120]} for r in results]
    })
    await asyncio.sleep(0.4)

    messages = payload.to_messages()
    messages.append({"role": "user", "content": QUERY})
    cce_tokens = count_tokens(messages)
    saved = without_tokens - cce_tokens
    pct_saved = round((saved / without_tokens) * 100) if without_tokens else 0

    await emit(ws, "tokens", {"label": "Tokens sent to Gemma 4 (with CCE)", "value": cce_tokens, "total": without_tokens, "pct": round((cce_tokens / without_tokens) * 100)})
    await asyncio.sleep(0.4)
    await emit(ws, "step", {"status": "loading", "text": f"Sending compressed context to Gemma 4... ({pct_saved}% fewer tokens)"})

    response, prompt_tok, completion_tok = await call_llm(messages)

    await emit(ws, "step", {"status": "done", "text": f"Response received — {prompt_tok} prompt tokens, {completion_tok} completion tokens"})
    await emit(ws, "response", {"label": "Gemma 4 (with CCE)", "text": response, "tokens_sent": cce_tokens})

    await emit(ws, "summary", {
        "without_tokens": without_tokens,
        "with_tokens": cce_tokens,
        "saved": saved,
        "pct_saved": pct_saved,
        "ratio": stats["compression_ratio"],
        "turns": len(CONVERSATION),
        "nodes": len(nodes),
    })
    store.close()


async def run_with_cce_mcp(ws: WebSocket, without_tokens: int):
    """Run CCE via MCP REST bridge."""
    await emit(ws, "section", {"title": "WITH CCE (via MCP server)", "subtitle": "Compress → retrieve → send only what's relevant"})
    await asyncio.sleep(0.3)

    await emit(ws, "step", {"status": "loading", "text": "Calling MCP tool: cce_ingest_history..."})
    result = await call_cce(CCE_ENDPOINT, "ingest_history", {"session_id": "mcp-visual-demo", "messages": CONVERSATION})
    if "error" in result:
        await emit(ws, "step", {"status": "error", "text": f"MCP bridge error: {result['error']}"})
        await emit(ws, "step", {"status": "info", "text": "Falling back to local CCE engine..."})
        await run_with_cce_local(ws, without_tokens)
        return

    await emit(ws, "step", {"status": "done", "text": f"MCP: {result.get('turns_loaded', '?')} turns loaded — {result.get('warm_nodes', 0)} warm nodes created"})
    await asyncio.sleep(0.4)

    await emit(ws, "step", {"status": "loading", "text": "Calling MCP tool: cce_retrieve_context..."})
    ctx = await call_cce(CCE_ENDPOINT, "retrieve_context", {"session_id": "mcp-visual-demo", "query": QUERY, "fmt": "messages", "top_k": 3})
    if "error" in ctx:
        await emit(ws, "step", {"status": "error", "text": f"Retrieval error: {ctx['error']}"})
        return

    compressed_messages = ctx.get("context", [])
    cce_tokens = count_tokens(compressed_messages)
    saved = without_tokens - cce_tokens
    pct_saved = round((saved / without_tokens) * 100) if without_tokens else 0

    await emit(ws, "step", {"status": "done", "text": f"MCP returned {ctx.get('past_blocks', '?')} compressed blocks + {ctx.get('recent_turns', '?')} recent turns"})
    await emit(ws, "tokens", {"label": "Tokens sent to Gemma 4 (with MCP+CCE)", "value": cce_tokens, "total": without_tokens, "pct": round((cce_tokens / without_tokens) * 100)})
    await asyncio.sleep(0.4)

    compressed_messages.append({"role": "user", "content": QUERY})
    await emit(ws, "step", {"status": "loading", "text": f"Sending compressed context to Gemma 4... ({pct_saved}% fewer tokens)"})

    response, prompt_tok, completion_tok = await call_llm(compressed_messages)
    await emit(ws, "step", {"status": "done", "text": f"Response received — {prompt_tok} prompt tokens, {completion_tok} completion tokens"})
    await emit(ws, "response", {"label": "Gemma 4 (with MCP + CCE)", "text": response, "tokens_sent": cce_tokens})

    await call_cce(CCE_ENDPOINT, "close_session", {"session_id": "mcp-visual-demo", "checkpoint": True})

    await emit(ws, "summary", {
        "without_tokens": without_tokens,
        "with_tokens": cce_tokens,
        "saved": saved,
        "pct_saved": pct_saved,
        "ratio": "—",
        "turns": len(CONVERSATION),
        "nodes": ctx.get("past_blocks", "?"),
    })


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        raw = await ws.receive_text()
        msg = json.loads(raw)
        use_mcp = msg.get("use_mcp", False)

        await emit(ws, "started", {})
        without_tokens, _ = await run_without_cce(ws)
        await asyncio.sleep(0.6)

        if use_mcp:
            await run_with_cce_mcp(ws, without_tokens)
        else:
            await run_with_cce_local(ws, without_tokens)

        await emit(ws, "done", {})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await emit(ws, "step", {"status": "error", "text": f"Unexpected error: {e}"})


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CCE Demo</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Syne:wght@400;500;700&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0b0d10;--bg2:#111419;--bg3:#181c22;--border:#1e232c;--border2:#252b36;
  --text:#dde2ec;--muted:#5a6478;--dim:#383e4d;
  --blue:#4f9eff;--purple:#8b7fff;--green:#34d399;--amber:#fbbf24;--red:#f87171;
  --font:'Syne',sans-serif;--mono:'JetBrains Mono',monospace;
}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:var(--font);overflow:hidden}

/* ── LANDING ── */
#landing{
  position:fixed;inset:0;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:0;z-index:10;background:var(--bg);
  transition:opacity .5s,transform .5s;
}
#landing.hide{opacity:0;transform:translateY(-20px);pointer-events:none}

.land-badge{
  font-family:var(--mono);font-size:10px;letter-spacing:2px;color:var(--muted);
  border:1px solid var(--border2);padding:5px 14px;border-radius:20px;margin-bottom:28px;
}
.land-title{
  font-size:clamp(28px,4vw,48px);font-weight:700;letter-spacing:-1px;
  text-align:center;line-height:1.1;margin-bottom:10px;
}
.land-title span{color:var(--blue)}
.land-sub{font-size:14px;color:var(--muted);text-align:center;margin-bottom:44px;max-width:420px;line-height:1.6}

.connector-box{
  background:var(--bg2);border:1px solid var(--border2);border-radius:14px;
  padding:20px 24px;width:340px;margin-bottom:28px;
}
.connector-label{font-size:11px;color:var(--muted);font-family:var(--mono);letter-spacing:.8px;margin-bottom:14px}
.connector-row{
  display:flex;align-items:center;justify-content:space-between;
  padding:12px 14px;border-radius:10px;border:1px solid var(--border2);
  background:var(--bg3);cursor:pointer;transition:border-color .2s;
  user-select:none;
}
.connector-row:hover{border-color:var(--border2)}
.connector-row.selected{border-color:var(--blue)}
.conn-left{display:flex;align-items:center;gap:10px}
.conn-icon{
  width:32px;height:32px;border-radius:8px;background:var(--bg2);border:1px solid var(--border2);
  display:flex;align-items:center;justify-content:center;font-size:14px;
}
.conn-name{font-size:13px;font-weight:500}
.conn-desc{font-size:11px;color:var(--muted);margin-top:1px}
.toggle{
  width:36px;height:20px;border-radius:10px;background:var(--border2);
  position:relative;transition:background .2s;flex-shrink:0;
}
.toggle.on{background:var(--blue)}
.toggle::after{
  content:'';position:absolute;top:2px;left:2px;width:16px;height:16px;
  border-radius:50%;background:#fff;transition:transform .2s;
}
.toggle.on::after{transform:translateX(16px)}

.start-btn{
  width:340px;padding:14px;border-radius:10px;font-size:14px;font-weight:600;
  font-family:var(--font);border:none;cursor:pointer;
  background:var(--blue);color:#000;transition:all .15s;letter-spacing:.3px;
}
.start-btn:hover{background:#6fb3ff;transform:translateY(-1px)}
.start-btn:active{transform:translateY(0)}

/* ── DEMO ── */
#demo{
  position:fixed;inset:0;display:flex;flex-direction:column;
  opacity:0;pointer-events:none;transition:opacity .4s;
}
#demo.show{opacity:1;pointer-events:all}

.demo-header{
  display:flex;align-items:center;justify-content:space-between;
  padding:10px 20px;border-bottom:1px solid var(--border);
  background:var(--bg2);flex-shrink:0;
}
.demo-logo{display:flex;align-items:center;gap:8px;font-size:13px;font-weight:600}
.pulse{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 2s infinite;box-shadow:0 0 6px var(--green)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.demo-status{font-size:11px;font-family:var(--mono);color:var(--muted)}

.demo-body{flex:1;display:flex;overflow:hidden;gap:1px;background:var(--border)}

/* Left — log */
.log-panel{flex:1;display:flex;flex-direction:column;background:var(--bg);min-width:0}
.panel-hdr{
  padding:10px 16px;border-bottom:1px solid var(--border);
  font-size:10px;font-family:var(--mono);color:var(--muted);letter-spacing:1px;
  text-transform:uppercase;display:flex;align-items:center;gap:6px;
}
.log-scroll{flex:1;overflow-y:auto;padding:14px 16px;display:flex;flex-direction:column;gap:6px}
.log-scroll::-webkit-scrollbar{width:3px}
.log-scroll::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}

.log-item{display:flex;gap:10px;align-items:flex-start;animation:fadeIn .25s ease}
@keyframes fadeIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.log-dot{width:6px;height:6px;border-radius:50%;margin-top:5px;flex-shrink:0}
.dot-info{background:var(--muted)}
.dot-loading{background:var(--amber);animation:pulse 1s infinite}
.dot-done{background:var(--green)}
.dot-error{background:var(--red)}
.log-text{font-size:12px;color:var(--muted);font-family:var(--mono);line-height:1.6;flex:1}
.log-text b{color:var(--text);font-weight:500}
.log-text .tag{
  display:inline-block;font-size:10px;padding:1px 7px;border-radius:3px;margin-right:4px;
}
.tag-blue{background:rgba(79,158,255,.15);color:var(--blue)}
.tag-purple{background:rgba(139,127,255,.15);color:var(--purple)}
.tag-green{background:rgba(52,211,153,.15);color:var(--green)}
.tag-amber{background:rgba(251,191,36,.15);color:var(--amber)}

.section-divider{
  margin:8px 0 4px;padding:6px 10px;border-radius:6px;
  background:var(--bg3);border:1px solid var(--border2);
  font-size:11px;font-weight:600;color:var(--text);
  display:flex;align-items:center;gap:6px;
}
.section-divider small{font-weight:400;color:var(--muted);font-size:10px}

/* Right — results */
.result-panel{width:380px;display:flex;flex-direction:column;background:var(--bg);flex-shrink:0}
.result-scroll{flex:1;overflow-y:auto;padding:14px}
.result-scroll::-webkit-scrollbar{width:3px}
.result-scroll::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}

/* Token bar */
.token-bar-wrap{background:var(--bg2);border:1px solid var(--border2);border-radius:8px;padding:12px 14px;margin-bottom:8px;animation:fadeIn .3s ease}
.token-bar-label{font-size:10px;font-family:var(--mono);color:var(--muted);margin-bottom:6px}
.token-bar-track{height:6px;background:var(--border);border-radius:3px;overflow:hidden;margin-bottom:6px}
.token-bar-fill{height:100%;border-radius:3px;transition:width .6s ease}
.token-bar-val{font-size:13px;font-weight:600;font-family:var(--mono);color:var(--text)}
.token-bar-sub{font-size:10px;color:var(--muted);font-family:var(--mono);margin-top:2px}

/* Memory nodes */
.memory-section{margin-bottom:10px;animation:fadeIn .3s ease}
.memory-hdr{font-size:10px;font-family:var(--mono);color:var(--muted);letter-spacing:1px;text-transform:uppercase;margin-bottom:6px}
.node-item{background:var(--bg2);border:1px solid var(--border2);border-left:2px solid var(--green);border-radius:6px;padding:8px 10px;margin-bottom:4px;cursor:pointer}
.node-item:hover{border-color:var(--green)}
.node-topic{font-size:11px;font-weight:500;color:var(--green);font-family:var(--mono);margin-bottom:2px}
.node-meta{font-size:10px;color:var(--muted);font-family:var(--mono)}
.node-summary{font-size:11px;color:var(--muted);margin-top:6px;line-height:1.5;display:none}
.node-item.open .node-summary{display:block}

/* Retrieved nodes */
.retrieved-item{background:var(--bg2);border:1px solid var(--border2);border-left:2px solid var(--blue);border-radius:6px;padding:8px 10px;margin-bottom:4px}
.retrieved-topic{font-size:11px;font-weight:500;color:var(--blue);font-family:var(--mono)}
.retrieved-score{font-size:10px;color:var(--green);font-family:var(--mono);margin-top:1px}
.retrieved-sum{font-size:11px;color:var(--muted);margin-top:4px;line-height:1.5}

/* Response box */
.response-box{background:var(--bg2);border:1px solid var(--border2);border-radius:8px;padding:12px 14px;margin-bottom:8px;animation:fadeIn .3s ease}
.response-label{font-size:10px;font-family:var(--mono);color:var(--muted);margin-bottom:6px;display:flex;justify-content:space-between;align-items:center}
.response-text{font-size:12px;color:var(--text);line-height:1.7}

/* Summary card */
.summary-card{background:var(--bg2);border:1px solid var(--blue);border-radius:10px;padding:16px;margin-top:4px;animation:fadeIn .4s ease}
.summary-title{font-size:12px;font-weight:600;color:var(--blue);margin-bottom:12px;font-family:var(--mono)}
.summary-row{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid var(--border)}
.summary-row:last-child{border-bottom:none}
.summary-key{font-size:11px;color:var(--muted);font-family:var(--mono)}
.summary-val{font-size:12px;font-weight:500;color:var(--text);font-family:var(--mono)}
.summary-val.green{color:var(--green)}
.summary-val.blue{color:var(--blue)}
</style>
</head>
<body>

<!-- LANDING -->
<div id="landing">
  <div class="land-badge">CONTEXT COMPRESSION ENGINE</div>
  <div class="land-title">See how CCE<br><span>saves tokens</span> in real time</div>
  <div class="land-sub">Watch a 14-turn SaaS conversation get compressed — then see Gemma 4 answer from compressed memory.</div>

  <div class="connector-box">
    <div class="connector-label">SELECT MODE</div>
    <div class="connector-row" id="connRow" onclick="toggleMCP()">
      <div class="conn-left">
        <div class="conn-icon">⚡</div>
        <div>
          <div class="conn-name">Use MCP Server</div>
          <div class="conn-desc">Requires cce_mcp.server_http running on :9000</div>
        </div>
      </div>
      <div class="toggle" id="mcpToggle"></div>
    </div>
  </div>

  <button class="start-btn" onclick="startDemo()">Start Demo</button>
</div>

<!-- DEMO -->
<div id="demo">
  <div class="demo-header">
    <div class="demo-logo">
      <div class="pulse"></div>
      CCE Visual Demo
    </div>
    <div class="demo-status" id="demoStatus">Running...</div>
  </div>
  <div class="demo-body">

    <!-- Left: log -->
    <div class="log-panel">
      <div class="panel-hdr">◈ Pipeline log</div>
      <div class="log-scroll" id="logScroll"></div>
    </div>

    <!-- Right: results -->
    <div class="result-panel">
      <div class="panel-hdr">◉ Results</div>
      <div class="result-scroll" id="resultScroll"></div>
    </div>

  </div>
</div>

<script>
var useMCP = false;
var ws = null;

function toggleMCP() {
  useMCP = !useMCP;
  document.getElementById('mcpToggle').classList.toggle('on', useMCP);
  document.getElementById('connRow').classList.toggle('selected', useMCP);
}

function startDemo() {
  document.getElementById('landing').classList.add('hide');
  setTimeout(function() {
    document.getElementById('demo').classList.add('show');
    connect();
  }, 400);
}

function connect() {
  ws = new WebSocket('ws://localhost:7861/ws');
  ws.onopen = function() {
    ws.send(JSON.stringify({use_mcp: useMCP}));
  };
  ws.onmessage = function(e) {
    handle(JSON.parse(e.data));
  };
  ws.onerror = function() {
    addLog('error', 'WebSocket error — is the server running?');
  };
}

function handle(msg) {
  if (msg.event === 'started') {
    document.getElementById('demoStatus').textContent = 'Running...';
  } else if (msg.event === 'section') {
    addSection(msg.title, msg.subtitle);
  } else if (msg.event === 'step') {
    addLog(msg.status, msg.text);
  } else if (msg.event === 'tokens') {
    addTokenBar(msg.label, msg.value, msg.total, msg.pct);
  } else if (msg.event === 'memory') {
    addMemoryNodes(msg.nodes);
  } else if (msg.event === 'retrieved') {
    addRetrieved(msg.nodes);
  } else if (msg.event === 'response') {
    addResponse(msg.label, msg.text, msg.tokens_sent);
  } else if (msg.event === 'summary') {
    addSummary(msg);
    document.getElementById('demoStatus').textContent = 'Complete';
  } else if (msg.event === 'done') {
    document.getElementById('demoStatus').textContent = 'Done';
  }
}

// Log helpers
function addLog(status, text) {
  var log = document.getElementById('logScroll');
  var div = document.createElement('div');
  div.className = 'log-item';
  var dotClass = 'dot-' + ({'info':'info','loading':'loading','done':'done','error':'error'}[status] || 'info');
  div.innerHTML = '<div class="log-dot ' + dotClass + '"></div><div class="log-text">' + esc(text) + '</div>';
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function addSection(title, subtitle) {
  var log = document.getElementById('logScroll');
  var div = document.createElement('div');
  div.className = 'section-divider';
  div.innerHTML = esc(title) + ' <small>' + esc(subtitle) + '</small>';
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

// Result helpers
function addTokenBar(label, value, total, pct) {
  var rs = document.getElementById('resultScroll');
  var color = pct >= 90 ? 'var(--red)' : pct >= 60 ? 'var(--amber)' : 'var(--green)';
  var div = document.createElement('div');
  div.className = 'token-bar-wrap';
  div.innerHTML =
    '<div class="token-bar-label">' + esc(label) + '</div>' +
    '<div class="token-bar-track"><div class="token-bar-fill" style="width:' + pct + '%;background:' + color + '"></div></div>' +
    '<div class="token-bar-val">' + value + ' tokens</div>' +
    '<div class="token-bar-sub">' + pct + '% of baseline (' + total + ' tokens)</div>';
  rs.appendChild(div);
  rs.scrollTop = rs.scrollHeight;
}

function addMemoryNodes(nodes) {
  var rs = document.getElementById('resultScroll');
  var wrap = document.createElement('div');
  wrap.className = 'memory-section';
  wrap.innerHTML = '<div class="memory-hdr">Warm tier — compressed memory nodes</div>';
  nodes.forEach(function(n) {
    var d = document.createElement('div');
    d.className = 'node-item';
    d.innerHTML =
      '<div class="node-topic">' + esc(n.topic) + '</div>' +
      '<div class="node-meta">turns ' + esc(n.turns) + '</div>' +
      '<div class="node-summary">' + esc(n.summary) + '</div>';
    d.onclick = function() { d.classList.toggle('open'); };
    wrap.appendChild(d);
  });
  rs.appendChild(wrap);
  rs.scrollTop = rs.scrollHeight;
}

function addRetrieved(nodes) {
  var rs = document.getElementById('resultScroll');
  var wrap = document.createElement('div');
  wrap.className = 'memory-section';
  wrap.innerHTML = '<div class="memory-hdr">Retrieved for this query</div>';
  nodes.forEach(function(n) {
    var d = document.createElement('div');
    d.className = 'retrieved-item';
    d.innerHTML =
      '<div class="retrieved-topic">' + esc(n.topic) + '</div>' +
      '<div class="retrieved-score">score: ' + n.score + '</div>' +
      '<div class="retrieved-sum">' + esc(n.summary) + '</div>';
    wrap.appendChild(d);
  });
  rs.appendChild(wrap);
  rs.scrollTop = rs.scrollHeight;
}

function addResponse(label, text, tokens) {
  var rs = document.getElementById('resultScroll');
  var div = document.createElement('div');
  div.className = 'response-box';
  div.innerHTML =
    '<div class="response-label"><span>' + esc(label) + '</span><span style="color:var(--muted)">' + tokens + ' tokens sent</span></div>' +
    '<div class="response-text">' + esc(text || '[no response]') + '</div>';
  rs.appendChild(div);
  rs.scrollTop = rs.scrollHeight;
}

function addSummary(d) {
  var rs = document.getElementById('resultScroll');
  var div = document.createElement('div');
  div.className = 'summary-card';
  var pct = d.pct_saved || 0;
  div.innerHTML =
    '<div class="summary-title">Final Results</div>' +
    row('Turns in conversation', d.turns) +
    row('Memory nodes created', d.nodes) +
    row('Tokens WITHOUT CCE', d.without_tokens) +
    row('Tokens WITH CCE', d.with_tokens, 'blue') +
    row('Tokens saved', d.saved + ' (' + pct + '%)', 'green') +
    row('Compression ratio', (d.ratio || '—') + 'x');
  rs.appendChild(div);
  rs.scrollTop = rs.scrollHeight;
}

function row(k, v, cls) {
  return '<div class="summary-row"><span class="summary-key">' + esc(String(k)) + '</span><span class="summary-val ' + (cls||'') + '">' + esc(String(v)) + '</span></div>';
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>
</body>
</html>"""


@app.get("/")
async def root():
    return HTMLResponse(HTML)


if __name__ == "__main__":
    print("\n  CCE Visual Demo")
    print("  Open: http://localhost:7861\n")
    uvicorn.run(app, host="0.0.0.0", port=7861, log_level="warning")