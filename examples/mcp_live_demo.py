"""
CCE MCP Live Demo
Two things happening here:
  1. WITHOUT CCE — raw conversation dumped to Gemma 4
  2. WITH CCE    — calls MCP server to compress, then sends to Gemma 4

Run this AFTER starting the MCP server:
    Terminal 1: uv run python -m cce_mcp.server_http
    Terminal 2: uv run python examples/mcp_live_demo.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

LLAMA_ENDPOINT = "http://localhost:8080/v1"
CCE_ENDPOINT   = "http://localhost:9000"   # CCE REST bridge (Phase 6)

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
    {"role": "assistant", "content": "BullMQ with Redis for Node, Celery for Python. Queues for emails, image processing, webhooks. Never block API handlers with long tasks."},
    {"role": "user",      "content": "How do I monitor in production?"},
    {"role": "assistant", "content": "Sentry for errors, Datadog for metrics, Axiom for logs. Structured JSON logging with request_id and user_id from day one."},
    {"role": "user",      "content": "How do I handle database migrations safely?"},
    {"role": "assistant", "content": "Alembic (Python) or Prisma Migrate (Node). Always backward-compatible: add nullable columns, backfill data, then add constraints."},
    {"role": "user",      "content": "Security must-haves?"},
    {"role": "assistant", "content": "HTTPS everywhere, parameterized queries, CORS configured, secrets in env vars, input validation on every endpoint, 2FA on cloud accounts."},
    {"role": "user",      "content": "How do I scale when traffic grows?"},
    {"role": "assistant", "content": "Vertical scaling first. Then horizontal with a load balancer. Read replicas before sharding. Profile first — bottlenecks are almost always N+1 queries."},
]

QUERY = "What did you say about authentication, payments, and security?"
SESSION = "mcp-live-demo"


def call_gemma(messages: list[dict], label: str) -> str:
    """Call Gemma 4 and return the response text."""
    print(f"  Calling Gemma 4... ", end="", flush=True)
    try:
        with httpx.Client(timeout=300.0) as client:
            t0 = time.time()
            resp = client.post(
                f"{LLAMA_ENDPOINT}/chat/completions",
                json={
                    "model": "gemma4",
                    "messages": messages,
                    "max_tokens": 1024,   # enough for thinking model to answer
                    "temperature": 0.7,
                },
            )
            resp.raise_for_status()
            elapsed = time.time() - t0
            data = resp.json()
            usage = data.get("usage", {})
            choice = data["choices"][0]
            content = choice.get("message", {}).get("content", "").strip()
            prompt_tok = usage.get("prompt_tokens", "?")
            completion_tok = usage.get("completion_tokens", "?")
            print(f"{elapsed:.1f}s  |  prompt={prompt_tok} completion={completion_tok} tokens")
            return content, int(prompt_tok) if str(prompt_tok).isdigit() else 0
    except httpx.ConnectError:
        print("FAILED — llama.cpp not running")
        return "", 0
    except Exception as e:
        print(f"FAILED — {e}")
        return "", 0


def call_cce_mcp(endpoint: str, tool: str, params: dict) -> dict:
    """Call a CCE REST bridge endpoint (wraps MCP tool calls over HTTP)."""
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(f"{endpoint}/{tool}", json=params)
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        return {"error": f"CCE REST bridge not running at {endpoint}"}
    except Exception as e:
        return {"error": str(e)}


def count_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        words = len(m.get("content", "").split())
        total += max(1, round(words / 0.75)) + 4
    return total


def sep(char="─", n=62):
    print("  " + char * n)


def main():
    print("\n  Context Compression Engine — MCP Live Demo")
    print("  " + "─" * 42)
    print(f"  Conversation : {len(CONVERSATION)} turns")
    print(f"  Query        : \"{QUERY}\"")
    print(f"  Model        : Gemma 4 E4B (llama.cpp @ port 8080)")
    print(f"  CCE          : REST bridge @ port 9000\n")

    raw_tokens = count_tokens(CONVERSATION)
    print(f"  Full conversation = {raw_tokens} tokens\n")

    # ── WITHOUT CCE ───────────────────────────────────────────────
    print("  ── WITHOUT CCE  (full context every call) " + "─" * 20)
    without_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        *CONVERSATION,
        {"role": "user", "content": QUERY},
    ]
    without_tokens = count_tokens(without_messages)
    print(f"  Tokens sent  : {without_tokens}  (system + all {len(CONVERSATION)} turns + query)")

    response_without, prompt_without = call_gemma(without_messages, "without")
    if response_without:
        print(f"\n  Gemma 4 says:\n")
        for line in response_without[:500].splitlines():
            print(f"    {line}")
    else:
        print("  [no response]")

    print()
    sep()

    # ── WITH CCE via REST bridge ───────────────────────────────────
    print("\n  ── WITH CCE  (MCP tool → compressed context) " + "─" * 17)
    print(f"  CCE endpoint : {CCE_ENDPOINT}\n")

    # Step 1: load history
    print("  [MCP] cce_ingest_history ...", end=" ", flush=True)
    t0 = time.time()
    result = call_cce_mcp(CCE_ENDPOINT, "ingest_history", {
        "session_id": SESSION,
        "messages": CONVERSATION,
    })
    print(f"{time.time()-t0:.1f}s")
    if "error" in result:
        print(f"\n  CCE not available: {result['error']}")
        print("  Start the REST bridge: uv run python -m cce_rest.app")
        print("\n  Falling back to local CCE engine for demo...\n")
        _local_fallback(CONVERSATION, QUERY, without_tokens)
        return

    print(f"  Loaded {result.get('turns_loaded', '?')} turns  |  "
          f"warm nodes: {result.get('warm_nodes', '?')}")

    # Step 2: retrieve compressed context
    print(f"\n  [MCP] cce_retrieve_context for query ...", end=" ", flush=True)
    t0 = time.time()
    ctx_result = call_cce_mcp(CCE_ENDPOINT, "retrieve_context", {
        "session_id": SESSION,
        "query": QUERY,
        "fmt": "messages",
        "top_k": 3,
    })
    print(f"{time.time()-t0:.1f}s")

    if "error" in ctx_result:
        print(f"  Error: {ctx_result['error']}")
        return

    compressed_messages = ctx_result.get("context", [])
    with_tokens = count_tokens(compressed_messages)
    past_blocks = ctx_result.get("past_blocks", "?")
    recent_turns = ctx_result.get("recent_turns", "?")

    saved = without_tokens - with_tokens
    pct = (saved / without_tokens) * 100 if without_tokens else 0

    print(f"  Past blocks  : {past_blocks} compressed topic chunks")
    print(f"  Recent turns : {recent_turns} verbatim turns")
    print(f"  Tokens sent  : {with_tokens}  ({pct:.0f}% fewer than {without_tokens})")

    # Step 3: add the actual query
    compressed_messages.append({"role": "user", "content": QUERY})

    response_with, prompt_with = call_gemma(compressed_messages, "with")
    if response_with:
        print(f"\n  Gemma 4 says:\n")
        for line in response_with[:500].splitlines():
            print(f"    {line}")
    else:
        print("  [no response]")

    # Step 4: close session
    call_cce_mcp(CCE_ENDPOINT, "close_session", {"session_id": SESSION, "checkpoint": True})

    sep()
    print(f"""
  Results
  ─────────────────────────────────────────────
  Without CCE   {without_tokens:>5} tokens sent to Gemma 4
  With CCE      {with_tokens:>5} tokens sent to Gemma 4
  Saved         {saved:>5} tokens  ({pct:.0f}% reduction)
  ─────────────────────────────────────────────
  Both answers came from the same model.
  CCE delivered relevant context in {pct:.0f}% less space.
""")


def _local_fallback(conversation, query, without_tokens):
    """Run CCE locally if the REST bridge isn't up yet."""
    from cce_core.config import CCEConfig
    from cce_core.engine import CCEEngine

    config = CCEConfig(summarizer_mode="extractive", hot_tier_max_turns=4, retrieval_top_k=3)
    engine = CCEEngine(config)

    print("  Compressing with local CCE...", end=" ", flush=True)
    t0 = time.time()
    store, turns, nodes = engine.full_pipeline(conversation, session_id="fallback")
    print(f"{time.time()-t0:.1f}s")

    payload = engine.build_context(store, query)
    messages = payload.to_messages()
    messages.append({"role": "user", "content": query})

    with_tokens = count_tokens(messages)
    saved = without_tokens - with_tokens
    pct = (saved / without_tokens) * 100

    print(f"  Tokens sent  : {with_tokens}  ({pct:.0f}% fewer than {without_tokens})")
    print(f"  Past blocks  : {len(payload.past_blocks)}")
    print(f"  Recent turns : {len(payload.recent_turns)}")

    response, _ = call_gemma(messages, "with-local")
    if response:
        print(f"\n  Gemma 4 says:\n")
        for line in response[:500].splitlines():
            print(f"    {line}")

    sep()
    print(f"""
  Results (local CCE, no MCP bridge)
  ─────────────────────────────────────────────
  Without CCE   {without_tokens:>5} tokens
  With CCE      {with_tokens:>5} tokens
  Saved         {saved:>5} tokens  ({pct:.0f}% reduction)
  ─────────────────────────────────────────────
""")
    store.close()


if __name__ == "__main__":
    main()