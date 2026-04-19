"""
CCE Showcase Demo
"Before vs After" proof that Context Compression Engine works.
Run this to show anyone how much token savings CCE delivers.

Usage:
    uv run python examples/showcase_demo.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from cce_core.config import CCEConfig
from cce_core.engine import CCEEngine
from cce_core.ingestion import tokenizer


# ── A long, realistic multi-topic conversation ────────────────────────────────

LONG_CONVERSATION = [
    {"role": "user",      "content": "I'm building a SaaS product from scratch. Where do I even start with the backend architecture?"},
    {"role": "assistant", "content": "Start with a monolith, not microservices. Pick FastAPI for Python or Express for Node. You want to move fast early. Use PostgreSQL as your primary database — it handles 95% of SaaS use cases perfectly."},
    {"role": "user",      "content": "What about authentication? Should I build it myself?"},
    {"role": "assistant", "content": "Never build auth from scratch. Use Auth0, Clerk, or Supabase Auth. They handle OAuth, MFA, session management, and security updates. Your time is better spent on your core product."},
    {"role": "user",      "content": "How should I structure my database schema for a multi-tenant SaaS?"},
    {"role": "assistant", "content": "Two main patterns: shared schema with tenant_id column on every table (simpler, cheaper), or separate schema per tenant (better isolation, harder to maintain). Start with shared schema and tenant_id — you can migrate later if needed."},
    {"role": "user",      "content": "What's the best way to handle payments and subscriptions?"},
    {"role": "assistant", "content": "Stripe is the standard. Use Stripe Billing for subscriptions. Never store card data yourself. Implement webhooks to sync subscription state — listen for invoice.paid, customer.subscription.deleted, and payment_intent.failed events."},
    {"role": "user",      "content": "Now let's talk frontend. React or Next.js?"},
    {"role": "assistant", "content": "Next.js for SaaS — you get SSR for marketing pages (better SEO), API routes to avoid a separate server, and the App Router handles auth layouts cleanly. Use Tailwind for styling, shadcn/ui for components."},
    {"role": "user",      "content": "What about state management in React?"},
    {"role": "assistant", "content": "For server state use React Query (TanStack Query) — it handles caching, refetching, and loading states. For client state, useState and useContext are enough for 90% of apps. Don't reach for Redux until you genuinely need it."},
    {"role": "user",      "content": "How do I handle file uploads — like profile pictures and documents?"},
    {"role": "assistant", "content": "Upload directly to S3 or Cloudflare R2 using presigned URLs. Never pipe files through your backend server. Frontend requests a presigned URL from your API, uploads directly to storage, then sends you the URL. R2 is cheaper than S3 with no egress fees."},
    {"role": "user",      "content": "What's the right way to send emails — transactional and marketing?"},
    {"role": "assistant", "content": "Resend or Postmark for transactional emails (receipts, verification, password reset). Use React Email for templates. For marketing emails, use Loops or Customer.io. Keep transactional and marketing on separate sending domains to protect deliverability."},
    {"role": "user",      "content": "How should I think about caching?"},
    {"role": "assistant", "content": "Redis for session storage and short-lived caches. CDN (Cloudflare) for static assets and edge caching. Database query caching with React Query on the frontend. Don't over-engineer caching early — add it when you see actual slow queries in production."},
    {"role": "user",      "content": "What about background jobs and queues?"},
    {"role": "assistant", "content": "BullMQ with Redis for Node, or Celery with Redis for Python. Use queues for emails, image processing, PDF generation, and webhooks. Don't run long tasks synchronously in API handlers — it kills your response times and creates timeouts."},
    {"role": "user",      "content": "How do I monitor my app in production?"},
    {"role": "assistant", "content": "Three layers: errors (Sentry), metrics (Datadog or Grafana), and logs (Axiom or Logtail). Add structured logging from day one — JSON logs with request_id, user_id, and timing. Set up uptime monitoring with Better Uptime. Alert on error rate spikes, not just downtime."},
    {"role": "user",      "content": "What's your advice on deployment and CI/CD?"},
    {"role": "assistant", "content": "Railway or Render for early stage — zero DevOps overhead. GitHub Actions for CI/CD. Run tests on every PR, deploy main branch automatically. Add Docker later when you need more control. Don't set up Kubernetes until you have real scaling problems."},
    {"role": "user",      "content": "Back to the database — how do I handle database migrations safely?"},
    {"role": "assistant", "content": "Alembic for Python, Prisma Migrate for Node. Always write backward-compatible migrations: add columns as nullable, backfill data, then add constraints. Never rename or drop columns in a single deployment. Blue-green deployments let you run old and new code simultaneously during migration."},
    {"role": "user",      "content": "What about API design — REST vs GraphQL?"},
    {"role": "assistant", "content": "REST for most SaaS products. GraphQL only if you have complex, nested data with many consumers (like a public API with mobile + web + third parties). REST is simpler to cache, easier to version, and most devs know it instantly. Use OpenAPI/Swagger for documentation."},
    {"role": "user",      "content": "How do I handle rate limiting to prevent abuse?"},
    {"role": "assistant", "content": "Rate limit at the API gateway level (Nginx, Cloudflare, or AWS API Gateway) before requests hit your server. Use sliding window rate limiting — token bucket algorithm. Limit by IP for unauthenticated endpoints, by user_id for authenticated ones. Return 429 with Retry-After header."},
    {"role": "user",      "content": "Security — what are the most important things I must not miss?"},
    {"role": "assistant", "content": "Must-haves: HTTPS everywhere, parameterized queries (never string concat SQL), CORS configured properly, secrets in environment variables never in code, input validation on every endpoint, dependency scanning in CI. Enable 2FA on all your cloud accounts. Review OWASP Top 10 once a quarter."},
    {"role": "user",      "content": "What's the best way to scale when I start getting real traffic?"},
    {"role": "assistant", "content": "Vertical scaling first (bigger server) — it's simpler and often enough. Then horizontal scaling with a load balancer. Read replicas for database before sharding. CDN for static assets reduces server load dramatically. Profile before optimizing — most bottlenecks are in N+1 database queries, not code."},
    {"role": "user",      "content": "One last thing — how do I structure my team and codebase as I hire?"},
    {"role": "assistant", "content": "Monorepo with Turborepo (Node) or a well-structured single repo (Python). Feature-based folder structure, not layer-based. Document your architecture decisions in ADRs (Architecture Decision Records). Hire generalists early, specialists later. Code review everything until trust is established."},
]

QUERIES = [
    "What did you say about authentication and payments?",
    "Remind me about the database migration advice and rate limiting.",
    "What was the scaling strategy you recommended?",
]

LLAMA_ENDPOINT = "http://localhost:8080/v1"


# ── Helpers ───────────────────────────────────────────────────────────────────

def call_llm(messages: list[dict]) -> tuple[str, int]:
    """Call Gemma 4. Returns (response_text, completion_tokens)."""
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(
                f"{LLAMA_ENDPOINT}/chat/completions",
                json={"model": "gemma4", "messages": messages, "max_tokens": 400, "temperature": 0.7},
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            return content, tokens
    except httpx.ConnectError:
        return "[llama.cpp not running on port 8080]", 0
    except Exception as e:
        return f"[Error: {e}]", 0


def box(title: str, width: int = 62):
    print("\n" + "╔" + "═" * width + "╗")
    print("║" + f" {title}".ljust(width) + "║")
    print("╚" + "═" * width + "╝")


def divider(char="─", width=64):
    print(char * width)


def count_raw_tokens(messages: list[dict]) -> int:
    return sum(tokenizer.count(m["content"]) + 4 for m in messages)


# ── Main showcase ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "█" * 64)
    print("█" + " " * 16 + "CONTEXT COMPRESSION ENGINE" + " " * 20 + "█")
    print("█" + " " * 18 + "Showcase Demo v1.0" + " " * 26 + "█")
    print("█" * 64)
    print(f"\n  Conversation: {len(LONG_CONVERSATION)} turns across 10 topics")
    print(f"  Test queries: {len(QUERIES)}")
    print(f"  LLM: Gemma 4 E4B via llama.cpp")

    # ── Setup CCE ─────────────────────────────────────────────────
    config = CCEConfig(
        summarizer_mode="extractive",
        hot_tier_max_turns=6,
        retrieval_top_k=4,
        context_max_tokens=2048,
    )
    engine = CCEEngine(config)

    # ── Measure WITHOUT CCE ────────────────────────────────────────
    box("BASELINE — Without CCE (naive full context)")

    raw_token_count = count_raw_tokens(LONG_CONVERSATION)
    print(f"\n  Full conversation tokens : {raw_token_count}")
    print(f"  Every LLM call sends     : ALL {len(LONG_CONVERSATION)} turns")
    print(f"  Context grows linearly   : ~{raw_token_count} tokens per call")
    print(f"\n  Problem: As conversation grows, you hit context limits.")
    print(f"  At 10k turns → millions of tokens → impossible to use.")

    divider()
    print("  BASELINE query (full context → LLM):")
    print(f"  Q: \"{QUERIES[0]}\"")
    baseline_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        *LONG_CONVERSATION,
        {"role": "user", "content": QUERIES[0]},
    ]
    baseline_tokens_sent = count_raw_tokens(baseline_messages)
    print(f"\n  Tokens sent to LLM  : {baseline_tokens_sent}")
    print("  Calling Gemma 4...", end=" ", flush=True)
    t0 = time.time()
    baseline_response, baseline_completion = call_llm(baseline_messages)
    baseline_time = time.time() - t0
    print(f"done ({baseline_time:.1f}s)")
    print(f"\n  Gemma 4 response:\n  {baseline_response[:300]}")

    # ── Compress with CCE ──────────────────────────────────────────
    box("STEP 1 — CCE Compression Pipeline")

    print("\n  Ingesting conversation into CCE...", end=" ", flush=True)
    t0 = time.time()
    store, turns, nodes = engine.full_pipeline(
        LONG_CONVERSATION, session_id="showcase-session"
    )
    compress_time = time.time() - t0
    print(f"done ({compress_time:.1f}s)")

    stats = engine.compression_stats(turns, nodes)

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  Raw turns           : {stats['total_turns']:>6}             │")
    print(f"  │  Compressed nodes    : {stats['total_chunks']:>6}             │")
    print(f"  │  Raw tokens          : {stats['raw_tokens']:>6}             │")
    print(f"  │  Compression ratio   : {stats['compression_ratio']:>5.2f}x            │")
    print(f"  └─────────────────────────────────────────┘")
    print(f"\n  Topics detected:")
    for i, topic in enumerate(stats["topics"], 1):
        print(f"    {i:>2}. {topic}")

    # ── Queries WITH CCE ───────────────────────────────────────────
    box("STEP 2 — CCE Retrieval + LLM (3 queries)")

    total_cce_tokens = 0
    total_cce_time = 0.0

    for i, query in enumerate(QUERIES, 1):
        divider("·")
        print(f"\n  Query {i}: \"{query}\"")

        payload = engine.build_context(store, query)
        messages = payload.to_messages()
        messages.append({"role": "user", "content": query})

        cce_tokens_sent = count_raw_tokens(messages)
        total_cce_tokens += cce_tokens_sent

        print(f"  Past blocks retrieved : {len(payload.past_blocks)}")
        print(f"  Recent turns          : {len(payload.recent_turns)}")
        print(f"  Tokens sent to LLM    : {cce_tokens_sent}")
        print(f"  Calling Gemma 4...", end=" ", flush=True)

        t0 = time.time()
        response, completion_toks = call_llm(messages)
        elapsed = time.time() - t0
        total_cce_time += elapsed
        print(f"done ({elapsed:.1f}s)")

        print(f"\n  Gemma 4 response:\n  {response[:300]}")

    # ── Final comparison ───────────────────────────────────────────
    box("RESULTS — CCE vs Baseline")

    avg_cce = total_cce_tokens // len(QUERIES)
    saving_per_call = baseline_tokens_sent - avg_cce
    saving_pct = (saving_per_call / baseline_tokens_sent) * 100

    print(f"""
  ╔══════════════════════════════════════════════════════════╗
  ║                  TOKEN USAGE COMPARISON                  ║
  ╠══════════════════════════════════════════════════════════╣
  ║  Without CCE (baseline)                                  ║
  ║    Tokens per call     : {baseline_tokens_sent:<6}                        ║
  ║    Context overhead    : ALL {len(LONG_CONVERSATION)} turns every call         ║
  ║                                                          ║
  ║  With CCE                                                ║
  ║    Avg tokens per call : {avg_cce:<6}                        ║
  ║    Context used        : compressed past + recent turns  ║
  ║                                                          ║
  ║  ▶ Tokens saved per call  : {saving_per_call:<6} ({saving_pct:.1f}% reduction)  ║
  ║  ▶ Compression ratio      : {stats['compression_ratio']:.2f}x                      ║
  ║  ▶ Topics auto-detected   : {len(stats['topics']):<2}                          ║
  ╚══════════════════════════════════════════════════════════╝
    """)

    print("  What CCE gives you:")
    print("    ✓ Conversation memory that scales to any length")
    print("    ✓ Relevant past context retrieved per query")
    print("    ✓ Verbatim recent turns always included")
    print("    ✓ Works with ANY LLM — local or cloud")
    print("    ✓ MCP server, pip package, npm SDK")
    print("    ✓ Stateful (persistent) + stateless (on-demand) modes")

    # ── Stateless mode bonus ───────────────────────────────────────
    box("BONUS — Stateless Mode (zero setup, one call)")

    print("\n  No session, no DB, no state. Just hand CCE a conversation.")
    proc = engine.create_stateless_processor()
    t0 = time.time()
    result = proc.process(LONG_CONVERSATION, query=QUERIES[1])
    sl_time = time.time() - t0
    sl_tokens = result.payload.token_count

    print(f"  Input turns       : {len(LONG_CONVERSATION)}")
    print(f"  Output tokens     : {sl_tokens}")
    print(f"  Time to compress  : {sl_time:.2f}s")
    print(f"  Compression ratio : {result.stats['compression_ratio']}x")
    print(f"\n  Ready-to-use messages: {len(result.to_messages())} (system + recent turns)")

    store.close()

    print("\n" + "█" * 64)
    print("█" + " " * 20 + "Demo complete." + " " * 29 + "█")
    print("█" + " " * 10 + "github.com/your-handle/context-compression-engine" + " " * 3 + "█")
    print("█" * 64 + "\n")


if __name__ == "__main__":
    main()