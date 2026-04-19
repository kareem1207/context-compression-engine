"""
CCE + Gemma 4 Local Demo
Full end-to-end demo: long conversation → CCE compression → Gemma 4 inference.

Prerequisites:
  1. llama.cpp server running with Gemma 4:
       llama-server -m gemma-4-... --port 8080 --ctx-size 4096
  2. CCE installed: uv run python examples/gemma_local_demo.py

What this demo shows:
  - Loading a long conversation into CCE
  - Compressing it into memory nodes
  - Building a compressed context payload
  - Sending it to Gemma 4 via llama.cpp's OpenAI-compat API
  - Showing compression ratio and token savings
"""

import sys
from pathlib import Path

# Add project root to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from cce_core.config import CCEConfig
from cce_core.engine import CCEEngine


# ── Long conversation to compress ─────────────────────────────────────────────

LONG_HISTORY = [
    {"role": "user",      "content": "I want to build a REST API with Python. Where do I start?"},
    {"role": "assistant", "content": "Great choice! Start with FastAPI — it's modern, fast, and auto-generates docs. Install it with pip install fastapi uvicorn."},
    {"role": "user",      "content": "How do I define an endpoint?"},
    {"role": "assistant", "content": "Use decorators: @app.get('/items') for GET, @app.post('/items') for POST. FastAPI uses type hints for automatic request/response validation."},
    {"role": "user",      "content": "What about database integration?"},
    {"role": "assistant", "content": "SQLAlchemy is the standard ORM for Python. With FastAPI, use async SQLAlchemy for best performance. Alembic handles migrations."},
    {"role": "user",      "content": "Can you explain async in Python?"},
    {"role": "assistant", "content": "Async allows concurrent I/O without threads. Use async def for coroutines and await for I/O operations. asyncio is the underlying event loop."},
    {"role": "user",      "content": "Now let's talk about Docker. How do I containerize my FastAPI app?"},
    {"role": "assistant", "content": "Create a Dockerfile: use python:3.12-slim as base, copy requirements.txt, run pip install, copy your app, then CMD uvicorn main:app --host 0.0.0.0."},
    {"role": "user",      "content": "How do I handle environment variables in Docker?"},
    {"role": "assistant", "content": "Use ENV in Dockerfile for defaults, and --env or --env-file at runtime. In Python, use os.getenv() or python-dotenv library."},
    {"role": "user",      "content": "What about Docker Compose for multiple services?"},
    {"role": "assistant", "content": "Docker Compose lets you define multi-container apps in docker-compose.yml. Define services (api, db, redis), networks, and volumes. Run with docker-compose up."},
    {"role": "user",      "content": "Let's switch to testing. How do I test my FastAPI app?"},
    {"role": "assistant", "content": "Use pytest with httpx's AsyncClient. FastAPI provides a TestClient. Write unit tests for business logic and integration tests for endpoints."},
    {"role": "user",      "content": "What is test-driven development?"},
    {"role": "assistant", "content": "TDD means writing tests before code. Red-Green-Refactor cycle: write a failing test, write minimal code to pass it, then refactor. It forces clear requirements."},
    {"role": "user",      "content": "Back to FastAPI — how do I add authentication?"},
    {"role": "assistant", "content": "Use OAuth2 with JWT tokens. FastAPI has built-in OAuth2PasswordBearer. Generate tokens with python-jose, verify on each request via a dependency."},
]

NEW_QUESTION = "Can you remind me what you said about Docker Compose and also how authentication works in FastAPI?"


def _extract_completion_fields(data: dict) -> tuple[str, str, dict]:
    """Extract assistant content and lightweight metadata from OpenAI-style response."""
    choices = data.get("choices") or []
    if not choices:
        return "", "no_choices", {}

    first = choices[0] or {}
    message = first.get("message") or {}
    content = message.get("content") or ""
    finish_reason = first.get("finish_reason") or "unknown"
    usage = data.get("usage") or {}
    meta = {
        "finish_reason": finish_reason,
        "prompt_tokens": usage.get("prompt_tokens", "?"),
        "completion_tokens": usage.get("completion_tokens", "?"),
        "total_tokens": usage.get("total_tokens", "?"),
    }
    return content, finish_reason, meta


def call_gemma(messages: list[dict], endpoint: str = "http://localhost:8080/v1") -> str:
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(
                f"{endpoint}/chat/completions",
                json={
                    "model": "gemma4",
                    "messages": messages,
                    "max_tokens": 512,
                    "temperature": 0.7,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return f"[No choices: {data}]"
            content = choices[0].get("message", {}).get("content", "").strip()
            finish = choices[0].get("finish_reason", "?")
            usage = data.get("usage", {})
            meta = f"finish={finish}, tokens={usage.get('completion_tokens','?')}"
            return f"{content}\n    Meta: {meta}" if content else f"[Empty]\n    Meta: {meta}"
    except httpx.ConnectError:
        return "[llama.cpp not running]"
    except httpx.TimeoutException:
        return "[Timed out]"
    except Exception as e:
        return f"[Error: {e}]"
    
def main():
    print("=" * 60)
    print("Context Compression Engine — Gemma 4 Demo")
    print("=" * 60)

    # ── Setup ──────────────────────────────────────────────────────
    config = CCEConfig(
        summarizer_mode="extractive",   # no LLM needed for compression
        hot_tier_max_turns=6,
        retrieval_top_k=3,
    )
    engine = CCEEngine(config)

    print(f"\n[1] Loading {len(LONG_HISTORY)} turns into CCE session...")
    store, turns, nodes = engine.full_pipeline(
        LONG_HISTORY,
        session_id="gemma-demo-session",
    )

    stats = engine.compression_stats(turns, nodes)
    print(f"    Raw tokens:        {stats['raw_tokens']}")
    print(f"    Compressed nodes:  {stats['total_chunks']}")
    print(f"    Compression ratio: {stats['compression_ratio']}x")
    print(f"    Topics: {stats['topics']}")

    # ── Build context for new question ─────────────────────────────
    print(f"\n[2] New question: '{NEW_QUESTION}'")
    print("\n[3] Building compressed context...")

    payload = engine.build_context(store, NEW_QUESTION)
    messages = payload.to_messages()

    # Append the actual new user question
    messages.append({"role": "user", "content": NEW_QUESTION})

    print(f"    Context tokens:  {payload.token_count}")
    print(f"    Past blocks:     {len(payload.past_blocks)}")
    print(f"    Recent turns:    {len(payload.recent_turns)}")
    print(f"    Total messages:  {len(messages)}")

    # ── Show what the model receives ───────────────────────────────
    print("\n[4] Context payload sent to Gemma 4:")
    print("-" * 60)
    for i, msg in enumerate(messages):
        role = msg["role"].upper()
        content_preview = msg["content"][:200].replace("\n", " ")
        print(f"  [{i}] {role}: {content_preview}...")
    print("-" * 60)

    # ── Call Gemma 4 ───────────────────────────────────────────────
    print("\n[5] Calling Gemma 4 via llama.cpp...")
    response, response_meta = call_gemma(messages)
    print("\n    Gemma 4 response:")
    if response.strip():
        print(f"    {response}")
    else:
        print("    [Empty content returned by model]")
    print(
        "    Meta: "
        f"attempt={response_meta.get('attempt', '?')}, "
        f"finish_reason={response_meta.get('finish_reason', '?')}, "
        f"prompt={response_meta.get('prompt_tokens', '?')}, "
        f"completion={response_meta.get('completion_tokens', '?')}, "
        f"total={response_meta.get('total_tokens', '?')}"
    )

    # ── Stateless comparison ───────────────────────────────────────
    print("\n[6] Stateless mode comparison (same question, no session)...")
    proc = engine.create_stateless_processor()
    sl_result = proc.process(LONG_HISTORY, query=NEW_QUESTION)
    print(f"    Stateless context tokens: {sl_result.payload.token_count}")
    print(f"    Stateless compression:    {sl_result.stats['compression_ratio']}x")

    # ── Cleanup ────────────────────────────────────────────────────
    macro = engine.macro_summary(nodes)
    store.checkpoint(macro)
    store.close()

    print("\n" + "=" * 60)
    print("Demo complete. Session checkpointed to cold tier.")
    print("=" * 60)


if __name__ == "__main__":
    main()