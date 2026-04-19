"""
CCE MCP Server
Exposes the Context Compression Engine as MCP tools.
Transport: stdio (local tool — runs as subprocess by Claude Desktop / any MCP client)

Tools exposed:
  cce_ingest_turn         — add one message to a session
  cce_ingest_history      — bulk-load conversation history into a session
  cce_retrieve_context    — get compressed context ready for LLM injection
  cce_summarize_session   — get macro summary of a session
  cce_session_stats       — diagnostic info for a session
  cce_close_session       — checkpoint and close a session
  cce_list_sessions       — list all known sessions
  cce_stateless_compress  — one-shot compress + retrieve (no persistence)

Usage:
  python -m cce_mcp.server
  or: uv run python -m cce_mcp.server
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP, Context

from cce_core.config import CCEConfig
from cce_core.engine import CCEEngine
from cce_core.session.manager import SessionManager
from cce_core.session.stateless import StatelessProcessor
from cce_mcp.schema import (
    IngestTurnInput,
    IngestHistoryInput,
    RetrieveContextInput,
    SummarizeSessionInput,
    CloseSessionInput,
    SessionStatsInput,
    StatelessCompressInput,
)


# ── Lifespan — engine + manager live for the server's lifetime ────────────────

@asynccontextmanager
async def cce_lifespan():
    """Initialize CCE engine and session manager on startup, clean up on shutdown."""
    config = CCEConfig(
        summarizer_mode=os.getenv("CCE_SUMMARIZER_MODE", "extractive"),
        llm_endpoint=os.getenv("CCE_LLM_ENDPOINT", "http://localhost:8080/v1"),
        llm_model=os.getenv("CCE_LLM_MODEL", "gemma4"),
        hot_tier_max_turns=int(os.getenv("CCE_HOT_TIER_TURNS", "10")),
        retrieval_top_k=int(os.getenv("CCE_TOP_K", "5")),
        context_max_tokens=int(os.getenv("CCE_MAX_TOKENS", "2048")),
    )

    engine = CCEEngine(config)
    manager = SessionManager(engine)
    processor = StatelessProcessor(engine)

    try:
        yield {
            "engine": engine,
            "manager": manager,
            "processor": processor,
        }
    finally:
        manager.close_all(checkpoint=True)


mcp = FastMCP("cce_mcp", lifespan=cce_lifespan)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_manager(ctx: Context) -> SessionManager:
    return ctx.request_context.lifespan_state["manager"]

def _get_processor(ctx: Context) -> StatelessProcessor:
    return ctx.request_context.lifespan_state["processor"]

def _ok(data: Any) -> str:
    return json.dumps({"status": "ok", **data} if isinstance(data, dict) else {"status": "ok", "result": data}, indent=2)

def _err(msg: str) -> str:
    return json.dumps({"status": "error", "message": msg}, indent=2)


# ── Tool: ingest_turn ─────────────────────────────────────────────────────────

@mcp.tool(
    name="cce_ingest_turn",
    annotations={
        "title": "Ingest a single conversation turn",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def cce_ingest_turn(params: IngestTurnInput, ctx: Context) -> str:
    """Add a single message turn to a CCE session.

    Call this after every user or assistant message to keep the session
    memory up to date. The CCE engine will automatically compress older
    turns into the warm tier as the conversation grows.

    Args:
        params (IngestTurnInput): Validated input containing:
            - session_id (str): Unique session identifier
            - role (str): Speaker role — 'user', 'assistant', or 'system'
            - content (str): Message content text

    Returns:
        str: JSON with status and turn metadata (turn_id, index, token_count)
    """
    try:
        manager = _get_manager(ctx)
        turn = manager.add_message(
            params.session_id,
            {"role": params.role, "content": params.content},
        )
        await ctx.report_progress(1.0, "Turn ingested")
        return _ok({
            "turn_id": turn.turn_id,
            "index": turn.index,
            "token_count": turn.token_count,
            "session_id": params.session_id,
        })
    except Exception as e:
        return _err(f"Failed to ingest turn: {e}")


# ── Tool: ingest_history ──────────────────────────────────────────────────────

@mcp.tool(
    name="cce_ingest_history",
    annotations={
        "title": "Bulk-load conversation history",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def cce_ingest_history(params: IngestHistoryInput, ctx: Context) -> str:
    """Load a full conversation history into a CCE session in one call.

    Use this to initialize a session with an existing conversation.
    More efficient than calling cce_ingest_turn repeatedly.
    Automatically compresses older turns into the warm memory tier.

    Args:
        params (IngestHistoryInput): Validated input containing:
            - session_id (str): Unique session identifier
            - messages (list[dict]): List of {'role': str, 'content': str} dicts

    Returns:
        str: JSON with status, turn count, and compression info
    """
    try:
        manager = _get_manager(ctx)
        await ctx.report_progress(0.1, "Opening session...")
        turns = manager.add_messages(params.session_id, params.messages)
        await ctx.report_progress(1.0, f"Loaded {len(turns)} turns")

        store = manager.get(params.session_id)
        stats = store.stats() if store else {}
        return _ok({
            "session_id": params.session_id,
            "turns_loaded": len(turns),
            "hot_turns": stats.get("hot_turns", 0),
            "warm_nodes": stats.get("warm_nodes", 0),
            "raw_tokens": stats.get("raw_tokens", 0),
        })
    except Exception as e:
        return _err(f"Failed to ingest history: {e}")


# ── Tool: retrieve_context ────────────────────────────────────────────────────

@mcp.tool(
    name="cce_retrieve_context",
    annotations={
        "title": "Retrieve compressed context for LLM injection",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def cce_retrieve_context(params: RetrieveContextInput, ctx: Context) -> str:
    """Build a compressed context payload ready for LLM injection.

    Call this right before every LLM inference call. It retrieves the most
    relevant past context from the session memory, combines it with the
    recent verbatim turns, and returns a compact payload that fits within
    the LLM context window.

    The 'messages' format returns an OpenAI-compatible list of message dicts.
    The system message contains compressed past context + a note explaining
    the CCE format. Subsequent messages are the verbatim recent turns.

    Args:
        params (RetrieveContextInput): Validated input containing:
            - session_id (str): Session to retrieve from
            - query (str): Current user query (used for semantic retrieval)
            - top_k (Optional[int]): Number of memory nodes to retrieve
            - fmt (str): Output format — 'messages', 'string', or 'dict'
            - include_micro (bool): Include per-turn micro summaries

    Returns:
        str: JSON with status and context payload in requested format
    """
    try:
        manager = _get_manager(ctx)
        await ctx.report_progress(0.3, "Retrieving relevant context...")
        payload = manager.build_context_payload(
            params.session_id,
            params.query,
            top_k=params.top_k,
        )
        await ctx.report_progress(0.9, "Building context payload...")

        exported = payload.export(params.fmt)
        return _ok({
            "context": exported,
            "token_count": payload.token_count,
            "past_blocks": len(payload.past_blocks),
            "recent_turns": len(payload.recent_turns),
            "was_truncated": payload.was_truncated,
            "retrieved_node_ids": payload.retrieved_node_ids,
        })
    except Exception as e:
        return _err(f"Failed to retrieve context: {e}")


# ── Tool: summarize_session ───────────────────────────────────────────────────

@mcp.tool(
    name="cce_summarize_session",
    annotations={
        "title": "Generate macro summary of a session",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def cce_summarize_session(params: SummarizeSessionInput, ctx: Context) -> str:
    """Generate a one-paragraph macro summary of the entire session so far.

    Useful for giving an LLM a quick 'story so far' without injecting
    the full compressed context. Also used before checkpointing.

    Args:
        params (SummarizeSessionInput): Validated input containing:
            - session_id (str): Session to summarize

    Returns:
        str: JSON with status and macro_summary text
    """
    try:
        manager = _get_manager(ctx)
        store = manager.get_or_open(params.session_id)
        nodes = store.warm.get_by_session(params.session_id)

        if not nodes:
            return _ok({
                "session_id": params.session_id,
                "macro_summary": "No compressed memory available yet for this session.",
                "node_count": 0,
            })

        engine: CCEEngine = ctx.request_context.lifespan_state["engine"]
        macro = engine.macro_summary(nodes)

        return _ok({
            "session_id": params.session_id,
            "macro_summary": macro,
            "node_count": len(nodes),
            "topics": [n.topic_label for n in nodes],
        })
    except Exception as e:
        return _err(f"Failed to summarize session: {e}")


# ── Tool: session_stats ───────────────────────────────────────────────────────

@mcp.tool(
    name="cce_session_stats",
    annotations={
        "title": "Get diagnostic stats for a session",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def cce_session_stats(params: SessionStatsInput, ctx: Context) -> str:
    """Return diagnostic statistics for a session.

    Includes turn count, token counts, compression ratio,
    hot/warm/cold tier occupancy, and session timing.

    Args:
        params (SessionStatsInput): Validated input containing:
            - session_id (str): Session to inspect

    Returns:
        str: JSON with full stats dict
    """
    try:
        manager = _get_manager(ctx)
        stats = manager.session_stats(params.session_id)
        if stats is None:
            return _err(f"Session '{params.session_id}' not found")
        return _ok(stats)
    except Exception as e:
        return _err(f"Failed to get stats: {e}")


# ── Tool: close_session ───────────────────────────────────────────────────────

@mcp.tool(
    name="cce_close_session",
    annotations={
        "title": "Checkpoint and close a session",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def cce_close_session(params: CloseSessionInput, ctx: Context) -> str:
    """Close a session, optionally writing a macro summary to cold tier storage.

    Call this when a conversation ends. With checkpoint=True (default),
    it flushes remaining hot-tier turns to warm, generates a macro summary,
    and persists it to disk — preserving the session's memory for future reference.

    Args:
        params (CloseSessionInput): Validated input containing:
            - session_id (str): Session to close
            - checkpoint (bool): Whether to write cold tier summary (default: True)

    Returns:
        str: JSON with status and checkpoint info
    """
    try:
        manager = _get_manager(ctx)
        success = manager.close(params.session_id, checkpoint=params.checkpoint)
        if not success:
            return _err(f"Session '{params.session_id}' was not active")
        return _ok({
            "session_id": params.session_id,
            "checkpointed": params.checkpoint,
        })
    except Exception as e:
        return _err(f"Failed to close session: {e}")


# ── Tool: list_sessions ───────────────────────────────────────────────────────

@mcp.tool(
    name="cce_list_sessions",
    annotations={
        "title": "List all known CCE sessions",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def cce_list_sessions(ctx: Context) -> str:
    """List all sessions known to this CCE instance (active and closed).

    Returns session metadata including creation time, turn count, and status.
    Useful for session management and monitoring.

    Returns:
        str: JSON with list of session metadata dicts
    """
    try:
        manager = _get_manager(ctx)
        sessions = manager.list_sessions()
        return _ok({
            "sessions": sessions,
            "total": len(sessions),
            "active": sum(1 for s in sessions if s.get("is_active")),
        })
    except Exception as e:
        return _err(f"Failed to list sessions: {e}")


# ── Tool: stateless_compress ──────────────────────────────────────────────────

@mcp.tool(
    name="cce_stateless_compress",
    annotations={
        "title": "One-shot stateless compression (no persistence)",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def cce_stateless_compress(params: StatelessCompressInput, ctx: Context) -> str:
    """Compress a full conversation history on-demand without storing anything.

    Zero persistence — no DB writes, no disk I/O. Takes the full conversation,
    compresses it in-memory, retrieves relevant past context for the query,
    and returns the ready-to-use context payload.

    Best for: one-shot integrations, plugins that can't maintain state,
    or testing compression quality.

    Args:
        params (StatelessCompressInput): Validated input containing:
            - messages (list[dict]): Full conversation as {'role', 'content'} dicts
            - query (str): Current user query for context retrieval
            - top_k (Optional[int]): Memory nodes to retrieve
            - fmt (str): Output format — 'messages', 'string', or 'dict'

    Returns:
        str: JSON with compressed context payload and compression stats
    """
    try:
        processor = _get_processor(ctx)
        await ctx.report_progress(0.2, "Segmenting conversation...")
        result = processor.process(
            params.messages,
            query=params.query,
            top_k=params.top_k,
        )
        await ctx.report_progress(0.9, "Building context payload...")
        exported = result.payload.export(params.fmt)

        return _ok({
            "context": exported,
            "stats": result.stats,
            "token_count": result.payload.token_count,
            "past_blocks": len(result.payload.past_blocks),
            "recent_turns": len(result.payload.recent_turns),
        })
    except Exception as e:
        return _err(f"Stateless compression failed: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()  # stdio transport — default for local MCP tools