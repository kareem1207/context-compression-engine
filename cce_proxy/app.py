"""
CCE proxy: point an OpenAI-compatible client at this server instead of the model server, and a long chat is
trimmed before it is sent.

    client  --POST /v1/chat/completions (whole history)-->  cce-proxy  --(trimmed messages)-->  upstream model server

What it does with each chat request:
  1. Finds the session (header `X-CCE-Session`, else the request's `user` field, else a hash of the first
     user message) and stores the messages it has not seen yet.
  2. If the whole request is under the token limit, forwards it unchanged. Otherwise it replaces the history
     with: your own system messages, one CCE system message holding the relevant older notes, and the most
     recent turns, which include the last user message.
  3. Forwards to the upstream server (`CCE_UPSTREAM_URL`, the client's `Authorization` header is passed on),
     streams or returns the answer unchanged, and stores the assistant's reply.

Responses carry `X-CCE-Session`, `X-CCE-Trimmed`, `X-CCE-Tokens-Before` and `X-CCE-Tokens-After`.

Left alone (forwarded unchanged, nothing stored): requests with tool calls or tool messages, images or other
non-text parts, and histories shorter than what the session already holds (the client started over).

Run:  cce-proxy --upstream http://localhost:8080/v1
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response, StreamingResponse

from cce_core.config import CCEConfig
from cce_core.engine import CCEEngine
from cce_core.ids import is_valid_session_id
from cce_core.ingestion import tokenizer
from cce_core.session.manager import SessionManager

_HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "content-encoding", "host"}


def session_id_for(request_headers, body: dict) -> str:
    """The session this chat belongs to; ids that are not valid session ids are hashed."""
    raw = request_headers.get("x-cce-session") or body.get("user")
    if not raw:
        first = next((m for m in body.get("messages", []) if m.get("role") == "user"), {})
        raw = "chat-" + hashlib.sha1(str(first.get("content", "")).encode("utf-8")).hexdigest()[:16]
    raw = str(raw)
    return raw if is_valid_session_id(raw) else "h-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def flatten(content) -> str | None:
    """Text of a message's content; None if it holds anything but text (images, ...)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            else:
                return None
        return "\n".join(parts)
    return None


def create_app(
    config: CCEConfig | None = None,
    upstream: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build the proxy. `transport` lets tests stand in for the upstream server."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg = config or CCEConfig(context_max_tokens=int(os.getenv("CCE_PROXY_MAX_TOKENS", "4096")))
        app.state.manager = SessionManager(CCEEngine(cfg))
        app.state.upstream = (upstream or os.getenv("CCE_UPSTREAM_URL", "http://localhost:8080/v1")).rstrip("/")
        app.state.client = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(600.0, connect=10.0))
        try:
            yield
        finally:
            await app.state.client.aclose()
            app.state.manager.close_all(checkpoint=True)

    app = FastAPI(title="CCE proxy", version="1.0.0", lifespan=lifespan)

    def _guard(request: Request):
        key = os.getenv("CCE_API_KEY")
        if key and request.headers.get("x-cce-key") != key:
            return JSONResponse({"error": {"message": "Missing or invalid X-CCE-Key", "type": "auth"}}, status_code=401)
        return None

    def _forward_headers(request: Request) -> dict:
        return {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP and not k.lower().startswith("x-cce-")}

    @app.get("/health")
    async def health(request: Request):
        return {"status": "ok", "upstream": request.app.state.upstream}

    @app.get("/v1/models")
    async def models(request: Request):
        denied = _guard(request)
        if denied:
            return denied
        r = await request.app.state.client.get(f"{request.app.state.upstream}/models", headers=_forward_headers(request))
        return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))

    def _prepare(manager: SessionManager, sid: str, body: dict) -> tuple[dict, dict]:
        """Store new messages and return (body to forward, info for the response headers)."""
        messages = body.get("messages") or []
        info = {"trimmed": False, "before": 0, "after": 0}
        texts = [flatten(m.get("content")) for m in messages]
        plain = (
            texts and all(t is not None for t in texts)
            and not any(m.get("tool_calls") or m.get("role") in ("tool", "function") for m in messages)
        )
        info["before"] = sum(tokenizer.count(t or "") + 4 for t in texts if t is not None)
        if not plain:
            return body, info
        chat = [(m["role"], t) for m, t in zip(messages, texts) if m.get("role") in ("user", "assistant")]
        system = [m for m in messages if m.get("role") in ("system", "developer")]
        if not chat or chat[-1][0] != "user":
            return body, info
        store = manager.get_or_open(sid)
        known = store.stats()["turns_ingested"]
        if len(chat) < known:
            return body, info                    # the client started a new history under the same session
        for role, text in chat[known:]:
            manager.add_message(sid, {"role": role, "content": text})
        if info["before"] <= manager.config.context_max_tokens:
            info["after"] = info["before"]
            return body, info
        payload = manager.build_context_payload(sid, chat[-1][1])
        out = payload.to_messages()                         # [CCE system message, recent turns...]
        trimmed = [*system, *out]
        info.update(trimmed=True, after=sum(tokenizer.count(m["content"]) + 4 for m in trimmed))
        return {**body, "messages": trimmed}, info

    def _store_reply(manager: SessionManager, sid: str, text: str) -> None:
        if text.strip():
            manager.add_message(sid, {"role": "assistant", "content": text})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        denied = _guard(request)
        if denied:
            return denied
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": {"message": "Body must be JSON", "type": "invalid_request"}}, status_code=400)
        manager: SessionManager = request.app.state.manager
        sid = session_id_for(request.headers, body)
        stateful = True
        try:
            forward, info = await run_in_threadpool(_prepare, manager, sid, body)
        except Exception as exc:                           # never block the chat because memory failed
            forward, info, stateful = body, {"trimmed": False, "before": 0, "after": 0}, False
            note = f"CCE skipped: {exc}"[:200]
        else:
            note = ""
        headers = {
            "X-CCE-Session": sid, "X-CCE-Trimmed": "1" if info["trimmed"] else "0",
            "X-CCE-Tokens-Before": str(info["before"]), "X-CCE-Tokens-After": str(info["after"]),
        }
        if note:
            headers["X-CCE-Note"] = note
        url = f"{request.app.state.upstream}/chat/completions"
        client: httpx.AsyncClient = request.app.state.client

        if forward.get("stream"):
            upstream_request = client.build_request("POST", url, json=forward, headers=_forward_headers(request))
            upstream_response = await client.send(upstream_request, stream=True)

            async def relay():
                collected: list[str] = []
                try:
                    async for line in upstream_response.aiter_lines():
                        if line.startswith("data:") and "[DONE]" not in line:
                            try:
                                delta = json.loads(line[5:].strip())["choices"][0].get("delta", {})
                                collected.append(delta.get("content") or "")
                            except (ValueError, KeyError, IndexError):
                                pass
                        yield line + "\n"
                finally:
                    await upstream_response.aclose()
                    if stateful and upstream_response.status_code == 200:
                        await run_in_threadpool(_store_reply, manager, sid, "".join(collected))

            return StreamingResponse(relay(), status_code=upstream_response.status_code,
                                     media_type="text/event-stream", headers=headers)

        r = await client.post(url, json=forward, headers=_forward_headers(request))
        if stateful and r.status_code == 200:
            try:
                text = r.json()["choices"][0]["message"].get("content") or ""
                await run_in_threadpool(_store_reply, manager, sid, text)
            except (ValueError, KeyError, IndexError):
                pass
        return Response(r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"), headers=headers)

    return app


app = create_app()


def main(argv: list[str] | None = None) -> None:
    """Console entry point: `cce-proxy`."""
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(
        prog="cce-proxy",
        description="OpenAI-compatible proxy: change your client's base URL to this server and long chats are trimmed with CCE.",
        epilog=("Environment: CCE_UPSTREAM_URL, CCE_PROXY_MAX_TOKENS (limit before trimming, default 4096), CCE_DATA_DIR, "
                "CCE_EMBEDDER, CCE_API_KEY (clients then send X-CCE-Key; required for a non-local host)."),
    )
    ap.add_argument("--host", default=os.getenv("CCE_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.getenv("CCE_PORT", "9200")))
    ap.add_argument("--upstream", default=None, help="base URL of the model server, e.g. http://localhost:8080/v1")
    args = ap.parse_args(argv)
    if args.upstream:
        os.environ["CCE_UPSTREAM_URL"] = args.upstream
    if args.host not in ("127.0.0.1", "localhost", "::1") and not os.getenv("CCE_API_KEY"):
        raise SystemExit("Refusing to listen on a public address without CCE_API_KEY set.")
    uvicorn.run("cce_proxy.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
