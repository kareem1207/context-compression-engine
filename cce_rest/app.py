"""
CCE REST API
A FastAPI wrapper around the Context Compression Engine, so any language or
app can use CCE over plain HTTP (the npm client in cce_sdk_js talks to this).

Run:  uv run python -m cce_rest.app          (http://localhost:9100)
Docs: http://localhost:9100/docs
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cce_core.config import CCEConfig
from cce_core.engine import CCEEngine
from cce_core.session.manager import SessionManager
from cce_rest.routes import router


def create_app(config: CCEConfig | None = None) -> FastAPI:
    """Build the app. Pass a CCEConfig (e.g. with a temp data dir) for tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg = config or CCEConfig(
            summarizer_mode=os.getenv("CCE_SUMMARIZER_MODE", "extractive"),
            hot_tier_max_turns=int(os.getenv("CCE_HOT_TIER_TURNS", "10")),
            retrieval_top_k=int(os.getenv("CCE_TOP_K", "5")),
            context_max_tokens=int(os.getenv("CCE_MAX_TOKENS", "2048")),
        )
        engine = CCEEngine(cfg)
        app.state.engine = engine
        app.state.manager = SessionManager(engine)
        app.state.processor = engine.create_stateless_processor()
        try:
            yield
        finally:
            app.state.manager.close_all(checkpoint=True)

    app = FastAPI(
        title="Context Compression Engine",
        version="1.0.0",
        description="Compress long conversations into compact memory and fetch just the context an LLM needs.",
        lifespan=lifespan,
    )

    origins = [o.strip() for o in os.getenv("CCE_CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    )
    app.include_router(router)
    return app


app = create_app()


def main(argv: list[str] | None = None) -> None:
    """Console entry point: `cce-rest`."""
    import argparse

    import uvicorn

    argparse.ArgumentParser(
        prog="cce-rest",
        description="Run the CCE REST API (docs at /docs).",
        epilog=(
            "Settings come from environment variables: CCE_HOST (default 127.0.0.1), CCE_PORT (9100), "
            "CCE_API_KEY (required for a non-local host), CCE_CORS_ORIGINS, CCE_DATA_DIR, CCE_EMBEDDER, "
            "CCE_HOT_TIER_TURNS, CCE_TOP_K, CCE_MAX_TOKENS, CCE_SUMMARIZER_MODE."
        ),
    ).parse_args(argv)

    host = os.getenv("CCE_HOST", "127.0.0.1")
    if host not in ("127.0.0.1", "localhost", "::1") and not os.getenv("CCE_API_KEY"):
        raise SystemExit("Refusing to listen on a public address without CCE_API_KEY set.")
    uvicorn.run("cce_rest.app:app", host=host, port=int(os.getenv("CCE_PORT", "9100")))


if __name__ == "__main__":
    main()
