"""Console entry point: `cce-ui`."""

from __future__ import annotations


def main(argv: list[str] | None = None) -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(
        prog="cce-ui",
        description="Run the CCE dashboard demo: a conversation is replayed through the memory tiers live.",
        epilog="Uses CCE_DATA_DIR for storage and CCE_LLM_ENDPOINT for the optional local model.",
    )
    parser.add_argument("--host", default=os.getenv("CCE_UI_HOST", "127.0.0.1"), help="default 127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.getenv("CCE_UI_PORT", "7860")), help="default 7860")
    args = parser.parse_args(argv)

    import uvicorn

    from cce_ui.app import app   # imported late: it opens the demo session as soon as it is loaded

    print(f"\n  CCE Dashboard -> http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
