"""The OpenAI-compatible proxy against a stand-in upstream, no servers needed.

    pip install "context-compression-engine[rest]"
    python examples/integrations/proxy_demo.py

With a real model server:   cce-proxy --upstream http://localhost:8080/v1
then point any OpenAI client at http://127.0.0.1:9200/v1 (see the README).
"""
import tempfile
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from cce_core.config import CCEConfig
from cce_proxy.app import create_app


def upstream(request: httpx.Request) -> httpx.Response:          # a model server that only counts what it receives
    messages = __import__("json").loads(request.content)["messages"]
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": f"got {len(messages)} messages"}}]})


config = CCEConfig(embedder="hash", base_dir=Path(tempfile.mkdtemp()), context_max_tokens=450, hot_tier_max_turns=4)
history = [{"role": "user", "content": "Our invoice retry limit is seven attempts."}, {"role": "assistant", "content": "Noted."}]
for topic in ("hiking boots", "pandas merge", "sourdough starter", "stripe webhooks", "guitar chords"):
    history += [{"role": "user", "content": f"Tell me about {topic} and everything related to {topic}, in detail please."},
                {"role": "assistant", "content": f"Here is a long answer about {topic}. " * 6}]
history.append({"role": "user", "content": "How many invoice retry attempts are allowed?"})

with TestClient(create_app(config, "http://upstream/v1", httpx.MockTransport(upstream))) as client:
    r = client.post("/v1/chat/completions", json={"model": "x", "messages": history}, headers={"X-CCE-Session": "demo"})
    print("sent", len(history), "messages ->", r.json()["choices"][0]["message"]["content"])
    print("trimmed:", r.headers["x-cce-trimmed"], "| tokens", r.headers["x-cce-tokens-before"], "->", r.headers["x-cce-tokens-after"])
