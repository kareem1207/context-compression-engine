"""Tests for the REST API (cce_rest)."""

import pytest
from fastapi.testclient import TestClient

from cce_core.config import CCEConfig
from cce_rest.app import create_app

CONVERSATION = [
    {"role": "user", "content": "I want to build a billing dashboard for my SaaS product."},
    {"role": "assistant", "content": "Start with plans, invoices and a webhook handler for payment events."},
    {"role": "user", "content": "How should the webhook handler verify signatures?"},
    {"role": "assistant", "content": "Verify the HMAC signature header against the raw request body before parsing."},
    {"role": "user", "content": "Different topic: what should I cook for dinner tonight?"},
    {"role": "assistant", "content": "Try a simple pasta with garlic, olive oil and fresh basil."},
    {"role": "user", "content": "Any tips for making the pasta sauce less bland?"},
    {"role": "assistant", "content": "Add salted pasta water, a pinch of chili and finish with lemon zest."},
] * 3


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    cfg = CCEConfig(base_dir=tmp_path_factory.mktemp("cce"), hot_tier_max_turns=4, retrieval_top_k=3)
    with TestClient(create_app(cfg)) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_stateless_compress_returns_context_and_stats(client):
    r = client.post("/v1/compress", json={"messages": CONVERSATION, "query": "how do I verify webhooks?"})
    assert r.status_code == 200
    body = r.json()
    assert body["context"][0]["role"] == "system"
    assert body["stats"]["total_turns"] == len(CONVERSATION)
    assert body["stats"]["compression_ratio"] > 0


def test_stateless_string_format(client):
    r = client.post("/v1/compress", json={"messages": CONVERSATION, "query": "pasta", "fmt": "string"})
    assert r.status_code == 200 and isinstance(r.json()["context"], str)


def test_session_lifecycle(client):
    sid = "rest-demo"
    add = client.post(f"/v1/sessions/{sid}/messages", json={"messages": CONVERSATION})
    assert add.status_code == 201 and add.json()["turns_added"] == len(CONVERSATION)

    ctx = client.post(f"/v1/sessions/{sid}/context", json={"query": "webhook signatures"})
    assert ctx.status_code == 200
    assert ctx.json()["token_count"] > 0
    assert any(m["role"] == "system" for m in ctx.json()["context"])

    assert client.get(f"/v1/sessions/{sid}/stats").status_code == 200
    summary = client.get(f"/v1/sessions/{sid}/summary").json()
    assert summary["session_id"] == sid
    assert any(s["session_id"] == sid for s in client.get("/v1/sessions").json()["sessions"])

    assert client.delete(f"/v1/sessions/{sid}").json()["deleted"] is True
    assert client.get(f"/v1/sessions/{sid}/stats").status_code == 404


def test_unknown_session_is_404(client):
    assert client.post("/v1/sessions/ghost/context", json={"query": "hi"}).status_code == 404
    assert client.get("/v1/sessions/ghost/summary").status_code == 404


@pytest.mark.parametrize("bad", ["../etc", "a b", "x" * 80, "a/b"])
def test_bad_session_ids_are_rejected(client, bad):
    r = client.post(f"/v1/sessions/{bad}/messages", json={"messages": CONVERSATION[:2]})
    assert r.status_code in (404, 422)


def test_validation(client):
    assert client.post("/v1/compress", json={"messages": [], "query": "x"}).status_code == 422
    assert client.post("/v1/compress", json={"messages": CONVERSATION[:2], "query": ""}).status_code == 422
    assert client.post("/v1/compress", json={"messages": CONVERSATION[:2], "query": "x", "top_k": 0}).status_code == 422
    assert client.post("/v1/compress", json={"messages": CONVERSATION[:2], "query": "x", "fmt": "xml"}).status_code == 422
