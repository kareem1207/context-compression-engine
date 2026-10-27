"""The OpenAI-compatible proxy (cce-proxy), with a stand-in for the upstream model server."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from cce_core.config import CCEConfig
from cce_core.ingestion import tokenizer
from cce_proxy.app import create_app, flatten, session_id_for

from tests.test_live_sessions import PLANTED, make_config, make_messages


class Upstream:
    """Records what the proxy forwards and answers like a chat completions server."""

    def __init__(self):
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        self.status = 200
        self.reply = "Seven attempts."

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.requests.append(body)
        self.headers.append(dict(request.headers))
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "stub"}]})
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "nope"}})
        if body.get("stream"):
            chunks = [{"choices": [{"delta": {"content": (" " if i else "") + w}}]} for i, w in enumerate(self.reply.split(" "))]
            text = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, content=text.encode(), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": self.reply}}]})


@pytest.fixture
def upstream():
    return Upstream()


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path, context_max_tokens=450, hot_tier_max_turns=4, retrieval_top_k=3)


def client_for(cfg, upstream, **kw):
    return TestClient(create_app(cfg, "http://upstream/v1", httpx.MockTransport(upstream), **kw))


def chat(messages, **extra):
    return {"model": "stub", "messages": messages, **extra}


def test_short_chats_are_forwarded_unchanged_and_remembered(cfg, upstream):
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hello there"}]
    with client_for(cfg, upstream) as c:
        r = c.post("/v1/chat/completions", json=chat(msgs), headers={"X-CCE-Session": "demo"})
        assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == upstream.reply
        assert r.headers["x-cce-trimmed"] == "0" and r.headers["x-cce-session"] == "demo"
        assert upstream.requests[0]["messages"] == msgs
        stats = c.app.state.manager.get_or_open("demo").stats()
        assert stats["turns_ingested"] == 2            # the user message and the assistant's reply


def test_long_history_is_trimmed_and_the_old_fact_survives(cfg, upstream):
    history = make_messages()
    question = {"role": "user", "content": "How many invoice retry attempts are allowed?"}
    full = [{"role": "system", "content": "You are a billing assistant."}, *history, question]
    with client_for(cfg, upstream) as c:
        r = c.post("/v1/chat/completions", json=chat(full), headers={"X-CCE-Session": "long"})
        assert r.status_code == 200 and r.headers["x-cce-trimmed"] == "1"
        sent = upstream.requests[0]["messages"]
        assert sent[0] == full[0]                                   # the caller's own system message stays first
        assert sent[-1] == question                                 # and the question stays last
        assert len(sent) < len(full) // 2
        assert PLANTED in "\n".join(m["content"] for m in sent)     # the fact from the old part is still there
        assert int(r.headers["x-cce-tokens-after"]) < int(r.headers["x-cce-tokens-before"])
        assert tokenizer.count_messages(sent) <= 450 + 150


def test_second_request_stores_only_the_new_messages(cfg, upstream):
    history = make_messages()[:10]
    with client_for(cfg, upstream) as c:
        c.post("/v1/chat/completions", json=chat([*history, {"role": "user", "content": "one"}]), headers={"X-CCE-Session": "inc"})
        n1 = c.app.state.manager.get_or_open("inc").stats()["turns_ingested"]
        again = [*history, {"role": "user", "content": "one"}, {"role": "assistant", "content": upstream.reply},
                 {"role": "user", "content": "two"}]
        c.post("/v1/chat/completions", json=chat(again), headers={"X-CCE-Session": "inc"})
        assert c.app.state.manager.get_or_open("inc").stats()["turns_ingested"] == n1 + 2     # "two" + the new reply


def test_streaming_is_relayed_and_the_reply_is_remembered(cfg, upstream):
    upstream.reply = "Seven attempts per invoice"
    with client_for(cfg, upstream) as c:
        with c.stream("POST", "/v1/chat/completions", json=chat([{"role": "user", "content": "how many?"}], stream=True),
                      headers={"X-CCE-Session": "st"}) as r:
            lines = [l for l in r.iter_lines() if l]
        assert lines[-1] == "data: [DONE]" and len(lines) == len(upstream.reply.split()) + 1
        texts = [t.content for t in c.app.state.manager.get_or_open("st").get_hot_turns()]
        assert texts == ["how many?", upstream.reply]


@pytest.mark.parametrize("message", [
    {"role": "tool", "content": "42", "tool_call_id": "1"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
    {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "http://x/y.png"}}]},
])
def test_requests_the_proxy_cannot_represent_pass_through(cfg, upstream, message):
    msgs = [{"role": "user", "content": "first"}, message, {"role": "user", "content": "next"}]
    with client_for(cfg, upstream) as c:
        r = c.post("/v1/chat/completions", json=chat(msgs), headers={"X-CCE-Session": "pt"})
        assert r.status_code == 200 and r.headers["x-cce-trimmed"] == "0"
        assert upstream.requests[0]["messages"] == msgs


def test_session_ids(cfg):
    assert session_id_for({"x-cce-session": "abc"}, {}) == "abc"
    assert session_id_for({}, {"user": "user-1"}) == "user-1"
    hashed = session_id_for({"x-cce-session": "../../etc"}, {})
    assert hashed.startswith("h-") and "/" not in hashed
    same = {"messages": [{"role": "user", "content": "hi"}]}
    assert session_id_for({}, same) == session_id_for({}, same) and session_id_for({}, same).startswith("chat-")


def test_flatten():
    assert flatten("a") == "a" and flatten(None) == "" and flatten([{"type": "text", "text": "x"}, {"type": "text", "text": "y"}]) == "x\ny"
    assert flatten([{"type": "image_url"}]) is None


def test_upstream_errors_and_headers_are_passed_on(cfg, upstream):
    with client_for(cfg, upstream) as c:
        r = c.post("/v1/chat/completions", json=chat([{"role": "user", "content": "hi"}]),
                   headers={"Authorization": "Bearer sk-test", "X-CCE-Session": "e"})
        assert upstream.headers[0]["authorization"] == "Bearer sk-test"
        assert "x-cce-session" not in upstream.headers[0]
        upstream.status = 429
        r = c.post("/v1/chat/completions", json=chat([{"role": "user", "content": "again"}]), headers={"X-CCE-Session": "e"})
        assert r.status_code == 429 and r.json()["error"]["message"] == "nope"
        assert c.get("/v1/models").json()["data"][0]["id"] == "stub"


def test_api_key_guard(cfg, upstream, monkeypatch):
    monkeypatch.setenv("CCE_API_KEY", "k")
    with client_for(cfg, upstream) as c:
        body = chat([{"role": "user", "content": "hi"}])
        assert c.post("/v1/chat/completions", json=body).status_code == 401
        assert c.post("/v1/chat/completions", json=body, headers={"X-CCE-Key": "k"}).status_code == 200
        assert c.get("/health").status_code == 200


def test_memory_failure_never_blocks_the_chat(cfg, upstream, monkeypatch):
    with client_for(cfg, upstream) as c:
        monkeypatch.setattr(c.app.state.manager, "get_or_open", lambda sid: (_ for _ in ()).throw(RuntimeError("disk full")))
        msgs = [{"role": "user", "content": "hi"}]
        r = c.post("/v1/chat/completions", json=chat(msgs))
        assert r.status_code == 200 and "disk full" in r.headers["x-cce-note"]
        assert upstream.requests[0]["messages"] == msgs


def test_bad_json_is_a_400(cfg, upstream):
    with client_for(cfg, upstream) as c:
        assert c.post("/v1/chat/completions", content=b"{nope", headers={"content-type": "application/json"}).status_code == 400
