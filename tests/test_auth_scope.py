"""API keys limited to some sessions (REST API and proxy)."""

import httpx
import pytest
from fastapi.testclient import TestClient

from cce_core.auth import allows, configured_keys, parse_keys, scope_for
from cce_proxy.app import create_app as create_proxy
from cce_rest.app import create_app

from tests.test_live_sessions import make_config

MSGS = [{"role": "user", "content": "hello there"}, {"role": "assistant", "content": "hi"}]
KEYS = "alice-key=user-1,user-2;team-key=team-*;admin-key=*"


def test_parsing_and_matching():
    keys = parse_keys(KEYS)
    assert keys == {"alice-key": ["user-1", "user-2"], "team-key": ["team-*"], "admin-key": ["*"]}
    assert allows(["user-1", "user-2"], "user-2") and not allows(["user-1"], "user-10")
    assert allows(["team-*"], "team-7") and not allows(["team-*"], "teams") and allows(["*"], "anything")
    assert not allows([], "x")
    assert scope_for("team-key", keys) == ["team-*"] and scope_for("nope", keys) is None and scope_for(None, keys) is None
    with pytest.raises(ValueError, match="key=pattern"):
        parse_keys("no-equals-sign")


def test_single_key_is_all_sessions_and_keys_are_merged(monkeypatch):
    monkeypatch.setenv("CCE_API_KEY", "single")
    monkeypatch.setenv("CCE_API_KEYS", "alice=user-1")
    assert configured_keys() == {"alice": ["user-1"], "single": ["*"]}
    monkeypatch.delenv("CCE_API_KEY")
    monkeypatch.delenv("CCE_API_KEYS")
    assert configured_keys() == {}


def auth(key):
    return {"Authorization": f"Bearer {key}"}


def test_rest_keys_only_reach_their_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("CCE_API_KEYS", KEYS)
    with TestClient(create_app(make_config(tmp_path))) as c:
        assert c.get("/v1/sessions").status_code == 401
        assert c.get("/v1/sessions", headers=auth("wrong")).status_code == 401
        assert c.get("/health").status_code == 200

        for key, sid in (("alice-key", "user-1"), ("team-key", "team-9"), ("admin-key", "anything")):
            assert c.post(f"/v1/sessions/{sid}/messages", json={"messages": MSGS}, headers=auth(key)).status_code == 201

        # outside the key's sessions: 403 whether or not the session exists
        assert c.post("/v1/sessions/user-3/messages", json={"messages": MSGS}, headers=auth("alice-key")).status_code == 403
        for route in ("stats", "summary", "export", "pins"):
            assert c.get(f"/v1/sessions/team-9/{route}", headers=auth("alice-key")).status_code == 403, route
            assert c.get(f"/v1/sessions/ghost/{route}", headers=auth("alice-key")).status_code == 403, route
        assert c.post("/v1/sessions/team-9/context", json={"query": "hi"}, headers=auth("alice-key")).status_code == 403
        assert c.post("/v1/sessions/team-9/forget", json={"start": 0, "end": 1}, headers=auth("alice-key")).status_code == 403
        assert c.delete("/v1/sessions/team-9", headers=auth("alice-key")).status_code == 403
        assert c.post("/v1/sessions/user-1/context", json={"query": "hi"}, headers=auth("alice-key")).status_code == 200
        assert c.post("/v1/sessions/user-1/context", json={"query": "hi"}, headers=auth("team-key")).status_code == 403

        # listings only show what the key may see; sessions that need no session still work
        seen = lambda key: sorted(s["session_id"] for s in c.get("/v1/sessions", headers=auth(key)).json()["sessions"])
        assert seen("alice-key") == ["user-1"] and seen("team-key") == ["team-9"] and seen("admin-key") == ["anything", "team-9", "user-1"]
        assert c.post("/v1/compress", json={"messages": MSGS, "query": "hi"}, headers=auth("alice-key")).status_code == 200


def test_the_old_single_key_still_means_everything(tmp_path, monkeypatch):
    monkeypatch.setenv("CCE_API_KEY", "secret")
    with TestClient(create_app(make_config(tmp_path))) as c:
        assert c.post("/v1/sessions/any/messages", json={"messages": MSGS}, headers={"X-API-Key": "secret"}).status_code == 201
        assert c.get("/v1/sessions", headers=auth("other")).status_code == 401


def test_proxy_keys_are_scoped_too(tmp_path, monkeypatch):
    monkeypatch.setenv("CCE_API_KEYS", KEYS)

    def upstream(request):
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    body = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    with TestClient(create_proxy(make_config(tmp_path), "http://up/v1", httpx.MockTransport(upstream))) as c:
        assert c.post("/v1/chat/completions", json=body).status_code == 401
        mine = c.post("/v1/chat/completions", json=body, headers={"X-CCE-Key": "alice-key", "X-CCE-Session": "user-1"})
        assert mine.status_code == 200
        theirs = c.post("/v1/chat/completions", json=body, headers={"X-CCE-Key": "alice-key", "X-CCE-Session": "team-1"})
        assert theirs.status_code == 403 and theirs.json()["error"]["type"] == "forbidden"
        assert c.post("/v1/chat/completions", json=body, headers={"X-CCE-Key": "admin-key", "X-CCE-Session": "team-1"}).status_code == 200
