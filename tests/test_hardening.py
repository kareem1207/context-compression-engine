"""Session-id validation, CORS preflight, embedder identity, and package metadata."""

import asyncio
import shutil
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from cce_core.config import CCEConfig
from cce_core.engine import CCEEngine, EmbedderMismatchError
from cce_core.ids import is_valid_session_id
from cce_core.memory.cold_tier import ColdTier, MacroSummary
from cce_core.memory.store import MemoryStore
from cce_mcp import schema, server
from cce_rest.app import create_app

from tests.test_live_sessions import make_config

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).parent / "fixtures" / "cce_0_1_data"

BAD_IDS = [
    "../../x", "../x", "..", ".", "a/b", "a\\b", "..\\..\\x", "/etc/passwd", "C:\\x", "",
    "a b", "x" * 65, "ok\n", "tab\tid", "id\x00", "naïve", "index",
]


def outside_files(base: Path) -> list[Path]:
    """Files under the data dir's parent that are not part of the data dir itself."""
    return [p for p in base.parent.rglob("*") if p.is_file() and base not in p.parents]


def summary(sid: str) -> MacroSummary:
    return MacroSummary(sid, "text", 1, 2, 10, 5, ["t"], datetime.now(timezone.utc))


# ── Fix 6.1: session ids ──────────────────────────────────────────────────────

def test_valid_ids_are_accepted():
    for ok in ("a", "user-123.chat_v2", "x" * 64, "1.2.3", "..a", "a.."):
        assert is_valid_session_id(ok), ok


@pytest.mark.parametrize("bad", BAD_IDS)
def test_invalid_ids_are_rejected(bad):
    assert not is_valid_session_id(bad)


@pytest.mark.parametrize("bad", BAD_IDS)
def test_cold_tier_never_writes_outside_its_folder(tmp_path, bad):
    cfg = CCEConfig(base_dir=tmp_path / "data")
    cold = ColdTier(cfg)
    with pytest.raises(ValueError):
        cold.write(summary(bad))
    assert cold.get(bad) is None and cold.exists(bad) is False and cold.delete(bad) is False
    assert outside_files(cfg.base_dir) == []
    assert list(cfg.cold_dir.glob("*.json")) == []


@pytest.mark.parametrize("bad", BAD_IDS)
def test_session_manager_and_store_reject_them(tmp_path, bad):
    cfg = make_config(tmp_path)
    mgr = CCEEngine(cfg).create_session_manager()
    for call in (
        lambda: mgr.open(bad),
        lambda: mgr.add_message(bad, {"role": "user", "content": "hi"}),
        lambda: mgr.add_messages(bad, [{"role": "user", "content": "hi"}]),
        lambda: mgr.build_context(bad, "hi"),
        lambda: mgr.delete(bad),
        lambda: MemoryStore(bad, cfg),
    ):
        with pytest.raises(ValueError):
            call()
    assert outside_files(cfg.base_dir) == []


@pytest.mark.parametrize("model", [
    schema.IngestTurnInput, schema.IngestHistoryInput, schema.RetrieveContextInput,
    schema.SummarizeSessionInput, schema.CloseSessionInput, schema.SessionStatsInput,
])
@pytest.mark.parametrize("bad", ["../../x", "..", "a/b", "x" * 65])
def test_mcp_inputs_validate_session_id(model, bad):
    extra = {
        schema.IngestTurnInput: {"role": "user", "content": "hi"},
        schema.IngestHistoryInput: {"messages": [{"role": "user", "content": "hi"}]},
        schema.RetrieveContextInput: {"query": "hi"},
    }.get(model, {})
    with pytest.raises(ValidationError):
        model(session_id=bad, **extra)
    model(session_id="fine-id", **extra)


def test_traversal_through_the_mcp_tools_writes_nothing_outside(tmp_path):
    """This used to create <data dir>/../escaped.json when the session was closed."""
    cfg = make_config(tmp_path)
    engine = CCEEngine(cfg)
    ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context={"engine": engine, "manager": engine.create_session_manager()}))

    async def progress(*a, **k):
        return None

    ctx.report_progress = progress
    with pytest.raises(ValidationError):
        server.cce_ingest_turn(schema.IngestTurnInput(session_id="../../escaped", role="user", content="hello there"), ctx)
    # and if a caller skipped the schema, the manager still refuses
    out = asyncio.run(server.cce_ingest_turn(
        schema.IngestTurnInput.model_construct(session_id="../../escaped", role="user", content="hello there"), ctx))
    assert '"status": "error"' in out
    assert outside_files(cfg.base_dir) == []


@pytest.mark.parametrize("bad", ["%2E%2E", "a%2Fb", "..%5Cx", "x" * 65, "a%20b"])
def test_rest_rejects_traversal_ids(tmp_path, bad):
    cfg = make_config(tmp_path)
    with TestClient(create_app(cfg)) as c:
        r = c.post(f"/v1/sessions/{bad}/messages", json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code in (404, 422)
        assert c.post(f"/v1/sessions/{bad}/close").status_code in (404, 422)
    assert outside_files(cfg.base_dir) == []


# ── Fix 6.2: CORS ─────────────────────────────────────────────────────────────

def test_cors_preflight_allows_the_auth_headers(tmp_path):
    with TestClient(create_app(make_config(tmp_path))) as c:
        r = c.options("/v1/compress", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization, x-api-key, content-type",
        })
        assert r.status_code == 200, r.text
        allowed = {h.strip().lower() for h in r.headers["access-control-allow-headers"].split(",")}
        assert {"authorization", "x-api-key", "content-type"} <= allowed
        assert r.headers["access-control-allow-origin"] == "http://localhost:3000"

        other = c.options("/v1/compress", headers={
            "Origin": "http://evil.example", "Access-Control-Request-Method": "POST",
        })
        assert "access-control-allow-origin" not in other.headers


# ── Fix 6.3: embedder identity ────────────────────────────────────────────────

def test_embedder_is_recorded_and_a_different_one_is_refused(tmp_path):
    cfg = make_config(tmp_path)                         # hash, 512 dimensions
    engine = CCEEngine(cfg)
    engine.open_session("a").close()
    assert engine.embedder_identity() == {"name": "hash", "dim": 512}
    CCEEngine(cfg).open_session("a").close()            # same embedder: fine

    other = CCEEngine(make_config(tmp_path, hash_dim=256))
    with pytest.raises(EmbedderMismatchError, match="512 dimensions"):
        other.open_session("a")
    with pytest.raises(ValueError, match="CCE_EMBEDDER"):   # the message says what to do
        other.open_session("b")


def test_a_0_1_folder_without_a_record_is_checked_by_its_vector_size(tmp_path):
    dst = tmp_path / "legacy"
    shutil.copytree(FIXTURE, dst)
    with pytest.raises(EmbedderMismatchError, match="512-dimension"):
        CCEEngine(CCEConfig(embedder="hash", hash_dim=256, base_dir=dst)).open_session("legacy")
    store = CCEEngine(CCEConfig(embedder="hash", base_dir=dst)).open_session("legacy")   # matching one works
    assert store.warm.get_meta("embedder") is not None                                   # and is recorded now
    store.close()


# ── Fix 6.4: packaging ────────────────────────────────────────────────────────

def _pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_every_package_in_the_repo_is_in_the_wheel_and_sdist():
    cfg = _pyproject()
    packages = {p.parent.name for p in ROOT.glob("cce_*/__init__.py")}
    assert {"cce_core", "cce_mcp", "cce_rest", "cce_ui"} <= packages
    assert packages <= set(cfg["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"])
    assert packages <= set(cfg["tool"]["hatch"]["build"]["targets"]["sdist"]["include"])


def test_console_scripts_point_at_real_functions():
    import importlib

    scripts = _pyproject()["project"]["scripts"]
    assert set(scripts) == {"cce-rest", "cce-mcp", "cce-ui", "cce", "cce-proxy"}
    for target in scripts.values():
        module, func = target.split(":")
        assert callable(getattr(importlib.import_module(module), func))


@pytest.mark.parametrize("target", ["cce_rest.app:main", "cce_mcp.server:main", "cce_ui.cli:main", "cce_proxy.app:main", "cce_core.cli:main"])
def test_help_prints_usage_and_exits(target, capsys):
    import importlib

    module, func = target.split(":")
    with pytest.raises(SystemExit) as e:
        getattr(importlib.import_module(module), func)(["--help"])
    assert e.value.code == 0
    assert "usage: cce" in capsys.readouterr().out
