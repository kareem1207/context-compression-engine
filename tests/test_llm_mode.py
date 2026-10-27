"""summarizer_mode='llm' is experimental: it warns, it works against an OpenAI-style server, and it falls back."""

import warnings

import httpx
import pytest

from cce_core.compression import summarizer as summ
from cce_core.compression.summarizer import ExperimentalFeatureWarning, Summarizer
from cce_core.config import CCEConfig
from cce_core.engine import CCEEngine

LONG = " ".join(f"Sentence number {i} says that the deployment window is Tuesday." for i in range(40))


def use_server(monkeypatch, handler):
    real = httpx.Client
    monkeypatch.setattr(summ.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


def test_extractive_is_the_default_and_does_not_warn():
    assert CCEConfig().summarizer_mode == "extractive"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Summarizer(CCEConfig())


def test_llm_mode_warns_that_it_is_experimental():
    with pytest.warns(ExperimentalFeatureWarning, match="experimental"):
        Summarizer(CCEConfig(summarizer_mode="llm"))


def test_llm_mode_uses_the_servers_answer(monkeypatch, tmp_path):
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, json={"choices": [{"message": {"content": "A short model-written summary."}}]})

    use_server(monkeypatch, handler)
    with pytest.warns(ExperimentalFeatureWarning):
        s = Summarizer(CCEConfig(summarizer_mode="llm", llm_endpoint="http://llm/v1", base_dir=tmp_path))
    assert s._summarize(LONG, 50) == "A short model-written summary."
    assert seen == ["/v1/chat/completions"]


def test_llm_mode_falls_back_to_extractive_when_the_server_is_down(monkeypatch, tmp_path, capsys):
    use_server(monkeypatch, lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    with pytest.warns(ExperimentalFeatureWarning):
        s = Summarizer(CCEConfig(summarizer_mode="llm", base_dir=tmp_path))
    out = s._summarize(LONG, 50)
    assert out and "Sentence number" in out
    assert "extractive fallback" in capsys.readouterr().out


def test_the_whole_pipeline_still_works_in_llm_mode_with_a_dead_server(monkeypatch, tmp_path):
    use_server(monkeypatch, lambda request: httpx.Response(500, json={}))
    with pytest.warns(ExperimentalFeatureWarning):
        engine = CCEEngine(CCEConfig(embedder="hash", summarizer_mode="llm", base_dir=tmp_path))
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"{LONG} variant {i}"} for i in range(6)]
    res = engine.create_stateless_processor().process(msgs, "when is the deployment window?")
    assert res.payload.past_blocks or res.payload.recent_turns
