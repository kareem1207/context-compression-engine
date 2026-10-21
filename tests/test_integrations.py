"""LangChain and LlamaIndex memory classes (each part is skipped when its library is not installed)."""

import pytest

from tests.test_live_sessions import PLANTED, make_messages

QUESTION = "How many invoice retry attempts are allowed?"


def options(tmp_path):
    return dict(data_dir=tmp_path / "mem", embedder="hash", hot_tier_max_turns=4, retrieval_top_k=3, context_max_tokens=450)


# ── Config snippets and examples ──────────────────────────────────────────────

def test_mcp_config_snippets_are_valid_and_name_a_real_command():
    import json
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    scripts = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    for name, key in (("claude_desktop_config.json", "mcpServers"), ("cursor_mcp.json", "mcpServers"), ("vscode_mcp.json", "servers")):
        cfg = json.loads((root / "examples" / "integrations" / name).read_text(encoding="utf-8"))
        server = cfg[key]["cce"]
        assert server["command"] in scripts, name
        assert "CCE_DATA_DIR" in server["env"], name


def test_examples_compile():
    import py_compile
    from pathlib import Path

    for path in (Path(__file__).resolve().parent.parent / "examples" / "integrations").glob("*.py"):
        py_compile.compile(str(path), doraise=True)


# ── LangChain ─────────────────────────────────────────────────────────────────

class TestLangChain:
    @pytest.fixture(autouse=True)
    def libs(self):
        pytest.importorskip("langchain_core")

    def history(self, tmp_path, session="lc", **extra):
        from cce_integrations.langchain import CCEChatMessageHistory
        return CCEChatMessageHistory(session, **{**options(tmp_path), **extra})

    @staticmethod
    def as_langchain(messages):
        from langchain_core.messages import AIMessage, HumanMessage
        return [(HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"]) for m in messages]

    def test_messages_hold_older_notes_and_recent_turns(self, tmp_path):
        from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
        h = self.history(tmp_path)
        assert h.messages == []                                      # nothing yet
        h.add_messages(self.as_langchain(make_messages()))
        h.add_messages([HumanMessage(content=QUESTION)])
        msgs = h.messages
        assert all(isinstance(m, BaseMessage) for m in msgs)
        assert isinstance(msgs[0], SystemMessage) and PLANTED in msgs[0].content     # the old fact, from the notes
        assert msgs[-1].content == QUESTION                                          # the latest turn stays verbatim
        assert len(msgs) < len(make_messages()) // 2

    def test_search_uses_another_question(self, tmp_path):
        h = self.history(tmp_path)
        h.add_messages(self.as_langchain(make_messages()))
        assert PLANTED in " ".join(m.content for m in h.search(QUESTION))

    def test_history_survives_a_restart_and_clear_forgets(self, tmp_path):
        h = self.history(tmp_path)
        h.add_messages(self.as_langchain(make_messages()))
        h.manager.close_all()
        again = self.history(tmp_path)
        assert PLANTED in " ".join(m.content for m in again.search(QUESTION))
        again.clear()
        assert not again.manager.exists("lc")
        assert again.messages == []          # (reading a session opens it again, empty)

    def test_works_inside_runnable_with_message_history(self, tmp_path):
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_core.runnables import RunnableLambda
        from langchain_core.runnables.history import RunnableWithMessageHistory

        seen = []

        def model(messages):
            seen.append(list(messages))
            return AIMessage(content=f"answer {len(seen)}")

        chain = RunnableWithMessageHistory(RunnableLambda(model), lambda sid: self.history(tmp_path, sid))
        cfg = {"configurable": {"session_id": "chain"}}
        for q in ("first question", "second question", "third question"):
            out = chain.invoke([HumanMessage(content=q)], config=cfg)
        assert out.content == "answer 3"
        assert [m.content for m in seen[-1]][-1] == "third question"
        assert len(seen[-1]) > len(seen[0])                                  # earlier turns came back as history
        store = self.history(tmp_path, "chain").manager.get_or_open("chain")
        assert store.stats()["turns_ingested"] == 6                          # 3 questions + 3 answers


# ── LlamaIndex ────────────────────────────────────────────────────────────────

class TestLlamaIndex:
    @pytest.fixture(autouse=True)
    def libs(self):
        pytest.importorskip("llama_index.core")

    def memory(self, tmp_path, session="li"):
        from cce_integrations.llamaindex import CCEChatMemory
        return CCEChatMemory.from_defaults(session_id=session, **options(tmp_path))

    @staticmethod
    def as_chat(messages):
        from llama_index.core.llms import ChatMessage, MessageRole
        return [ChatMessage(role=MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT, content=m["content"])
                for m in messages]

    def test_get_returns_notes_for_the_input(self, tmp_path):
        from llama_index.core.llms import MessageRole
        mem = self.memory(tmp_path)
        for m in self.as_chat(make_messages()):
            mem.put(m)
        got = mem.get(QUESTION)
        assert got[0].role == MessageRole.SYSTEM and PLANTED in got[0].content
        assert len(got) < len(make_messages()) // 2
        assert len(mem.get_all()) == len(make_messages())                    # every message is still stored

    def test_set_reset_and_restart(self, tmp_path):
        mem = self.memory(tmp_path)
        mem.set(self.as_chat(make_messages()[:6]))
        assert len(mem.get_all()) == 6
        mem.set(self.as_chat(make_messages()[:2]))
        assert len(mem.get_all()) == 2                                       # set replaces
        mem.reset()
        assert mem.get_all() == [] and mem.get() == []

        mem.set(self.as_chat(make_messages()))
        mem._manager.close_all()
        again = self.memory(tmp_path)
        assert PLANTED in " ".join(m.content for m in again.get(QUESTION))

    def test_works_with_a_chat_engine(self, tmp_path):
        from llama_index.core.chat_engine import SimpleChatEngine
        from llama_index.core.llms import MockLLM

        mem = self.memory(tmp_path, "engine")
        engine = SimpleChatEngine.from_defaults(llm=MockLLM(max_tokens=8), memory=mem)
        engine.chat("hello there")
        engine.chat("and again")
        roles = [m.role.value for m in mem.get_all()]
        assert roles == ["user", "assistant", "user", "assistant"]
