"""
LangChain chat history backed by CCE.

    from cce_integrations.langchain import CCEChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory

    chain_with_memory = RunnableWithMessageHistory(
        chain, lambda session_id: CCEChatMessageHistory(session_id, data_dir="./memory"),
        input_messages_key="question", history_messages_key="history")

`messages` is what the model should see: the older notes that match the latest user message (as one system
message) followed by the recent turns. `search(query)` does the same for another question. The whole
conversation is kept in the CCE session, so history can grow without the prompt growing with it.
Needs `langchain-core`.
"""

from __future__ import annotations

try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
except ImportError as exc:                                                           # pragma: no cover
    raise ImportError("cce_integrations.langchain needs langchain-core: pip install langchain-core") from exc

from cce_integrations._common import context_pairs, last_user_text, make_manager

_ROLES = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
_CLASSES = {"user": HumanMessage, "assistant": AIMessage, "system": SystemMessage, "tool": None}


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")


class CCEChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, manager=None, *, data_dir=None, query: str | None = None, **config):
        """`manager`: an existing SessionManager; else one is made on `data_dir` with CCEConfig options in `config`.
        `query`: fixed question used to pick older notes; default is the latest user message."""
        self.session_id = session_id
        self.query = query
        self.manager = make_manager(manager, data_dir, config)

    def _convert(self, pairs: list[tuple[str, str]]) -> list[BaseMessage]:
        out: list[BaseMessage] = []
        for role, text in pairs:
            cls = _CLASSES.get(role)
            out.append(cls(content=text) if cls else ToolMessage(content=text, tool_call_id="cce"))
        return out

    def search(self, query: str) -> list[BaseMessage]:
        """The CCE context for `query` as LangChain messages (a system message with older notes, then recent turns)."""
        return self._convert(context_pairs(self.manager, self.session_id, query))

    @property
    def messages(self) -> list[BaseMessage]:           # type: ignore[override]
        query = self.query or last_user_text(self.manager, self.session_id)
        if query:
            return self.search(query)
        return self._convert(context_pairs(self.manager, self.session_id, None))

    def add_messages(self, messages) -> None:
        for m in messages:
            role = _ROLES.get(m.type, "user")
            self.manager.add_message(self.session_id, {"role": role, "content": _text(m)})

    def clear(self) -> None:
        if self.manager.exists(self.session_id):
            self.manager.delete(self.session_id)
