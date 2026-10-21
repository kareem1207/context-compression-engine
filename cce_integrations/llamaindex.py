"""
LlamaIndex chat memory backed by CCE.

    from cce_integrations.llamaindex import CCEChatMemory

    memory = CCEChatMemory.from_defaults(session_id="user-42", data_dir="./memory")
    agent = SimpleChatEngine.from_defaults(memory=memory)      # or any engine/agent that takes a memory

`get(input)` returns the older notes that match `input` (one system message) followed by the recent turns;
`get_all()` returns every stored message (all of them if `keep_raw_turns` is on). Needs `llama-index-core`.
"""

from __future__ import annotations

from typing import Any, List, Optional

try:
    from llama_index.core.llms import ChatMessage, MessageRole
    from llama_index.core.memory import BaseMemory
    from pydantic import PrivateAttr
except ImportError as exc:                                                           # pragma: no cover
    raise ImportError("cce_integrations.llamaindex needs llama-index-core: pip install llama-index-core") from exc

from cce_integrations._common import context_pairs, make_manager

_TO_LI = {"user": MessageRole.USER, "assistant": MessageRole.ASSISTANT, "system": MessageRole.SYSTEM, "tool": MessageRole.TOOL}
_FROM_LI = {v: k for k, v in _TO_LI.items()}


class CCEChatMemory(BaseMemory):
    session_id: str = "default"
    _manager: Any = PrivateAttr()

    @classmethod
    def class_name(cls) -> str:
        return "CCEChatMemory"

    @classmethod
    def from_defaults(cls, session_id: str = "default", manager=None, data_dir=None, **config: Any) -> "CCEChatMemory":
        """`manager`: an existing SessionManager; else one is made on `data_dir` with CCEConfig options in `config`."""
        memory = cls(session_id=session_id)
        memory._manager = make_manager(manager, data_dir, config)
        return memory

    def _messages(self, pairs) -> List[ChatMessage]:
        return [ChatMessage(role=_TO_LI.get(role, MessageRole.USER), content=text) for role, text in pairs]

    def get(self, input: Optional[str] = None, **kwargs: Any) -> List[ChatMessage]:
        return self._messages(context_pairs(self._manager, self.session_id, input))

    def get_all(self) -> List[ChatMessage]:
        store = self._manager.get_or_open(self.session_id)
        return self._messages((t.role, t.content) for t in store.get_all_turns())

    def put(self, message: ChatMessage) -> None:
        self._manager.add_message(
            self.session_id, {"role": _FROM_LI.get(message.role, "user"), "content": message.content or ""})

    def set(self, messages: List[ChatMessage]) -> None:
        self.reset()
        for m in messages:
            self.put(m)

    def reset(self) -> None:
        if self._manager.exists(self.session_id):
            self._manager.delete(self.session_id)
