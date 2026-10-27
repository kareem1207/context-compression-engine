"""CCE as LlamaIndex chat memory. Runs without an API key (MockLLM).

    pip install "context-compression-engine" llama-index-core
    python examples/integrations/llamaindex_memory.py
"""
import tempfile

from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.llms import MockLLM

from cce_integrations.llamaindex import CCEChatMemory

memory = CCEChatMemory.from_defaults(session_id="demo", data_dir=tempfile.mkdtemp(), embedder="hash", context_max_tokens=450, hot_tier_max_turns=4)
engine = SimpleChatEngine.from_defaults(llm=MockLLM(max_tokens=12), memory=memory)    # use your own llm here

engine.chat("Our invoice retry limit is seven attempts.")
for topic in ("hiking boots", "pandas merge", "sourdough starter", "stripe webhooks"):
    engine.chat(f"Tell me about {topic} and everything related to {topic}.")

print(len(memory.get_all()), "messages stored")
for m in memory.get("How many invoice retry attempts are allowed?"):
    print(m.role.value, "|", (m.content or "")[:90].replace("\n", " "))
