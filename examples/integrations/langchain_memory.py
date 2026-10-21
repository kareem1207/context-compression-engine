"""CCE as LangChain chat history. Runs without an API key (a stand-in model answers).

    pip install "context-compression-engine" langchain-core
    python examples/integrations/langchain_memory.py
"""
import tempfile

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from cce_integrations.langchain import CCEChatMessageHistory

memory_dir = tempfile.mkdtemp()


def model(messages):                      # replace with your chat model
    return AIMessage(content=f"(I can see {len(messages)} messages)")


chain = RunnableWithMessageHistory(
    RunnableLambda(model),
    lambda session_id: CCEChatMessageHistory(session_id, data_dir=memory_dir, embedder="hash", context_max_tokens=450, hot_tier_max_turns=4),
)
config = {"configurable": {"session_id": "demo"}}
chain.invoke([HumanMessage(content="Our invoice retry limit is seven attempts.")], config=config)
for topic in ("hiking boots", "pandas merge", "sourdough starter", "stripe webhooks"):
    chain.invoke([HumanMessage(content=f"Tell me about {topic} and everything related to {topic}.")], config=config)

history = CCEChatMessageHistory("demo", data_dir=memory_dir, embedder="hash", context_max_tokens=450, hot_tier_max_turns=4)
for m in history.search("How many invoice retry attempts are allowed?"):
    print(type(m).__name__, "|", m.content[:90].replace("\n", " "))
