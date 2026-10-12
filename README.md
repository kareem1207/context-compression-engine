# Context Compression Engine (CCE)

A memory layer for AI assistants. CCE turns a long conversation into compact, searchable notes, then hands a language model only the context it needs for the next question. Chats stay fast and cheap because the model reads a few hundred relevant tokens instead of the whole history.

It runs fully offline by default (no LLM is needed to compress) and plugs into any model through an **MCP server**, or directly as a Python library.

> **Status: working prototype (v0.1).** The core engine, memory tiers, retrieval, sessions, MCP server and a live dashboard are implemented. The standalone REST package and JavaScript SDK are placeholders (see [Roadmap](#roadmap)). Compression-ratio and quality benchmarks have not been published yet, so this README makes no numeric claims.

---

## How it works

```
 messages ──► segment ──► semantic chunking ──► hierarchical summaries ──► memory nodes
                                                                              │
              ┌───────────────────────────────────────────────────────────────┘
              ▼
   ┌───────────────────────────────  memory tiers  ───────────────────────────────┐
   │  HOT   last N turns, verbatim, in RAM                                        │
   │  WARM  topic chunks + embeddings in SQLite, searched by cosine similarity    │
   │  COLD  one macro summary per finished session, as JSON files on disk         │
   └───────────────────────────────────────────────────────────────────────────────┘
              ▲
 user query ──┴─► retrieve (similarity + recency + keyword bonus) ──► context payload ──► your LLM
```

1. **Ingestion** splits raw messages into turns and counts tokens.
2. **Semantic chunking** embeds the turns (`all-MiniLM-L6-v2`) and starts a new chunk when the topic shifts (cosine similarity under a threshold, with min/max chunk sizes).
3. **Hierarchical summarising** writes three levels: *micro* (one sentence per turn), *meso* (2 to 4 sentences per topic chunk) and *macro* (one paragraph per session). The default extractive summariser needs no model; an optional `llm` mode calls any OpenAI-compatible local server such as llama.cpp.
4. **Memory store** keeps recent turns verbatim (hot), topic chunks with embeddings in SQLite (warm), and a finished-session summary on disk (cold).
5. **Retrieval** scores warm nodes by *semantic similarity + recency boost + keyword bonus*, and the **context builder** assembles a payload under a hard token cap.
6. **Sessions** can be *stateful* (persistent, checkpointed, many sessions at once) or *stateless* (compress and retrieve in one call, nothing saved).

---

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/kareem1207/context-compression-engine.git
cd context-compression-engine
uv sync
```

The first run downloads the embedding model (`sentence-transformers/all-MiniLM-L6-v2`, roughly 90 MB) from Hugging Face. After that it works offline.

### Use it as a library

```python
from cce_core.engine import CCEEngine

engine = CCEEngine()

messages = [
    {"role": "user", "content": "I want to build a SaaS billing dashboard."},
    {"role": "assistant", "content": "Start with plans, invoices and a webhook handler."},
    # ... many more turns ...
]

# Compress a conversation and store it in a session
store, turns, nodes = engine.full_pipeline(messages, session_id="demo")
print(engine.compression_stats(turns, nodes))

# Right before each LLM call: get only the relevant context
payload = engine.build_context(store, "How should the webhook handler work?")
llm_messages = payload.to_messages()      # or .to_string() / .to_dict()

store.close()
```

No persistence needed? Use the stateless processor:

```python
processor = engine.create_stateless_processor()
result = processor.process(messages, query="what did we decide about billing?")
llm_messages = result.to_messages()
```

Several live sessions at once:

```python
manager = engine.create_session_manager()
manager.add_message("session-a", {"role": "user", "content": "hello"})
context = manager.build_context("session-a", "hello")
manager.close("session-a")        # checkpoints a macro summary to the cold tier
```

### Use it from an MCP client (Claude Desktop and others)

CCE exposes eight MCP tools over stdio:

| Tool | What it does |
|---|---|
| `cce_ingest_turn` | add one message to a session |
| `cce_ingest_history` | bulk-load a conversation |
| `cce_retrieve_context` | get compressed context ready to inject |
| `cce_summarize_session` | macro summary of a session |
| `cce_session_stats` | diagnostics for a session |
| `cce_close_session` | checkpoint and close |
| `cce_list_sessions` | list known sessions |
| `cce_stateless_compress` | one-shot compress and retrieve, nothing stored |

Run the server:

```bash
uv run python -m cce_mcp.server
```

Example MCP client configuration (adjust the path):

```json
{
  "mcpServers": {
    "cce": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/context-compression-engine", "python", "-m", "cce_mcp.server"]
    }
  }
}
```

See [`examples/mcp_claude_demo.md`](examples/mcp_claude_demo.md) for a walkthrough.

### See it work

```bash
uv run python -m cce_ui.app                     # dashboard: http://localhost:7860
uv run python examples/summarizer_showcase.py   # summariser showcase: http://localhost:7862
uv run python examples/visual_demo.py
```

The dashboard replays a 24-turn conversation through CCE and shows the hot, warm and cold tiers filling up live, plus each summarising step. You can type your own messages afterwards.

The end-to-end demos (`examples/showcase_demo.py`, `examples/mcp_live_demo.py`, `examples/gemma_local_demo.py`) compare answers with and without compression using a local model served by [llama.cpp](https://github.com/ggml-org/llama.cpp) at `http://localhost:8080/v1`. They were written against a Gemma 4 GGUF; point `CCE_LLM_ENDPOINT` and `CCE_LLM_MODEL` at whatever you run. The MCP live demo also needs the small HTTP bridge: `uv run python -m cce_mcp.server_http` (port 9000, demo use only).

---

## Configuration

All tunables live in [`cce_core/config.py`](cce_core/config.py) (`CCEConfig`). Highlights:

| Setting | Default | Meaning |
|---|---|---|
| `summarizer_mode` | `extractive` | `extractive` (offline) or `llm` (local model) |
| `chunk_similarity_threshold` | `0.45` | cosine similarity below this starts a new topic |
| `chunk_min_turns` / `chunk_max_turns` | `2` / `20` | chunk size bounds |
| `hot_tier_max_turns` | `10` | recent turns kept verbatim |
| `warm_tier_max_chunks` | `500` | topic chunks kept in SQLite |
| `cold_tier_max_summaries` | `1000` | session summaries kept on disk |
| `retrieval_top_k` | `5` | nodes retrieved per query |
| `retrieval_recency_boost` | `0.2` | bonus for recent chunks |
| `context_max_tokens` | `2048` | hard cap on the assembled context |

Environment variables: `CCE_DATA_DIR` (storage folder, default `.cce_data`), `CCE_LLM_ENDPOINT`, `CCE_LLM_MODEL`, and for the MCP server `CCE_SUMMARIZER_MODE`, `CCE_HOT_TIER_TURNS`, `CCE_TOP_K`, `CCE_MAX_TOKENS`.

---

## Project layout

```
cce_core/
  ingestion/     segmenter, tokenizer
  compression/   semantic chunker, hierarchical summariser, merger
  memory/        hot tier (RAM), warm tier (SQLite + NumPy search), cold tier (JSON), store
  retrieval/     retriever (scoring), context builder (token-capped payloads)
  session/       stateful session manager, stateless processor
  engine.py      CCEEngine: the single entry point
cce_mcp/         MCP server (stdio) and a small HTTP demo bridge
cce_ui/          live dashboard
examples/        demos and showcases
tests/           pytest suites for chunker, summariser, memory store, retriever, sessions
                 and an end-to-end test that needs a local LLM
```

## Tests

```bash
uv run pytest --ignore=tests/test_e2e_gemma.py
```

`tests/test_e2e_gemma.py` needs a running llama.cpp server. (Note: `tests/` is currently listed in `.gitignore`; remove that line so contributors and CI see the tests.)

## Design notes and limits

- **Search is brute-force cosine similarity over a NumPy matrix.** For the default warm-tier size (hundreds of chunks) this beats building an ANN index and keeps dependencies small. It is not meant for millions of chunks.
- **The extractive summariser picks sentences; it does not paraphrase.** Use `llm` mode for more fluent summaries when a model is available.
- **SQLite is accessed with a lock** (single writer, many readers): fine for local use, not a multi-server deployment.
- Compression ratio is reported per run by `engine.compression_stats(...)`. There is no published benchmark yet.

## Roadmap

- [ ] Implement the `cce_rest` FastAPI package (currently empty files; `cce_mcp/server_http.py` is a demo-only stand-in)
- [ ] Implement the `cce_sdk_js` npm client (currently empty placeholders)
- [ ] Packaging for PyPI and npm
- [ ] Published benchmarks: compression ratio and answer quality with and without CCE
- [ ] Add a licence file
- [ ] Optional ANN index for large warm tiers

## Author

Built by [Shaik Kareem](https://github.com/kareem1207).
