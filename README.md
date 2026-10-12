# Context Compression Engine (CCE)

A memory layer for AI assistants. CCE turns a long conversation into compact, searchable notes, then hands a language model only the context it needs for the next question. Chats stay fast and cheap because the model reads a few hundred relevant tokens instead of the whole history.

It runs fully offline by default (no LLM is needed to compress) and plugs into any model through an **MCP server**, or directly as a Python library.

**[Try the live demo in your browser](https://kareem1207.github.io/context-compression-engine/)**: paste a conversation, see it split into topics, ask a question and compare the context a model would receive with the full history.

> **Status: working prototype (v0.1).** The core engine, memory tiers, retrieval, sessions, MCP server, REST API, JavaScript client and a live dashboard are implemented and tested. Benchmarks are in [`benchmarks/`](https://github.com/kareem1207/context-compression-engine/blob/main/benchmarks/README.md). On local models, CCE found facts said long ago with about 90% fewer tokens than resending the chat, and far more often than sending the newest messages that fit. **But a plain search over raw messages does just as well**: CCE tied it on single facts, lists, corrections and "what did we talk about?" questions, and lost to it on lists and overviews until we fixed that. So treat CCE as a convenient, packaged memory layer (sessions, tiers, MCP, REST, JS client), not as better retrieval than ordinary search. Read the limits before relying on it.

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

Or install it into your own project straight from GitHub (it is not on PyPI yet):

```bash
pip install "context-compression-engine[all] @ git+https://github.com/kareem1207/context-compression-engine"
# extras: [neural] recommended embedder, [rest] REST API, [mcp] MCP server, [ui] dashboard;
# the base install is the light engine (hashing embedder, no PyTorch)
```

This also gives you two commands: `cce-rest` (the REST API) and `cce-mcp` (the MCP server).

**Two embedders.** The engine needs one to find topics and match questions:

| | `neural` (recommended) | `hash` |
|---|---|---|
| Install | `pip install "...[neural]"` (pulls in PyTorch, about 1 GB) | nothing extra |
| Understands paraphrase | yes | no, matches shared words only |
| Held-out recall in our test | 99% | 55% (lexical 100%, paraphrase 10%) |

`CCE_EMBEDDER=auto` (the default) uses neural when installed and hashing otherwise. The first neural run downloads `sentence-transformers/all-MiniLM-L6-v2` (about 90 MB) from Hugging Face, then works offline. Do not mix embedders within one stored session: the engine will raise a clear error if you try.

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

See [`examples/mcp_claude_demo.md`](https://github.com/kareem1207/context-compression-engine/blob/main/examples/mcp_claude_demo.md) for a walkthrough.

### Use it over HTTP (any language)

```bash
uv run python -m cce_rest.app        # http://localhost:9100  (interactive docs at /docs)
```

| Endpoint | What it does |
|---|---|
| `POST /v1/compress` | one-shot: send `messages` and a `query`, get the context back; nothing is stored |
| `POST /v1/sessions/{id}/messages` | add messages to a stored session |
| `POST /v1/sessions/{id}/context` | get compressed context for a query |
| `GET /v1/sessions/{id}/summary` | macro summary of the session |
| `GET /v1/sessions/{id}/stats` | tier occupancy and token counts |
| `POST /v1/sessions/{id}/close`, `DELETE /v1/sessions/{id}` | checkpoint and close, or delete |
| `GET /v1/sessions`, `GET /health` | list sessions, health check |

```bash
curl -X POST localhost:9100/v1/compress -H "Content-Type: application/json"   -d '{"messages":[{"role":"user","content":"..."}], "query":"what did we decide?"}'
```

Configuration comes from the `CCE_*` variables below; set `CCE_CORS_ORIGINS` (comma separated) to allow browser apps.

**Security.** The API listens on localhost only by default. Set `CCE_API_KEY` to require a key (`Authorization: Bearer <key>` or `X-API-Key`) on every `/v1` route; the server refuses to start on a public address without one. Anyone with access can read every stored session, so do not expose it to the internet without a key and HTTPS in front.

### Use it from JavaScript

[`cce_sdk_js`](https://github.com/kareem1207/context-compression-engine/blob/main/cce_sdk_js) has two parts, no dependencies, Node 18+ and browsers:

```js
// 1) Talk to the REST API above
import { CCEClient } from "cce-client";
const cce = new CCEClient({ baseUrl: "http://localhost:9100", apiKey: process.env.CCE_API_KEY });
await cce.addMessages("chat-1", messages);
const { context } = await cce.getContext("chat-1", "what did we decide about billing?");

// 2) Or run the engine offline, in-process (this is what the live demo uses)
import { createEngine } from "cce-client/lite";
const out = createEngine().process(messages, "what did we decide about billing?");
console.log(out.stats.savedPercent, out.payload.toMessages());
```

The package is not on npm yet (it will be `cce-client`); import the files from the repo for now. **CCE Lite** swaps the neural sentence model for a lightweight word-hashing embedder, so it needs no downloads but is less semantic than the Python engine.

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

All tunables live in [`cce_core/config.py`](https://github.com/kareem1207/context-compression-engine/blob/main/cce_core/config.py) (`CCEConfig`). Highlights:

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

Environment variables: `CCE_EMBEDDER` (`auto`, `neural` or `hash`), `CCE_DATA_DIR` (storage folder, default `.cce_data`), `CCE_LLM_ENDPOINT`, `CCE_LLM_MODEL`, and for the MCP server `CCE_SUMMARIZER_MODE`, `CCE_HOT_TIER_TURNS`, `CCE_TOP_K`, `CCE_MAX_TOKENS`.

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
cce_rest/        FastAPI REST API
cce_sdk_js/      JavaScript client (client.js) and offline engine (lite.js), with tests
cce_ui/          live dashboard
docs/            the GitHub Pages demo (built together with cce_sdk_js/lite.js)
examples/        demos and showcases
tests/           pytest suites for chunker, summariser, memory store, retriever, sessions
                 and an end-to-end test that needs a local LLM
```

## Tests

```bash
uv run pytest --ignore=tests/test_e2e_gemma.py      # Python: engine, memory, sessions, REST
cd cce_sdk_js && npm test                            # JavaScript: client and offline engine
```

`tests/test_e2e_gemma.py` needs a running llama.cpp server, so CI skips it. The first Python run downloads the embedding model.

## Design notes and limits

- **Search is brute-force cosine similarity over a NumPy matrix.** For the default warm-tier size (hundreds of chunks) this beats building an ANN index and keeps dependencies small. It is not meant for millions of chunks.
- **Compression only kicks in once a topic is longer than the summary budget** (`meso_max_tokens`). Short topics are kept as they are.
- **The extractive summariser picks sentences; it does not paraphrase.** Use `llm` mode for more fluent summaries when a model is available.
- **SQLite is accessed with a lock** (single writer, many readers): fine for local use, not a multi-server deployment.
- Compression ratio is reported per run by `engine.compression_stats(...)`. There is no published benchmark yet.

## Roadmap

- [x] REST API (`cce_rest`), JavaScript client and offline engine (`cce_sdk_js`), browser demo
- [x] Benchmarks on real chat text (retrieval, window overflow, two local models), see [`benchmarks/`](https://github.com/kareem1207/context-compression-engine/blob/main/benchmarks/README.md)
- [x] MIT licence, packaging metadata, optional API key for the REST API
- [ ] Test what plain retrieval cannot do: questions that need a summary of a whole topic or session
- [ ] Benchmark on messy real conversations and with a hosted model
- [ ] Better summaries: the default extractive summariser only picks sentences (a model-written option exists but is untested)
- [ ] Publish to PyPI (`context-compression-engine`) and npm (`cce-client`)
- [ ] Optional ANN index for large warm tiers

## License

[MIT](https://github.com/kareem1207/context-compression-engine/blob/main/LICENSE). Copyright (c) 2026 Shaik Kareem.

## Author

Built by [Shaik Kareem](https://github.com/kareem1207).
