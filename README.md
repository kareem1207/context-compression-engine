# Context Compression Engine (CCE)

A memory layer for AI assistants. CCE turns a long conversation into compact, searchable notes, then hands a language model only the context it needs for the next question. Chats stay fast and cheap because the model reads a few hundred relevant tokens instead of the whole history.

It runs fully offline by default (no LLM is needed to compress) and plugs into any model through an **MCP server**, or directly as a Python library.

**[Try the live demo in your browser](https://kareem1207.github.io/context-compression-engine/)**: paste a conversation, see it split into topics, ask a question and compare the context a model would receive with the full history.

> **Status: v1.0.** The core engine, memory tiers, retrieval, sessions, MCP server, REST API, JavaScript client and a live dashboard are implemented and tested. Benchmarks are in [`benchmarks/`](https://github.com/kareem1207/context-compression-engine/blob/main/benchmarks/README.md). On local models, CCE found facts said long ago with about 90% fewer tokens than resending the chat, and far more often than sending the newest messages that fit. **But a plain search over raw messages does just as well**: CCE tied it on single facts, lists, corrections and "what did we talk about?" questions, and lost to it on lists and overviews until we fixed that. So treat CCE as a convenient, packaged memory layer (sessions, tiers, MCP, REST, JS client), not as better retrieval than ordinary search. Read the limits before relying on it.

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
 user query ──┴─► retrieve (similarity + recency, fused with BM25) ──► context payload ──► your LLM
```

1. **Ingestion** splits raw messages into turns and counts tokens.
2. **Semantic chunking** embeds the turns (`all-MiniLM-L6-v2`) and starts a new chunk when the topic shifts (cosine similarity under a threshold, with min/max chunk sizes).
3. **Hierarchical summarising** writes three levels: *micro* (one sentence per turn), *meso* (2 to 4 sentences per topic chunk) and *macro* (one paragraph per session). The default extractive summariser needs no model; an optional `llm` mode calls any OpenAI-compatible local server such as llama.cpp.
4. **Memory store** keeps recent turns verbatim (hot), topic chunks with embeddings in SQLite (warm), and a finished-session summary on disk (cold).
5. **Retrieval** scores warm nodes by *semantic similarity (best message or sentence) + recency boost*, and fuses that ranking with BM25 over the notes' text when the question contains identifiers, codes, acronyms, names or numbers, and the **context builder** assembles a payload under a hard token cap.
6. **Sessions** can be *stateful* (persistent, checkpointed, many sessions at once) or *stateless* (compress and retrieve in one call, nothing saved).

---

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/kareem1207/context-compression-engine.git
cd context-compression-engine
uv sync
```

Or install it into your own project from PyPI:

```bash
pip install "context-compression-engine[neural]"          # recommended: engine + the neural embedder
pip install "context-compression-engine[all]"             # everything: neural, REST API, MCP server, dashboard
pip install context-compression-engine                    # the light engine only (hashing embedder, no PyTorch)
# extras: [neural] English embedder (PyTorch), [onnx] multilingual embedder (no PyTorch), [rest] REST API, [mcp] MCP server, [ui] dashboard
```

This also gives you two commands: `cce-rest` (the REST API) and `cce-mcp` (the MCP server).

**Three embedders.** The engine needs one to find topics and match questions. `neural` and `onnx` understand paraphrase; `onnx` does it in 50+ languages without PyTorch (see Design notes for sizes and numbers):

| | `neural` (recommended) | `hash` |
|---|---|---|
| Install | `pip install "...[neural]"` (pulls in PyTorch, about 1 GB) | nothing extra |
| Understands paraphrase | yes | no, matches shared words only |
| Held-out recall in our test | 99% | 55% (lexical 100%, paraphrase 10%) |

`CCE_EMBEDDER=auto` (the default) uses neural when installed, then onnx when installed, and hashing otherwise. The first neural run downloads `sentence-transformers/all-MiniLM-L6-v2` (about 90 MB) from Hugging Face, then works offline. Do not mix embedders within one stored session: the engine will raise a clear error if you try.

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
| `GET /v1/sessions/{id}/stats` | tier occupancy and token counts (see [Session stats](#session-stats)) |
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

Install it with `npm install cce-client`. **CCE Lite** swaps the neural sentence model for a lightweight word-hashing embedder, so it needs no downloads but is less semantic than the Python engine.

### Use it with other tools

Each of these has a runnable example in [`examples/integrations/`](https://github.com/kareem1207/context-compression-engine/tree/main/examples/integrations).

| You use | Do this |
|---|---|
| Claude Desktop, Cursor, VS Code (MCP) | install `context-compression-engine[mcp]`, then copy `claude_desktop_config.json`, `cursor_mcp.json` or `vscode_mcp.json` from the examples folder into the tool's MCP settings (`command` is `cce-mcp`, set `CCE_DATA_DIR` to where memory should live) |
| Anything that speaks the OpenAI chat API | run `cce-proxy --upstream <model server base URL>` (extra `rest`) and change your client's base URL to `http://127.0.0.1:9200/v1`. Long histories are trimmed to the relevant notes plus the recent turns; short ones pass through. Send `X-CCE-Session: <id>` (else the `user` field, else a hash of the first message). Streaming works. Requests with tool calls, tool messages or images are forwarded unchanged. |
| LangChain | `from cce_integrations.langchain import CCEChatMessageHistory` (needs `langchain-core`); use it with `RunnableWithMessageHistory`. `.messages` is the older notes that match the latest user message plus the recent turns; `.search(query)` for another question |
| LlamaIndex | `from cce_integrations.llamaindex import CCEChatMemory` (needs `llama-index-core`); `CCEChatMemory.from_defaults(session_id=..., data_dir=...)` goes wherever a `memory=` is accepted |
| TypeScript / JavaScript | `cce_sdk_js` ships type definitions for the REST client and the offline engine (`CCEClient.getContext(..., { explain: true })`, `SessionStats`, `Explain`) |

The LangChain and LlamaIndex classes were checked against langchain-core 1.6.12 and llama-index-core 0.14.25 (2026-10-21). LangChain now recommends LangGraph persistence over `RunnableWithMessageHistory`; `CCEChatMessageHistory` is a standard `BaseChatMessageHistory`, so it works with either.

### See why something was retrieved

`POST /v1/sessions/{id}/context?explain=1` (or `"explain": true`, or `explain=true` on the `cce_retrieve_context` MCP tool) returns, next to the context, which notes and quotes were chosen, their scores (dense, recency, BM25, fused), the token budget split (system note, recent turns, pending turns, notes, overhead) and the notes that were retrieved but dropped to fit. From a terminal:

```bash
cce inspect                                   # sessions in the data folder
cce inspect my-session                        # its notes
cce inspect my-session --query "what is the retry limit?"   # why that question gets that context (add --json)
```

### See it work

```bash
uv run python -m cce_ui.app                     # dashboard: http://localhost:7860
cce-ui                                           # same, after `pip install "context-compression-engine[ui]"` (listens on 127.0.0.1)
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
| `warm_tier_max_chunks` | `500` | topic notes kept per session. Over the limit, the least-retrieved neighbouring old notes are merged into coarser notes (the newest are left alone, nothing is deleted). `0` = unlimited |
| `cold_tier_max_summaries` | `1000` | session summaries kept on disk; the oldest are deleted beyond this. `0` = unlimited |
| `live_batch_turns` | `8` | live sessions: compress evicted turns once this many are waiting, in topic-sized notes (`1` = each eviction on its own) |
| `live_pending_max_turns` / `live_pending_max_tokens` | `24` / `3000` | live sessions: flush the waiting turns even if the last topic is still going |
| `sentence_index_min_words` / `sentence_discount` | `40` / `0.9` | messages with at least this many words also get one vector per sentence (`0` = off); a sentence's score counts at this fraction against whole-message scores |
| `lexical_weight` / `lexical_terms` | `1.0` / `distinctive` | weight of the BM25 ranking in the fusion (`0` = dense only); `distinctive` sends only identifiers, codes, acronyms, names and numbers of the question to BM25, `all` sends every non-stopword |
| `onnx_model` / `onnx_max_length` | `Xenova/paraphrase-multilingual-MiniLM-L12-v2` / `256` | model and input length of the `onnx` embedder |
| `keep_raw_turns` | `true` | keep the original text of compressed turns in the local SQLite file so a topic can quote the exact sentence that answers a question (env `CCE_KEEP_RAW_TURNS=0` turns it off; see [Raw text and privacy](#raw-text-and-privacy)) |
| `overview_max_topics` / `overview_budget_share` | `24` / `0.65` | "what did we talk about?" questions list up to this many topics, in conversation order, sharing this fraction of `context_max_tokens` evenly |
| `retrieval_top_k` | `5` | nodes retrieved per query |
| `retrieval_recency_boost` | `0.2` | bonus for recent chunks |
| `context_max_tokens` | `2048` | hard cap on the assembled context |

Environment variables: `CCE_EMBEDDER` (`auto`, `neural`, `onnx` or `hash`), `CCE_DATA_DIR` (storage folder, default `.cce_data`), `CCE_LLM_ENDPOINT`, `CCE_LLM_MODEL`, and for the MCP server `CCE_SUMMARIZER_MODE`, `CCE_HOT_TIER_TURNS`, `CCE_TOP_K`, `CCE_MAX_TOKENS`.

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

## Raw text and privacy

A topic note holds a summary and one short line per message, and a summary drops details: a fact in the third sentence of a long answer may not survive it. So, by default (`keep_raw_turns = true`), CCE also keeps the **original text of every compressed turn** in the `turns` table of the SQLite file `warm.db` in your data folder. When a topic matches a question, CCE scores the sentences of its best-matching messages against the question and quotes the best ones, instead of the one-line summary of the message. Messages of 40 words or more also get one vector per sentence when they are compressed (stored as float16 with the note), so a fact inside a long message is found and quoted on its own; shorter messages are scored at query time and those vectors are remembered between queries.

- **It stays on your machine.** The text is in `<data dir>/warm.db` (default `.cce_data/`), next to the notes that were already there. CCE sends it nowhere; it only reaches a model if it is quoted in a context you then send. The cold-tier JSON summaries are separate and hold no raw text.
- **Turn it off** with `CCEConfig(keep_raw_turns=False)` or `CCE_KEEP_RAW_TURNS=0`. Turns are then dropped once summarised and quotes fall back to the one-line summaries (details inside longer messages can be lost). Turning it off does not remove text that was already stored.
- **Delete it:** `DELETE /v1/sessions/{id}` (REST) or `SessionManager.delete(id)` removes a session's notes, its raw text and its summary. To remove everything, stop CCE and delete the data folder. Notes made by CCE 1.0 or earlier have no raw text and keep quoting summaries; the file is upgraded in place the first time it is opened.
- The raw text adds roughly the size of the conversation to `warm.db`.

## Session stats

`GET /v1/sessions/{id}/stats` (and `cce_session_stats`) report, among others: `raw_tokens` (the whole session as it was said), `tokens_sent` (what sending the whole session would cost now: the notes as they are sent plus the hot and pending turns, verbatim), `tokens_saved` (`raw_tokens - tokens_sent`, never below 0) and `compression_ratio` (`raw_tokens / tokens_sent`). `compression_ratio` is `null` while the session is not smaller than its raw text, which is normal for short chats where everything is still verbatim. `warm_tokens_original` is the raw size of the turns the notes cover and `warm_tokens_compressed` the size of those notes as sent; `raw_turns_stored` counts the turns whose original text is kept.

## What the results do and do not show

- **Live sessions and one-shot mode.** "Live sessions" (the MCP server and the REST sessions) and "one-shot" mode (`/v1/compress`, `full_pipeline`) use the same compression. Before 1.0, live sessions did not group topics: every message that left the recent-turns window became a note of its own. The benchmarks in [`benchmarks/README.md`](https://github.com/kareem1207/context-compression-engine/blob/main/benchmarks/README.md) were produced with one-shot mode, and before that live-session fix. **They have not been re-run for 1.0.**
- **Plain search is a strong baseline.** In our tests, a plain search over the raw messages matched CCE on finding single facts. What we measured in CCE's favour is token use, and results against two simpler ways of fitting a chat into a prompt: sending everything, and sending the newest messages that fit.
- **What we claim**, each from the tables in the [benchmark write-up](https://github.com/kareem1207/context-compression-engine/blob/main/benchmarks/README.md):
  - about 90% fewer tokens than resending the chat;
  - on a 6.7B local model, 79% of questions correct with CCE against 42% when the whole chat was sent ([table](https://github.com/kareem1207/context-compression-engine/blob/main/benchmarks/README.md#chats-that-fit-the-models-window-about-5300-tokens-context-8192));
  - on chats longer than the model's window, 75% to 100% correct with CCE against 4% to 6% with the newest messages that fit ([table](https://github.com/kareem1207/context-compression-engine/blob/main/benchmarks/README.md#chats-longer-than-the-window-about-9300-tokens-context-4096));
  - it runs offline, and it is MIT licensed.
- **What we do not claim.** Those are small tests (16 to 24 questions per cell) on two local models with planted facts, not a promise about your data or your model. Results depend on the conversation, the embedder and the model. See [what is not tested](https://github.com/kareem1207/context-compression-engine/blob/main/benchmarks/README.md#what-is-not-tested).

## Design notes and limits

- **Search is brute-force cosine similarity over a NumPy matrix.** For the default warm-tier size (hundreds of chunks) this beats building an ANN index and keeps dependencies small. It is not meant for millions of chunks.
- **One search index per data folder, shared by all sessions in the process.** It follows the SQLite file (`PRAGMA data_version` plus a change log), so rows written by another process show up without a restart. Scoring a query runs no SQL; only the winning notes are read from disk.
- **Compression only kicks in once a topic is longer than the summary budget** (`meso_max_tokens`). Short topics are kept as they are.
- **The extractive summariser picks sentences; it does not paraphrase.** Use `llm` mode for more fluent summaries when a model is available.
- **Live sessions keep their state in the same SQLite file.** The last `hot_tier_max_turns` turns, the turns waiting to be compressed, and the turn counters are saved as messages arrive, so a restart or crash picks up where it stopped. Waiting turns are sent to the model verbatim until they are compressed, so nothing falls in a gap. After compression, the original text of each turn stays in the same file unless you turn that off (`keep_raw_turns`, see [Raw text and privacy](#raw-text-and-privacy)).
- **Embedders.** `neural` is all-MiniLM-L6-v2 through sentence-transformers (English, pulls in PyTorch, about 1 GB installed). `onnx` is `paraphrase-multilingual-MiniLM-L12-v2` (int8) run with onnxruntime: no PyTorch, 384 dimensions, 50+ languages. `pip install "context-compression-engine[onnx]"` adds about 115 MB of packages (numpy included) and downloads 135 MB of model files on first use (`onnx_model` in the config). `hash` needs nothing and matches shared words only (Latin script). With `embedder="auto"` CCE uses `neural` if sentence-transformers is installed, else `onnx` if onnxruntime is installed, else `hash`; a data folder remembers its embedder, so switching is refused rather than silently wrong. Measured on the hand-written Spanish, Hindi and Chinese set in `benchmarks/retrieval_multilingual.py` (5 conversations x 4 questions per language, 2026-10-21): onnx 60/60, MiniLM 49/60, hash 23/60; on 96 English questions onnx finds 96/96, as MiniLM does. BM25 matches Latin-script terms only, so exact-term search does not help Chinese or Hindi text yet.
- **Overview questions** ("what did we talk about?") list every topic in conversation order. Each topic shows the first user message of its note, then the sentences that carry the most terms particular to that topic (tf-idf, one new term set per line), within an even share of the budget. The previous behaviour showed the extractive summary of each topic, which used more tokens for the same coverage.
- **BM25 is used for exact terms only.** Ids such as `ERR-4021-X` or a UUID, acronyms, names and numbers in a question are matched exactly against the notes and fused with the dense ranking (reciprocal rank). Ordinary lower-case words are not, because exact matching of them added noise to paraphrased questions in our tests (two of 48 were lost). This helps when the dense embedder cannot tell such tokens apart; it does not make the hash embedder understand paraphrases.
- **Token counts are estimates.** Offline, CCE estimates from words and characters (CJK text, code and JSON count higher than plain English). With the optional `tokens` extra (`tiktoken`) and its vocabulary already cached, counts are exact for `cl100k_base`; CCE never downloads it. Other models tokenize differently. `context_max_tokens` is a cap on this estimate.
- **Session ids** are 1-64 characters of letters, digits, `_`, `.` and `-` (not `.`, `..` or `index`), in the library, the REST API and the MCP tools, because they become file names in the cold tier.
- **A data folder remembers its embedder.** The first time a folder is opened, the embedder (name and vector size) is recorded. Opening it later with a different one (`CCE_EMBEDDER`, `hash_dim`, another model) fails with a clear error instead of returning bad matches. Notes hold no raw text, so they cannot be re-embedded; use the original embedder or a new folder.
- **Databases are versioned** (`PRAGMA user_version`). A 0.1.x `.cce_data` folder is upgraded in place the first time it is opened; its notes are kept.
- **SQLite is accessed with a lock** (single writer, many readers): fine for local use, not a multi-server deployment.
- Compression ratio is reported per run by `engine.compression_stats(...)`.

## Roadmap

- [x] REST API (`cce_rest`), JavaScript client and offline engine (`cce_sdk_js`), browser demo
- [x] Benchmarks on real chat text (retrieval, window overflow, two local models), see [`benchmarks/`](https://github.com/kareem1207/context-compression-engine/blob/main/benchmarks/README.md)
- [x] MIT licence, packaging metadata, optional API key for the REST API
- [ ] Test what plain retrieval cannot do: questions that need a summary of a whole topic or session
- [ ] Benchmark on messy real conversations and with a hosted model
- [ ] Better summaries: the default extractive summariser only picks sentences (a model-written option exists but is untested)
- [x] Published to PyPI (`context-compression-engine`) and npm (`cce-client`)
- [ ] Optional ANN index for large warm tiers

## License

[MIT](https://github.com/kareem1207/context-compression-engine/blob/main/LICENSE). Copyright (c) 2026 Shaik Kareem.

## Author

Built by [Shaik Kareem](https://github.com/kareem1207).
