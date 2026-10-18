# Changelog

## 1.0.0 (2026-10-18)

Fixes for live sessions, counting, limits and search, found while checking how MCP and REST users run CCE. Existing `.cce_data` folders from 0.1.x keep working; they are upgraded in place the first time they are opened.

Live sessions (MCP, REST)
- Evicted turns are compressed in topic-sized batches with the same chunker as one-shot mode, instead of one note per message. Turns waiting to be compressed are kept in a persisted pending buffer and sent verbatim, so there is no gap. Flushed on checkpoint, close and when the buffer gets big. New settings: `live_batch_turns`, `live_pending_max_turns`, `live_pending_max_tokens`.
- State survives restarts and crashes: the hot tier, pending buffer and counters are saved in SQLite as messages arrive, and turn numbering continues instead of starting from 0 (which had broken the recency boost and overview ordering).
- Database schema is versioned (`PRAGMA user_version`, now 4) with in-place migration from 0.1.x. The embedder (name, dimension, model) is recorded; opening a folder with a different embedder fails with a clear error.

Counting and limits
- Token counts: 400 Chinese characters or 400 characters of minified JSON used to count as 1 token. A character-aware estimate replaces the word count; exact `cl100k_base` counts are used when the optional `tokens` extra is installed and its vocabulary is already cached (nothing is downloaded). The context cap is now enforced for such text.
- `warm_tier_max_chunks` and `cold_tier_max_summaries` are now enforced (they were documented but unused). Over the warm limit, the least-retrieved neighbouring old notes are merged into coarser ones; over the cold limit, the oldest summaries are dropped. `0` means unlimited.

Search
- One shared index per data folder (instead of one per store), kept current from SQLite so a second process sees new notes; scoring a query runs no SQL per note.

Security and packaging
- Session ids are validated everywhere (MCP, REST, library, cold tier): an id such as `../../x` could write a file outside the data folder.
- REST CORS allows the `Authorization` and `X-API-Key` headers.
- `cce_ui` is in the wheel, with a `cce-ui` command; `cce-rest` and `cce-mcp` answer `--help`; `mcp` is limited to `<2` (mcp 2.x broke a clean install of `cce-mcp`); the unused `pyvis` extra is gone. CI builds the wheel and smoke-tests it in a clean venv.

Docs
- README: roadmap and status updated, and a section on what the benchmarks do and do not show (they predate the live-session fix and have not been re-run).

## 0.1.0 (2026-10-12)

First public version.

- Engine: segmenting, topic chunking, summaries (micro/meso/macro), hot/warm/cold memory, retrieval, context builder, stateful and stateless sessions.
- Retrieval judges each topic by its best-matching message (per-message vectors are stored with every topic) and quotes that message when the summary omitted it.
- Two embedders: neural (sentence-transformers, optional extra) and a dependency-free hashing embedder; `CCE_EMBEDDER=auto|neural|hash`.
- Answers spread over several messages: a topic now quotes every message in it that matches the question nearly as well as the best one (found by a multi-message test where the first version lost lists to summaries); default top topics 5.
- Overview questions ("what did we talk about?") are recognised and answered with every topic in compact form, in order.
- Detail-aware extractive summaries (favour numbers, names and "our/my ..." statements, drop acknowledgements and questions).
- MCP server (`cce-mcp`), REST API (`cce-rest`, optional `CCE_API_KEY`), JavaScript client and offline engine (`cce_sdk_js`), browser demo.
- Benchmarks on real chat text with local models; see `benchmarks/README.md` for results and limits.
- MIT licence.
