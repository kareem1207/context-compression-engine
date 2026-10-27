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

Details in longer messages
- A fact buried in a multi-sentence message could be lost once its turn was compressed, because topics only quoted a one-line summary per message. The original text of compressed turns is now kept in SQLite (`turns` table, state `done`), and a matching topic quotes the best-matching sentences of its messages (plus a neighbouring sentence for very short ones). Setting `keep_raw_turns` / `CCE_KEEP_RAW_TURNS=0` turns this off. The data stays on the user's machine and is removed by deleting the session or the data folder; deleting a note removes the text it covered. Databases from 1.0 are upgraded in place (schema 5); their existing notes keep the old behaviour.

Overview questions
- "What did we talk about?" is answered with every topic (up to `overview_max_topics`, now 24) in conversation order. Each topic shows its first user message and the sentences that cover the most terms particular to it, within an even share of the token budget (`overview_budget_share`), instead of its extractive summary. Notes record which turns were the user's (`roles`).

Sentence index and BM25
- Messages of 40 words or more get one vector per sentence at compression time (`sentence_index_min_words`), stored as float16 with the note, so a fact buried in a long message is ranked by its own sentence and quoted without embedding it again at query time. Notes saved by earlier versions keep working.
- Notes are also indexed for BM25 (term counts stored with the note; from the original messages when `keep_raw_turns` is on, otherwise from the summaries). Identifiers, codes, acronyms, names and numbers of a question are matched exactly and fused with the dense ranking by reciprocal rank; the old ad-hoc keyword bonus is gone. `retrieval_live.py` takes `--sentence-min-words` and `--lexical-weight` to switch the parts off.

Token counter calibration
- The first 1.0 token estimate counted runs of one repeated character (a line of underscores in a contract template) at 2.2 characters per token: 1,711 tokens where tiktoken counts 158. One such message in the recent turns crowded the past context out of the payload and cost 8 of 96 held-out recall questions. The estimate is now the larger of the old word count and a feature estimate fitted to cl100k_base on 25,000 real chat messages plus JSON, code and CJK samples (a run of one character is one token). English chat is counted as before; the worst case against tiktoken is about 1.5x instead of 4x.
- CI has a recall gate (`benchmarks/retrieval_recall.py` and `benchmarks/retrieval_live.py` with pinned seeds and `--min` thresholds) so a drop in recall fails the build.

Session stats
- `compression_ratio` was `raw / (note tokens + 50 per hot turn)`, a guess that could show 0.41 for a short chat, and the "note tokens" were the raw size of the turns the notes cover. Stats now use the real size of what would be sent (notes as sent + hot and pending turns): new `tokens_sent` and `tokens_saved`, `compression_ratio` is `null` when the session is not smaller than its raw text, `warm_tokens_compressed` is now the size of the notes as sent and `warm_tokens_original` the raw size they cover.

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
