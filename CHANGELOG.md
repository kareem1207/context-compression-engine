# Changelog

## 0.1.0 (unreleased)

First public version.

- Engine: segmenting, topic chunking, summaries (micro/meso/macro), hot/warm/cold memory, retrieval, context builder, stateful and stateless sessions.
- Retrieval judges each topic by its best-matching message (per-message vectors are stored with every topic) and quotes that message when the summary omitted it.
- Two embedders: neural (sentence-transformers, optional extra) and a dependency-free hashing embedder; `CCE_EMBEDDER=auto|neural|hash`.
- Detail-aware extractive summaries (favour numbers, names and "our/my ..." statements, drop acknowledgements and questions).
- MCP server (`cce-mcp`), REST API (`cce-rest`, optional `CCE_API_KEY`), JavaScript client and offline engine (`cce_sdk_js`), browser demo.
- Benchmarks on real chat text with local models; see `benchmarks/README.md` for results and limits.
- MIT licence.
