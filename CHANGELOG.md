# Changelog

## 0.1.0 (unreleased)

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
