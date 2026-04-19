Phase 1 — Core compression (cce_core/ingestion + compression) — segmenter, SBERT chunker, hierarchical summarizer, merger. This is pure NLP, no DB yet, fully testable in isolation.
Phase 2 — Memory store (cce_core/memory) — hot/warm/cold tiers with the SQLite vector index. Once this is solid, we plug Phase 1 output directly into it.
Phase 3 — Retrieval + context builder (cce_core/retrieval) — ANN search, recency boosting, context assembly. This is what the LLM actually sees.
Phase 4 — Session layer (cce_core/session) — stateful manager + stateless pass-through mode.
Phase 5 — MCP server (cce_mcp/) — FastMCP wrapper. Test with Gemma 4 via llama.cpp here.
Phase 6 — REST bridge + npm SDK (cce_rest/ + cce_sdk_js/) — so your Electron browser can use it too.
Phase 7 — Packaging — pyproject.toml for pip, package.json for npm, README.