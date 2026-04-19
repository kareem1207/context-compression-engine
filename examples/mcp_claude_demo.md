# Connecting CCE MCP Server to Claude Desktop

## Prerequisites

- CCE installed and working (all 4 phases passing)
- Claude Desktop installed
- Python 3.12 with `uv` and the CCE venv

---

## Step 1 — Verify the server starts

```bash
# From your project root with venv active
uv run python -m cce_mcp.server
```

You should see no errors. Press Ctrl+C to stop.

---

## Step 2 — Find Claude Desktop config location

| OS      | Path |
|---------|------|
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| macOS   | `~/Library/Application Support/Claude/claude_desktop_config.json` |

---

## Step 3 — Add CCE to claude_desktop_config.json

```json
{
  "mcpServers": {
    "cce": {
      "command": "uv",
      "args": [
        "--directory",
        "E:\\context-compression-engine",
        "run",
        "python",
        "-m",
        "cce_mcp.server"
      ],
      "env": {
        "CCE_SUMMARIZER_MODE": "extractive",
        "CCE_HOT_TIER_TURNS": "10",
        "CCE_TOP_K": "5",
        "CCE_MAX_TOKENS": "2048",
        "CCE_LLM_ENDPOINT": "http://localhost:8080/v1",
        "CCE_LLM_MODEL": "gemma4"
      }
    }
  }
}
```

> **Windows note**: Use double backslashes `\\` or forward slashes `/` in the path.

---

## Step 4 — Restart Claude Desktop

Fully quit and reopen Claude Desktop. The CCE tools should appear in the tools panel.

---

## Step 5 — Test in Claude Desktop

Try these prompts:

```
Use cce_ingest_history to load this conversation into session "test-01":
[paste a conversation here]
```

```
Use cce_retrieve_context to get compressed context for session "test-01"
with query "what did we discuss about Python?"
```

```
Use cce_session_stats to show me the compression stats for session "test-01"
```

---

## Step 6 — Use with Gemma 4 (llama.cpp)

Start llama.cpp server first:
```bash
llama-server -m path/to/gemma-4-it.gguf --port 8080 --ctx-size 4096
```

Then set `CCE_SUMMARIZER_MODE=llm` in the env config above for
higher-quality abstractive summaries from Gemma 4 itself.

---

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `cce_ingest_turn` | Add one message to a session |
| `cce_ingest_history` | Bulk-load full conversation history |
| `cce_retrieve_context` | Get compressed context for LLM injection |
| `cce_summarize_session` | Get macro summary of entire session |
| `cce_session_stats` | Diagnostic stats (tokens, compression ratio) |
| `cce_close_session` | Checkpoint and close a session |
| `cce_list_sessions` | List all known sessions |
| `cce_stateless_compress` | One-shot compress with no persistence |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CCE_SUMMARIZER_MODE` | `extractive` | `extractive` (offline) or `llm` (uses Gemma 4) |
| `CCE_LLM_ENDPOINT` | `http://localhost:8080/v1` | llama.cpp API endpoint |
| `CCE_LLM_MODEL` | `gemma4` | Model name for LLM summarization |
| `CCE_HOT_TIER_TURNS` | `10` | Recent turns kept verbatim in RAM |
| `CCE_TOP_K` | `5` | Memory nodes retrieved per query |
| `CCE_MAX_TOKENS` | `2048` | Max tokens in assembled context payload |
| `CCE_DATA_DIR` | `.cce_data` | Where SQLite DB and cold summaries live |