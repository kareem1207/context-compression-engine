# Benchmark: does CCE keep answers right with far fewer tokens?

**Short answer from this run:** on both local models tested, CCE answered as accurately as sending the whole conversation (Gemma 11/12 vs 11/12, DeepSeek 8/12 vs 7/12) while sending about **86% fewer tokens**. A naive "just send the most recent messages" baseline with the same token budget found **0 of 12** facts. On Gemma, which cannot reuse a cached prompt, answering was also about **7.6x faster** on a CPU; on DeepSeek, where the server cached the shared history, the speed advantage disappeared (see the note below the table).

This is a small, synthetic test. Read [What this does and does not show](#what-this-does-and-does-not-show) before quoting it.

## The test

A 120-turn conversation (about 3,500 tokens) drifts across six everyday topics (billing, cooking, running, Python, travel, fitness). Twelve facts are mentioned **once, in passing** (for example "our staging database password rotates every 45 days"), somewhere in the first 80% of the chat, between 12 and 56 exchanges before the end. Afterwards the model is asked 12 questions, one per fact. Six reuse the fact's own words ("lexical"), six rephrase it ("paraphrase"). An answer counts as correct if it contains the expected keywords.

Three ways of building the prompt for each question:

| Condition | What the model receives |
|---|---|
| **full** | the entire conversation, then the question |
| **recent** | only the newest turns, cut to the same token budget CCE used (a fair naive baseline) |
| **cce** | CCE's context: the best-matching topic blocks plus the last 4 turns, then the question |

CCE settings: extractive summariser, MiniLM embeddings, 4 recent turns kept verbatim, top 3 topics, 200-token summary budget per topic. Models run through llama.cpp on a laptop CPU (AMD Ryzen 7 7730U, no GPU), temperature 0, thinking mode off.

## Results

| Model | Condition | Correct | Mean prompt tokens | Mean prompt processing time |
|---|---|---|---|---|
| Gemma 4 E4B (Q4_K_M) | full | 11 / 12 | 3,521 | 87.7 s |
| | recent (same budget) | 0 / 12 | 429 | 7.1 s |
| | **CCE** | **11 / 12** | **495** | **11.5 s** |
| DeepSeek-Coder 6.7B Instruct | full | 7 / 12 | 3,980 | 22.4 s |
| | recent (same budget) | 0 / 12 | 524 | 14.3 s |
| | **CCE** | **8 / 12** | **534** | 20.0 s |

**Speed note.** Prompt-processing time is only a fair comparison when the server cannot reuse earlier work. Gemma 4 uses sliding-window attention, so llama.cpp reprocessed the whole history for every question (87.7 s vs 11.5 s). DeepSeek-Coder let llama.cpp cache the history that every "full" prompt shares, so those prompts were cheap (22.4 s vs 20.0 s). The token saving is real in both cases, and it is what you pay for with a hosted API; the time saving depends on your server.

By question type (Gemma): lexical 6/6 for both full and CCE; paraphrase 5/6 for both. Raw per-question answers are in [`results/`](results). CCE also costs about 2.7 s per question to build the context on this CPU (it recompresses the whole chat each time in stateless mode; a stored session does this once).

Two questions are worth a look:
- Gemma with the full history said "not mentioned" for one fact that CCE's shorter context got right: **more context is not always better** for a small model.
- CCE missed one paraphrase ("how much do we lose to processing costs on each sale") that the full prompt got right: paraphrases are where retrieval can fail.

## What this does and does not show

It shows that, when the needed fact lives in an older topic, retrieving the right topic block recovers it, and that doing so is much cheaper than resending everything.

It does **not** show:
- **General quality.** Twelve questions on one synthetic conversation is a smoke test. One question is 8 percentage points, so the DeepSeek result (CCE 8/12 vs full 7/12) is a tie, not a win.
- **Summarisation quality.** Most topic blocks in this chat were shorter than the 200-token summary budget, so they were kept verbatim. The savings here come from **retrieval** (sending only relevant topics), not from summarising. Longer topics would be summarised by a simple sentence-picking method that can drop details.
- **Long contexts.** The whole conversation fit in the model's window. The gap should widen for chats that do not fit, but that is not measured here.
- **Other models, other languages, or messy real conversations.**
- The two prompt-processing times are single CPU runs and depend on this machine.

## Reproduce

```bash
# 1. start a llama.cpp server with any GGUF model
llama-server -m model.gguf --port 8080 --ctx-size 8192
# 2. run the benchmark (writes benchmarks/results/<label>.json)
uv run python benchmarks/run_benchmark.py --label my-model
```

`dataset.py` is deterministic (seeded), so the same conversation and questions are generated every time. The first run in `results/v1-invalid-harness/` is kept on purpose: the model's thinking mode used up the token limit and returned empty answers, which made CCE look like 25%. That was a flaw in the test harness, not in CCE, and was fixed before the results above.
