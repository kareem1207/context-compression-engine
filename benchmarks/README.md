# Benchmarks

**The honest summary.** CCE lets a small local model answer questions about things said long ago, using about 90% fewer tokens than resending the chat, and it beats "send everything" and "send the newest messages that fit". But **a plain search over the raw messages with the same token budget finds everything just as reliably**: in the model-free checks below CCE and plain search tie (96/96 facts, and 12/12 on each multi-message task). So these tests show that CCE's *retrieval* works; they do **not** show that its topic structure or summaries beat ordinary retrieval, and on overview questions ("what did we talk about?") it first did worse than plain search before a fix brought it level. On the one test that went against CCE (answers spread over several messages) it first lost badly, which we fixed; it now only ties. We have not tested the cases where summaries should matter (see [What is not tested](#what-is-not-tested)).

**Version note.** These results were produced with one-shot mode (the whole conversation compressed in one call), before the 1.0 fix that makes live sessions (MCP, REST) group topics the same way. They have not been re-run for 1.0.

All tests use local models through llama.cpp on a laptop CPU (AMD Ryzen 7 7730U, no GPU), temperature 0, thinking mode off.

## How the tests work

A conversation is built from **real chat text** (UltraChat 200k, MIT licence, downloaded on first run and not stored in this repo) with 8 facts planted once, in passing, in the first 80% of the chat (for example "our staging database password rotates every 45 days", with randomised values). After the chat, the model is asked one question per fact: half reuse the fact's wording ("lexical"), half rephrase it ("paraphrase"). An answer is correct if it contains the expected keywords. Conversations are different for every seed, so nothing can be memorised.

Prompts compared for the same question:

| Condition | What the model receives |
|---|---|
| **full** | the whole conversation, then the question (only when it fits the model's window) |
| **window** | the newest turns that fit the window, then the question: what most apps do when a chat gets too long |
| **recent** | the newest turns, cut to the **same token budget CCE used** (fair, naive) |
| **cce** | CCE's context: best-matching topic blocks plus the last 4 turns (neural embedder) |
| **cce_hash** | the same, with the dependency-free hashing embedder |

## Results

### Chats that fit the model's window (about 5,300 tokens, context 8,192)

DeepSeek-Coder 6.7B Instruct, 3 conversations x 8 questions = 24 questions:

| Condition | Correct | Mean prompt tokens |
|---|---|---|
| full | 10 / 24 (42%) | 5,910 |
| recent (same budget as CCE) | 0 / 24 (0%) | 641 |
| **CCE** | **19 / 24 (79%)** | **697** |
| CCE, hashing embedder | 11 / 24 (46%) | 851 |

More context made this model worse, not better: with the whole chat in the prompt it got fewer answers right than with CCE's short context.

### Chats longer than the window (about 9,300 tokens, context 4,096)

| Model | window | recent | **CCE** | CCE hashing |
|---|---|---|---|---|
| DeepSeek-Coder 6.7B (24 questions) | 1 / 24 (4%) | 0 / 24 | **18 / 24 (75%)** | 12 / 24 (50%) |
| Gemma 4 E4B (16 questions) | 1 / 16 (6%) | 0 / 16 | **16 / 16 (100%)** | 8 / 16 (50%) |

Mean CCE prompt size: 778 and 687 tokens, against 3,675 and 3,184 for the window. This is the case CCE is for: the model cannot see most of the chat, and CCE brings back the relevant part.

### Model-free check: is the needed fact in the context?

96 questions on 12 conversations that were **not used while developing** (seeds 500 to 511). No LLM involved, so it is fast and exact.

| Method | Fact present | Mean context tokens |
|---|---|---|
| CCE (neural embedder), first version | 95 / 96 (99%) | 611 |
| **CCE after the multi-message fix** (top 5 topics, several quotes per topic) | **96 / 96 (100%)** | 831 |
| CCE, detail-aware summaries switched off | 95 / 96 (99%) | 607 |
| CCE, hashing embedder | 53 / 96 (55%): lexical 48/48, paraphrase 5/48 | 752 |
| **Plain RAG over raw messages, same budget** | **96 / 96 (100%)** | same as CCE |

Findings:
- Turn-level retrieval is what makes CCE find facts: ranking topics only by their average vector missed 5 of 32 on the first look (84%), judging topics by their best-matching message fixed that (32 of 32 on those conversations, 99% on the held-out ones).
- The detail-aware summariser made no measurable difference here, because facts are recovered by retrieval and quoting, not by summarising.
- The hashing embedder needs no downloads but only matches shared words: perfect on lexical questions, almost useless on paraphrased ones.
- **Plain RAG matches CCE.** On "find one stated fact" tasks, CCE adds nothing over simply retrieving the best raw messages.

### Answers spread over several messages (model-free)

The case where plain search was expected to struggle less than CCE's summaries: a list built up one message at a time, and a value that is stated and later corrected. 12 conversations per task, chats of about 5,000 tokens, same token budget for both methods:

| Task | CCE, first version | **CCE, fixed** | Plain RAG |
|---|---|---|---|
| List built over 4 consecutive exchanges | 2 / 12 | **12 / 12** | 12 / 12 |
| List with items scattered through the chat | 1 / 12 | **12 / 12** | 12 / 12 |
| Value stated, then corrected later (new value present) | 12 / 12 | **12 / 12** | 12 / 12 |

This test found a real flaw. CCE sent topic summaries plus one quoted message per topic, and extractive summaries dropped list items, so it lost almost every list. The fix: when a topic matches, also quote every message in it that matches nearly as well as the best one, and retrieve the top 5 topics instead of 3. That repaired lists, at the price of a larger context (about 830 tokens instead of 610). It makes CCE equal to plain search here, not better.

Note: the language-model results above were produced **before** this fix (3 topics, one quote each). They have not been re-run.

### Overview questions: "what did we talk about?" (model-free)

Nothing in such a question matches a single message, so this is where topic summaries were expected to help. 20 conversations x 3 phrasings = 60 questions. Each conversation is six real dialogues back to back (six topics); a topic counts as covered if the context mentions any of its three rare keywords. Plain RAG gets the same token budget as CCE.

| Method | Topics covered | All six covered | Mean context tokens |
|---|---|---|---|
| CCE, first version (5 closest topics) | 60% | 1 / 60 | 1,140 |
| Plain RAG, same budget | 77% | 18 / 60 | 1,088 |
| **CCE with overview mode** (every topic, compact) | **87%** | **31 / 60** | 1,809 |
| Plain RAG, same budget | 89% | 37 / 60 | 1,716 |
| CCE + session summary | 87% | 31 / 60 | 2,198 |
| Plain RAG, same budget | 91% | 42 / 60 | 1,991 |

Predicted going in: CCE would clearly win here. **It did not.** The first version lost because it only sent the 5 closest topics and an overview question matches none of them; plain search happens to spread across the whole chat. Overview mode (recognise recap-style questions and send every topic in order) closed the gap, but CCE still only ties plain search, and the context grew by about 60% to get there. The coverage measure favours raw text (it looks for words from the first message of each topic, which raw quoting includes), so this slightly flatters plain search.

**Bottom line of all the model-free tests: CCE ties plain search over raw messages.** What CCE adds is the packaging (sessions, hot/warm/cold memory, persistence, an MCP server, a REST API, a JavaScript client, a compact context format) and not a measured quality advantage. A language-model test of summary questions is unlikely to change that unless the summaries themselves are written by a model, which we have not tried.

### Earlier, smaller run (12 questions, one hand-written chat)

Gemma 11/12 with the whole chat and 11/12 with CCE; DeepSeek 7/12 and 8/12; newest-turns-only 0/12 for both. Kept in `results/gemma4-e4b.json` and `results/deepseek-coder-6.7b.json`. `results/v1-invalid-harness/` is a first attempt that wrongly showed CCE at 25%: the model's thinking mode used up the token limit and returned empty answers. That was a flaw in the test, not in CCE; it is kept on purpose.

## What is not tested

- **Questions that need a written summary.** We tested overview questions model-free (above) but not with a language model judging answers, and CCE's summaries are picked sentences, not written text. A model-written summariser (`summarizer_mode="llm"`) is untested.
- **Messy real conversations.** Filler is real text, but the facts are planted statements. Real chats have corrections, references like "that one" and facts spread across many turns.
- **Much longer chats**, hosted models, other languages, and larger sample sizes. 16 to 24 questions per cell is small: one question is 4 to 6 percentage points, so differences of one or two questions are noise.
- **Speed.** Prompt-processing time depends on the server: Gemma 4 cannot reuse a cached prompt, DeepSeek can. The token saving is real in both cases; the time saving depends on your setup.
- **The language model summariser** (`summarizer_mode="llm"`).

## Reproduce

```bash
uv sync --group bench
# model-free recall (fast, no LLM)
uv run python benchmarks/retrieval_recall.py --conversations 12 --variants cce,cce_hash
uv run python benchmarks/retrieval_multi.py --conversations 12 --top-k 5
uv run python benchmarks/retrieval_overview.py --conversations 20
# model benchmarks: needs llama.cpp and two GGUF files, about two hours on a CPU
bash benchmarks/run_all_v2.sh              # or: RUNS="overflow" bash benchmarks/run_all_v2.sh
```

Data is deterministic per seed (`dataset_real.py`). Raw per-question answers are in `results/`.
