# Benchmarks

CCE 1.0 (release branch), re-measured on 2026-10-21 (model-free tests and speed) and on the dates given in each language-model table. Every number below says which version and which day produced it; numbers from an older version are marked as such and were not re-run.

**The honest summary.**
- CCE lets a small local model answer questions about things said long ago with about 88% fewer tokens than resending the whole chat (681 against 5,821 tokens in the latest fit run), and it beat "send everything" and "send the newest messages that fit" in every language-model run (tables below).
- **A plain search over the raw messages with the same token budget finds single facts just as reliably.** In the model-free tests CCE and plain search tie on facts (96/96 each) and on lists and corrections (12/12 each). These tests show that CCE's retrieval works; they do not show that its topic structure beats ordinary retrieval for finding facts.
- On overview questions ("what did we talk about?") CCE is ahead of plain search on the same budget on one set of conversations (36/60 against 29/60 with all six topics covered), but on a second, held-out set the two tie (12/60 each).
- With the neural English embedder CCE finds every planted fact; with the dependency-free hashing embedder it finds 61%; with the multilingual ONNX embedder it finds every fact in English, Spanish, Hindi and Chinese in our small test.
- The language-model results are small (24 questions per cell) and on two local models. Read [what is not tested](#what-is-not-tested).

Hardware for everything: AMD Ryzen 7 7730U laptop (16 logical cores), Windows 11, Python 3.12, no GPU. Language models run through llama.cpp, temperature 0, thinking mode off.

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

## Language-model results

### DeepSeek-Coder 6.7B, chats that fit the window (about 5,300 tokens, context 8,192; 3 conversations x 8 questions)

| Condition | CCE 0.1.0 (2026-10-12, published) | CCE 1.0, code of PR #13 (2026-10-20) |
|---|---|---|
| full | 10 / 24 (42%), 5,910 tokens | 10 / 24 (42%), 5,821 tokens |
| recent (same budget as CCE) | 0 / 24 (0%), 641 tokens | 0 / 24 (0%), 610 tokens |
| **CCE** | **19 / 24 (79%)**, 697 tokens | **17 / 24 (71%)**, 681 tokens |
| CCE, hashing embedder | 11 / 24 (46%), 851 tokens | 12 / 24 (50%), 876 tokens |

In the 2026-10-20 run, split by question type: lexical questions CCE 100%, full 83%; paraphrased questions CCE 42%, full 0%. More context made this model worse, not better: with the whole chat in the prompt it got fewer answers right than with CCE's short context. CCE used 88% fewer tokens than the full chat (681 against 5,821). The 1.0 column was measured before the token-estimate fix (which at that time cost 8 of 96 questions in the model-free recall test); it was not re-run on the final code.

<!--LLM-NEW-->

### DeepSeek-Coder 6.7B, chats longer than the window (about 9,300 tokens, context 4,096; 3 conversations x 8 questions)

| Condition | CCE 0.1.0 (2026-10-12, published) | CCE 1.0, code of PR #13 (2026-10-20, run log) |
|---|---|---|
| window (newest turns that fit) | 1 / 24 (4%) | 3 / 24 (12%) |
| recent (same budget as CCE) | 0 / 24 | 1 / 24 (4%) |
| **CCE** | **18 / 24 (75%)** | **18 / 24 (75%)** |
| CCE, hashing embedder | 12 / 24 (50%) | 9 / 16 (56%), the run stopped before the last conversation |

Gemma 4 E4B, 0.1.0 only (2026-10-12): window 1/16 (6%), recent 0/16, **CCE 16/16 (100%)**, hashing 8/16 (50%). It has not been re-run for 1.0. Mean CCE prompt size in the 0.1.0 runs was 778 and 687 tokens, against 3,675 and 3,184 for the window. This is the case CCE is for: the model cannot see most of the chat, and CCE brings back the relevant part. The raw per-question answers of the 0.1.0 runs are in `results/`; the 2026-10-20 fit run is `results/deepseek-coder-6.7b-fit-v1.0-2026-10-20.json`.

## Model-free results (CCE 1.0 code, 2026-10-21)

These need no language model: does the context CCE builds contain the fact (or topic) needed? Fast and exact, so they run on many conversations. Plain RAG is the best-matching raw messages under the same token budget.

### Held-out recall: 12 conversations not used while developing (seeds 500 to 511), 96 questions

`uv run --group bench python benchmarks/retrieval_recall.py --conversations 12 --seed-start 500 --top-k 5 --variants cce,cce_hash,cce_onnx`

| Method | Fact present | Lexical | Paraphrase | Mean context tokens |
|---|---|---|---|---|
| CCE, MiniLM (neural, English) | **96 / 96** | 48 / 48 | 48 / 48 | 947 |
| CCE, multilingual ONNX | **96 / 96** | 48 / 48 | 48 / 48 | 1,032 |
| CCE, hashing embedder | 59 / 96 (61%) | 48 / 48 | 11 / 48 | 1,140 |
| Plain RAG over raw messages, same budget as the first row | 96 / 96 | 48 / 48 | 48 / 48 | same |

Plain RAG matches CCE. The hashing embedder needs no downloads and finds every lexical question but almost no paraphrased one (it matches shared words).

How this number moved during the 1.0 work (same command, MiniLM): 0.1.0 96/96; the first 1.0 code 88/96; after the token-estimate fix 96/96. The cause of the dip was one message made of long runs of underscores that the new token counter counted at 1,711 tokens (158 by cl100k_base), which crowded the past context out of the payload. A CI job now runs a pinned part of this test and fails below a threshold, so this cannot slip again.

### Facts buried inside longer messages (live sessions, `retrieval_live.py`, 6 conversations, 48 questions)

`uv run --group bench python benchmarks/retrieval_live.py --conversations 6 --top-k 5`

| Mode | Fact present (lexical / paraphrase) |
|---|---|
| one-shot (stateless processor) | 48 / 48 (24 / 24) |
| live: messages added one by one | 48 / 48 (24 / 24) |
| live, after close and reopen with a new engine | 47 / 48 (24 / 23) |
| live, after a simulated crash (no close) | 48 / 48 (24 / 24) |
| facts buried in a long message, live after reopen | 47 / 48 (24 / 23) |
| facts buried in a long message, one-shot | 48 / 48 (24 / 24) |

Before the sentence index the buried rows were 33/48 (23 / 10) and 34/48 (19 / 15).

### Answers spread over several messages (`retrieval_multi.py --top-k 5`, 12 conversations per task)

| Task | CCE | Plain RAG |
|---|---|---|
| List built over 4 consecutive exchanges | 12 / 12 | 12 / 12 |
| List with items scattered through the chat | 12 / 12 | 12 / 12 |
| Value stated, then corrected later (new value present) | 12 / 12 | 12 / 12 |

### Overview questions: "what did we talk about?" (`retrieval_overview.py --top-k 5`, 20 conversations x 3 phrasings = 60 questions)

Each conversation is six real dialogues back to back; a topic counts as covered if the context mentions any of three rare words of its first message. Plain RAG gets the same token budget as CCE.

| Set | Method | Topics covered | All six covered | Mean context tokens |
|---|---|---|---|---|
| seeds 900 to 919 (the set the overview mode was tuned on) | **CCE** | 88% | **36 / 60** | 1,577 |
| | plain RAG, same budget | 85% | 29 / 60 | 1,492 |
| | CCE + session summary | 90% | 36 / 60 | 1,954 |
| | plain RAG, that larger budget | 90% | 39 / 60 | 1,841 |
| seeds 1000 to 1019 (held out) | CCE | 80% | 12 / 60 | 1,593 |
| | plain RAG, same budget | 79% | 12 / 60 | 1,500 |

On the tuned set CCE is ahead of plain RAG at the same budget; on the held-out set they tie, so do not read the first set as a general advantage. Before the overview change CCE scored 32/60 on seeds 900 to 919 with 1,787 tokens. Adding the session summary does not help CCE (36 against 36). The coverage measure has a ceiling: its keywords come from the unclipped first message, so a perfect context covers all six topics in only 14 of the 20 conversations (42 of 60 questions) on seeds 900 to 919.

### Other languages (`retrieval_multilingual.py --seeds 5`)

Hand-written conversations in Spanish, Hindi and Chinese (English as a control), four planted facts each, asked as paraphrases in the same language; 20 questions per language.

| Embedder | Spanish | Hindi | Chinese | English | Spanish + Hindi + Chinese |
|---|---|---|---|---|---|
| multilingual ONNX | 20 / 20 | 20 / 20 | 20 / 20 | 20 / 20 | **60 / 60** |
| MiniLM (English) | 18 / 20 | 15 / 20 | 16 / 20 | 20 / 20 | 49 / 60 |
| hashing | 17 / 20 | 4 / 20 | 2 / 20 | 4 / 20 | 23 / 60 |

This set is small and easy (conversations of about 24 turns, shared-script filler); it shows the model works in these languages, not how it ranks at scale.

### Speed and size (`performance.py`, 2026-10-21)

1,000 real chat messages ingested live one by one, then 100 queries (see the README for what is measured and the breakdown):

| Embedder | Messages ingested per second | Query p50 / p95 (ms) | Database growth (MB per 1,000 messages) |
|---|---|---|---|
| hashing | 166 | 16.5 / 22.6 | 7.6 |
| multilingual ONNX | 6.1 | 44.8 / 159.2 | 6.4 |
| MiniLM | 8.9 | 63.1 / 151.0 | 6.4 |

Search index memory: 383 MB for 10,000 synthetic notes. Ingestion with a neural embedder is slow (6 to 9 messages per second here) and has not been optimised.

## What is not tested

- **Questions that need a written summary.** Overview questions were tested model-free, not with a language model judging answers, and CCE's summaries are picked sentences, not written text. The model-written summariser (`summarizer_mode="llm"`) is experimental and untested.
- **Messy real conversations.** Filler is real text, but the facts are planted statements. Real chats have corrections, references like "that one" and facts spread across many turns.
- **Much longer chats**, hosted models, and larger sample sizes. 16 to 24 questions per language-model cell is small: one question is 4 to 6 percentage points, so differences of one or two questions are noise. Gemma was not re-run for 1.0.
- **The language-model results on the final code** are limited to the runs shown above; the model-free tests on the final code are complete.
- **Speed on other hardware**, and long-running servers (memory growth over days).

## Reproduce

```bash
uv sync --group bench
# model-free (fast, no language model; needs the MiniLM and ONNX models cached, HF_HUB_OFFLINE=1 after the first run)
uv run --group bench python benchmarks/retrieval_recall.py --conversations 12 --seed-start 500 --top-k 5 --variants cce,cce_hash,cce_onnx
uv run --group bench python benchmarks/retrieval_multi.py --top-k 5
uv run --group bench python benchmarks/retrieval_overview.py --top-k 5            # add --seed 1000 for the held-out set
uv run --group bench python benchmarks/retrieval_live.py --conversations 6 --top-k 5
uv run --group bench python benchmarks/retrieval_multilingual.py --seeds 5
uv run --group bench python benchmarks/performance.py --embedders hash,onnx,neural
# language models: needs llama.cpp and two GGUF files, a few hours on a CPU
bash benchmarks/run_all_v2.sh              # or: RUNS="overflow" bash benchmarks/run_all_v2.sh
```

Data is deterministic per seed (`dataset_real.py`, `dataset_multilingual.py`). The CI job `recall-gate` runs pinned parts of the recall, live-session and multilingual tests and fails if recall drops below a threshold. Raw per-question answers are in `results/`.
