"""
CCE Global Configuration
All tuneable constants live here. Override via environment variables or
instantiate CCEConfig with custom values in your app.
"""

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class CCEConfig:
    # ── Embedding model ───────────────────────────────────────────────────────
    # "auto" uses the neural model if sentence-transformers is installed, otherwise the
    # dependency-free hashing embedder. Force one with "neural" or "hash" (or CCE_EMBEDDER).
    embedder: str = os.getenv("CCE_EMBEDDER", "auto")
    hash_dim: int = 512
    # Hash embeddings score lower than neural ones for the same topic, so they need a lower threshold.
    hash_chunk_similarity_threshold: float = 0.10
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384          # all-MiniLM-L6-v2 output dimension
    embedding_batch_size: int = 32    # how many sentences to embed at once

    # ── Semantic chunker ──────────────────────────────────────────────────────
    chunk_similarity_threshold: float = 0.45   # cosine sim below this = new topic
    chunk_min_turns: int = 2                   # minimum turns per chunk
    chunk_max_turns: int = 20                  # hard cap — force split after this
    chunk_window_size: int = 3                 # sliding window for boundary detection

    # ── Sentence index / lexical search ───────────────────────────────────────
    # Messages with at least this many words also get one vector per sentence (float16, stored with the
    # note), so a fact inside a long message is ranked and quoted on its own. 0 = off.
    sentence_index_min_words: int = 40
    sentence_index_max_per_turn: int = 40
    # Notes are also ranked with BM25 over their text; the dense and BM25 rankings are fused (reciprocal rank).
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    rrf_k: int = 60
    # A note's dense score is the larger of its message score and its best sentence score times this.
    # Below 1, because with one vector per sentence there are many more chances for an unrelated sentence to match.
    sentence_discount: float = 0.9
    lexical_max_df: float = 1.0       # BM25 ignores query terms found in more than this share of a session's notes
    # Which question terms go to BM25: "distinctive" = identifiers, codes, acronyms, names and numbers;
    # "all" = every non-stopword. Exact matching of ordinary words only adds noise to paraphrased questions.
    lexical_terms: str = "distinctive"
    lexical_weight: float = 1.0       # weight of the BM25 ranking against the dense ranking (0 = dense only)

    # ── Summarizer ────────────────────────────────────────────────────────────
    # micro  = single turn condensed to one sentence
    # meso   = topic chunk condensed to 2-4 sentences
    # macro  = full session condensed to one paragraph
    micro_max_tokens: int = 60
    meso_max_tokens: int = 200
    macro_max_tokens: int = 400

    # summarizer backend: "extractive" (no LLM needed) | "llm" (calls local LLM)
    # Extractive summaries favour sentences that carry facts (numbers, names, "our/my ..." statements)
    # over generic ones. Turn off to get the plain sentence-frequency behaviour.
    detail_summaries: bool = True
    summarizer_mode: str = "extractive"
    llm_endpoint: str = os.getenv("CCE_LLM_ENDPOINT", "http://localhost:8080/v1")
    llm_model: str = os.getenv("CCE_LLM_MODEL", "gemma4")

    # ── Memory tiers ──────────────────────────────────────────────────────────
    hot_tier_max_turns: int = 10          # last N turns always in RAM
    warm_tier_max_chunks: int = 500       # max topic chunks in SQLite
    cold_tier_max_summaries: int = 1000   # max macro summaries on disk

    # ── Live sessions (MCP / REST) ────────────────────────────────────────────
    # Turns that leave the hot tier wait in a persisted "pending" buffer and are compressed in
    # topic-sized batches (the same chunker as one-shot mode). Pending turns stay available to
    # retrieval, verbatim, until they are compressed. Everything left is flushed on checkpoint/close.
    live_batch_turns: int = 8             # compress once this many turns are pending (<= 1: compress each eviction at once)
    live_pending_max_turns: int = 24      # flush everything pending beyond this, even if the last topic is still open
    live_pending_max_tokens: int = 3000   # same, by size

    # Keep the original text of compressed turns in the local SQLite file, so a retrieved topic can quote
    # the exact sentence that answers a question instead of the one-line summary of the turn.
    # False: compressed turns are dropped once they are summarised (CCE_KEEP_RAW_TURNS=0).
    keep_raw_turns: bool = field(
        default_factory=lambda: os.getenv("CCE_KEEP_RAW_TURNS", "1").strip().lower() not in ("0", "false", "no", "off")
    )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = 5              # how many chunks to retrieve per query
    overview_max_topics: int = 12         # overview questions ("what did we cover?") show up to this many topics
    retrieval_recency_boost: float = 0.2  # added to score for recent chunks
    context_max_tokens: int = 2048        # hard cap on assembled context payload

    # ── Storage paths ─────────────────────────────────────────────────────────
    base_dir: Path = field(
        default_factory=lambda: Path(os.getenv("CCE_DATA_DIR", ".cce_data"))
    )

    @property
    def db_path(self) -> Path:
        return self.base_dir / "warm.db"

    @property
    def cold_dir(self) -> Path:
        return self.base_dir / "cold"

    def ensure_dirs(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.cold_dir.mkdir(parents=True, exist_ok=True)


# Module-level singleton — import this everywhere
DEFAULT_CONFIG = CCEConfig()