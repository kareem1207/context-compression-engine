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
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384          # all-MiniLM-L6-v2 output dimension
    embedding_batch_size: int = 32    # how many sentences to embed at once

    # ── Semantic chunker ──────────────────────────────────────────────────────
    chunk_similarity_threshold: float = 0.45   # cosine sim below this = new topic
    chunk_min_turns: int = 2                   # minimum turns per chunk
    chunk_max_turns: int = 20                  # hard cap — force split after this
    chunk_window_size: int = 3                 # sliding window for boundary detection

    # ── Summarizer ────────────────────────────────────────────────────────────
    # micro  = single turn condensed to one sentence
    # meso   = topic chunk condensed to 2-4 sentences
    # macro  = full session condensed to one paragraph
    micro_max_tokens: int = 60
    meso_max_tokens: int = 200
    macro_max_tokens: int = 400

    # summarizer backend: "extractive" (no LLM needed) | "llm" (calls local LLM)
    summarizer_mode: str = "extractive"
    llm_endpoint: str = os.getenv("CCE_LLM_ENDPOINT", "http://localhost:8080/v1")
    llm_model: str = os.getenv("CCE_LLM_MODEL", "gemma4")

    # ── Memory tiers ──────────────────────────────────────────────────────────
    hot_tier_max_turns: int = 10          # last N turns always in RAM
    warm_tier_max_chunks: int = 500       # max topic chunks in SQLite
    cold_tier_max_summaries: int = 1000   # max macro summaries on disk

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = 5              # how many chunks to retrieve per query
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