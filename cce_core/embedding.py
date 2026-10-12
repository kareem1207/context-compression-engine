"""
CCE embedders.

The engine needs one thing from an embedder: turn text into unit-length vectors.
Two are available:

  neural  sentence-transformers (default model: all-MiniLM-L6-v2). Understands paraphrase.
          Needs the optional `sentence-transformers` package (and PyTorch), about 1 GB installed.
  hash    feature hashing of words and word pairs. No downloads, no extra packages,
          runs anywhere, but matches on shared words rather than meaning.

`load_embedder(config)` picks one. With `config.embedder == "auto"` (the default) it uses the
neural model when sentence-transformers is installed and falls back to hashing otherwise.
"""

from __future__ import annotations

import re
import zlib

import numpy as np

_WORD = re.compile(r"[a-z0-9]{2,}")
_STOP = frozenset(
    "the a an is it to do of and or in on at for with that this was are be have has had you your we can will "
    "would could should what how why when where about just so but if not no i me my tell explain describe "
    "give show help".split()
)


def _stem(w: str) -> str:
    # Light stemming so "webhooks"/"webhook" and "running"/"run" share a bucket.
    return re.sub(r"(ing|ed|es|s)$", "", w) if len(w) > 4 else w


class HashingEmbedder:
    """Dependency-free embedder. Same call shape as SentenceTransformer.encode for what CCE uses."""

    name = "hash"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        words = [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP]
        prev = None
        for w in words:
            h = zlib.crc32(w.encode())
            v[h % self.dim] += 1.0 if h & 0x80000000 == 0 else -1.0
            if prev is not None:
                h2 = zlib.crc32(f"{prev}_{w}".encode())
                v[h2 % self.dim] += 0.5 if h2 & 0x80000000 == 0 else -0.5
            prev = w
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-8 else v

    def encode(self, texts, batch_size: int = 32, normalize_embeddings: bool = True, show_progress_bar: bool = False):
        if isinstance(texts, str):
            return self._vector(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts])


class NeuralEmbedder:
    """Thin lazy wrapper around SentenceTransformer."""

    name = "neural"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self._model = SentenceTransformer(model_name)
        getter = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = int(getter())

    def encode(self, texts, **kwargs):
        return self._model.encode(texts, **kwargs)


def neural_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def load_embedder(config):
    mode = getattr(config, "embedder", "auto")
    if mode == "hash":
        return HashingEmbedder(getattr(config, "hash_dim", 512))
    if mode == "neural":
        if not neural_available():
            raise ImportError(
                "embedder='neural' needs sentence-transformers: pip install 'context-compression-engine[neural]'"
            )
        return NeuralEmbedder(config.embedding_model)
    if mode != "auto":
        raise ValueError(f"Unknown embedder {mode!r}; use 'auto', 'neural' or 'hash'")
    if neural_available():
        return NeuralEmbedder(config.embedding_model)
    return HashingEmbedder(getattr(config, "hash_dim", 512))
