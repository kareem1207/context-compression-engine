"""
CCE embedders.

The engine needs one thing from an embedder: turn text into unit-length vectors.
Three are available:

  neural  sentence-transformers (default model: all-MiniLM-L6-v2). Understands paraphrase.
          Needs the optional `sentence-transformers` package (and PyTorch), about 1 GB installed.
  onnx    a multilingual sentence model (paraphrase-multilingual-MiniLM-L12-v2, int8) run with onnxruntime:
          no PyTorch, about 135 MB of model files downloaded on first use, 50+ languages, 384 dimensions.
          Needs the optional `onnx` extra.
  hash    feature hashing of words and word pairs. No downloads, no extra packages,
          runs anywhere, but matches on shared words rather than meaning.

`load_embedder(config)` picks one. With `config.embedder == "auto"` (the default) it uses the neural model
when sentence-transformers is installed, else the ONNX model when onnxruntime is installed, else hashing.
(sentence-transformers stays first so data folders made with it keep opening; a folder remembers its embedder.)
"""

from __future__ import annotations

import importlib.util
import os
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


class OnnxEmbedder:
    """Sentence embeddings from an int8 ONNX model: onnxruntime + tokenizers, no PyTorch."""

    name = "onnx"

    def __init__(self, repo: str, max_length: int = 256, model_file: str = "onnx/model_quantized.onnx"):
        import numpy  # noqa: F401  (already a dependency)
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        # The model is public: `token=False` never sends stored credentials (an expired token would make a public download fail).
        model_path = hf_hub_download(repo, model_file, token=False)
        tokenizer_path = hf_hub_download(repo, "tokenizer.json", token=False)
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        self._session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._session.get_inputs()}
        self._tok = Tokenizer.from_file(tokenizer_path)
        self._tok.enable_truncation(max_length=max_length)
        self._tok.enable_padding()
        self.model_name = repo
        self.dim = int(self._embed_batch(["dimension probe"]).shape[1])

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        enc = self._tok.encode_batch([t if t.strip() else " " for t in texts])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self._session.run(None, feed)[0]
        pooled = (hidden * mask[..., None]).sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.maximum(norms, 1e-8)).astype(np.float32)

    def encode(self, texts, batch_size: int = 32, normalize_embeddings: bool = True, show_progress_bar: bool = False):
        if isinstance(texts, str):
            return self._embed_batch([texts])[0]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))      # similar lengths pad less
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            out[idx] = self._embed_batch([texts[i] for i in idx])
        return out


def onnx_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("onnxruntime", "tokenizers", "huggingface_hub"))


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
    if mode == "onnx":
        if not onnx_available():
            raise ImportError(
                "embedder='onnx' needs onnxruntime, tokenizers and huggingface_hub: "
                "pip install 'context-compression-engine[onnx]'"
            )
        return OnnxEmbedder(config.onnx_model, config.onnx_max_length)
    if mode != "auto":
        raise ValueError(f"Unknown embedder {mode!r}; use 'auto', 'neural', 'onnx' or 'hash'")
    if neural_available():
        return NeuralEmbedder(config.embedding_model)
    if onnx_available():
        return OnnxEmbedder(config.onnx_model, config.onnx_max_length)
    return HashingEmbedder(getattr(config, "hash_dim", 512))
