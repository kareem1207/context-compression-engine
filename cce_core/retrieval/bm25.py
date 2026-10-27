"""BM25 over a small set of documents (the notes of one session) and rank fusion with the dense score."""

from __future__ import annotations

import math

import numpy as np


class Bm25:
    """Okapi BM25 with incremental document adds. Documents are term-frequency dicts."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = {}   # term -> [(doc, tf)]
        self.lengths: list[int] = []

    def add(self, tf: dict[str, int]) -> int:
        doc = len(self.lengths)
        self.lengths.append(sum(tf.values()))
        for term, n in tf.items():
            self.postings.setdefault(term, []).append((doc, n))
        return doc

    def __len__(self) -> int:
        return len(self.lengths)

    def score(self, query_terms, max_df_ratio: float | None = None) -> np.ndarray:
        """BM25 per document. With `max_df_ratio`, terms found in more than that share of the documents are
        ignored: common words only add noise, while ids, names and numbers are rare."""
        n = len(self.lengths)
        scores = np.zeros(n, dtype=np.float32)
        if not n:
            return scores
        avg = max(sum(self.lengths) / n, 1.0)
        lengths = np.asarray(self.lengths, dtype=np.float32)
        norm = self.k1 * (1.0 - self.b + self.b * lengths / avg)
        for term in set(query_terms):
            plist = self.postings.get(term)
            if not plist:
                continue
            if max_df_ratio is not None and len(plist) > max(1, int(max_df_ratio * n)):
                continue
            idf = math.log(1.0 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            docs = np.fromiter((d for d, _ in plist), dtype=np.int64, count=len(plist))
            tfs = np.fromiter((t for _, t in plist), dtype=np.float32, count=len(plist))
            scores[docs] += idf * tfs * (self.k1 + 1.0) / (tfs + norm[docs])
        return scores


def fuse(dense: np.ndarray, *extra: tuple[np.ndarray, np.ndarray, float], k: int = 60, w_dense: float = 1.0) -> np.ndarray:
    """
    Reciprocal rank fusion of the dense score with other rankings of the same documents.
    Each extra is (scores, mask, weight): only documents where `mask` is true are ranked in that list and
    get a share, so a document with no lexical match (or no sentence vectors) gets nothing from it and a
    question that shares no words with the notes is ranked by the dense score alone.
    """
    n = len(dense)
    fused = np.zeros(n, dtype=np.float32)
    if not n:
        return fused
    d_rank = np.empty(n, dtype=np.int64)
    d_rank[np.argsort(-dense, kind="stable")] = np.arange(n)
    fused += w_dense / (k + 1 + d_rank)
    for scores, mask, weight in extra:
        hit = np.flatnonzero(mask)
        if not len(hit) or weight <= 0:
            continue
        order = hit[np.argsort(-scores[hit], kind="stable")]
        rank = np.arange(len(order))
        fused[order] += weight / (k + 1 + rank)
    return fused
