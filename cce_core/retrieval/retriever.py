"""
CCE Retriever
Given a query (text or embedding), retrieves the most relevant MemoryNodes
from the warm tier and returns them ranked by a composite score.

Composite score = semantic_similarity + recency_boost + keyword_bonus

  semantic_similarity  — cosine sim between query embedding and node centroid
  recency_boost        — small additive bonus for more recent nodes
  keyword_bonus        — bonus if query keywords appear in topic_label/meso_summary

The retriever is stateless — it takes a MemoryStore and a query, returns results.
It does NOT modify any state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from cce_core.config import CCEConfig, DEFAULT_CONFIG
from cce_core.compression.merger import MemoryNode, decode_turn_vecs

if TYPE_CHECKING:
    from cce_core.memory.store import MemoryStore


_WORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")

_STOPWORDS = {
    "the", "a", "an", "is", "it", "to", "do", "of", "and", "or",
    "in", "on", "at", "for", "with", "that", "this", "was", "are",
    "be", "have", "has", "had", "you", "your", "we", "can", "will",
    "would", "could", "should", "what", "how", "why", "when", "where",
    "about", "just", "so", "but", "if", "not", "no", "i", "me", "my",
    "tell", "explain", "describe", "give", "show", "help",
}


@dataclass
class RetrievalResult:
    """A single retrieved memory node with its composite score breakdown."""
    node: MemoryNode
    semantic_score: float      # raw cosine similarity
    recency_score: float       # normalized recency contribution
    keyword_score: float       # keyword overlap bonus
    composite_score: float     # final ranking score
    best_turn: str = ""        # text of the best-matching message in the node (for quoting)

    @property
    def compressed_text(self) -> str:
        return self.node.compressed_text

    def to_dict(self) -> dict:
        return {
            "node_id": self.node.node_id,
            "topic_label": self.node.topic_label,
            "meso_summary": self.node.meso_summary,
            "turn_range": [self.node.turn_start, self.node.turn_end],
            "semantic_score": round(self.semantic_score, 4),
            "recency_score": round(self.recency_score, 4),
            "keyword_score": round(self.keyword_score, 4),
            "composite_score": round(self.composite_score, 4),
        }

    def __repr__(self):
        return (
            f"RetrievalResult(topic={self.node.topic_label!r}, "
            f"score={self.composite_score:.3f}, "
            f"turns={self.node.turn_start}-{self.node.turn_end})"
        )


def node_similarity(node: MemoryNode, qvec: np.ndarray) -> tuple[float, str]:
    """
    Similarity of a query to a node, judged by its best-matching message rather than only
    the topic centroid. A single specific fact inside a long, mixed topic barely moves the
    centroid, so centroid-only search misses it; the best single message does not.

    Returns (score, text of the best-matching message). Nodes saved before per-turn vectors
    existed fall back to the centroid.
    """
    centroid = float(node.embedding @ qvec)
    vecs = decode_turn_vecs(node.metadata)
    if vecs is None or vecs.shape[1] != qvec.shape[0] or len(vecs) == 0:
        return centroid, ""
    sims = vecs @ qvec
    i = int(np.argmax(sims))
    best = float(sims[i])
    # If a question matched best, the useful part is the answer that follows it.
    micros = node.micro_summaries
    text = micros[i] if i < len(micros) else ""
    return max(best, centroid), text


class Retriever:
    """
    Stateless retriever. Takes a store + query, returns ranked RetrievalResults.

    Usage:
        retriever = Retriever(embed_fn=chunker.embed_text)
        results = retriever.retrieve(store, query="what did we say about SQL?", top_k=3)
    """

    def __init__(
        self,
        embed_fn,              # Callable[[str], np.ndarray]
        config: CCEConfig = DEFAULT_CONFIG,
    ):
        self.embed_fn = embed_fn
        self.config = config

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        store: "MemoryStore",
        query: str,
        top_k: int | None = None,
        recency_boost: float | None = None,
        keyword_bonus_weight: float = 0.15,
    ) -> list[RetrievalResult]:
        """
        Main retrieval entry point.

        Args:
            store: The MemoryStore to search.
            query: Natural language query string.
            top_k: Number of results. Defaults to config.retrieval_top_k.
            recency_boost: Additive recency weight. Defaults to config value.
            keyword_bonus_weight: Weight for keyword overlap bonus.

        Returns:
            Ranked list of RetrievalResult objects, best first.
        """
        top_k = top_k or self.config.retrieval_top_k
        boost = recency_boost if recency_boost is not None else self.config.retrieval_recency_boost

        query_emb = self.embed_fn(query)
        query_keywords = self._extract_keywords(query)

        # Get raw semantic results from warm tier (already recency-boosted at DB level)
        qvec = query_emb.astype(np.float32)
        nodes = store.warm.get_all(session_id=store.session_id)
        if nodes and nodes[0].embedding.shape[0] != qvec.shape[0]:
            raise ValueError(
                f"Stored embeddings have {nodes[0].embedding.shape[0]} dimensions but the active embedder produces "
                f"{qvec.shape[0]}. Use the same embedder (CCE_EMBEDDER) that created these sessions."
            )
        raw_results = [(n, *node_similarity(n, qvec)) for n in nodes]

        if not raw_results:
            return []

        max_turn_end = max(n.turn_end for n, _, _ in raw_results)

        results: list[RetrievalResult] = []
        for node, sem_score, best_text in raw_results:
            recency = (node.turn_end / max(max_turn_end, 1)) * boost
            keyword = self._keyword_score(node, query_keywords) * keyword_bonus_weight
            composite = sem_score + recency + keyword

            results.append(RetrievalResult(
                node=node,
                semantic_score=float(sem_score),
                recency_score=float(recency),
                keyword_score=float(keyword),
                composite_score=float(composite),
                best_turn=best_text,
            ))

        results.sort(key=lambda r: -r.composite_score)
        return results[:top_k]

    def retrieve_by_turn(
        self,
        store: "MemoryStore",
        turn_index: int,
    ) -> list[RetrievalResult]:
        """
        Retrieve all nodes that cover a specific turn index.
        Useful for 'what happened at turn N?' queries.
        """
        nodes = store.warm.get_by_turn_range(turn_index, turn_index)
        return [
            RetrievalResult(
                node=n,
                semantic_score=1.0,
                recency_score=0.0,
                keyword_score=0.0,
                composite_score=1.0,
            )
            for n in nodes
        ]

    def retrieve_all(self, store: "MemoryStore") -> list[RetrievalResult]:
        """
        Return all warm tier nodes for this session, ordered by turn position.
        Used for full context reconstruction (e.g. macro summary generation).
        """
        nodes = store.warm.get_all(session_id=store.session_id)
        return [
            RetrievalResult(
                node=n,
                semantic_score=1.0,
                recency_score=float(i / max(len(nodes) - 1, 1)),
                keyword_score=0.0,
                composite_score=1.0,
            )
            for i, n in enumerate(nodes)
        ]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _extract_keywords(self, text: str) -> set[str]:
        words = _WORD_RE.findall(text.lower())
        return {w for w in words if w not in _STOPWORDS}

    def _keyword_score(self, node: MemoryNode, keywords: set[str]) -> float:
        """
        Compute keyword overlap between the query and node text.
        Checks topic_label + meso_summary.
        Returns a score in [0, 1].
        """
        if not keywords:
            return 0.0

        target_text = f"{node.topic_label} {node.meso_summary}".lower()
        target_words = set(_WORD_RE.findall(target_text)) - _STOPWORDS

        if not target_words:
            return 0.0

        overlap = len(keywords & target_words)
        return overlap / len(keywords)