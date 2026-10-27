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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from cce_core.config import CCEConfig, DEFAULT_CONFIG
from cce_core.compression.merger import MemoryNode, decode_turn_vecs
from cce_core.retrieval.keywords import STOPWORDS, WORD_RE, keyword_set

if TYPE_CHECKING:
    from cce_core.memory.store import MemoryStore


_WORD_RE = WORD_RE
_STOPWORDS = STOPWORDS


@dataclass
class RetrievalResult:
    """A single retrieved memory node with its composite score breakdown."""
    node: MemoryNode
    semantic_score: float      # raw cosine similarity
    recency_score: float       # normalized recency contribution
    keyword_score: float       # keyword overlap bonus
    composite_score: float     # final ranking score
    best_turn: str = ""        # text of the best-matching message in the node (for quoting)
    evidence: list[str] = field(default_factory=list)   # all strongly matching messages in the node

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


_OVERVIEW = re.compile(
    r"(overview|recap|summar(?:y|ise|ize)|main (?:topics|points)|what (?:did|have) we (?:talk|discuss|cover|go)\w*|"
    r"topics? (?:we|did)|everything we (?:covered|discussed|talked)|remind me of|so far)",
    re.IGNORECASE,
)


def is_overview_query(query: str) -> bool:
    """True for "what did we talk about?"-style questions: nothing in them matches one message,
    so the right context is a compact view of every topic, not the few best matches."""
    return bool(_OVERVIEW.search(query or ""))


def node_evidence(node: MemoryNode, qvec: np.ndarray, max_items: int = 4, min_ratio: float = 0.65,
                  min_sim: float = 0.2) -> list[str]:
    """
    The messages in a node that match the query almost as well as its best one (in conversation order).

    Needed for answers spread over several messages (a list built up turn by turn): a summary tends to
    drop some of the items, and one quote per topic only shows one of them.
    """
    vecs = decode_turn_vecs(node.metadata)
    if vecs is None or vecs.shape[1] != qvec.shape[0] or len(vecs) == 0:
        return []
    sims = vecs @ qvec
    best = float(sims.max())
    if best < min_sim:
        return []
    picked = [i for i in np.argsort(-sims)[:max_items] if sims[i] >= max(min_sim, min_ratio * best)]
    micros = node.micro_summaries
    return [micros[i] for i in sorted(picked) if i < len(micros)]


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

        qvec = query_emb.astype(np.float32)
        scores = store.warm.index.session_scores(store.session_id, qvec)
        if scores is None:
            return []

        # Vectorised composite score = semantic + recency + keyword, no SQL per note.
        max_turn_end = max(int(scores.turn_end.max()), 1)
        recency = scores.turn_end / max_turn_end * boost
        if query_keywords:
            overlap = np.array([len(query_keywords & kw) for kw in scores.keywords], dtype=np.float32)
            keyword = overlap / len(query_keywords) * keyword_bonus_weight
        else:
            keyword = np.zeros(len(recency), dtype=np.float32)
        composite = scores.semantic + recency + keyword

        if is_overview_query(query):
            # Every topic, in the order they happened (capped), instead of the few closest matches.
            order = np.argsort(scores.turn_start, kind="stable")[: self.config.overview_max_topics]
        else:
            # best first; equal scores keep conversation order
            order = np.lexsort((scores.turn_start, -composite))[:top_k]

        nodes = {n.node_id: n for n in store.warm.get_many([scores.node_ids[i] for i in order])}
        results: list[RetrievalResult] = []
        for i in order:
            node = nodes.get(scores.node_ids[i])
            if node is None:
                continue   # deleted since the index was read
            results.append(RetrievalResult(
                node=node,
                semantic_score=float(scores.semantic[i]),
                recency_score=float(recency[i]),
                keyword_score=float(keyword[i]),
                composite_score=float(composite[i]),
                best_turn=node_similarity(node, qvec)[1],
            ))

        if is_overview_query(query):
            return results
        for r in results:
            r.evidence = node_evidence(r.node, qvec)
        store.warm.record_hits([r.node.node_id for r in results])
        return results

    def rank_turns(self, turns: list, query: str) -> list:
        """Order turns by how well they match the query, best first (used for pending turns)."""
        qvec = self.embed_fn(query).astype(np.float32)
        scored = [(float(self.embed_fn(t.content).astype(np.float32) @ qvec), t) for t in turns]
        scored.sort(key=lambda p: -p[0])
        return [t for _, t in scored]

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