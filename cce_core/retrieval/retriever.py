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

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

from cce_core.config import CCEConfig, DEFAULT_CONFIG
from cce_core.compression.merger import MemoryNode, decode_sentence_vecs, decode_turn_vecs
from cce_core.retrieval.bm25 import Bm25, fuse
from cce_core.retrieval.keywords import STOPWORDS, WORD_RE, keyword_set
from cce_core.text import distinctive_terms, term_counts, terms


def query_terms(query: str, config) -> list[str]:
    """The terms of a question that go to BM25 (see config.lexical_terms)."""
    if config.lexical_weight <= 0:
        return []
    return terms(query) if config.lexical_terms == "all" else distinctive_terms(query)

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
    keyword_score: float       # BM25 score of the note's text for the query
    composite_score: float     # final ranking score
    best_turn: str = ""        # text of the best-matching message in the node (for quoting)
    evidence: list[str] = field(default_factory=list)   # all strongly matching messages in the node
    overview_lines: list[str] = field(default_factory=list)   # representative sentences (overview questions only)

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


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# raw_fn(node, positions) -> {position: original text of that turn}; positions index node.micro_summaries
RawFn = Callable[["MemoryNode", "list[int]"], "dict[int, str]"]
# embed_many(texts) -> (N, dim) unit vectors
EmbedManyFn = Callable[["list[str]"], np.ndarray]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def sentence_evidence(
    node: MemoryNode,
    qvec: np.ndarray,
    sims: np.ndarray,
    raw_fn: RawFn,
    embed_many: EmbedManyFn,
    stored=None,
    max_turns: int = 12,
    max_candidates: int = 40,
    max_sentences: int = 4,
    min_ratio: float = 0.5,
    short_words: int = 6,
) -> list[str]:
    """
    The sentences of a note's own messages that answer the query best.

    Long messages carry a vector per sentence, stored with the note (`stored`, see
    config.sentence_index_min_words): all of them are scored for free, so a fact inside a long message is
    found even when the message as a whole does not look like the question. Other messages are handled as
    before: the note's per-message vectors rank them (`sims`), and the sentences of the most promising ones
    (at most `max_candidates`) are embedded and scored at query time. The original text of the winning
    sentences comes from `raw_fn`; a very short sentence brings its neighbour along. One quote per
    message, in conversation order; [] when no original text is kept.
    """
    scored: list[tuple[float, int, int]] = []      # (score, position of the message, sentence number)
    covered: set[int] = set()
    if stored is not None and stored[0].shape[1] == qvec.shape[0]:
        for score, (pos, k) in zip(stored[0] @ qvec, stored[1]):
            scored.append((float(score), int(pos), int(k)))
            covered.add(int(pos))

    order = [int(i) for i in np.argsort(-sims)[:max_turns]]
    texts = raw_fn(node, sorted(set(order) | covered))
    per_turn: dict[int, list[str]] = {pos: split_sentences(t) for pos, t in texts.items()}

    cands: list[tuple[int, int, str]] = []
    for pos in order:                                   # most promising message first
        sents = per_turn.get(pos)
        if pos in covered or not sents or (cands and len(cands) + len(sents) > max_candidates):
            continue
        cands += [(pos, k, s) for k, s in enumerate(sents)]
    if cands:
        vecs = embed_many([c[2] for c in cands])
        if vecs.shape[0] == len(cands) and vecs.shape[1] == qvec.shape[0]:
            scored += [(float(v), c[0], c[1]) for v, c in zip(vecs @ qvec, cands)]

    # a sentence can only be quoted if its text is known; a long message without kept text is quoted by its summary
    micros = node.micro_summaries
    scored = [x for x in scored
              if (x[1] in per_turn and x[2] < len(per_turn[x[1]])) or (x[1] in covered and x[1] < len(micros))]
    if not scored:
        return []
    best = max(x[0] for x in scored)
    if best <= 0.0:
        return []
    scored.sort(key=lambda x: -x[0])
    ranked = [x for x in scored if x[0] >= min_ratio * best][: max_sentences * 2]

    chosen: dict[int, set[int]] = {}
    for score, pos, k in ranked:
        if pos not in per_turn:
            chosen.setdefault(pos, set())       # summary fallback below
            continue
        if len(chosen) >= max_sentences and pos not in chosen:
            continue
        picks = chosen.setdefault(pos, set())
        picks.add(k)
        sents = per_turn[pos]
        if len(sents[k].split()) < short_words:      # "Yes, that one." means little alone
            picks.add(k + 1 if k + 1 < len(sents) else max(k - 1, 0))
    quotes = []
    for pos in sorted(chosen)[:max_sentences]:
        if pos in per_turn:
            quotes.append(" ".join(per_turn[pos][k] for k in sorted(chosen[pos])))
        else:
            quotes.append(micros[pos])
    return quotes


def node_evidence(node: MemoryNode, qvec: np.ndarray, max_items: int = 4, min_ratio: float = 0.65,
                  min_sim: float = 0.2, raw_fn: RawFn | None = None,
                  embed_many: EmbedManyFn | None = None) -> list[str]:
    """
    The messages in a node that match the query almost as well as its best one (in conversation order).

    Needed for answers spread over several messages (a list built up turn by turn): a summary tends to
    drop some of the items, and one quote per topic only shows one of them. When the original text of the
    messages is available (`raw_fn`), the quotes are the best-matching sentences from it instead of the
    one-line summaries of the messages.
    """
    vecs = decode_turn_vecs(node.metadata)
    if vecs is None or vecs.shape[1] != qvec.shape[0] or len(vecs) == 0:
        return []
    if raw_fn is not None and embed_many is not None:
        quotes = sentence_evidence(node, qvec, vecs @ qvec, raw_fn, embed_many,
                                   stored=decode_sentence_vecs(node.metadata))
        if quotes:
            return quotes
    sims = vecs @ qvec
    best = float(sims.max())
    if best < min_sim:
        return []
    picked = [i for i in np.argsort(-sims)[:max_items] if sims[i] >= max(min_sim, min_ratio * best)]
    micros = node.micro_summaries
    return [micros[i] for i in sorted(picked) if i < len(micros)]


def node_similarity(node: MemoryNode, qvec: np.ndarray) -> tuple[float, str]:
    """
    Similarity of a query to a node, judged by its best-matching message (or sentence of a long message)
    rather than only the topic centroid. A single specific fact inside a long, mixed topic barely moves the
    centroid, so centroid-only search misses it; the best single message or sentence does not.

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


def overview_lines(
    node: MemoryNode,
    raw_fn: RawFn | None = None,
    budget_tokens: int = 100,
    doc_freq: dict[str, int] | None = None,
    n_notes: int = 1,
    opener_tokens: int = 60,
) -> list[str]:
    """
    Representative lines of a topic for "what did we talk about?" questions, within `budget_tokens`.

    What the user asked names the subject of a topic better than the answers do, so the first user message
    of the note is always the first line. The rest of the budget goes to the sentences that cover the most
    of what is particular to this topic: its terms weighted by how rare they are among the session's notes
    (tf-idf), picked greedily by new weight per token, so each line adds something the earlier ones did not.
    Text comes from the original messages when they are kept, else the summary lines. Lines are returned in
    conversation order.
    """
    from cce_core.ingestion import tokenizer

    micros = node.micro_summaries
    n = len(micros)
    if n == 0 or budget_tokens <= 0:
        return []
    roles = node.metadata.get("roles")
    has_roles = roles is not None and len(roles) == n
    first = next((i for i in range(n) if not has_roles or roles[i] == "u"), 0)
    raw = raw_fn(node, list(range(n))) if raw_fn is not None else {}

    # candidate sentences: (position of the message, sentence)
    cands: list[tuple[int, str]] = []
    for pos in range(n):
        if raw.get(pos):
            cands += [(pos, s) for s in split_sentences(raw[pos]) if len(s.split()) >= 4]
        else:
            cands.append((pos, micros[pos]))

    tf = node.metadata.get("tf") or {}
    df = doc_freq or {}
    weight = {term: (count ** 0.5) * math.log(1.0 + n_notes / max(df.get(term, 1), 1)) for term, count in tf.items()}

    lead = raw.get(first) or micros[first]
    lead = tokenizer.truncate_to_tokens(" ".join(lead.split()), min(opener_tokens, budget_tokens))
    chosen: list[tuple[int, str]] = [(first, lead)]
    covered = set(terms(lead))
    used = tokenizer.count(lead)

    pool = [(pos, " ".join(s.split())) for pos, s in cands]
    pool = [(pos, tokenizer.truncate_to_tokens(s, 36)) for pos, s in pool]
    while pool:
        best_i, best_gain = -1, 0.0
        for i, (pos, sent) in enumerate(pool):
            cost = tokenizer.count(sent)
            if used + cost > budget_tokens:
                continue
            gain = sum(weight.get(term, 0.0) for term in set(terms(sent)) - covered) / max(cost, 8)
            if gain > best_gain:
                best_i, best_gain = i, gain
        if best_i < 0:
            break
        pos, sent = pool.pop(best_i)
        chosen.append((pos, sent))
        used += tokenizer.count(sent)
        covered |= set(terms(sent))
    seen, lines = set(), []
    for pos, sent in sorted(chosen, key=lambda x: x[0]):
        if sent and sent not in seen:
            seen.add(sent)
            lines.append(sent)
    return lines


def fill_overviews(results, raw_fn: RawFn | None, config: CCEConfig) -> None:
    """Give every result of an overview question its representative lines, sharing the budget between topics."""
    if not results:
        return
    doc_freq: dict[str, int] = {}
    for r in results:
        for term in (r.node.metadata.get("tf") or {}):
            doc_freq[term] = doc_freq.get(term, 0) + 1
    per_topic = int(min(max(config.context_max_tokens * config.overview_budget_share / len(results), 30), 160))
    for r in results:
        r.overview_lines = overview_lines(r.node, raw_fn, per_topic, doc_freq, len(results))


def sentence_similarity(node: MemoryNode, qvec: np.ndarray) -> float:
    """Best-matching sentence of the note's long messages, or NaN when it has no sentence vectors."""
    stored = decode_sentence_vecs(node.metadata)
    if stored is None or stored[0].shape[1] != qvec.shape[0]:
        return float("nan")
    return float((stored[0] @ qvec).max())


def rank_in_memory(nodes: list[MemoryNode], qvec: np.ndarray, query: str, config: CCEConfig):
    """
    Score notes that are not in a database (one-shot mode): dense score fused with BM25 over the notes' text.
    Returns (node, dense score, text of the best-matching message, fused score) sorted best first.
    """
    if not nodes:
        return []
    dense_parts = [node_similarity(n, qvec) for n in nodes]
    sentence = np.array([sentence_similarity(n, qvec) for n in nodes], dtype=np.float32)
    dense = np.fmax(np.array([d[0] for d in dense_parts], dtype=np.float32), sentence * config.sentence_discount)
    bm25 = Bm25(config.bm25_k1, config.bm25_b)
    for n in nodes:
        bm25.add(n.metadata.get("tf") or term_counts(f"{n.topic_label} {n.meso_summary}"))
    q_terms = query_terms(query, config)
    extra = []
    if q_terms:
        lexical = bm25.score(q_terms, config.lexical_max_df)
        extra.append((lexical, lexical > 0, config.lexical_weight))
    fused = fuse(dense, *extra, k=config.rrf_k)
    order = np.lexsort((np.arange(len(nodes)), -fused))
    return [(nodes[i], float(dense[i]), dense_parts[i][1], float(fused[i])) for i in order]


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
        embed_many_fn=None,    # Callable[[list[str]], np.ndarray]; batch version, used for sentence evidence
    ):
        self.embed_fn = embed_fn
        self.config = config
        self._embed_many = embed_many_fn
        self._sentence_vecs: dict[str, np.ndarray] = {}

    def embed_sentences(self, texts: list[str]) -> np.ndarray:
        """Embed sentences for evidence, remembering recent ones (the same messages come up query after query)."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._sentence_vecs]
        if missing:
            if len(self._sentence_vecs) > 4096:
                self._sentence_vecs.clear()
            vecs = self._embed_many(missing) if self._embed_many else np.stack([self.embed_fn(t) for t in missing])
            for t, v in zip(missing, vecs):
                self._sentence_vecs[t] = np.asarray(v, dtype=np.float32)
        return np.stack([self._sentence_vecs[t] for t in texts])

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

        qvec = query_emb.astype(np.float32)
        q_terms = query_terms(query, self.config)
        scores = store.warm.index.session_scores(store.session_id, qvec, q_terms, self.config.lexical_max_df)
        if scores is None:
            return []

        # Dense score (best message or sentence + recency) fused with BM25 over the notes' text
        # by reciprocal rank; vectorised, no SQL per note.
        max_turn_end = max(int(scores.turn_end.max()), 1)
        recency = scores.turn_end / max_turn_end * boost
        semantic = np.fmax(scores.semantic, scores.sentence * self.config.sentence_discount)
        dense = semantic + recency
        lexical = scores.lexical if scores.lexical is not None else np.zeros(len(dense), dtype=np.float32)
        extra = []
        if scores.lexical is not None:
            extra.append((lexical, lexical > 0, self.config.lexical_weight))
        composite = fuse(dense, *extra, k=self.config.rrf_k)

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
                semantic_score=float(semantic[i]),
                recency_score=float(recency[i]),
                keyword_score=float(lexical[i]),
                composite_score=float(composite[i]),
                best_turn=node_similarity(node, qvec)[1],
            ))

        raw_fn = getattr(store, "raw_texts", None)
        if is_overview_query(query):
            fill_overviews(results, raw_fn, self.config)
            return results
        for r in results:
            r.evidence = node_evidence(
                r.node, qvec, raw_fn=raw_fn, embed_many=self.embed_sentences if raw_fn else None
            )
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