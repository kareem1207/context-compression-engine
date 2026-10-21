"""Text helpers shared by compression, the search index and retrieval."""

from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# A term is a run of letters/digits, optionally joined by - _ . / : so that ids such as
# "ERR-4021-X" or "3f2a9c1e-8b7d-4e1a" stay whole; the pieces are indexed as well.
_TERM = re.compile(r"[A-Za-z0-9À-ɏ]+(?:[-_.:/][A-Za-z0-9À-ɏ]+)*")
_PIECE = re.compile(r"[-_.:/]")

STOPWORDS = frozenset(
    "the a an is it to do of and or in on at for with that this was are be have has had you your we can will "
    "would could should what how why when where about just so but if not no i me my tell explain describe "
    "give show help from by as its were been which who whom there their them then than too very also into "
    "some any all more most other such only own same our out up down over under again further once".split()
)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def terms(text: str) -> list[str]:
    """Lower-cased search terms: whole ids plus their pieces, without stopwords or single characters."""
    out: list[str] = []
    for m in _TERM.finditer(text.lower()):
        tok = m.group(0)
        pieces = _PIECE.split(tok)
        if len(pieces) > 1:
            out.append(tok)
        out.extend(pieces)
    return [t for t in out if len(t) >= 2 and t not in STOPWORDS]


def term_counts(text: str, max_terms: int = 600) -> dict[str, int]:
    """Term frequencies of a text, capped at its `max_terms` most frequent terms."""
    counts: dict[str, int] = {}
    for t in terms(text):
        counts[t] = counts.get(t, 0) + 1
    if len(counts) > max_terms:
        counts = dict(sorted(counts.items(), key=lambda kv: -kv[1])[:max_terms])
    return counts


_CAMEL_OR_CAPS = re.compile(r"^[A-Z][A-Z0-9]+$")


def distinctive_terms(text: str) -> list[str]:
    """
    The search terms of a question that are worth an exact-match search: identifiers and codes (a digit, or
    parts joined by - _ . : /, as in "ERR-4021-X" or a UUID), acronyms, and capitalised words that are not
    the start of the question (names). Ordinary lower-case words are left to the dense search: matching
    them exactly adds noise to paraphrased questions ("sibling" vs "sister").
    """
    out: list[str] = []
    for n, m in enumerate(_TERM.finditer(text)):
        tok = m.group(0)
        pieces = _PIECE.split(tok)
        lowered = tok.lower()
        if len(pieces) > 1:
            if lowered not in STOPWORDS:
                out.append(lowered)
            out += [p.lower() for p in pieces if any(c.isdigit() for c in p) and len(p) >= 2]
        elif any(c.isdigit() for c in tok) and len(tok) >= 2:
            out.append(lowered)
        elif _CAMEL_OR_CAPS.match(tok) and len(tok) >= 2:
            out.append(lowered)
        elif tok[0].isupper() and n > 0 and lowered not in STOPWORDS and len(tok) >= 3:
            out.append(lowered)
    return out
