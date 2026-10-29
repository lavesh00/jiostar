"""Candidate scoring, fusion, overlap suppression and span refinement."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from .chunking import Chunk
from .config import RetrievalConfig
from .indexing import LexicalIndex


@dataclass(frozen=True)
class Candidate:
    chunk: Chunk
    score: float  # ranking score (BM25-normalised [+ RRF] [+ document prior])
    confidence: float  # normalised BM25 of this chunk in [0, 1): the abstention signal
    idx: int = -1  # position in the chunk list


def rank_candidates(
    index: LexicalIndex,
    chunks: list[Chunk],
    q_tokens: list[str],
    cfg: RetrievalConfig,
    doc_prior: dict[str, float] | None = None,
    allowed: set[str] | None = None,
) -> list[Candidate]:
    """Score chunks for a query and return candidates sorted by (-score, file_path, start).

    ``allowed`` restricts candidates to those contracts (explicit contract choice or document routing).
    """
    raw, bound = index.bm25(q_tokens)
    if allowed is not None:
        raw = {i: s for i, s in raw.items() if chunks[i].file_path in allowed}
    if not raw or bound <= 0:
        return []  # no candidate shares any term with the query
    conf = {i: s / bound for i, s in raw.items()}
    prior = doc_prior or {}

    if cfg.fusion == "rrf":
        by_bm25 = sorted(raw, key=lambda i: (-raw[i], i))[: cfg.candidates]
        tf = index.tfidf(q_tokens)
        if allowed is not None:
            tf = {i: s for i, s in tf.items() if chunks[i].file_path in allowed}
        by_tfidf = sorted(tf, key=lambda i: (-tf[i], i))[: cfg.candidates]
        rrf: dict[int, float] = {}
        for ranking in (by_bm25, by_tfidf):
            for r, i in enumerate(ranking, 1):
                rrf[i] = rrf.get(i, 0.0) + 1.0 / (cfg.rrf_k + r)
        top = 2.0 / (cfg.rrf_k + 1)
        base = {i: s / top for i, s in rrf.items()}
    else:
        pool = sorted(raw, key=lambda i: (-raw[i], i))[: cfg.candidates]
        base = {i: conf[i] for i in pool}

    cands = []
    for i, s in base.items():
        ch = chunks[i]
        score = s + cfg.doc_boost * prior.get(ch.file_path, 0.0)
        cands.append(Candidate(ch, score, conf.get(i, 0.0), i))
    cands.sort(key=lambda c: (-c.score, c.chunk.file_path, c.chunk.start))
    return cands


def suppress_overlaps(cands: list[Candidate], max_overlap_frac: float, top_k: int) -> list[Candidate]:
    """Greedy: drop a candidate if more than ``max_overlap_frac`` of it is already covered by kept ones."""
    kept: list[Candidate] = []
    for c in cands:
        length = c.chunk.end - c.chunk.start
        covered = 0
        for k in kept:
            if k.chunk.file_path == c.chunk.file_path:
                covered += max(0, min(k.chunk.end, c.chunk.end) - max(k.chunk.start, c.chunk.start))
        if covered / length <= max_overlap_frac:
            kept.append(c)
            if len(kept) == top_k:
                break
    return kept


_UNIT_BREAK = re.compile(r"(?<=[.;:])\s+|\n[ \t]*\n\s*")


def refine_span(
    text: str, start: int, end: int, weights: dict[str, float], tokenize: Callable[[str], list[str]], max_chars: int
) -> tuple[int, int]:
    """Shrink ``[start, end)`` to the contiguous run of sentence-like units (split after '.', ';', ':' or at
    blank lines) of at most ``max_chars`` characters whose distinct query terms carry the most idf weight.
    Ties prefer the shorter window, then the earlier one. A single unit longer than ``max_chars`` is kept whole.
    Returns the original span if no unit matches. The result is a whitespace-trimmed sub-interval of the
    original text, so offsets stay exact."""
    units, pos = [], start
    for m in _UNIT_BREAK.finditer(text, start, end):
        if m.start() > pos:
            units.append((pos, m.start()))
        pos = m.end()
    if pos < end:
        units.append((pos, end))
    if len(units) <= 1:
        return start, end
    terms = [set(tokenize(text[a:b])) & weights.keys() for a, b in units]
    best: tuple[float, int, int, int] | None = None  # (-score, length, i, j)
    for i in range(len(units)):
        seen: set[str] = set()
        for j in range(i, len(units)):
            length = units[j][1] - units[i][0]
            if length > max_chars and j > i:
                break
            seen |= terms[j]
            key = (-sum(weights[t] for t in seen), length, i, j)
            if best is None or key < best:
                best = key
    if best is None or best[0] == 0:
        return start, end
    s, e = units[best[2]][0], units[best[3]][1]
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return (s, e) if e > s else (start, end)
