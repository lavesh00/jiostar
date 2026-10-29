"""Candidate scoring, fusion and overlap suppression."""

from __future__ import annotations

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
) -> list[Candidate]:
    """Score chunks for a query and return candidates sorted by (-score, file_path, start)."""
    raw, bound = index.bm25(q_tokens)
    if not raw or bound <= 0:
        return []  # no candidate shares any term with the query
    conf = {i: s / bound for i, s in raw.items()}
    prior = doc_prior or {}

    if cfg.fusion == "rrf":
        by_bm25 = sorted(raw, key=lambda i: (-raw[i], i))[: cfg.candidates]
        tf = index.tfidf(q_tokens)
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
