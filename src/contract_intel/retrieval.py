"""Retrieval service shared by the CLI, the Streamlit app, evaluation and the optimizer.

Result schema (``SearchResponse.to_dict``)::

    {"query": str, "abstained": bool, "abstain_reason": str,
     "results": [{"rank": int, "file_path": "cuad/<name>.txt", "start": int, "end": int,
                  "score": float, "confidence": float, "text": str}],   # text == document[start:end]
     "meta": {"config": str, "config_hash": str, "top1_confidence": float, "top2_confidence": float,
              "elapsed_ms": float}}

Retrieved passages are evidence for a human to read, not legal conclusions.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .abstention import decide
from .chunking import Chunk, chunk_document
from .config import Config
from .documents import DocumentStore
from .indexing import LexicalIndex, Tokenizer, strip_query_context
from .lexicon import expand
from .ranking import Candidate, rank_candidates, refine_span, suppress_overlaps


@dataclass(frozen=True)
class Passage:
    rank: int
    file_path: str
    start: int
    end: int
    score: float
    confidence: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "file_path": self.file_path,
            "start": self.start,
            "end": self.end,
            "score": round(self.score, 6),
            "confidence": round(self.confidence, 6),
            "text": self.text,
        }


@dataclass
class RawSearch:
    """Ranked passages *before* the abstention decision, plus the signals abstention uses."""

    query: str
    passages: list[Passage]
    top1: float
    top2: float
    elapsed_ms: float
    doc_match: float = 1.0  # best document prior (1.0 without a doc index or with an explicit contract)


@dataclass
class SearchResponse:
    query: str
    abstained: bool
    abstain_reason: str
    results: list[Passage]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "results": [p.to_dict() for p in self.results],
            "meta": self.meta,
        }


def allowed_contract_paths(manifest_file: str | Path) -> list[str]:
    """Contract list from the manifest. Deliberately reads *only* ``contracts`` (no queries/gold)."""
    manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
    return [c["file_path"] for c in manifest["contracts"]]


class ContractSearcher:
    """Chunk + index the allowed contracts once, then answer queries."""

    def __init__(self, cfg: Config, docs: DocumentStore) -> None:
        self.cfg = cfg
        self.docs = docs
        t0 = time.perf_counter()
        self.tokenize = Tokenizer(cfg.retrieval)
        self.chunks: list[Chunk] = []
        for fp in docs.paths():
            self.chunks.extend(chunk_document(fp, docs.text(fp), cfg.chunking))
        r = cfg.retrieval
        self.index = LexicalIndex(
            [self.tokenize(docs.slice(c.file_path, c.start, c.end)) for c in self.chunks], r.k1, r.b
        )
        self.doc_chunks: dict[str, list[int]] = {}
        for i, ch in enumerate(self.chunks):
            self.doc_chunks.setdefault(ch.file_path, []).append(i)
        self.doc_index: LexicalIndex | None = None
        if (r.doc_boost > 0 or r.doc_filter_ratio > 0 or cfg.abstention.topical_min_docs > 0
                or cfg.abstention.min_doc_prior > 0):
            self.doc_index = LexicalIndex([self.tokenize(docs.text(fp)) for fp in docs.paths()], r.k1, r.b)
        self.build_seconds = time.perf_counter() - t0

    @classmethod
    def from_config(cls, cfg: Config) -> ContractSearcher:
        paths = allowed_contract_paths(cfg.paths.manifest_file)
        return cls(cfg, DocumentStore.from_files(cfg.paths.corpus_root, paths))

    def _doc_prior(self, query: str) -> dict[str, float]:
        assert self.doc_index is not None
        raw, bound = self.doc_index.bm25(self.tokenize(query))
        paths = self.docs.paths()
        return {paths[i]: s / bound for i, s in raw.items()} if bound > 0 else {}

    def _topical_confidence(self, q_tokens: list[str], idxs: list[int]) -> dict[int, float]:
        """Normalised BM25 using only query terms that occur in >= N contracts (drops party/contract names; unseen terms are kept)."""
        assert self.doc_index is not None
        n = self.cfg.abstention.topical_min_docs
        topical = [t for t in q_tokens if (df := len(self.doc_index.postings.get(t, ()))) == 0 or df >= n]
        raw, bound = self.index.bm25(topical)
        return {i: (raw.get(i, 0.0) / bound if bound > 0 else 0.0) for i in idxs}

    def _peakedness_confidence(self, question: str, kept: list[Candidate]) -> dict[int, float]:
        """Confidence = how far the best literal-question match in the passage's contract stands above that
        contract's median chunk: (peak - median) / peak in [0, 1). Flat score profiles (nothing specific to the
        question) give ~0; a clause that literally answers it gives a clear peak. Uses unexpanded question terms
        (optionally only terms found in >= topical_min_docs contracts, so contract names do not count)."""
        tokens = self.tokenize(question)
        n = self.cfg.abstention.topical_min_docs
        if n > 0:
            assert self.doc_index is not None
            tokens = [t for t in tokens if (df := len(self.doc_index.postings.get(t, ()))) == 0 or df >= n]
        raw, _ = self.index.bm25(tokens)
        by_doc: dict[str, float] = {}
        out = {}
        for c in kept:
            fp = c.chunk.file_path
            if fp not in by_doc:
                vals = sorted(raw.get(i, 0.0) for i in self.doc_chunks[fp])
                peak, med = vals[-1], statistics.median(vals)
                by_doc[fp] = (peak - med) / peak if peak > 0 else 0.0
            out[c.idx] = by_doc[fp]
        return out

    def contracts_matching(self, contract: str) -> list[str]:
        """Indexed contract paths equal to, or containing (case-insensitive), ``contract``."""
        if contract in self.docs:
            return [contract]
        return [p for p in self.docs.paths() if contract.lower() in p.lower()]

    def search_raw(self, query: str, top_k: int | None = None, contract: str | None = None) -> RawSearch:
        """Ranked passages without applying abstention (used for fast threshold sweeps).

        ``contract`` optionally restricts the search to indexed contracts matching it (path or substring).
        """
        t0 = time.perf_counter()
        r = self.cfg.retrieval
        top_k = top_k or self.cfg.eval.top_k
        base = strip_query_context(query) if r.strip_query_context else query
        q_text = expand(base) if r.expand_query else base
        conf_text = strip_query_context(query) if (r.strip_query_context or self.cfg.abstention.strip_context) else query
        prior = self._doc_prior(query) if self.doc_index is not None else None
        allowed: set[str] | None = None
        doc_match = 1.0
        if contract:
            allowed = set(self.contracts_matching(contract))
            if not allowed:
                raise ValueError(f"no indexed contract matches {contract!r}")
            prior = None  # the user chose the contract: no routing needed
        elif prior is not None:
            doc_match = max(prior.values(), default=0.0)
            if r.doc_filter_ratio > 0 and prior:
                allowed = {fp for fp, v in prior.items() if v >= r.doc_filter_ratio * doc_match}
        q_tokens = self.tokenize(q_text)
        cands = rank_candidates(self.index, self.chunks, q_tokens, r, prior, allowed)
        kept = suppress_overlaps(cands, r.max_overlap_frac, top_k)
        if r.min_rel_score > 0 and kept:
            kept = [c for c in kept if c.score >= r.min_rel_score * kept[0].score]
        conf = {c.idx: c.confidence for c in kept}
        if self.cfg.abstention.signal == "doc_peakedness" and kept:
            conf = self._peakedness_confidence(strip_query_context(query), kept)
        elif self.cfg.abstention.topical_min_docs > 0 and kept:
            conf = self._topical_confidence(self.tokenize(conf_text), [c.idx for c in kept])
        weights = {t: self.index.idf[t] for t in set(q_tokens) if t in self.index.idf}
        passages = []
        for i, c in enumerate(kept, 1):
            fp, s, e = c.chunk.file_path, c.chunk.start, c.chunk.end
            if r.refine_max_chars > 0:
                s, e = refine_span(self.docs.text(fp), s, e, weights, self.tokenize, r.refine_max_chars)
            passages.append(Passage(i, fp, s, e, c.score, conf[c.idx], self.docs.slice(fp, s, e)))
        top1 = passages[0].confidence if passages else 0.0
        top2 = passages[1].confidence if len(passages) > 1 else 0.0
        return RawSearch(query, passages, top1, top2, (time.perf_counter() - t0) * 1000, doc_match)

    def search(self, query: str, top_k: int | None = None, contract: str | None = None) -> SearchResponse:
        raw = self.search_raw(query, top_k, contract)
        abstain, reason = decide(self.cfg.abstention, raw.top1, raw.top2, len(raw.passages), raw.doc_match)
        return SearchResponse(
            query=query,
            abstained=abstain,
            abstain_reason=reason,
            results=[] if abstain else raw.passages,
            meta={
                "config": self.cfg.name,
                "config_hash": self.cfg.hash(),
                "top1_confidence": round(raw.top1, 6),
                "top2_confidence": round(raw.top2, 6),
                "contract_match": round(raw.doc_match, 6),
                "contract_filter": contract,
                "elapsed_ms": round(raw.elapsed_ms, 3),
            },
        )
