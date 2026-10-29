"""Lexical indexes over chunks: BM25 and TF-IDF cosine (pure Python, deterministic).

Text is tokenised from the original chunk text; the index stores only chunk ids and term statistics,
so every hit maps straight back to an exact ``[start, end)`` span of the original document.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

from .config import RetrievalConfig

_TOKEN = re.compile(r"\w+")
_CONTEXT_PREFIX = re.compile(r"^\s*Consider\s+(?:the\s+)?[^;]{0,500};\s*", re.I)

STOPWORDS = frozenset(
    "a an and are as at be been but by can could did do does for from had has have if in into is it its "
    "may not of on or shall should so such that the their them then there these they this to under upon "
    "was were what when where which who whom whose will with would any all each other".split()
)


def strip_query_context(query: str) -> str:
    """Drop a leading 'Consider the <contract description>;' clause; keep the original if nothing remains."""
    stripped = _CONTEXT_PREFIX.sub("", query, count=1).strip()
    return stripped or query


def _stem(tok: str) -> str:
    for suf in ("ations", "ation", "ings", "ing", "edly", "ed", "ies", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)] + ("y" if suf == "ies" else "")
    return tok


class Tokenizer:
    def __init__(self, cfg: RetrievalConfig) -> None:
        self.lowercase, self.stop, self.stem = cfg.lowercase, cfg.remove_stopwords, cfg.stem

    def __call__(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        toks = [t for t in _TOKEN.findall(text) if len(t) > 1 or t.isdigit()]
        if self.stop:
            toks = [t for t in toks if t not in STOPWORDS]
        if self.stem:
            toks = [_stem(t) for t in toks]
        return toks


class LexicalIndex:
    """Inverted index with BM25 and TF-IDF-cosine scoring over a fixed list of token lists."""

    def __init__(self, docs: Sequence[Iterable[str]], k1: float, b: float) -> None:
        self.k1, self.b = k1, b
        counts = [Counter(d) for d in docs]
        self.n = len(counts)
        self.lengths = [sum(c.values()) for c in counts]
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, c in enumerate(counts):
            for t, tf in c.items():
                self.postings.setdefault(t, []).append((i, tf))
        self.idf = {
            t: math.log(1.0 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()
        }
        # TF-IDF cosine: sublinear tf, smooth idf, L2-normalised document vectors
        self._tfidf_idf = {t: math.log((1 + self.n) / (1 + len(p))) + 1.0 for t, p in self.postings.items()}
        norms = [0.0] * self.n
        for t, p in self.postings.items():
            for i, tf in p:
                norms[i] += ((1 + math.log(tf)) * self._tfidf_idf[t]) ** 2
        self._norms = [math.sqrt(x) or 1.0 for x in norms]

    def bm25(self, q_tokens: Sequence[str]) -> tuple[dict[int, float], float]:
        """Raw BM25 scores per doc id and the saturation upper bound used for normalisation.

        The bound sums ``idf*(k1+1)`` over *all* distinct query terms; terms absent from the index use the
        maximum idf, so a question whose key term occurs nowhere cannot look fully supported.
        """
        scores: dict[int, float] = {}
        bound = 0.0
        oov_idf = math.log(1.0 + (self.n + 0.5) / 0.5)  # a term seen nowhere counts as maximally rare
        for t in sorted(set(q_tokens)):
            post = self.postings.get(t)
            if not post:
                bound += oov_idf * (self.k1 + 1.0)  # unmatched query terms lower confidence
                continue
            idf = self.idf[t]
            bound += idf * (self.k1 + 1.0)
            for i, tf in post:
                norm = self.k1 * (1.0 - self.b + self.b * self.lengths[i] / (self.avgdl or 1.0))
                scores[i] = scores.get(i, 0.0) + idf * tf * (self.k1 + 1.0) / (tf + norm)
        return scores, bound

    def tfidf(self, q_tokens: Sequence[str]) -> dict[int, float]:
        q = Counter(t for t in q_tokens if t in self.postings)
        if not q:
            return {}
        qw = {t: (1 + math.log(c)) * self._tfidf_idf[t] for t, c in q.items()}
        qn = math.sqrt(sum(w * w for w in qw.values())) or 1.0
        scores: dict[int, float] = {}
        for t, w in qw.items():
            for i, tf in self.postings[t]:
                scores[i] = scores.get(i, 0.0) + w * (1 + math.log(tf)) * self._tfidf_idf[t]
        return {i: s / (qn * self._norms[i]) for i, s in scores.items()}
