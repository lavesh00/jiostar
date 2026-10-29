"""Official-style character-level precision / recall, plus supplementary document-level metrics.

Convention (follows LegalBench-RAG ``run_benchmark.py``, MIT, zeroentropy-ai/legalbenchrag):
  * a span is ``(file_path, start, end)`` with half-open character offsets;
  * only characters in the *same file* that lie inside a gold span are relevant;
  * precision = relevant retrieved characters / retrieved characters   (0 if nothing retrieved)
  * recall    = relevant retrieved characters / gold characters          (0 if no gold characters)
  * per-query values are averaged with a macro mean over queries.

Differences from the reference implementation (documented, not hidden):
  * retrieved and gold intervals are first merged into disjoint unions per file, so overlapping or
    duplicate retrieved passages (and duplicate gold spans) are never double counted. The reference sums
    per-snippet lengths and relies on the retriever not returning overlaps; the two agree whenever
    returned passages do not overlap, which our overlap suppression guarantees by default.
  * zero-length spans are ignored; inverted/negative/non-integer spans raise ``MetricError``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

Span = tuple[str, int, int]


class MetricError(ValueError):
    """Raised for malformed spans."""


def merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union of half-open intervals as a sorted list of disjoint intervals (touching ones merge)."""
    out: list[list[int]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _by_file(spans: Sequence[Span]) -> dict[str, list[tuple[int, int]]]:
    d: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for fp, s, e in spans:
        if not (isinstance(s, int) and isinstance(e, int)) or isinstance(s, bool) or isinstance(e, bool):
            raise MetricError(f"non-integer span offsets: {(fp, s, e)!r}")
        if s < 0 or e < s:
            raise MetricError(f"malformed span: {(fp, s, e)!r}")
        if e > s:
            d[fp].append((s, e))
    return {fp: merge_intervals(iv) for fp, iv in d.items()}


def _length(iv: Sequence[tuple[int, int]]) -> int:
    return sum(e - s for s, e in iv)


def _intersection(a: Sequence[tuple[int, int]], b: Sequence[tuple[int, int]]) -> int:
    """Total length of the intersection of two sorted disjoint interval lists."""
    i = j = total = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def char_overlap(retrieved: Sequence[Span], gold: Sequence[Span]) -> tuple[int, int, int]:
    """Return ``(relevant_retrieved_chars, retrieved_chars, gold_chars)`` over merged unions."""
    ret, gld = _by_file(retrieved), _by_file(gold)
    inter = sum(_intersection(iv, gld.get(fp, [])) for fp, iv in ret.items())
    return inter, sum(_length(iv) for iv in ret.values()), sum(_length(iv) for iv in gld.values())


def char_precision_recall(retrieved: Sequence[Span], gold: Sequence[Span]) -> tuple[float, float]:
    inter, n_ret, n_gold = char_overlap(retrieved, gold)
    return (inter / n_ret if n_ret else 0.0, inter / n_gold if n_gold else 0.0)


def doc_hit(retrieved: Sequence[Span], gold: Sequence[Span]) -> bool:
    """True if any retrieved passage comes from a gold file (supplementary)."""
    return bool({fp for fp, _, _ in retrieved} & {fp for fp, _, _ in gold})


def span_hit(retrieved: Sequence[Span], gold: Sequence[Span]) -> bool:
    """True if any retrieved character lies in a gold span (supplementary)."""
    return char_overlap(retrieved, gold)[0] > 0


def reciprocal_rank_span(ranked: Sequence[Span], gold: Sequence[Span]) -> float:
    """1/rank of the first ranked passage overlapping a gold span; 0 if none (supplementary)."""
    for r, sp in enumerate(ranked, 1):
        if span_hit([sp], gold):
            return 1.0 / r
    return 0.0
