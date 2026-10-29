"""Abstention policy based on retrieval signals only (no LLM).

Signals (all computed from the lexical retriever, never from gold labels):
  top1  normalised BM25 confidence of the best passage, in [0, 1): raw BM25 / sum_t idf_t*(k1+1)
        over the query terms present in the index (the saturation upper bound), so the value does
        not scale with query length.
  gap   (top1 - top2) / top1, the relative margin to the second passage.

Policies
  top_score      abstain iff top1 < threshold
  top_score_gap  abstain iff top1 < threshold or gap < min_gap
Abstain always when nothing matched (no candidate passage). ``threshold == 0`` disables the score test.
"""

from __future__ import annotations

from .config import AbstentionConfig


def decide(cfg: AbstentionConfig, top1: float, top2: float, n_results: int) -> tuple[bool, str]:
    """Return ``(abstain, reason)``; ``reason`` is '' when not abstaining."""
    if n_results == 0:
        return True, "no passage matched any query term"
    if top1 < cfg.threshold:
        return True, f"top confidence {top1:.3f} below threshold {cfg.threshold:.3f}"
    if cfg.policy == "top_score_gap":
        gap = max(0.0, (top1 - top2) / top1)  # top-2 confidence can exceed top-1 (ranking uses another score)
        if gap < cfg.min_gap:
            return True, f"relative gap {gap:.3f} below min_gap {cfg.min_gap:.3f}"
    return False, ""
