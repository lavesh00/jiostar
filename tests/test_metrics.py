from __future__ import annotations

import pytest

from contract_intel.metrics import (
    MetricError,
    char_overlap,
    char_precision_recall,
    doc_hit,
    merge_intervals,
    reciprocal_rank_span,
    span_hit,
)

G = [("f", 100, 200)]


def test_merge_intervals():
    assert merge_intervals([(5, 10), (0, 6), (20, 30), (30, 31)]) == [(0, 10), (20, 31)]
    assert merge_intervals([]) == []


def test_perfect_and_disjoint():
    assert char_precision_recall([("f", 100, 200)], G) == (1.0, 1.0)
    assert char_precision_recall([("f", 300, 400)], G) == (0.0, 0.0)


def test_hand_calculated_partial_overlap():
    # retrieved 150-300 (150 chars) overlaps gold 100-200 in 50 chars -> P=50/150, R=50/100
    p, r = char_precision_recall([("f", 150, 300)], G)
    assert p == pytest.approx(1 / 3) and r == pytest.approx(0.5)


def test_other_file_is_not_relevant():
    assert char_precision_recall([("g", 100, 200)], G) == (0.0, 0.0)


def test_duplicate_and_overlapping_retrieved_not_double_counted():
    # two identical passages must count once; union = 100 chars, all relevant
    assert char_precision_recall([("f", 100, 200), ("f", 100, 200)], G) == (1.0, 1.0)
    # overlapping retrieved 100-150 & 120-200 -> union 100-200
    assert char_overlap([("f", 100, 150), ("f", 120, 200)], G) == (100, 100, 100)


def test_duplicate_overlapping_gold_not_double_counted():
    gold = [("f", 100, 200), ("f", 100, 200), ("f", 150, 250)]  # union 100-250 = 150 chars
    p, r = char_precision_recall([("f", 100, 250)], gold)
    assert (p, r) == (1.0, 1.0)
    assert char_overlap([("f", 100, 175)], gold) == (75, 75, 150)


def test_multi_snippet_gold_across_files():
    gold = [("a", 0, 10), ("b", 0, 30)]  # 40 gold chars
    p, r = char_precision_recall([("a", 0, 10), ("b", 0, 10), ("c", 0, 20)], gold)  # 40 retrieved, 20 relevant
    assert p == pytest.approx(0.5) and r == pytest.approx(0.5)


def test_empty_cases():
    assert char_precision_recall([], G) == (0.0, 0.0)  # nothing retrieved -> 0 (reference convention)
    assert char_precision_recall([("f", 0, 10)], []) == (0.0, 0.0)  # no gold characters
    assert char_precision_recall([("f", 5, 5)], G) == (0.0, 0.0)  # zero-length ignored


def test_malformed_spans_raise():
    for bad in [("f", 10, 5), ("f", -1, 5), ("f", 1.5, 5)]:
        with pytest.raises(MetricError):
            char_precision_recall([bad], G)
    with pytest.raises(MetricError):
        char_precision_recall([("f", 0, 5)], [("f", 9, 3)])


def test_boundaries_are_half_open():
    assert char_overlap([("f", 0, 100)], G)[0] == 0  # touching at 100 is not overlap
    assert char_overlap([("f", 199, 300)], G)[0] == 1


def test_supplementary_metrics():
    ranked = [("g", 0, 5), ("f", 150, 160), ("f", 100, 120)]
    assert doc_hit(ranked[:1], G) is False and doc_hit(ranked, G) is True
    assert span_hit(ranked[:1], G) is False and span_hit(ranked, G) is True
    assert reciprocal_rank_span(ranked, G) == 0.5
    assert reciprocal_rank_span([("g", 0, 5)], G) == 0.0
