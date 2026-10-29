from __future__ import annotations

import dataclasses

from contract_intel.abstention import decide
from contract_intel.config import AbstentionConfig
from contract_intel.data import EvalQuery, GoldSnippet, load_queries
from contract_intel.diagnostics import categorize, diagnose
from contract_intel.evaluation import QueryRun, evaluate_runs, run_queries, score_run
from contract_intel.retrieval import ContractSearcher, Passage, RawSearch


def _raw(conf, spans):
    ps = [Passage(i, f, s, e, 1.0, conf, "x") for i, (f, s, e) in enumerate(spans, 1)]
    return RawSearch("q", ps, conf if ps else 0.0, 0.0, 1.0)


def _run(answerable, conf, spans, gold=(("f", 0, 100),)):
    q = EvalQuery("id", "q", "train", answerable, tuple(GoldSnippet(*g, "x") for g in gold) if answerable else ())
    return QueryRun(q, _raw(conf, spans))


def test_decide_threshold_and_gap():
    ab = AbstentionConfig(threshold=0.5)
    assert decide(ab, 0.49, 0.0, 3)[0] and not decide(ab, 0.5, 0.0, 3)[0]
    assert decide(ab, 0.0, 0.0, 0)[0]  # nothing retrieved
    assert not decide(AbstentionConfig(threshold=0.0), 0.0, 0.0, 2)[0]  # threshold 0 disables the score test
    gap = AbstentionConfig(policy="top_score_gap", threshold=0.1, min_gap=0.2)
    assert decide(gap, 0.5, 0.45, 2)[0] and not decide(gap, 0.5, 0.2, 2)[0]
    assert not decide(dataclasses.replace(gap, min_gap=0.0), 0.5, 0.9, 2)[0]  # negative gap clipped


def test_both_abstention_error_directions_counted():
    ab = AbstentionConfig(threshold=0.5)
    runs = [
        _run(True, 0.9, [("f", 0, 100)]),  # answerable, answered
        _run(True, 0.1, [("f", 0, 100)]),  # answerable, falsely abstained
        _run(False, 0.9, [("f", 0, 100)]),  # unanswerable, falsely accepted
        _run(False, 0.1, [("f", 0, 100)]),  # unanswerable, correctly abstained
        _run(False, 0.2, [("f", 0, 100)]),  # unanswerable, correctly abstained
    ]
    _, agg = evaluate_runs(runs, ab, (1, 3, 5))
    assert (agg["n_answerable"], agg["n_unanswerable"]) == (2, 3)
    assert agg["false_abstentions"] == 1 and agg["false_abstention_rate"] == 0.5 and agg["coverage"] == 0.5
    assert agg["correct_abstentions"] == 2 and agg["false_acceptances"] == 1
    assert agg["correct_abstention_rate"] == 2 / 3 and agg["false_acceptance_rate"] == 1 / 3
    assert agg["recall@5"] == 0.5 and agg["precision@5"] == 0.5  # abstained answerable scores 0 (macro over 2)


def test_rates_over_empty_denominators_are_none():
    _, agg = evaluate_runs([_run(True, 0.9, [("f", 0, 100)])], AbstentionConfig(), (1, 3, 5))
    assert agg["correct_abstention_rate"] is None and agg["false_acceptance_rate"] is None
    _, agg = evaluate_runs([_run(False, 0.9, [("f", 0, 5)])], AbstentionConfig(), (1, 3, 5))
    assert agg["coverage"] is None and agg["recall@5"] is None


def test_precision_recall_at_k_uses_first_k_results():
    run = _run(True, 0.9, [("f", 500, 600), ("f", 0, 50), ("f", 50, 100)])
    rec = score_run(run, AbstentionConfig(), (1, 3, 5))
    assert rec["recall@1"] == 0.0 and rec["recall@3"] == 1.0
    assert rec["precision@3"] == 100 / 200 and rec["doc_hit@1"] and not rec["span_hit@1"]


def test_unanswerable_with_abstention_has_no_results():
    rec = score_run(_run(False, 0.1, [("f", 0, 10)]), AbstentionConfig(threshold=0.5), (1, 3, 5))
    assert rec["abstained"] and rec["results"] == []


def test_end_to_end_fixture_workflow_and_reproducibility(cfg):
    p = cfg.paths
    qs = load_queries(p.benchmark_file, p.corpus_root, p.unanswerable_file, "train")
    c = cfg.with_changes({"chunking.size": 150, "retrieval.strip_query_context": True, "retrieval.doc_boost": 1.0})

    def go():
        s = ContractSearcher.from_config(c)
        recs, agg = evaluate_runs(run_queries(s, qs), c.abstention, c.eval.k_values)
        agg.pop("mean_latency_ms")
        return [{k: v for k, v in r.items() if k != "latency_ms"} for r in recs], agg

    (r1, a1), (r2, a2) = go(), go()
    assert r1 == r2 and a1 == a2  # reproducible
    assert a1["n_answerable"] == 6 and a1["n_unanswerable"] == 4
    assert a1["doc_hit@5"] >= 0.8 and a1["recall@5"] > 0.4  # loose smoke bounds on the toy corpus


def test_search_raw_receives_query_text_only(cfg, monkeypatch):
    s = ContractSearcher.from_config(cfg)
    calls = []
    orig = s.search_raw
    monkeypatch.setattr(s, "search_raw", lambda *a, **k: calls.append((a, k)) or orig(*a, **k))
    qs = load_queries(cfg.paths.benchmark_file, cfg.paths.corpus_root, cfg.paths.unanswerable_file, "train")
    run_queries(s, qs)
    assert len(calls) == len(qs) and all(len(a) == 2 and isinstance(a[0], str) and not k for a, k in calls)


def test_diagnostic_categories():
    base = {"answerable": True, "abstained": False, "doc_hit@5": True, "span_hit@5": True, "recall@5": 0.9, "precision@5": 0.5}
    assert categorize(base, 5) == "success"
    assert categorize({**base, "abstained": True}, 5) == "false_abstention"
    assert categorize({**base, "doc_hit@5": False, "span_hit@5": False}, 5) == "wrong_document"
    assert categorize({**base, "span_hit@5": False}, 5) == "right_doc_wrong_section"
    assert categorize({**base, "recall@5": 0.2}, 5) == "incomplete_coverage"
    assert categorize({**base, "precision@5": 0.05}, 5) == "overly_broad"
    assert categorize({"answerable": False, "abstained": False}, 5) == "false_acceptance"
    assert categorize({"answerable": False, "abstained": True}, 5) == "correct_abstention"
    rec = {**base, "id": "a", "query": "q", "results": [], "top1_confidence": 0.1}
    assert diagnose([rec], 5)["categories"] == {"success": 1}
