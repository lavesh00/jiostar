from __future__ import annotations

import json

import pytest
from conftest import fixture_paths

import contract_intel.optimize as opt
from contract_intel import cli
from contract_intel.abstention import decide
from contract_intel.config import (
    AbstentionConfig,
    Config,
    ConfigError,
    RetrievalConfig,
    dump_config,
    from_dict,
)
from contract_intel.data import EvalQuery, load_queries
from contract_intel.evaluation import QueryRun
from contract_intel.indexing import Tokenizer
from contract_intel.ranking import refine_span
from contract_intel.retrieval import ContractSearcher, Passage, RawSearch

TOK = Tokenizer(RetrievalConfig())
TEXT = "Intro words here. The licensee may not sublicense the software. Fees are due monthly; late fees apply. End."


def test_refine_span_picks_best_sentence_and_keeps_offsets_exact():
    w = {"sublicense": 3.0, "software": 1.0}
    s, e = refine_span(TEXT, 0, len(TEXT), w, TOK, 60)
    assert TEXT[s:e] == "The licensee may not sublicense the software."
    assert TEXT[s:e].strip() == TEXT[s:e]


def test_refine_span_no_match_or_single_unit_returns_original():
    assert refine_span(TEXT, 0, len(TEXT), {"zzz": 1.0}, TOK, 60) == (0, len(TEXT))
    assert refine_span("one unit only", 0, 13, {"unit": 1.0}, TOK, 5) == (0, 13)


def test_refine_span_respects_max_chars_and_offsets_within_bounds():
    w = {"fees": 2.0, "monthly": 1.0, "late": 1.0, "sublicense": 1.0}
    for n in (20, 50, 80, 500):
        s, e = refine_span(TEXT, 5, len(TEXT) - 2, w, TOK, n)
        assert 5 <= s < e <= len(TEXT) - 2


def test_decide_contract_gate():
    ab = AbstentionConfig(min_doc_prior=0.5)
    abstain, reason = decide(ab, 0.9, 0.1, 3, doc_match=0.3)
    assert abstain and "contract" in reason
    assert not decide(ab, 0.9, 0.1, 3, doc_match=0.6)[0]
    assert not decide(ab, 0.9, 0.1, 3)[0]  # default doc_match=1.0 (explicit contract / no doc index)


def test_fit_doc_gate_uses_only_answerable_and_budget():
    def run(answerable, dm):
        q = EvalQuery("i", "q", "train", answerable)
        return QueryRun(q, RawSearch("q", [Passage(1, "f", 0, 1, 1.0, 0.5, "x")], 0.5, 0.0, 1.0, dm))

    runs = [run(True, 0.6 + i * 0.002) for i in range(200)] + [run(False, 0.0) for _ in range(5)]
    g = opt.fit_doc_gate(runs)
    below = sum(r.raw.doc_match < g for r in runs if r.query.answerable)
    assert below <= 0.01 * 200 and g == pytest.approx(0.6)  # 0.61 would reject 5 answerable (> 1%)
    # unanswerable values do not move the gate
    assert opt.fit_doc_gate(runs[:200]) == g


@pytest.fixture
def s(cfg):
    return ContractSearcher.from_config(cfg.with_changes({"chunking.size": 150, "retrieval.doc_boost": 1.0,
                                                          "retrieval.strip_query_context": True}))


def test_contract_filter_restricts_results(s):
    r = s.search("payment within thirty days", contract="gamma")
    assert r.results and all(p.file_path.endswith("gamma.txt") for p in r.results)
    assert r.meta["contract_filter"] == "gamma" and r.meta["contract_match"] == 1.0
    with pytest.raises(ValueError):
        s.search("x", contract="no-such-contract")


def test_doc_filter_and_cutoff_and_refinement_keep_exact_text(cfg):
    c = cfg.with_changes({"chunking.size": 300, "retrieval.doc_boost": 1.0, "retrieval.strip_query_context": True,
                          "retrieval.doc_filter_ratio": 1.0, "retrieval.min_rel_score": 0.5,
                          "retrieval.refine_max_chars": 120})
    s = ContractSearcher.from_config(c)
    for q in load_queries(c.paths.benchmark_file, c.paths.corpus_root, c.paths.unanswerable_file, "train"):
        raw = s.search_raw(q.text)
        assert len({p.file_path for p in raw.passages}) <= 1  # ratio 1.0 keeps only the best-matching contract
        for p in raw.passages:
            assert s.docs.text(p.file_path)[p.start:p.end] == p.text and p.text.strip() == p.text
        if raw.passages:
            assert all(p.score >= 0.5 * raw.passages[0].score for p in raw.passages)


def test_new_config_fields_validated():
    for bad in ({"retrieval": {"doc_filter_ratio": 1.5}}, {"retrieval": {"refine_max_chars": -1}},
                {"retrieval": {"min_rel_score": 2}}, {"abstention": {"min_doc_prior": -0.1}}):
        with pytest.raises(ConfigError):
            from_dict(bad)


def test_cli_contract_option_and_probe_commands(tmp_path, capsys):
    cfg = Config(name="fx", paths=fixture_paths(tmp_path)).with_changes({"chunking.size": 150})
    p = tmp_path / "c.yaml"
    dump_config(cfg, p)
    assert cli.main(["search", "payment days", "--config", str(p), "--contract", "beta", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert all(r["file_path"].endswith("beta.txt") for r in d["results"])
    probe = tmp_path / "probe.jsonl"
    probe.write_text(json.dumps({"id": "ooc-1", "query": "Consider the Lease between X and Y; What is the rent?",
                                 "label": "out_of_corpus", "contract_not_indexed": "cuad/z.txt", "split": "train"}) + "\n",
                     encoding="utf-8")
    assert cli.main(["probe", "--config", str(p), "--probe", str(probe), "--out", str(tmp_path / "o")]) == 0
    assert json.loads(capsys.readouterr().out)["n"] == 1
