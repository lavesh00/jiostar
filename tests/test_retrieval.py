from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import contract_intel.retrieval as R
from contract_intel.data import load_queries
from contract_intel.documents import DocumentStore
from contract_intel.retrieval import ContractSearcher, allowed_contract_paths


@pytest.fixture
def searcher(cfg):
    return ContractSearcher.from_config(cfg.with_changes({"chunking.size": 120, "chunking.overlap": 20}))


def test_only_manifest_contracts_are_indexed(cfg, searcher):
    assert searcher.docs.paths() == sorted(allowed_contract_paths(cfg.paths.manifest_file))
    assert {c.file_path for c in searcher.chunks} <= set(searcher.docs.paths())


def test_result_schema_and_exact_text(searcher):
    d = searcher.search("Which law governs this agreement? Delaware", top_k=3).to_dict()
    assert set(d) == {"query", "abstained", "abstain_reason", "results", "meta"}
    assert d["results"], "expected passages"
    for r in d["results"]:
        assert set(r) == {"rank", "file_path", "start", "end", "score", "confidence", "text"}
        assert searcher.docs.text(r["file_path"])[r["start"] : r["end"]] == r["text"]  # exact substring
        assert r["text"] and r["start"] < r["end"]
    assert [r["rank"] for r in d["results"]] == list(range(1, len(d["results"]) + 1))
    json.dumps(d)  # JSON serialisable


def test_ranking_is_deterministic(searcher):
    a = searcher.search("sublicense the software to a third party").to_dict()
    b = searcher.search("sublicense the software to a third party").to_dict()
    for x in (a, b):
        x["meta"].pop("elapsed_ms")
    assert a == b
    scores = [r["score"] for r in a["results"]]
    assert scores == sorted(scores, reverse=True)


def test_relevant_passage_ranks_first(searcher):
    top = searcher.search("sublicense rent lease the Software third party").results[0]
    assert top.file_path.endswith("beta.txt") and "sublicense" in top.text


def test_results_do_not_overlap_beyond_limit(searcher):
    res = searcher.search("agreement notice party days payment", top_k=5).results
    for i, a in enumerate(res):
        for b in res[i + 1 :]:
            if a.file_path == b.file_path:
                inter = max(0, min(a.end, b.end) - max(a.start, b.start))
                assert inter / (b.end - b.start) <= searcher.cfg.retrieval.max_overlap_frac


def test_abstention_returns_empty_results(cfg):
    s = ContractSearcher.from_config(cfg.with_changes({"abstention.threshold": 0.99}))
    r = s.search("sublicense the software")
    assert r.abstained and r.results == [] and r.abstain_reason
    assert r.to_dict()["results"] == []


def test_no_matching_terms_abstains_even_without_threshold(searcher):
    r = searcher.search("zzzqqq xxyyzz")
    assert r.abstained and r.results == []


def test_search_raw_matches_search_when_not_abstaining(searcher):
    raw = searcher.search_raw("payment invoices thirty days")
    resp = searcher.search("payment invoices thirty days")
    assert [(p.file_path, p.start, p.end) for p in raw.passages] == [(p.file_path, p.start, p.end) for p in resp.results]


def test_retrieval_module_only_reads_contract_list_from_manifest(cfg):
    m = json.loads(Path(cfg.paths.manifest_file).read_text(encoding="utf-8"))
    assert "queries" in m  # the manifest does hold gold offsets; the service must ignore them
    assert R.allowed_contract_paths(cfg.paths.manifest_file) == [c["file_path"] for c in m["contracts"]]
    src = inspect.getsource(R)
    assert all(tok not in src for tok in ("GoldSnippet", "EvalQuery", '"queries"', "snippets", '"span"'))


@pytest.mark.parametrize(
    "flags",
    [
        {"chunking.strategy": "paragraph"},
        {"chunking.strategy": "section"},
        {"retrieval.fusion": "rrf"},
        {"retrieval.doc_boost": 0.5, "retrieval.strip_query_context": True},
        {"retrieval.expand_query": True, "retrieval.stem": True, "retrieval.remove_stopwords": True},
        {"abstention.signal": "doc_peakedness"},
        {"abstention.topical_min_docs": 2},
        {"retrieval.stem": True, "retrieval.stemmer": "extended"},
        {"abstention.signal": "doc_peakedness", "abstention.peak_skip_opening": True},
    ],
)
def test_every_option_preserves_exact_offsets(cfg, flags):
    s = ContractSearcher.from_config(cfg.with_changes({"chunking.size": 150, **flags}))
    qs = load_queries(cfg.paths.benchmark_file, cfg.paths.corpus_root, cfg.paths.unanswerable_file, "train")
    for q in qs:
        for p in s.search_raw(q.text).passages:
            assert s.docs.text(p.file_path)[p.start : p.end] == p.text and p.text.strip() == p.text


def test_document_store_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        DocumentStore.from_files(tmp_path, ["cuad/nope.txt"])
