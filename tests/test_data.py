from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest
from conftest import FIX, REAL_BENCH, REAL_CORPUS, ROOT, needs_data

from contract_intel.data import (
    DataValidationError,
    EvalQuery,
    GoldSnippet,
    assign_split,
    build_manifest,
    count_overlaps,
    load_answerable,
    load_queries,
    load_unanswerable,
    manifest_hash,
    mini,
    parse_gold,
    validate_span,
)


def _fake_tests(n: int) -> list[dict]:
    return [{"query": f"q{i}", "snippets": [{"file_path": f"cuad/f{i % 37}.txt", "span": [0, 1], "answer": "x"}]} for i in range(n)]


def test_mini_matches_reference_and_is_deterministic():
    tests = _fake_tests(500)

    def ref(ts, n=194):  # verbatim copy of the prescribed function
        if len(ts) <= n:
            return ts

        def key(t):
            random.seed(t["snippets"][0]["file_path"])
            return random.random()

        return sorted(ts, key=key)[:n]

    a, b = mini(tests), mini(list(reversed(tests)))
    assert len(a) == 194 and a == ref(tests) and mini(tests) == a
    assert [t["query"] for t in a] != [t["query"] for t in tests[:194]]  # really a keyed selection
    assert b == ref(list(reversed(tests)))


def test_mini_returns_all_when_small():
    tests = _fake_tests(10)
    assert mini(tests) == tests


def test_assign_split_exact_hash_rule():
    for q in ["a", "What is the term?", "  padded  ", "ünïcode — query"]:
        h = int(hashlib.sha256(q.strip().encode("utf-8")).hexdigest(), 16)
        assert assign_split(q) == ("test" if h % 10 < 3 else "train")
    assert assign_split("  x ") == assign_split("x")  # strip() applied


def test_validate_span_cases():
    doc = "Héllo wörld — end"
    validate_span(doc, [0, 5], "Héllo")  # unicode, start boundary
    validate_span(doc, [len(doc) - 3, len(doc)], "end")  # end boundary
    validate_span(doc, [0, len(doc)], doc)
    for span, ans in [([0, 5], "Hello"), ([5, 5], ""), ([3, 2], "x"), ([-1, 2], "H"), ([0, len(doc) + 1], doc),
                      ([0], "H"), ("0:5", "H"), ([0.0, 5], "Héllo"), ([0, 5], None), ([True, 5], "Héllo")]:
        with pytest.raises(DataValidationError):
            validate_span(doc, span, ans)


def test_crlf_files_use_universal_newlines(cfg):
    # gamma.txt is stored with CRLF; offsets in the benchmark refer to the newline-translated text
    raw = (FIX / "corpus/cuad/gamma.txt").read_bytes()
    assert b"\r\n" in raw
    queries, _ = load_answerable(cfg.paths.benchmark_file, cfg.paths.corpus_root)
    gamma = [q for q in queries if "Epsilon" in q.text]
    assert gamma and all("\r" not in g.answer for q in gamma for g in q.gold)


def test_fixture_load_counts_and_multi_snippet(cfg):
    queries, stats = load_answerable(cfg.paths.benchmark_file, cfg.paths.corpus_root)
    assert stats["n_selected"] == 9 and stats["n_gold_spans"] == 10 and stats["n_files"] == 3
    assert max(len(q.gold) for q in queries) == 2
    assert {q.split for q in queries} == {"train", "test"}


def test_source_path_mismatch_and_bad_offsets_fail_loudly(tmp_path, cfg):
    root = tmp_path / "corpus/cuad"
    root.mkdir(parents=True)
    (root / "a.txt").write_text("hello world", encoding="utf-8")
    ok = {"query": "q", "snippets": [{"file_path": "cuad/a.txt", "span": [0, 5], "answer": "hello"}]}
    assert parse_gold(ok, tmp_path / "corpus", {})[0].end == 5
    missing = {"query": "q", "snippets": [{"file_path": "cuad/missing.txt", "span": [0, 1], "answer": "h"}]}
    with pytest.raises(DataValidationError, match="not found"):
        parse_gold(missing, tmp_path / "corpus", {})
    bad = {"query": "q", "snippets": [{"file_path": "cuad/a.txt", "span": [1, 6], "answer": "hello"}]}
    with pytest.raises(DataValidationError, match="!="):
        parse_gold(bad, tmp_path / "corpus", {})
    with pytest.raises(DataValidationError):
        parse_gold({"query": "q", "snippets": []}, tmp_path / "corpus", {})


def test_duplicate_and_overlapping_gold_counted():
    g = (GoldSnippet("f", 0, 10, "x" * 10), GoldSnippet("f", 5, 15, "x" * 10), GoldSnippet("g", 0, 10, "x" * 10),
         GoldSnippet("f", 0, 10, "x" * 10))
    assert count_overlaps(g) == 3  # f-snippets 0,1,3 overlap pairwise; g is disjoint


def test_unanswerable_schema_and_split_consistency(cfg, tmp_path):
    qs = load_unanswerable(cfg.paths.unanswerable_file)
    assert len(qs) == 5 and all(not q.answerable and not q.gold for q in qs)
    bad = tmp_path / "bad.jsonl"
    rec = json.loads((FIX / "unanswerable.jsonl").read_text(encoding="utf-8").splitlines()[0])
    rec["split"] = "train" if rec["split"] == "test" else "test"
    bad.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="split"):
        load_unanswerable(bad)
    rec2 = {k: v for k, v in rec.items() if k != "rationale"}
    bad.write_text(json.dumps(rec2) + "\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="missing field"):
        load_unanswerable(bad)


def test_load_queries_split_isolation(cfg):
    tr = load_queries(cfg.paths.benchmark_file, cfg.paths.corpus_root, cfg.paths.unanswerable_file, "train")
    te = load_queries(cfg.paths.benchmark_file, cfg.paths.corpus_root, cfg.paths.unanswerable_file, "test")
    assert tr and te and not ({q.id for q in tr} & {q.id for q in te})
    assert all(q.split == "train" for q in tr) and all(q.split == "test" for q in te)
    with pytest.raises(ValueError):
        load_queries(cfg.paths.benchmark_file, cfg.paths.corpus_root, cfg.paths.unanswerable_file, "val")


def test_manifest_contains_only_referenced_contracts(cfg):
    m = build_manifest(cfg.paths.benchmark_file, cfg.paths.corpus_root)
    referenced = {s["file_path"] for t in json.loads(Path(cfg.paths.benchmark_file).read_text(encoding="utf-8"))["tests"] for s in t["snippets"]}
    assert {c["file_path"] for c in m["contracts"]} == referenced
    assert m["manifest_hash"] == manifest_hash(m)


# --- frozen unanswerable set (real file, no download needed) ------------------------------------

UNANSWERABLE = ROOT / "data/unanswerable_queries.jsonl"
UNANSWERABLE_SHA256 = "5b156c306254e0bbc4bc98c4e33aba33d3457a1125386eef119f492a95070e89"


def test_frozen_unanswerable_set_unchanged_and_valid():
    raw = UNANSWERABLE.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == UNANSWERABLE_SHA256, "unanswerable set was modified after freezing"
    qs = load_unanswerable(UNANSWERABLE)
    assert len(qs) >= 30
    assert all(q.files_checked for q in qs) and len({q.text for q in qs}) == len(qs)


# --- real-data checks ---------------------------------------------------------------------------


@needs_data
@pytest.mark.data
def test_real_mini_selection_and_offsets():
    queries, stats = load_answerable(REAL_BENCH, REAL_CORPUS)
    assert stats["n_selected"] == 194 and stats["n_gold_spans"] == 272 and stats["n_files"] == 29
    assert sum(q.split == "train" for q in queries) == 140 and sum(q.split == "test" for q in queries) == 54
    assert isinstance(queries[0], EvalQuery)


@needs_data
@pytest.mark.data
def test_real_manifest_is_current():
    committed = json.loads((ROOT / "data/manifests/cuad_mini.json").read_text(encoding="utf-8"))
    rebuilt = build_manifest(REAL_BENCH, REAL_CORPUS)
    assert committed["manifest_hash"] == rebuilt["manifest_hash"]
    assert len(committed["contracts"]) == 29


@needs_data
@pytest.mark.data
def test_unanswerable_files_exist_in_manifest():
    manifest = json.loads((ROOT / "data/manifests/cuad_mini.json").read_text(encoding="utf-8"))
    files = {c["file_path"] for c in manifest["contracts"]}
    for q in load_unanswerable(UNANSWERABLE):
        assert set(q.files_checked) <= files
