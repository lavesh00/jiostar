from __future__ import annotations

import json
import subprocess
import sys

import pytest
import yaml
from conftest import FIX, ROOT, fixture_paths

import contract_intel.data as data_mod
import contract_intel.optimize as opt
from contract_intel import app as app_mod
from contract_intel import cli
from contract_intel.config import Config, ConfigError, dump_config, from_dict, load_config

# ---- config ----------------------------------------------------------------------------------


def test_shipped_configs_load_and_validate():
    for name in ("baseline", "final"):
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        assert cfg.eval.top_k >= max(cfg.eval.k_values)
    assert load_config(ROOT / "configs/baseline.yaml").abstention.threshold == 0.0


def test_config_roundtrip_and_hash_ignores_name_and_paths(tmp_path):
    cfg = Config(name="a")
    dump_config(cfg, tmp_path / "c.yaml")
    assert load_config(tmp_path / "c.yaml") == cfg
    other = from_dict({**cfg.to_dict(), "name": "b"})
    assert other.hash() == cfg.hash() and cfg.with_changes({"chunking.size": 500}).hash() != cfg.hash()


@pytest.mark.parametrize(
    "raw",
    [
        {"bogus": 1},
        {"chunking": {"strategy": "words"}},
        {"chunking": {"size": 10}},
        {"chunking": {"size": 100, "overlap": 100}},
        {"retrieval": {"k1": 0}},
        {"retrieval": {"b": 1.5}},
        {"retrieval": {"fusion": "x"}},
        {"abstention": {"threshold": 2}},
        {"abstention": {"signal": "magic"}},
        {"eval": {"k_values": [1, 3], "top_k": 2}},
        {"chunking": {"nope": 1}},
        {"chunking": 5},
    ],
)
def test_invalid_config_rejected(raw):
    with pytest.raises(ConfigError):
        from_dict(raw)


def test_config_file_errors(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")
    (tmp_path / "bad.yaml").write_text("a: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="malformed"):
        load_config(tmp_path / "bad.yaml")
    with pytest.raises(ConfigError, match="unknown config key"):
        Config().with_changes({"chunking.nope": 1})


# ---- optimizer: train-only isolation + append-only log ----------------------------------------


@pytest.fixture
def fixture_config_file(tmp_path):
    cfg = Config(name="baseline", paths=fixture_paths(tmp_path)).with_changes({"chunking.size": 200})
    path = tmp_path / "base.yaml"
    dump_config(cfg, path)
    return path


def test_optimizer_cannot_load_test_split(fixture_config_file, tmp_path, monkeypatch):
    seen_splits = []
    real = data_mod.load_queries

    def guarded(bench, corpus, un, split):
        seen_splits.append(split)
        assert split != "test", "optimizer attempted to load the test split"
        return real(bench, corpus, un, split)

    monkeypatch.setattr(data_mod, "load_queries", guarded)
    out_cfg = tmp_path / "final.yaml"
    rc = opt.optimize(str(fixture_config_file), 3, str(out_cfg), str(tmp_path / "exp"), note="test")
    assert rc == 0 and seen_splits == ["train"] and out_cfg.exists()
    assert not hasattr(opt, "load_queries") and not hasattr(opt, "load_test_queries")


def test_optimizer_log_is_append_only_and_complete(fixture_config_file, tmp_path):
    exp = tmp_path / "exp"
    opt.optimize(str(fixture_config_file), 2, str(tmp_path / "f1.yaml"), str(exp))
    first = (exp / "iterations.jsonl").read_text(encoding="utf-8")
    opt.optimize(str(fixture_config_file), 2, str(tmp_path / "f2.yaml"), str(exp))
    both = (exp / "iterations.jsonl").read_text(encoding="utf-8")
    assert both.startswith(first) and len(both) > len(first)  # earlier history preserved
    rows = [json.loads(line) for line in both.splitlines()]
    its = [r for r in rows if r.get("type") == "iteration"]
    assert its and all(r["decision"] in ("accepted", "rejected") for r in its)
    need = {"run_id", "timestamp", "git_commit", "parent_config_hash", "candidate_config_hash", "changed_parameters",
            "hypothesis", "proposer", "metrics_before", "metrics_after", "guardrails", "decision", "decision_rationale",
            "elapsed_s"}
    assert all(need <= set(r) for r in its)
    assert any(r.get("type") == "finalize" for r in rows) and any(r.get("type") == "baseline" for r in rows)


def test_guardrails_reject_precision_collapse():
    class Ev(dict):
        pass

    def ev(p5, r5, chunks=100, lat=1.0):
        return Ev(retrieval={"recall@5": r5, "precision@5": p5, **{f"{m}@{k}": 0.0 for m in ("recall", "precision") for k in (1, 3)}},
                  n_chunks=chunks, latency_ms=lat, build_s=0.1,
                  fit={"threshold": 0, "min_gap": 0, "coverage": 1.0, "correct_abstention_rate": 0.0})

    base = ev(0.10, 0.2)
    ok = opt.guardrails("A", ev(0.095, 0.5), base, base)
    assert all(v["pass"] for v in ok.values())
    bad = opt.guardrails("A", ev(0.05, 0.9), base, base)
    assert not bad["G1:precision@5"]["pass"]
    assert not opt.guardrails("A", ev(0.1, 0.5, chunks=5000), base, base)["G5:n_chunks"]["pass"]


def test_threshold_fit_respects_coverage():
    from contract_intel.config import AbstentionConfig
    from contract_intel.data import EvalQuery
    from contract_intel.evaluation import QueryRun
    from contract_intel.retrieval import Passage, RawSearch

    def run(answerable, conf):
        q = EvalQuery("i", "q", "train", answerable)
        return QueryRun(q, RawSearch("q", [Passage(1, "f", 0, 1, 1.0, conf, "x")], conf, 0.0, 1.0))

    runs = [run(True, 0.1 * i) for i in range(1, 11)] + [run(False, 0.05), run(False, 0.15), run(False, 0.95)]
    ab, fit = opt.fit_abstention(runs, AbstentionConfig())
    assert fit["coverage"] >= opt.MIN_COVERAGE
    assert 0.1 <= ab.threshold <= 0.2 and fit["correct_abstention_rate"] == pytest.approx(2 / 3)


# ---- CLI ---------------------------------------------------------------------------------------


@pytest.fixture
def cli_cfg(tmp_path):
    cfg = Config(name="fixture", paths=fixture_paths(tmp_path)).with_changes({"chunking.size": 150})
    p = tmp_path / "cfg.yaml"
    dump_config(cfg, p)
    return str(p)


def test_cli_search_json_and_text(cli_cfg, capsys):
    assert cli.main(["search", "sublicense the software to third parties", "--config", cli_cfg, "--json", "--top-k", "2"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["results"] and len(d["results"]) <= 2
    assert cli.main(["search", "sublicense the software", "--config", cli_cfg]) == 0
    assert "not legal advice" in capsys.readouterr().out


def test_cli_prints_explicit_abstention(tmp_path, capsys):
    cfg = Config(name="x", paths=fixture_paths(tmp_path)).with_changes({"abstention.threshold": 0.99})
    p = tmp_path / "abst.yaml"
    dump_config(cfg, p)
    assert cli.main(["search", "sublicense", "--config", str(p)]) == 0
    assert "ABSTAINED" in capsys.readouterr().out
    assert cli.main(["search", "sublicense", "--config", str(p), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["results"] == []


def test_cli_exit_codes(tmp_path, cli_cfg, capsys):
    assert cli.main(["search", "x", "--config", str(tmp_path / "nope.yaml")]) == 2
    assert cli.main(["search", "   ", "--config", cli_cfg]) == 2
    assert cli.main(["evaluate", "--split", "test", "--config", cli_cfg]) == 2  # needs --final
    assert "--final" in capsys.readouterr().err
    assert cli.main(["evaluate", "--split", "train", "--final", "--config", cli_cfg]) == 2
    cfg = Config(paths=fixture_paths(tmp_path)).with_changes({})
    broken = tmp_path / "broken.yaml"
    d = cfg.to_dict()
    d["paths"]["manifest_file"] = str(tmp_path / "missing_manifest.json")
    broken.write_text(yaml.safe_dump(d), encoding="utf-8")
    assert cli.main(["search", "x", "--config", str(broken)]) == 2
    assert "setup" in capsys.readouterr().err


def test_cli_evaluate_train_and_final_logging(cli_cfg, tmp_path, capsys):
    out = tmp_path / "res"
    assert cli.main(["evaluate", "--split", "train", "--config", cli_cfg, "--out", str(out)]) == 0
    m = json.loads((out / "fixture_train_metrics.json").read_text(encoding="utf-8"))
    assert m["meta"]["split"] == "train" and m["metrics"]["n_answerable"] == 6
    assert (out / "fixture_train_per_query.jsonl").exists()
    for n in (1, 2):
        assert cli.main(["evaluate", "--split", "test", "--final", "--config", cli_cfg, "--out", str(out)]) == 0
        assert json.loads((out / f"fixture_test_run{n}_metrics.json").read_text(encoding="utf-8"))["meta"]["test_run_number"] == n
    log = (tmp_path / "experiments" / "final_test_runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log) == 2


def test_cli_build_index_and_help(cli_cfg, capsys):
    assert cli.main(["build-index", "--config", cli_cfg]) == 0
    assert json.loads(capsys.readouterr().out)["contracts"] == 3
    r = subprocess.run([sys.executable, "-m", "contract_intel.cli", "--help"], capture_output=True, text=True, check=False)
    assert r.returncode == 0 and "evaluate" in r.stdout and "optimize" in r.stdout


def test_cli_validate_writes_manifest(tmp_path, capsys):
    cfg = Config(paths=fixture_paths(tmp_path))
    d = cfg.to_dict()
    d["paths"]["manifest_file"] = str(tmp_path / "m" / "manifest.json")
    p = tmp_path / "v.yaml"
    p.write_text(yaml.safe_dump(d), encoding="utf-8")
    assert cli.main(["validate", "--config", str(p)]) == 0
    assert json.loads((tmp_path / "m" / "manifest.json").read_text(encoding="utf-8"))["stats"]["n_selected"] == 9


# ---- app service layer -------------------------------------------------------------------------


def test_app_run_query_handles_results_abstention_and_errors(cli_cfg, tmp_path):
    r = app_mod.run_query("sublicense the software", cli_cfg, 3)
    assert not r.get("error") and r["results"] and r["results"][0]["start"] < r["results"][0]["end"]
    assert "error" in app_mod.run_query("   ", cli_cfg, 3)
    assert "error" in app_mod.run_query("x", str(tmp_path / "missing.yaml"), 3)
    cfg = Config(paths=fixture_paths(tmp_path)).with_changes({"abstention.threshold": 0.99})
    p = tmp_path / "a.yaml"
    dump_config(cfg, p)
    a = app_mod.run_query("sublicense the software", str(p), 3)
    assert a["abstained"] and a["results"] == []


def test_app_and_cli_return_identical_results(cli_cfg, capsys):
    cli.main(["search", "payment invoices thirty days", "--config", cli_cfg, "--json"])
    c = json.loads(capsys.readouterr().out)
    a = app_mod.run_query("payment invoices thirty days", cli_cfg, 5)
    for x in (c, a):
        x["meta"].pop("elapsed_ms")
    assert c == a


def test_streamlit_app_boots_when_streamlit_available():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "src/contract_intel/app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    # on a machine without the corpus the app must show a clear error instead of crashing
    assert at.error or at.title
    assert FIX.exists()
