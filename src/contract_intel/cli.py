"""Command-line interface: ``python -m contract_intel.cli <command>`` (or ``contract-intel``)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import ConfigError, load_config
from .data import (
    DataValidationError,
    build_manifest,
    build_out_of_corpus_probe,
    load_out_of_corpus_probe,
    load_queries,
    manifest_hash,
)
from .diagnostics import diagnose
from .evaluation import (
    append_jsonl,
    evaluate_runs,
    run_metadata,
    run_queries,
    write_json,
    write_jsonl,
)
from .service import SetupError, load_searcher

DISCLAIMER = "Retrieved passages are evidence for a human reviewer, not legal advice or a definitive answer."
DATASET_REPO = "awinml/legalbench-rag"
FINAL_RUNS_LOG = "final_test_runs.jsonl"


def _cmd_setup(a: argparse.Namespace) -> int:
    cfg = load_config(a.config)
    raw_root = Path(cfg.paths.benchmark_file).parent.parent
    if not a.skip_download:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("error: huggingface_hub missing; `pip install -e .[data]` or use --skip-download", file=sys.stderr)
            return 2
        print(f"downloading CUAD files from {DATASET_REPO} into {raw_root} ...")
        snapshot_download(DATASET_REPO, repo_type="dataset", local_dir=str(raw_root),
                          allow_patterns=["benchmarks/cuad.json", "corpus/cuad/*", "LICENSE", "README.md", "SHA256SUMS"])
    return _write_manifest(cfg)


def _write_manifest(cfg: Any) -> int:
    manifest = build_manifest(cfg.paths.benchmark_file, cfg.paths.corpus_root)  # raises on any bad span
    Path(cfg.paths.manifest_file).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.manifest_file).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    s, c = manifest["stats"], manifest["split_counts"]
    print(f"validated {s['n_gold_spans']} gold spans over {s['n_files']} contracts: 0 invalid")
    print(f"selected {s['n_selected']} answerable queries (train {c['train']} / test {c['test']}); "
          f"manifest {cfg.paths.manifest_file} hash {manifest['manifest_hash']}")
    return 0


def _cmd_validate(a: argparse.Namespace) -> int:
    return _write_manifest(load_config(a.config))


def _cmd_build_index(a: argparse.Namespace) -> int:
    s = load_searcher(a.config)
    print(json.dumps({"config": s.cfg.name, "config_hash": s.cfg.hash(), "contracts": len(s.docs),
                      "chunks": len(s.chunks), "terms": len(s.index.postings), "build_seconds": round(s.build_seconds, 3)}, indent=2))
    return 0


def _cmd_search(a: argparse.Namespace) -> int:
    s = load_searcher(a.config)
    if not a.query.strip():
        print("error: empty query", file=sys.stderr)
        return 2
    resp = s.search(a.query.strip(), a.top_k, a.contract)
    if a.json:
        print(json.dumps(resp.to_dict(), indent=2, ensure_ascii=False))
        return 0
    if resp.abstained:
        print(f"ABSTAINED: no passage is sufficiently relevant ({resp.abstain_reason}). No passages returned.")
    for p in resp.results:
        print(f"\n#{p.rank}  {p.file_path}  [{p.start}:{p.end}]  score={p.score:.3f} confidence={p.confidence:.3f}\n{p.text}")
    print(f"\n{DISCLAIMER}")
    return 0


def _final_runs_path(cfg: Any) -> Path:
    return Path(cfg.paths.experiments_dir) / FINAL_RUNS_LOG


def count_final_runs(cfg: Any) -> int:
    p = _final_runs_path(cfg)
    return sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()) if p.exists() else 0


def _cmd_evaluate(a: argparse.Namespace) -> int:
    cfg = load_config(a.config)
    if a.split == "test" and not a.final:
        print("error: the held-out test split requires --final (evaluate it only after the config is frozen)", file=sys.stderr)
        return 2
    if a.final and a.split != "test":
        print("error: --final only applies to --split test", file=sys.stderr)
        return 2
    p = cfg.paths
    queries = load_queries(p.benchmark_file, p.corpus_root, p.unanswerable_file, a.split)
    searcher = load_searcher(a.config, cfg)
    runs = run_queries(searcher, queries)
    records, agg = evaluate_runs(runs, cfg.abstention, cfg.eval.k_values)
    manifest = json.loads(Path(p.manifest_file).read_text(encoding="utf-8"))
    meta = run_metadata(cfg.hash(), manifest.get("manifest_hash") or manifest_hash(manifest), a.split,
                        {"config": cfg.name, "config_file": a.config, "n_answerable": agg["n_answerable"],
                         "n_unanswerable": agg["n_unanswerable"], "kind": "final_test" if a.final else "train",
                         "index_build_s": round(searcher.build_seconds, 3), "n_chunks": len(searcher.chunks)})
    out = Path(a.out or Path(p.experiments_dir) / "results")
    stem = f"{cfg.name}_{a.split}"
    if a.final:
        meta["test_run_number"] = count_final_runs(cfg) + 1
        stem += f"_run{meta['test_run_number']}"
    write_json(out / f"{stem}_metrics.json", {"meta": meta, "metrics": agg, "diagnostics": diagnose(records, 5, searcher.docs)})
    write_jsonl(out / f"{stem}_per_query.jsonl", records)
    if a.final:
        append_jsonl(_final_runs_path(cfg), {**meta, "metrics": agg})
    print(json.dumps({"split": a.split, "config": cfg.name, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in agg.items()}}, indent=2))
    print(f"wrote {out / (stem + '_metrics.json')}")
    return 0


PROBE_FILE = "data/out_of_corpus_probe.jsonl"


def _cmd_build_probe(a: argparse.Namespace) -> int:
    cfg = load_config(a.config)
    rows = build_out_of_corpus_probe(cfg.paths.benchmark_file, a.n)
    write_jsonl(a.out, rows)
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "test")}
    print(f"wrote {len(rows)} out-of-corpus questions to {a.out} (train {counts['train']} / test {counts['test']})")
    return 0


def _cmd_probe(a: argparse.Namespace) -> int:
    """Abstention on questions about contracts that are not indexed (evaluation only)."""
    cfg = load_config(a.config)
    queries = load_out_of_corpus_probe(a.probe)
    searcher = load_searcher(a.config, cfg)
    records, agg = evaluate_runs(run_queries(searcher, queries), cfg.abstention, cfg.eval.k_values)
    by_split = {s: [r for r in records if r["split"] == s] for s in ("train", "test")}
    summary = {"config": cfg.name, "n": len(records), "correct_abstentions": agg["correct_abstentions"],
               "correct_abstention_rate": agg["correct_abstention_rate"],
               "by_split": {s: {"n": len(v), "correct_abstentions": sum(r["abstained"] for r in v)} for s, v in by_split.items()},
               "abstain_reasons": dict(sorted({k: sum(1 for r in records if r["abstain_reason"].startswith(k))
                                               for k in ("the question does not match", "top confidence", "no passage")}.items()))}
    out = Path(a.out or Path(cfg.paths.experiments_dir) / "results")
    write_json(out / f"{cfg.name}_out_of_corpus_metrics.json",
               {"meta": run_metadata(cfg.hash(), None, "out_of_corpus_probe", {"config": cfg.name}), "summary": summary})
    write_jsonl(out / f"{cfg.name}_out_of_corpus_per_query.jsonl", records)
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_diagnose(a: argparse.Namespace) -> int:
    cfg = load_config(a.config)
    if a.split == "test" and not a.final:
        print("error: diagnosing the test split requires --final", file=sys.stderr)
        return 2
    p = cfg.paths
    searcher = load_searcher(a.config, cfg)
    runs = run_queries(searcher, load_queries(p.benchmark_file, p.corpus_root, p.unanswerable_file, a.split))
    records, _ = evaluate_runs(runs, cfg.abstention, cfg.eval.k_values)
    print(json.dumps(diagnose(records, 5, searcher.docs, a.examples), indent=2, ensure_ascii=False))
    return 0


def _cmd_optimize(a: argparse.Namespace) -> int:
    from .optimize import main as opt_main

    argv = ["--config", a.config, "--iterations", str(a.iterations), "--out-config", a.out_config]
    if a.note:
        argv += ["--note", a.note]
    return opt_main(argv)


def _cmd_test(a: argparse.Namespace) -> int:
    return subprocess.call([sys.executable, "-m", "pytest", *a.pytest_args])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="contract-intel", description="Contract Intelligence Assistant: exact-span passage retrieval over CUAD contracts. " + DISCLAIMER)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name: str, fn: Any, help_: str, config_default: str = "configs/final.yaml") -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_, description=help_)
        sp.add_argument("--config", default=config_default, help=f"YAML config (default {config_default})")
        sp.set_defaults(fn=fn)
        return sp

    sp = add("setup", _cmd_setup, "Download the CUAD subset of LegalBench-RAG, validate every gold offset, write the manifest")
    sp.add_argument("--skip-download", action="store_true", help="only validate/rebuild the manifest from files already in data/raw")
    add("validate", _cmd_validate, "Validate gold offsets against the source files and rebuild the manifest")
    add("build-index", _cmd_build_index, "Chunk and index the manifest contracts and print index statistics")
    sp = add("search", _cmd_search, "Search the contracts with a natural-language question")
    sp.add_argument("query")
    sp.add_argument("--top-k", type=int, default=None, help="number of passages (default: eval.top_k from config)")
    sp.add_argument("--json", action="store_true", help="print the stable JSON result schema")
    sp.add_argument("--contract", default=None, help="restrict to one indexed contract (path or case-insensitive substring)")
    sp = add("evaluate", _cmd_evaluate, "Evaluate a split (train, or test with --final) and save metrics + per-query results", "configs/baseline.yaml")
    sp.add_argument("--split", choices=["train", "test"], default="train")
    sp.add_argument("--final", action="store_true", help="declare this the final held-out test run (logged; required for --split test)")
    sp.add_argument("--out", default=None, help="output directory (default experiments/results)")
    sp = add("build-probe", _cmd_build_probe, "Derive the out-of-corpus probe (questions about non-indexed CUAD contracts)")
    sp.add_argument("--n", type=int, default=60)
    sp.add_argument("--out", default=PROBE_FILE)
    sp = add("probe", _cmd_probe, "Measure abstention on the out-of-corpus probe (evaluation only, never used for tuning)")
    sp.add_argument("--probe", default=PROBE_FILE)
    sp.add_argument("--out", default=None)
    sp = add("diagnose", _cmd_diagnose, "Categorise failures on a split with concrete examples")
    sp.add_argument("--split", choices=["train", "test"], default="train")
    sp.add_argument("--final", action="store_true", help="required for --split test")
    sp.add_argument("--examples", type=int, default=3)
    sp = add("optimize", _cmd_optimize, "Run the train-only improvement loop (never touches the test split)", "configs/baseline.yaml")
    sp.add_argument("--iterations", type=int, default=10)
    sp.add_argument("--out-config", default="configs/final.yaml")
    sp.add_argument("--note", default="")
    sp = sub.add_parser("test", help="Run the pytest suite")
    sp.add_argument("pytest_args", nargs="*", help="extra pytest arguments")
    sp.set_defaults(fn=_cmd_test)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except (ConfigError, SetupError, DataValidationError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 - last-resort CLI guard
        print(f"unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
