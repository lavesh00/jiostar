"""Train-only self-improvement loop (deterministic grid proposer, rule-gated by diagnostics).

    python -m contract_intel.optimize --config configs/baseline.yaml --iterations 9

Protocol (declared in docs/optimization_objective.md, revision 1.1):
  * Phase A (structure / lexical): maximise macro char Recall@5 with abstention disabled, subject to
    guardrails on Precision@5, latency and index size.
  * Phase B (abstention): maximise the correct-abstention rate on train unanswerable queries at the
    train-fitted threshold subject to answerable coverage >= 0.90 and no loss of retrieval quality.
  * Each iteration tests one hypothesis (a small grid), takes the best guardrail-passing grid point as the
    candidate and accepts it only if it beats the incumbent by the declared minimum improvement.
  * Only ``load_train_queries`` is imported: the optimizer cannot load the test split.
Every iteration (accepted, rejected or skipped) is appended to ``experiments/iterations.jsonl``.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .abstention import decide
from .config import AbstentionConfig, Config, ConfigError, dump_config, load_config
from .data import EvalQuery, load_train_queries, manifest_hash
from .diagnostics import diagnose
from .documents import DocumentStore
from .evaluation import (
    QueryRun,
    append_jsonl,
    evaluate_runs,
    git_commit,
    run_metadata,
    run_queries,
    write_json,
)
from .retrieval import ContractSearcher, allowed_contract_paths

# --- declared constants (docs/optimization_objective.md) --------------------------------------
MIN_IMPROVEMENT_A = 0.002  # absolute Recall@5
MIN_IMPROVEMENT_B = 0.02  # absolute correct-abstention rate (>= 1 of 33 train unanswerable queries)
PRECISION_FLOOR_FRAC = 0.90  # G1: P@5 >= 0.90 * baseline P@5
MAX_LATENCY_MS = 500.0  # G4
LATENCY_FACTOR = 5.0
CHUNK_FACTOR = 10.0  # G5
MIN_COVERAGE = 0.90  # G2
PHASE_B_RETRIEVAL_TOLERANCE = 0.002
THRESHOLD_GRID = [round(i * 0.005, 3) for i in range(0, 191)]  # 0.000 .. 0.950
GAP_GRID = [round(i * 0.05, 2) for i in range(0, 11)]


@dataclasses.dataclass(frozen=True)
class Hypothesis:
    name: str
    phase: str  # "A" | "B"
    hypothesis: str
    grid: list[dict[str, Any]]
    trigger: str | None = None  # diagnosis-based gate: category name whose share must reach 10%


HYPOTHESES: list[Hypothesis] = [
    Hypothesis("chunk_strategy", "A",
               "Gold clauses are paragraph/section shaped; structure-aware chunks should align better than blind windows.",
               [{"chunking.strategy": "paragraph"}, {"chunking.strategy": "section"}]),
    Hypothesis("chunk_size", "A",
               "Gold spans average ~400 chars; smaller chunks should raise precision and may raise top-5 recall.",
               [{"chunking.size": s} for s in (300, 500, 700, 1500, 2500)]),
    Hypothesis("overlap_boundaries", "A",
               "Fixed windows cut clauses in half; overlap and sentence/whitespace snapping should reduce split clauses.",
               [{"chunking.overlap": o, "chunking.snap_boundaries": s} for o in (0, 100, 200) for s in (False, True)
                if (o, s) != (0, False)]),
    Hypothesis("query_processing", "A",
               "Queries start with 'Consider the <contract>;' whose name tokens dominate BM25; stripping it, dropping stopwords or stemming should focus scoring on the clause topic.",
               [{"retrieval.strip_query_context": True},
                {"retrieval.strip_query_context": True, "retrieval.remove_stopwords": True},
                {"retrieval.strip_query_context": True, "retrieval.stem": True},
                {"retrieval.strip_query_context": True, "retrieval.remove_stopwords": True, "retrieval.stem": True}]),
    Hypothesis("bm25_params", "A",
               "Default k1/b are tuned for web text; long, repetitive legal chunks may prefer different saturation/length normalisation.",
               [{"retrieval.k1": k1, "retrieval.b": b} for k1 in (0.8, 1.2, 1.6, 2.0) for b in (0.5, 0.75, 0.9)
                if (k1, b) != (1.2, 0.75)]),
    Hypothesis("doc_prior", "A",
               "If many failures come from the wrong contract, a document-level BM25 prior over the full query should route to the right contract.",
               [{"retrieval.doc_boost": x} for x in (0.1, 0.25, 0.5, 1.0)], trigger="wrong_document"),
    Hypothesis("rrf_fusion", "A",
               "A second lexical view (TF-IDF cosine) fused by reciprocal rank can correct BM25 ranking errors.",
               [{"retrieval.fusion": "rrf", "retrieval.rrf_k": k} for k in (20, 60)]),
    Hypothesis("topical_confidence", "B",
               "Contract-identifying names inflate confidence for unsupported questions; scoring confidence only on terms shared by >= N contracts should separate answerable from unanswerable.",
               [{"abstention.topical_min_docs": n} for n in (3, 5, 8, 12, 16)]),
    Hypothesis("gap_policy", "B",
               "Unsupported questions produce flat score distributions; requiring a relative gap between top-1 and top-2 may add separation.",
               [{"abstention.policy": "top_score_gap", "abstention.min_gap": g} for g in (0.05, 0.1, 0.2, 0.3)]),
]


# --- threshold fitting (train only) -------------------------------------------------------------


def _coverage_and_correct(runs: list[QueryRun], ab: AbstentionConfig) -> tuple[float, float]:
    ans = [r for r in runs if r.query.answerable]
    un = [r for r in runs if not r.query.answerable]
    cov = sum(not decide(ab, r.raw.top1, r.raw.top2, len(r.raw.passages))[0] for r in ans) / max(len(ans), 1)
    cor = sum(decide(ab, r.raw.top1, r.raw.top2, len(r.raw.passages))[0] for r in un) / max(len(un), 1)
    return cov, cor


def fit_abstention(runs: list[QueryRun], base: AbstentionConfig) -> tuple[AbstentionConfig, dict[str, float]]:
    """Rule R: highest threshold with answerable coverage >= MIN_COVERAGE (maximises correct abstention).

    For the gap policy the same rule is applied for each min_gap in GAP_GRID and the best correct-abstention
    rate wins (ties -> smaller min_gap). Uses train runs only.
    """
    gaps = GAP_GRID if base.policy == "top_score_gap" else [0.0]
    best: tuple[float, float, AbstentionConfig, float] | None = None
    for gap in gaps:
        chosen = None
        for th in THRESHOLD_GRID:
            ab = dataclasses.replace(base, threshold=th, min_gap=gap)
            cov, cor = _coverage_and_correct(runs, ab)
            if cov >= MIN_COVERAGE:
                chosen = (th, ab, cov, cor)
        if chosen is None:  # even threshold 0 (+gap) violates coverage: fall back to disabled abstention
            ab = dataclasses.replace(base, threshold=0.0, min_gap=0.0)
            cov, cor = _coverage_and_correct(runs, ab)
            chosen = (0.0, ab, cov, cor)
        _, ab, cov, cor = chosen
        if best is None or cor > best[0]:
            best = (cor, cov, ab, gap)
    assert best is not None
    cor, cov, ab, _ = best
    return ab, {"threshold": ab.threshold, "min_gap": ab.min_gap, "coverage": cov, "correct_abstention_rate": cor}


# --- evaluation of one configuration ------------------------------------------------------------


def evaluate_config(cfg: Config, queries: list[EvalQuery]) -> dict[str, Any]:
    searcher = ContractSearcher.from_config(cfg)
    runs = run_queries(searcher, queries)
    ks = cfg.eval.k_values
    off = dataclasses.replace(cfg.abstention, threshold=0.0, min_gap=0.0)
    records, agg = evaluate_runs(runs, off, ks)
    fitted, fit = fit_abstention(runs, cfg.abstention)
    return {
        "cfg": cfg,
        "records": records,
        "retrieval": agg,
        "n_chunks": len(searcher.chunks),
        "build_s": searcher.build_seconds,
        "latency_ms": agg["mean_latency_ms"],
        "fitted_abstention": fitted,
        "fit": fit,
        "docs": searcher.docs,
    }


def summary(ev: dict[str, Any]) -> dict[str, Any]:
    a = ev["retrieval"]
    out = {f"{m}@{k}": a[f"{m}@{k}"] for m in ("recall", "precision") for k in (1, 3, 5)}
    out.update(
        n_chunks=ev["n_chunks"],
        mean_latency_ms=round(ev["latency_ms"], 3),
        build_s=round(ev["build_s"], 3),
        fitted_threshold=ev["fit"]["threshold"],
        fitted_min_gap=ev["fit"]["min_gap"],
        fitted_coverage=ev["fit"]["coverage"],
        fitted_correct_abstention=ev["fit"]["correct_abstention_rate"],
    )
    return out


def guardrails(phase: str, cand: dict[str, Any], inc: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    c, i, b = summary(cand), summary(inc), summary(base)
    g: dict[str, Any] = {}

    def add(name: str, value: float, limit: float, ok: bool) -> None:
        g[name] = {"value": value, "limit": limit, "pass": bool(ok), "delta_vs_incumbent": value - i.get(name.split(":")[1], value) if ":" in name else None}

    if phase == "A":
        lim = PRECISION_FLOOR_FRAC * b["precision@5"]
        add("G1:precision@5", c["precision@5"], lim, c["precision@5"] >= lim)
        lim = min(LATENCY_FACTOR * b["mean_latency_ms"], MAX_LATENCY_MS)
        add("G4:mean_latency_ms", c["mean_latency_ms"], lim, c["mean_latency_ms"] <= lim)
        lim = CHUNK_FACTOR * b["n_chunks"]
        add("G5:n_chunks", c["n_chunks"], lim, c["n_chunks"] <= lim)
    else:
        add("G2:fitted_coverage", c["fitted_coverage"], MIN_COVERAGE, c["fitted_coverage"] >= MIN_COVERAGE)
        for m in ("recall@5", "precision@5"):
            lim = i[m] - PHASE_B_RETRIEVAL_TOLERANCE
            add(f"GB:{m}", c[m], lim, c[m] >= lim)
        lim = min(LATENCY_FACTOR * b["mean_latency_ms"], MAX_LATENCY_MS)
        add("G4:mean_latency_ms", c["mean_latency_ms"], lim, c["mean_latency_ms"] <= lim)
    return g


def objective(phase: str, ev: dict[str, Any]) -> float:
    return ev["retrieval"]["recall@5"] if phase == "A" else ev["fit"]["correct_abstention_rate"]


def with_fitted_abstention(ev: dict[str, Any]) -> Config:
    return dataclasses.replace(ev["cfg"], abstention=ev["fitted_abstention"])


# --- the loop -----------------------------------------------------------------------------------


def optimize(config_path: str, iterations: int, out_config: str, experiments_dir: str | None = None) -> int:
    base_cfg = load_config(config_path)
    exp = Path(experiments_dir or base_cfg.paths.experiments_dir)
    log_path = exp / "iterations.jsonl"
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = base_cfg.paths
    queries = load_train_queries(p.benchmark_file, p.corpus_root, p.unanswerable_file)
    assert all(q.split == "train" for q in queries)
    n_ans = sum(q.answerable for q in queries)
    mh = manifest_hash(__import__("json").loads(Path(p.manifest_file).read_text(encoding="utf-8")))
    print(f"[optimize] run {run_id}: {n_ans} answerable + {len(queries) - n_ans} unanswerable TRAIN queries; "
          f"{len(allowed_contract_paths(p.manifest_file))} contracts")

    base_ev = evaluate_config(base_cfg, queries)
    baseline_path = exp / "baseline.json"
    if not baseline_path.exists():
        write_json(baseline_path, {
            "meta": run_metadata(base_cfg.hash(), mh, "train", {"n_answerable": n_ans, "n_unanswerable": len(queries) - n_ans}),
            "config": base_cfg.to_dict(),
            "retrieval_only_metrics": base_ev["retrieval"],
            "summary": summary(base_ev),
            "fitted_abstention_rule_R": dataclasses.asdict(base_ev["fitted_abstention"]),
            "diagnostics": diagnose(base_ev["records"], 5, base_ev["docs"]),
        })
        print(f"[optimize] wrote {baseline_path}")
    append_jsonl(log_path, {"run_id": run_id, "iteration": 0, "type": "baseline", "timestamp": run_metadata("", None, "train")["timestamp"],
                            "git_commit": git_commit(), "config_hash": base_cfg.hash(), "metrics": summary(base_ev)})

    incumbent = base_ev
    done = 0
    for hyp in HYPOTHESES:
        if done >= iterations:
            break
        diag = diagnose(incumbent["records"], 5, None, 0)
        row_common = {
            "run_id": run_id, "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(), "phase": hyp.phase, "hypothesis_name": hyp.name, "hypothesis": hyp.hypothesis,
            "proposer": "grid_search" if hyp.trigger is None else "rule+grid_search",
            "parent_config_hash": incumbent["cfg"].hash(), "diagnosis_before": diag["answerable_failure_share"],
        }
        if hyp.trigger and diag["answerable_failure_share"].get(hyp.trigger, 0.0) < 0.10:
            append_jsonl(log_path, {**row_common, "iteration": None, "type": "skipped", "decision": "skipped",
                                    "decision_rationale": f"trigger '{hyp.trigger}' share {diag['answerable_failure_share'].get(hyp.trigger, 0.0):.3f} < 0.10; not a measured problem"})
            print(f"[optimize] skip {hyp.name}: trigger '{hyp.trigger}' not met")
            continue

        done += 1
        t0 = time.perf_counter()
        grid_rows, evals = [], []
        for changes in hyp.grid:
            try:
                cfg = incumbent["cfg"].with_changes(changes)
            except ConfigError as e:
                grid_rows.append({"changes": changes, "error": str(e)})
                continue
            if cfg.hash() == incumbent["cfg"].hash():
                continue
            ev = evaluate_config(cfg, queries)
            g = guardrails(hyp.phase, ev, incumbent, base_ev)
            obj = objective(hyp.phase, ev)
            grid_rows.append({"changes": changes, "config_hash": cfg.hash(), "objective": obj,
                              "guardrails_pass": all(v["pass"] for v in g.values()), "summary": summary(ev)})
            evals.append((changes, ev, g, obj))
        if not evals:
            append_jsonl(log_path, {**row_common, "iteration": done, "type": "iteration", "decision": "rejected",
                                    "decision_rationale": "no valid candidate in grid", "grid": grid_rows})
            continue

        passing = [e for e in evals if all(v["pass"] for v in e[2].values())]
        pool = passing or evals
        changes, cand, g, obj = max(pool, key=lambda e: e[3])  # first max wins ties (grid order)
        cur = objective(hyp.phase, incumbent)
        need = MIN_IMPROVEMENT_A if hyp.phase == "A" else MIN_IMPROVEMENT_B
        gain = obj - cur
        if not passing:
            decision, why = "rejected", "every grid point violated at least one guardrail; best candidate shown"
        elif gain > need:
            decision, why = "accepted", f"objective {cur:.4f} -> {obj:.4f} (+{gain:.4f} > {need}); all guardrails pass"
        else:
            decision, why = "rejected", f"objective gain {gain:+.4f} does not exceed minimum improvement {need}"
        cand_diag = diagnose(cand["records"], 5, None, 0)
        out = {
            **row_common, "iteration": done, "type": "iteration",
            "objective": "recall@5 (no abstention)" if hyp.phase == "A" else "correct_abstention_rate at rule-R threshold",
            "candidate_config_hash": cand["cfg"].hash(), "changed_parameters": changes,
            "metrics_before": summary(incumbent), "metrics_after": summary(cand),
            "objective_before": cur, "objective_after": obj,
            "guardrails": g, "diagnosis_after": cand_diag["answerable_failure_share"],
            "decision": decision, "decision_rationale": why, "grid": grid_rows,
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
        append_jsonl(log_path, out)
        write_json(exp / "results" / f"{run_id}_iter{done:02d}_candidate.json",
                   {"changes": changes, "decision": decision, "config": cand["cfg"].to_dict(), "summary": summary(cand)})
        print(f"[optimize] it{done:02d} {hyp.name:20s} {decision:8s} {cur:.4f} -> {obj:.4f}  {changes}")
        if decision == "accepted":
            incumbent = cand

    final = with_fitted_abstention(incumbent)
    final = dataclasses.replace(final, name="final")
    ab_changed = final.abstention != incumbent["cfg"].abstention
    append_jsonl(log_path, {"run_id": run_id, "iteration": None, "type": "finalize", "proposer": "threshold_fit(rule R)",
                            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                            "git_commit": git_commit(), "final_config_hash": final.hash(),
                            "fitted_abstention": dataclasses.asdict(final.abstention), "fit": incumbent["fit"],
                            "abstention_changed_by_fit": ab_changed, "metrics": summary(incumbent)})
    dump_config(final, out_config)
    print(f"[optimize] iterations run: {done}; wrote {out_config} (hash {final.hash()}); threshold {final.abstention.threshold}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train-only improvement loop (never touches the test split).")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--iterations", type=int, default=9)
    ap.add_argument("--out-config", default="configs/final.yaml")
    ap.add_argument("--experiments-dir", default=None)
    a = ap.parse_args(argv)
    try:
        return optimize(a.config, a.iterations, a.out_config, a.experiments_dir)
    except (ConfigError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
