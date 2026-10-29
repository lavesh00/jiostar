"""Per-query scoring and aggregation.

Retrieval is invoked with the query text only. Gold spans are used here (and in diagnostics) purely
for scoring; they never reach the searcher.

Denominator conventions
  answerable metrics   : macro mean over all answerable queries in the split (abstained queries score 0)
  coverage             : answerable queries NOT abstained / answerable queries
  false_abstention_rate: answerable queries abstained / answerable queries
  correct_abstention   : unanswerable queries abstained / unanswerable queries
  false_acceptance     : unanswerable queries that returned >= 1 passage / unanswerable queries
Rates are reported with their numerators and denominators; a rate over zero queries is ``None``.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .abstention import decide
from .config import AbstentionConfig
from .data import EvalQuery
from .metrics import (
    Span,
    char_overlap,
    doc_hit,
    reciprocal_rank_span,
    span_hit,
)
from .retrieval import ContractSearcher, RawSearch


@dataclass
class QueryRun:
    query: EvalQuery
    raw: RawSearch


def run_queries(searcher: ContractSearcher, queries: list[EvalQuery], top_k: int | None = None) -> list[QueryRun]:
    """Retrieve for each query using ONLY its text (no gold, no labels)."""
    return [QueryRun(q, searcher.search_raw(q.text, top_k)) for q in queries]


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def score_run(run: QueryRun, abstention: AbstentionConfig, k_values: tuple[int, ...]) -> dict[str, Any]:
    q, raw = run.query, run.raw
    abstained, reason = decide(abstention, raw.top1, raw.top2, len(raw.passages))
    shown = [] if abstained else raw.passages
    rec: dict[str, Any] = {
        "id": q.id,
        "query": q.text,
        "split": q.split,
        "answerable": q.answerable,
        "abstained": abstained,
        "abstain_reason": reason,
        "top1_confidence": raw.top1,
        "top2_confidence": raw.top2,
        "latency_ms": raw.elapsed_ms,
        "results": [[p.file_path, p.start, p.end, round(p.score, 6)] for p in shown],
    }
    if q.answerable:
        gold: list[Span] = [(g.file_path, g.start, g.end) for g in q.gold]
        ranked: list[Span] = [(p.file_path, p.start, p.end) for p in shown]
        rec["gold"] = [list(g) for g in gold]
        for k in k_values:
            inter, n_ret, n_gold = char_overlap(ranked[:k], gold)
            rec[f"precision@{k}"] = inter / n_ret if n_ret else 0.0
            rec[f"recall@{k}"] = inter / n_gold if n_gold else 0.0
            rec[f"inter@{k}"], rec[f"retrieved_chars@{k}"], rec["gold_chars"] = inter, n_ret, n_gold
            rec[f"doc_hit@{k}"] = doc_hit(ranked[:k], gold)
            rec[f"span_hit@{k}"] = span_hit(ranked[:k], gold)
        rec["mrr_span"] = reciprocal_rank_span(ranked, gold)
    return rec


def aggregate(records: list[dict[str, Any]], k_values: tuple[int, ...]) -> dict[str, Any]:
    ans = [r for r in records if r["answerable"]]
    un = [r for r in records if not r["answerable"]]
    n_a, n_u = len(ans), len(un)

    def mean(key: str) -> float | None:
        return sum(r[key] for r in ans) / n_a if n_a else None

    out: dict[str, Any] = {"n_answerable": n_a, "n_unanswerable": n_u}
    for k in k_values:
        out[f"precision@{k}"] = mean(f"precision@{k}")
        out[f"recall@{k}"] = mean(f"recall@{k}")
        out[f"doc_hit@{k}"] = mean(f"doc_hit@{k}")
        out[f"span_hit@{k}"] = mean(f"span_hit@{k}")
        ti = sum(r[f"inter@{k}"] for r in ans)
        tr = sum(r[f"retrieved_chars@{k}"] for r in ans)
        tg = sum(r["gold_chars"] for r in ans)
        out[f"micro_precision@{k}"] = ti / tr if tr else None
        out[f"micro_recall@{k}"] = ti / tg if tg else None
    out["mrr_span"] = mean("mrr_span")
    fa = sum(r["abstained"] for r in ans)
    out["false_abstentions"] = fa
    out["false_abstention_rate"] = _rate(fa, n_a)
    out["coverage"] = _rate(n_a - fa, n_a)
    ca = sum(r["abstained"] for r in un)
    out["correct_abstentions"] = ca
    out["correct_abstention_rate"] = _rate(ca, n_u)
    out["false_acceptances"] = n_u - ca
    out["false_acceptance_rate"] = _rate(n_u - ca, n_u)
    lat = [r["latency_ms"] for r in records]
    out["mean_latency_ms"] = sum(lat) / len(lat) if lat else None
    return out


def evaluate_runs(
    runs: list[QueryRun], abstention: AbstentionConfig, k_values: tuple[int, ...]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = [score_run(r, abstention, k_values) for r in runs]
    return records, aggregate(records, k_values)


# --- run metadata / persistence ----------------------------------------------------------------


def git_commit(cwd: str | Path = ".") -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10, check=False
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def run_metadata(cfg_hash: str, manifest_hash: str | None, split: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = {
        "timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config_hash": cfg_hash,
        "manifest_hash": manifest_hash,
        "split": split,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    meta.update(extra or {})
    return meta


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
