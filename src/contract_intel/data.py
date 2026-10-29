"""Dataset access: exact LegalBench-RAG-mini selection, split, offset validation, manifests.

Dataset schema (verified against awinml/legalbench-rag, benchmarks/cuad.json):
    {"tests": [{"query": str,
                "snippets": [{"file_path": "cuad/<name>.txt", "span": [start, end], "answer": str}]}]}
Documents live at ``<corpus_root>/<file_path>`` and are read with Python's default text mode
(``encoding="utf-8"``, universal newlines). Offsets are Python str offsets into that text.

Gold fields (``span``/``answer``) are consumed only by validation, scoring and diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MINI_N = 194


class DataValidationError(ValueError):
    """Raised when dataset invariants (offsets, schema, paths) are violated."""


# --- prescribed functions (verbatim behaviour) -------------------------------------------------


def mini(tests: list[dict[str, Any]], n: int = MINI_N) -> list[dict[str, Any]]:
    """Exact LegalBench-RAG-mini selection function as specified for this task."""
    if len(tests) <= n:
        return tests

    def key(t: dict[str, Any]) -> float:
        random.seed(t["snippets"][0]["file_path"])
        return random.random()

    return sorted(tests, key=key)[:n]


def assign_split(query: str) -> str:
    """Prescribed deterministic split on the query text."""
    h = int(hashlib.sha256(query.strip().encode("utf-8")).hexdigest(), 16)
    return "test" if h % 10 < 3 else "train"


# --- records -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldSnippet:
    file_path: str
    start: int
    end: int
    answer: str


@dataclass(frozen=True)
class EvalQuery:
    """One evaluation item. ``gold`` is empty for unanswerable queries."""

    id: str
    text: str
    split: str
    answerable: bool
    gold: tuple[GoldSnippet, ...] = ()
    files_checked: tuple[str, ...] = field(default=())  # unanswerable only


def query_id(text: str) -> str:
    return "cuad-" + hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:10]


def read_text(path: Path) -> str:
    """Read a source document exactly as the benchmark offsets assume."""
    return path.read_text(encoding="utf-8")


def load_raw_tests(benchmark_file: str | Path) -> list[dict[str, Any]]:
    p = Path(benchmark_file)
    if not p.exists():
        raise DataValidationError(
            f"benchmark file not found: {p}. Run `contract-intel setup` to download it."
        )
    try:
        return json.loads(p.read_text(encoding="utf-8"))["tests"]
    except (KeyError, json.JSONDecodeError) as e:
        raise DataValidationError(f"malformed benchmark file {p}: {e}") from e


# --- validation --------------------------------------------------------------------------------


def validate_span(doc: str, span: Any, answer: Any, where: str = "") -> None:
    """Raise DataValidationError unless ``doc[start:end] == answer`` for a well-formed span."""
    if not (isinstance(span, (list, tuple)) and len(span) == 2):
        raise DataValidationError(f"{where}: span must be [start, end], got {span!r}")
    start, end = span
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (start, end)):
        raise DataValidationError(f"{where}: span offsets must be ints, got {span!r}")
    if not isinstance(answer, str) or not answer:
        raise DataValidationError(f"{where}: empty or non-string answer")
    if start < 0 or end <= start:
        raise DataValidationError(f"{where}: empty or inverted span {span!r}")
    if end > len(doc):
        raise DataValidationError(f"{where}: span {span!r} exceeds document length {len(doc)}")
    if doc[start:end] != answer:
        raise DataValidationError(f"{where}: document[{start}:{end}] != expected answer text")


def parse_gold(test: dict[str, Any], corpus_root: str | Path, cache: dict[str, str]) -> tuple[GoldSnippet, ...]:
    """Validate and parse the snippets of one benchmark test (strict: raises on any mismatch)."""
    snippets = test.get("snippets")
    if not snippets:
        raise DataValidationError(f"no snippets for query {test.get('query')!r}")
    out = []
    for s in snippets:
        fp = s.get("file_path")
        if not isinstance(fp, str) or not fp:
            raise DataValidationError(f"missing file_path in snippet of {test.get('query')!r}")
        if fp not in cache:
            path = Path(corpus_root) / fp
            if not path.exists():
                raise DataValidationError(f"source document not found for gold path {fp!r}: {path}")
            cache[fp] = read_text(path)
        validate_span(cache[fp], s.get("span"), s.get("answer"), where=f"{fp} :: {test['query'][:60]!r}")
        out.append(GoldSnippet(fp, s["span"][0], s["span"][1], s["answer"]))
    return tuple(out)


def count_overlaps(gold: tuple[GoldSnippet, ...]) -> int:
    """Number of overlapping/duplicate gold snippet pairs within one query (same file)."""
    n = 0
    for i, a in enumerate(gold):
        for b in gold[i + 1 :]:
            if a.file_path == b.file_path and min(a.end, b.end) > max(a.start, b.start):
                n += 1
    return n


def load_answerable(
    benchmark_file: str | Path, corpus_root: str | Path
) -> tuple[list[EvalQuery], dict[str, Any]]:
    """Select the mini set, validate every gold span, assign splits. Fails loudly on bad spans."""
    tests = mini(load_raw_tests(benchmark_file))
    cache: dict[str, str] = {}
    queries, overlaps = [], 0
    seen: set[str] = set()
    for t in tests:
        text = t["query"]
        if text in seen:
            raise DataValidationError(f"duplicate query text in selection: {text[:80]!r}")
        seen.add(text)
        gold = parse_gold(t, corpus_root, cache)
        overlaps += count_overlaps(gold)
        queries.append(EvalQuery(query_id(text), text, assign_split(text), True, gold))
    stats = {
        "n_total_benchmark_tests": len(load_raw_tests(benchmark_file)),
        "n_selected": len(queries),
        "n_gold_spans": sum(len(q.gold) for q in queries),
        "n_invalid_spans": 0,  # any invalid span raises above
        "n_overlapping_gold_pairs": overlaps,
        "n_files": len(cache),
    }
    return queries, stats


def load_unanswerable(path: str | Path) -> list[EvalQuery]:
    """Load data/unanswerable_queries.jsonl; the stored split must equal assign_split(query)."""
    p = Path(path)
    if not p.exists():
        raise DataValidationError(f"unanswerable set not found: {p}")
    out, ids = [], set()
    for ln, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        for k in ("id", "query", "label", "files_checked", "rationale", "split"):
            if k not in r:
                raise DataValidationError(f"{p}:{ln}: missing field {k!r}")
        if r["label"] != "unanswerable":
            raise DataValidationError(f"{p}:{ln}: label must be 'unanswerable'")
        if r["split"] != assign_split(r["query"]):
            raise DataValidationError(f"{p}:{ln}: stored split disagrees with assign_split()")
        if r["id"] in ids:
            raise DataValidationError(f"{p}:{ln}: duplicate id {r['id']}")
        ids.add(r["id"])
        out.append(EvalQuery(r["id"], r["query"], r["split"], False, (), tuple(r["files_checked"])))
    return out


# --- manifest ----------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(benchmark_file: str | Path, corpus_root: str | Path) -> dict[str, Any]:
    """Manifest of selected queries (no answer text) and exactly the contracts they reference."""
    queries, stats = load_answerable(benchmark_file, corpus_root)
    files = sorted({g.file_path for q in queries for g in q.gold})
    contracts = []
    for fp in files:
        p = Path(corpus_root) / fp
        contracts.append(
            {"file_path": fp, "n_chars": len(read_text(p)), "sha256": sha256_file(p)}
        )
    counts = {s: sum(q.split == s for q in queries) for s in ("train", "test")}
    manifest = {
        "dataset": "awinml/legalbench-rag (CUAD subset)",
        "benchmark_file_sha256": sha256_file(Path(benchmark_file)),
        "selection": f"mini(tests, n={MINI_N}) exactly as specified",
        "split_rule": "assign_split: sha256(query.strip()) % 10 < 3 -> test else train",
        "stats": stats,
        "split_counts": counts,
        "contracts": contracts,
        "queries": [
            {
                "id": q.id,
                "query": q.text,
                "split": q.split,
                "gold": [[g.file_path, g.start, g.end] for g in q.gold],
            }
            for q in queries
        ],
    }
    manifest["manifest_hash"] = manifest_hash(manifest)
    return manifest


def manifest_hash(manifest: dict[str, Any]) -> str:
    body = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
