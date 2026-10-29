"""Failure categorisation from per-query records.

Gold annotations are used here, after retrieval, only to explain errors (never passed to retrieval).

Primary category per answerable query (first match wins; thresholds are module constants):
  false_abstention        abstained although the query has annotated evidence
  wrong_document          no top-k passage comes from a gold contract
  right_doc_wrong_section gold contract present, but no retrieved character is inside a gold span
  incomplete_coverage     some gold characters retrieved, recall@k < RECALL_OK
  overly_broad            recall@k >= RECALL_OK but precision@k < PRECISION_OK
  success                 recall@k >= RECALL_OK and precision@k >= PRECISION_OK
Unanswerable queries: ``correct_abstention`` or ``false_acceptance``.

Secondary tags (not exclusive): duplicate_overlap, query_wording_mismatch, cross_reference_dependency.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .config import RetrievalConfig
from .documents import DocumentStore
from .indexing import Tokenizer, strip_query_context

RECALL_OK = 0.5
PRECISION_OK = 0.25
WORDING_JACCARD = 0.10
_XREF = re.compile(r"\b(?:Section|Article|Exhibit|Schedule|Paragraph|clause)\s+[\dIVXA-Z]|as defined|hereof|set forth (?:above|below|in)", re.IGNORECASE)
_TOK = Tokenizer(RetrievalConfig(remove_stopwords=True))


def categorize(rec: dict[str, Any], k: int) -> str:
    if not rec["answerable"]:
        return "correct_abstention" if rec["abstained"] else "false_acceptance"
    if rec["abstained"]:
        return "false_abstention"
    if not rec[f"doc_hit@{k}"]:
        return "wrong_document"
    if not rec[f"span_hit@{k}"]:
        return "right_doc_wrong_section"
    r, p = rec[f"recall@{k}"], rec[f"precision@{k}"]
    if r < RECALL_OK:
        return "incomplete_coverage"
    return "overly_broad" if p < PRECISION_OK else "success"


def _overlapping_results(results: list[list[Any]]) -> bool:
    for i, a in enumerate(results):
        for b in results[i + 1 :]:
            if a[0] == b[0] and min(a[2], b[2]) > max(a[1], b[1]):
                return True
    return False


def tags(rec: dict[str, Any], docs: DocumentStore | None) -> list[str]:
    out = []
    if _overlapping_results(rec["results"]):
        out.append("duplicate_overlap")
    if rec["answerable"] and docs is not None and rec.get("gold"):
        gold_text = " ".join(docs.slice(f, s, e) for f, s, e in rec["gold"])
        q = set(_TOK(strip_query_context(rec["query"])))
        g = set(_TOK(gold_text))
        if q and g and len(q & g) / len(q | g) < WORDING_JACCARD:
            out.append("query_wording_mismatch")
        if _XREF.search(gold_text):
            out.append("cross_reference_dependency")
    return out


def diagnose(
    records: list[dict[str, Any]], k: int = 5, docs: DocumentStore | None = None, max_examples: int = 3
) -> dict[str, Any]:
    """Counts per category / tag and a few concrete examples per failure category."""
    cats: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        c = categorize(rec, k)
        cats[c] += 1
        t = tags(rec, docs)
        tag_counts.update(t)
        if c in ("success", "correct_abstention"):
            continue
        ex = examples.setdefault(c, [])
        if len(ex) < max_examples:
            item: dict[str, Any] = {"id": rec["id"], "query": rec["query"], "tags": t,
                                    "top1_confidence": round(rec["top1_confidence"], 4)}
            if docs is not None:
                if rec["answerable"]:
                    item["gold"] = [{"file": f, "start": s, "end": e, "text": docs.slice(f, s, e)[:300]} for f, s, e in rec["gold"][:2]]
                item["top_result"] = (
                    {"file": rec["results"][0][0], "start": rec["results"][0][1], "end": rec["results"][0][2],
                     "text": docs.slice(*rec["results"][0][:3])[:300]} if rec["results"] else None
                )
            ex.append(item)
    n_ans = sum(r["answerable"] for r in records)
    return {
        "k": k,
        "categories": dict(cats),
        "tags": dict(tag_counts),
        "answerable_failure_share": {
            c: round(n / n_ans, 4) for c, n in cats.items()
            if n_ans and c not in ("success", "correct_abstention", "false_acceptance")
        },
        "examples": examples,
    }
