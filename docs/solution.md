# Solution overview

A short, end-to-end account of the problem, the approach and where it stands. Full evidence is in [REPORT.md](../REPORT.md). Every
number below is copied from it, and it in turn reads from files under `experiments/`.

## Problem

Someone reviewing a contract asks a question in plain language ("Is there an anti-assignment clause?"). The system must return the
exact clause, with its contract and character offsets, or say that no passage is relevant enough. Quality is measured with
LegalBench-RAG character-level precision/recall on the 194 CUAD queries in LegalBench-RAG-mini (29 contracts; 140 train / 54 test).
Abstention is measured on 39 hand-written unanswerable questions (33 train / 6 test).

## Approach in one picture

```
question ──► split off "Consider the <contract>;" ──► legal query expansion
    │                                                   │
    └──► which contract? (whole-contract BM25 prior) ──►│──► keep contracts near the best match
                                                        ▼
               1500-char chunks (exact offsets) ──► BM25 + TF-IDF ──► RRF fusion + prior ──► top 5, no overlaps
                                                        ▼
              contract match < 0.59 or clause confidence < 0.66 ? ──► abstain, with a reason
                                                        ▼
                                  passages with text == document[start:end]
```

## Why this design

| Choice | Reason |
|---|---|
| Lexical retrieval (BM25 + TF-IDF), no LLM | Contracts are confidential. It runs fully offline and deterministically, needs no API keys and no cache to reproduce, and costs nothing per query. The gains measured without an LLM were already large. |
| Offsets as intervals of untouched text | Exact citations by construction: no re-assembled or normalised text can drift. |
| Separate routing from clause scoring | Benchmark questions name their contract in a long prefix. Scored together, the prefix overwhelms the clause words (67% of train queries were "right contract, wrong section" before this change). Scored separately, the prefix picks the contract and the question picks the clause. |
| Query-expansion lexicon | CUAD questions are short templates. Adding typical clause vocabulary was the largest single gain (train Recall@5 0.394 → 0.717). |
| "Peakedness" confidence | Normalised BM25 mostly tracked query length. Asking "does one clause stand out from the rest of this contract?" separates answerable from unanswerable questions much better (train correct abstention 0.333 → 0.727). |
| Peak outside the opening block | Every remaining false acceptance in Run 5 was "won" by the title/parties block, whose names and entity words match any question about that contract. Leaving it out of the peak raised train correct abstention 0.727 → 0.818 without changing any answerable result (Run 6). |
| Contract gate | Questions about contracts that are not indexed must not get an answer. The gate is calibrated to reject ≤ 1% of train answerable queries. |
| Metric-driven loop | Every choice above was proposed, measured on train, and accepted or rejected by a rule declared in advance ([improvement_loop.md](improvement_loop.md)). |

## Results (from REPORT.md §4)

| Metric | Baseline test | Final test (54 answerable / 6 unanswerable) |
|---|---|---|
| Char Recall@5 · Precision@5 | 0.136 · 0.009 | **0.798 · 0.061** |
| Char Recall@1 · Precision@1 | 0.019 · 0.003 | 0.535 · 0.146 |
| Coverage (answered / answerable) | 54/54 | 50/54 |
| Correct abstention (unanswerable) | 0/6 | 4/6 |
| Correct abstention, out-of-corpus probe (test part) | 0/20 | 20/20 |

Train for the final config: Recall@5 0.750, Precision@5 0.052, correct abstention 27/33. The test split was evaluated 4 times in
total, each run logged in `experiments/final_test_runs.jsonl`.

## What is weak, and what would come next

* **Precision.** Passages are about 4× longer than gold clauses. Shrinking them raised precision, but recall fell past the guardrail
  (REPORT §5 trade-off table). Next would be a reranker or clause-level segmentation that keeps recall.
* **Abstention on questions about indexed contracts** is 4/6 on test (27/33 on train). Party names that recur in signature blocks and exhibits
  still lift the confidence of some free-form questions, and redacted values (`***`) look like answers. Next would be removing party names from the confidence signal, plus per-question-type thresholds.
* **Cross-references and definitions** are not resolved (41 of 140 train queries have gold text that refers elsewhere).
* **Query form.** The lexicon and gate rely on CUAD-style templates. Free-form questions should use the contract selector.

## Where to look

| Need | File |
|---|---|
| Run it | [README.md](../README.md) |
| Code walkthrough | [implementation.md](implementation.md) |
| Improvement loop | [improvement_loop.md](improvement_loop.md) |
| Objective and guardrails (frozen) | [optimization_objective.md](optimization_objective.md) |
| Full evaluation, iteration table, error analysis | [REPORT.md](../REPORT.md) |
| Iteration log, test-run log, configs | `experiments/iterations.jsonl`, `experiments/final_test_runs.jsonl`, `configs/` |
