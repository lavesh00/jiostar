# Contract Intelligence Assistant

## Overview
Ask a natural-language question about a contract. The system returns a ranked list of passages from the indexed contracts, each with its
source file and exact character offsets (`document[start:end]` equals the returned text). When no passage is relevant enough, it returns
nothing and says why. It is fully offline: lexical retrieval (BM25 + TF-IDF with reciprocal-rank fusion), document routing and a
small legal query-expansion lexicon, with no LLM and no API keys. It has a CLI and a Streamlit web app built on the same service code,
plus an evaluation harness and a train-only self-improvement loop. The data is the CUAD part of LegalBench-RAG-mini (194 queries over 29 contracts).
Results and analysis are in [REPORT.md](REPORT.md).

## Setup
Python 3.11 or 3.12 (developed on 3.12.0, Windows 11).

```bash
python -m venv .venv
.venv\Scripts\activate                 # macOS/Linux: source .venv/bin/activate
pip install -e ".[data,app,dev]"       # PyYAML (core); huggingface_hub, streamlit, pytest, ruff (extras)
contract-intel setup                   # download data, validate offsets, write the manifest (~30 s)
```

`setup` downloads `benchmarks/cuad.json` and `corpus/cuad/` from `huggingface.co/datasets/awinml/legalbench-rag` into the git-ignored
`data/raw/legalbench-rag/` folder (~32 MB). It selects the 194 mini queries, checks `document[start:end] == answer` for all 272 gold spans
(it stops with an error on any mismatch), and writes `data/manifests/cuad_mini.json`. No LLM is used, so no keys or local models are
needed. `.env.example` lists only an optional `HF_TOKEN` for faster downloads.

## How to run

| Task | Command |
|---|---|
| Web app | `streamlit run src/contract_intel/app.py` (choose a contract, or name it in the question) |
| CLI search | `contract-intel search "Consider the Sponsorship Agreement between R. C. Boyd Enterprises, LLC and Cano Petroleum, Inc. for 'Honey Hole'; Is there an anti-assignment clause in this contract?"` |
| CLI search in one contract, JSON | `contract-intel search "Is there an anti-assignment clause?" --contract CANOPETROLEUM --top-k 3 --json` |
| Index statistics | `contract-intel build-index` |
| Train evaluation | `contract-intel evaluate --split train --config configs/final.yaml` (or `configs/baseline.yaml`) |
| Test evaluation (final reporting only) | `contract-intel evaluate --split test --final --config configs/final.yaml` |
| Out-of-corpus abstention probe | `contract-intel probe --config configs/final.yaml` |
| Failure diagnosis | `contract-intel diagnose --split train` |
| Improvement loop | `contract-intel optimize --config configs/baseline.yaml --iterations 14` |
| Tests | `contract-intel test` or `pytest` |

`python -m contract_intel.cli …` works the same as `contract-intel …`.

* **Exit codes:** 0 success (including abstention), 2 usage/config/data error, 1 unexpected error.
* **Test-split guard:** `--split test` refuses to run without `--final`, and every test run is appended to `experiments/final_test_runs.jsonl` so the count is auditable.
* **JSON output:** `{"query", "abstained", "abstain_reason", "results": [{"rank", "file_path", "start", "end", "score", "confidence", "text"}], "meta"}`.
  * `file_path` uses the dataset's form (`cuad/<name>.txt`).
  * Offsets are Python string offsets into the file read with `Path.read_text(encoding="utf-8")`.

## Architecture

```
contract-intel setup ─> data.py (exact mini(), assign_split(), offset validation) ─> manifest (29 contracts, no answer text)
                                                                          │
search query ─> retrieval.ContractSearcher ──────────────────────────────┤ documents.py (original text, never modified)
   (CLI, app,       ├ indexing.py   tokenizer, BM25 + TF-IDF index        │ chunking.py  fixed / paragraph / section chunks as [start,end)
    evaluation)     ├ lexicon.py    query expansion                       
                    ├ ranking.py    RRF fusion, document prior, route filter, overlap suppression, span refinement
                    └ abstention.py contract gate + confidence threshold  ─> SearchResponse (text == doc[start:end])
evaluation.py (per-query scoring) ─ metrics.py (character P/R@k, as in LegalBench-RAG) ─ diagnostics.py (failure categories)
optimize.py (train-only loop: diagnose -> propose -> evaluate -> accept/reject -> experiments/iterations.jsonl)
service.py (shared by cli.py and app.py)
```

* **Configuration** lives in `configs/baseline.yaml` and `configs/final.yaml`, parsed into typed dataclasses in `config.py`. It covers chunking,
  BM25 parameters, query processing, fusion, routing, refinement, abstention signal and thresholds, k values and paths.
* **Gold spans** are read only by `data.py` (validation), `evaluation.py`, `metrics.py` and `diagnostics.py`. The searcher receives only the query text.
* **Objective and guardrails:** `docs/optimization_objective.md`.
* **Unanswerable set:** `data/unanswerable_queries.jsonl` (frozen).
* **Out-of-corpus probe:** `data/out_of_corpus_probe.jsonl`.

## Reproducing results
After `setup` (about 30 s), run these from the repository root:

| Step | Command | Time |
|---|---|---|
| Train metrics, baseline and final | `contract-intel evaluate --split train --config configs/baseline.yaml` and `… configs/final.yaml` | ~3 s each |
| Re-derive the final config | `contract-intel optimize --config configs/baseline.yaml --iterations 14` → `configs/final.yaml`, config hash `78a53ef0d1aa` | ~50 s |
| Test metrics | `contract-intel evaluate --split test --final --config configs/final.yaml` (logged as an extra test run) | ~3 s |
| Out-of-corpus probe | `contract-intel probe --config configs/final.yaml` | ~3 s |
| Tests | `pytest`: 203 tests; 3 of them need `data/raw` and are skipped without it | ~7 s |

* **Outputs:** metrics and per-query results go to `experiments/results/`.
* **Committed results:** the runs used in the report are already committed (`experiments/results/`, `experiments/final_test_runs.jsonl`,
  `experiments/iterations.jsonl`), so you can compare a fresh run against them.
* **Determinism:** everything is deterministic; only latencies vary.

## Assumptions
* **Visibility.** The brief asks for a private repository with collaborator access in one place and a public repository in another. I assumed
  **private + collaborator access**, because that is the explicit submission instruction.
* **Offsets.** Offsets are taken as Python `str` indices after reading in text mode with UTF-8 (universal newlines). All 272 gold spans match exactly
  under this reading.
* **Corpus.** One index over all 29 contracts. Benchmark questions name their contract in free text ("Consider the … between X and Y; …"),
  so the system has to work out which contract is meant. Nothing gold-derived is used for routing.
* **Unanswerable split.** `assign_split` put 33 of the 39 unanswerable questions in train and 6 in test. The brief expects about a third in test; I kept
  the function's output rather than editing the frozen set.
* **Questions about contracts outside the corpus** are answered with an abstention. To measure this I derived a separate probe set from benchmark
  questions about non-indexed CUAD contracts. It is used only for reporting.
* **Explicit contract choice.** A question that names no contract can be scoped with `--contract` or the app's selector. Without one, it searches
  every indexed contract.
* **"Relevant enough"** means passing two train-calibrated thresholds: the contract match (≤ 1% answerable loss) and passage confidence
  (≥ 90% answerable coverage). The rationale is in `docs/optimization_objective.md`.
* **Character metrics** follow the LegalBench-RAG reference implementation. Overlapping passages are merged so they are never double counted; see `metrics.py`.

## AI tool usage
* **Development:** Claude Code (Anthropic) was used throughout. It audited a candidate starting repository (`Akshitha024/legal-retrieval-benchmark`,
  MIT); only ideas were reused, no code was copied. It wrote the code, the tests and the first draft of the 39 unanswerable questions, which were
  checked against the contract text with scripts and by reading (REPORT §7 says how thoroughly). It also ran every experiment and drafted these documents.
* **The submitted system** contains no LLM.

The author reviewed the design and is responsible for being able to explain and change every part of it.

## Licenses
* **Data:** CUAD / LegalBench-RAG under CC BY 4.0. It is downloaded at setup time and not redistributed.
* **Dependencies:** see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
* **This repository:** no license has been chosen yet (private submission).
