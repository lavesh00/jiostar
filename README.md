# JioStar Contract Intelligence Assistant

Retrieval-focused assistant that finds the **exact answer-bearing passages** in commercial contracts. For a
natural-language question it returns ranked passages with the source file and exact character offsets
(`document[start:end] == text`), or **abstains** (returns nothing) when no passage is sufficiently relevant.
It runs fully offline: BM25 + TF-IDF lexical retrieval, no LLM, no API keys.

Data: the CUAD subset of [LegalBench-RAG](https://huggingface.co/datasets/awinml/legalbench-rag) restricted to the
194-query "mini" selection (29 contracts). Numbers and analysis are in [REPORT.md](REPORT.md); the plan and repository
audit are in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). The tool retrieves evidence; it is **not legal advice**.

## Setup (Python 3.11 or 3.12)

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on macOS/Linux
pip install -e ".[data,app,dev]"
contract-intel setup              # downloads CUAD files (~32 MB) into data/raw/, validates all 272 gold offsets, writes data/manifests/cuad_mini.json
```

`setup` fails loudly if any gold `document[start:end]` differs from its annotated text. To rebuild only the manifest from
files you already have: `contract-intel setup --skip-download` (or `contract-intel validate`).
`data/raw/` is git-ignored; nothing from the corpus is committed except question text and offsets in the manifest.

## Use

```bash
contract-intel build-index --config configs/final.yaml
contract-intel search "Consider the Sponsorship Agreement between Jacksonville Jaguars, LLC and The ARC Group, Inc.; Is there an anti-assignment clause in this contract?" --top-k 5
contract-intel search "..." --json                  # stable JSON schema (below)
streamlit run src/contract_intel/app.py             # web app (reads configs/final.yaml; override with CONTRACT_INTEL_CONFIG)
```

(`python -m contract_intel.cli ...` is equivalent to `contract-intel ...`.) Abstention prints
`ABSTAINED: no passage is sufficiently relevant (...)` and returns exit code 0 with an empty `results` list in JSON mode.
Exit codes: 0 success, 1 unexpected error, 2 usage/config/data problem.

JSON result schema: `{"query", "abstained", "abstain_reason", "results": [{"rank", "file_path", "start", "end", "score", "confidence", "text"}], "meta": {...}}`.
`file_path` is dataset-style (`cuad/<name>.txt`, resolved under `data/raw/legalbench-rag/corpus/`); offsets are Python
`str` offsets of the file read with `Path.read_text(encoding="utf-8")`.

## Evaluate, optimize, test

```bash
contract-intel evaluate --split train --config configs/baseline.yaml     # also configs/final.yaml
contract-intel optimize --config configs/baseline.yaml --iterations 10   # train-only loop, ~35 s; writes configs/final.yaml
contract-intel evaluate --split test --final --config configs/final.yaml # held-out test; --final is mandatory and every run is logged
contract-intel diagnose --split train --config configs/final.yaml        # failure categories with examples
contract-intel test                                                       # = pytest (194 tests; 3 need the downloaded data)
```

Outputs: `experiments/baseline.json`, `experiments/iterations.jsonl` (append-only), `experiments/results/*` (metrics +
per-query JSONL), `experiments/final_test_runs.jsonl` (every test-split evaluation). Approximate runtimes on a laptop:
index build 0.2-0.5 s, train evaluation ~3 s, optimizer ~35 s, test suite ~6 s.

## Reproduce the reported numbers

1. `contract-intel setup` (manifest hash must be `e2a5024fd7e7ef0b`; 194 queries; 140 train / 54 test).
2. `contract-intel evaluate --split train --config configs/baseline.yaml` and `... configs/final.yaml` reproduce the train rows of REPORT.md.
3. `contract-intel evaluate --split test --final --config configs/final.yaml` (and baseline) reproduce the test rows. The committed
   `experiments/final_test_runs.jsonl` already records the two official test runs; re-running appends further (clearly numbered) runs.
4. `contract-intel optimize ...` re-derives `configs/final.yaml` deterministically (compare `config_hash 1cec43a77086`).

## Architecture

```
manifest (contract list) -> DocumentStore (original text) -> chunking (start,end on original text)
  -> LexicalIndex (BM25 / TF-IDF) -> rank_candidates (+RRF, +document prior, +query expansion)
  -> overlap suppression -> abstention.decide -> SearchResponse (text == doc[start:end])
```
Modules in `src/contract_intel/`: `config` (typed YAML), `data` (exact `mini()`, `assign_split()`, offset validation, manifest),
`documents`, `chunking` (fixed / paragraph / section, offset-exact), `indexing`, `ranking`, `lexicon`, `retrieval`
(the service), `abstention`, `metrics` (official-style char precision/recall), `evaluation`, `diagnostics`, `optimize`, `service`, `cli`, `app`.
CLI and Streamlit both call `service.load_searcher` / `ContractSearcher.search`, so results are identical (tested).

## Configuration

`configs/baseline.yaml` (fixed 1000-char windows, plain BM25, abstention off) and `configs/final.yaml` (the configuration
accepted by the train-only loop, with its train-fitted abstention threshold). Every pipeline choice (chunking, BM25 parameters,
query processing/expansion, RRF fusion, document prior, abstention signal/threshold, k values, paths) lives in these files.

## Assumptions and ambiguities

* Submission visibility: the brief is inconsistent (private repo with collaborator access vs. public); a **private** repository is assumed.
* Selection/split functions are implemented verbatim; split counts are reported as they fall (no balancing).
* A single index over all 29 contracts is searched; there is no gold-based routing.
* The unanswerable set (39 queries; 33 train / 6 test) was verified by an AI assistant: 17 items by reading the whole contract,
  22 by automated keyword-absence checks plus a section-level read (see REPORT.md, "Limitations"). It was frozen before tuning.
* No dense retrieval, reranker or LLM is used; adding them is an extension, not a requirement of the baseline.

## AI tool usage

This project was built with Claude Code (Claude Sonnet 5.5) acting autonomously on a written specification: it audited the
candidate repository, wrote the code, tests, unanswerable questions, configs and reports, and ran every experiment reported here.
All figures come from files under `experiments/`. A human owner should review the code and the unanswerable set before relying on them.

## License and third-party material

No project license has been chosen yet (private repository; the owner must pick one). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for data and dependency licenses. No code was copied from `Akshitha024/legal-retrieval-benchmark` (MIT, inspected only) or from the official
LegalBench-RAG repository (MIT, used to confirm the metric definition).
