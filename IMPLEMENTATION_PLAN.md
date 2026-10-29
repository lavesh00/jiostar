# Implementation plan and repository audit

## 1. Audit of the candidate repository
Candidate: `https://github.com/Akshitha024/legal-retrieval-benchmark` (cloned at `./legal-retrieval-benchmark/`, HEAD `8936867`, MIT license, (c) 2026 Akshitha Reddy Lingampally). It is treated as untrusted input; nothing from it is executed.

| Aspect | Finding (verified in code) |
|---|---|
| Purpose | Doc-level IR benchmark (nDCG/MRR/Recall/MAP) over BM25, dense (BGE-small + FAISS), RRF, cross-encoder rerank. |
| Data | `loader.py` builds docs/queries from `theatticusproject/cuad-qa` (SQuAD rows) with **document-level** qrels. LegalBench-RAG loading is a `NotImplementedError` stub. |
| Offsets / spans | **None.** Retrievers return `Hit(doc_id, score, rank)` for whole documents; no chunking, no character spans. |
| Metrics | Doc-level only; no character precision/recall, no abstention metrics. |
| Split / selection | None of the prescribed `mini()` selection or `assign_split()`. |
| Abstention, optimizer, CLI/Streamlit | No abstention; no optimization loop; Typer CLI over doc-level eval; no web app. |
| Tests | Small suite on doc-level metrics/BM25/hybrid with a toy fixture. |
| Dependencies | datasets, sentence-transformers, faiss-cpu, torch stack (heavy; pinned `numpy<2`). |
| README claims | Reports doc-level numbers that are not comparable to character-level LegalBench-RAG metrics; not relied upon. |

**Verdict:** poor foundation for a span-level system (wrong retrieval unit, no offsets, heavy deps). Per the brief we do not silently switch bases: we keep the repository's *ideas* (BM25 baseline, RRF fusion, configurable retrievers, benchmark-then-compare discipline) and implement a clean, small package `contract_intel` in the project root. **No code is copied** from the candidate (so its MIT notice is not triggered; it is credited as inspiration in the README). Its `.git` history is left untouched and the folder is git-ignored.

## 2. Reuse / modify / missing
* Reusable as concepts: BM25 baseline, RRF fusion, config-driven retrievers, optional cross-encoder rerank (not implemented; see non-goals).
* Missing (all built new): exact mini selection, split, offset validation, manifest, chunking with offsets, BM25 on chunks, span metrics, abstention, unanswerable set, diagnostics, optimizer, CLI, Streamlit, tests.

## 3. Licensing and provenance
| Component | License | Use |
|---|---|---|
| LegalBench-RAG data export (HF `awinml/legalbench-rag`); CUAD v1 | CC BY 4.0 (CUAD, per the dataset's own LICENSE table) | Downloaded at setup into git-ignored `data/raw/`; not redistributed. Only question text and offsets (no contract text) are in the committed manifest. |
| Official LegalBench-RAG code (`zeroentropy-ai/legalbenchrag`, MIT) | MIT | Read only to confirm metric conventions (`run_benchmark.py`); metrics re-implemented from the definition. |
| `legal-retrieval-benchmark` | MIT | Not copied; credited as inspiration. |
| Python deps: PyYAML, pytest, streamlit, huggingface_hub | MIT/Apache-2.0/BSD | Standard; no copied code. |

## 4. Risks
* Python: system default is 3.10; project pins 3.11-3.12; a 3.12 venv is used.
* Only 29 contracts / 194 queries: high variance; test split has 54 answerable and only 6 unanswerable queries (small denominators reported as such).
* `Path.read_text` universal-newline behaviour matches the benchmark; validated for all 272 gold spans (0 mismatches).
* Query text embeds the contract description (`Consider the X agreement between ...; question`) which dominates lexical scores — handled by an optional, config-gated query-context stripper and document-level prior, evaluated on train only.
* Unanswerable-set quality: items were verified by reading or by keyword probes plus structural reads; some contracts incorporate other documents (leases, ordinances, SPAs) that are not indexed; recorded in rationales.

## 5. Architecture (package `src/contract_intel`)
`config` → `data` (mini, split, validation, manifest) → `documents` → `chunking` → `indexing`/`ranking` (own BM25 + TF-IDF + RRF) → `retrieval` (service returning the stable result schema) → `abstention` → `metrics` / `evaluation` / `diagnostics` → `optimize` (train-only) → `cli`, `app` (Streamlit; both call `retrieval.ContractSearcher`).

Data flow: manifest → allowed contract list → original text → chunks (start/end on original text) → inverted index → query → candidates → overlap suppression → abstention decision → `SearchResult` JSON with `text == doc[start:end]`.

## 6. Milestones and acceptance tests
1. Data layer: 194 selected, 272 spans valid, split 140/54. (`tests/test_data.py`)
2. Frozen unanswerable set + objective committed *before* retrieval code. (git history)
3. Chunking/retrieval/metrics/abstention unit tests green.
4. Baseline train metrics saved to `experiments/baseline.json`.
5. >= 5 genuine optimizer iterations logged to `experiments/iterations.jsonl`.
6. CLI/Streamlit smoke tests; final test evaluation once; README/REPORT; acceptance audit.

## 7. Assumptions
* Private repository is the default submission mode (conflicting logistics vs. deliverables text in the brief).
* Gold file paths are dataset-style (`cuad/<name>.txt`), resolved under `corpus_root`; results use the same string.
* A single index over all 29 contracts is searched (mirrors the benchmark, where the query names the contract in free text); no gold-based document routing.
* Test-split unanswerable questions (6) are too few for tight intervals; reported with raw counts.
* No LLM and no dense model are used in the default pipeline (offline, deterministic).
