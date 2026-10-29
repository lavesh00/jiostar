# Implementation guide

This walks through the code in `src/contract_intel/`. It covers what each module does, how data flows between modules, and
which rules the code always keeps. Use it to find your way around the code or to make a change. Results are in
[REPORT.md](../REPORT.md); run commands are in [README.md](../README.md).

## 1. Data flow

```
data/raw/legalbench-rag/ (downloaded by `contract-intel setup`, git-ignored)
   │
   ▼
data.py ── mini() + assign_split() (verbatim from the brief) ── offset check doc[start:end] == answer
   │        └─> data/manifests/cuad_mini.json  (29 contract paths, query ids, splits, offsets; no contract text)
   ▼
documents.py  DocumentStore: Path.read_text(encoding="utf-8"), the text is never modified
   ▼
chunking.py   Chunk(file_path, start, end): intervals of the original text (fixed / paragraph / section)
   ▼
indexing.py   Tokenizer (lowercase, stopwords, light stemmer) + LexicalIndex (BM25 and TF-IDF over chunks,
              and a second index over whole contracts for the document prior)
   ▼
retrieval.py  ContractSearcher.search_raw(query)            ◄── lexicon.py (query expansion)
   │            strip "Consider the …;" → expand → document prior → route filter
   │            → ranking.py: BM25 + TF-IDF fused by RRF (+ prior) → overlap suppression → (optional) span refinement
   │            → confidence signal (peakedness)
   ▼
abstention.py decide(): no match / contract gate / confidence threshold / (optional) gap
   ▼
SearchResponse {query, abstained, abstain_reason, results[rank, file_path, start, end, score, confidence, text], meta}
   │
   ├── service.py ──> cli.py (`contract-intel search`) and app.py (Streamlit)
   └── evaluation.py ──> metrics.py (character P/R@k) ──> diagnostics.py (failure categories) ──> optimize.py (loop)
```

## 2. Module reference

| Module | Responsibility | Key functions / classes |
|---|---|---|
| `config.py` | Typed dataclasses for every pipeline choice: chunking, retrieval, abstention, eval, paths. YAML in, validated, hashed. `with_changes({"chunking.size": 700})` makes a modified copy (used by the loop). | `load_config`, `dump_config`, `Config.hash`, `Config.with_changes` |
| `data.py` | Exact `mini()` and `assign_split()` from the brief; loads the benchmark, checks every gold span against the file text (stops on any mismatch), builds the manifest, loads the unanswerable set and the out-of-corpus probe. `load_train_queries` is the only loader the optimizer imports. | `mini`, `assign_split`, `validate_span`, `build_manifest`, `load_train_queries` |
| `documents.py` | Read-only store of the original contract text, keyed by dataset path (`cuad/<name>.txt`). | `DocumentStore.text`, `.slice` |
| `chunking.py` | Builds chunks as `[start, end)` offsets into the original text. Trimming only moves bounds inward, so `text[start:end]` is always exact. | `chunk_document` |
| `indexing.py` | Tokenizer and a pure-Python inverted index. `bm25()` also returns the score's upper bound so scores can be normalised to `[0, 1)`. `strip_query_context` removes the benchmark's "Consider the …;" prefix. | `Tokenizer`, `LexicalIndex.bm25`, `.tfidf` |
| `lexicon.py` | Hand-written trigger → expansion rules for common contract-review questions (governing law, assignment, liability cap, …). | `expand` |
| `ranking.py` | Candidate scoring, RRF fusion, document-prior boost, greedy overlap suppression, sentence-window span refinement. | `rank_candidates`, `suppress_overlaps`, `refine_span` |
| `retrieval.py` | `ContractSearcher` builds the indexes once and answers queries; computes the routing prior and the peakedness confidence. Supports an explicit `contract=` filter. | `ContractSearcher.search_raw`, `.search` |
| `abstention.py` | Turns retrieval signals into an answer-or-abstain decision with a reason the user can read. | `decide` |
| `metrics.py` | Character-level precision/recall (LegalBench-RAG definition), document hit, span hit, MRR. Merges intervals so overlaps are never double counted. | `char_precision_recall`, `merge_intervals` |
| `evaluation.py` | Runs a query set, scores each query, aggregates the metrics, writes `experiments/results/*`. Guards the test split. | `run_queries`, `score_run`, `aggregate` |
| `diagnostics.py` | Puts each query into one failure category (wrong document / wrong section / partial coverage / overly broad / false abstention / false acceptance) plus secondary tags. | `categorize`, `diagnose` |
| `optimize.py` | The train-only self-improvement loop (see [improvement_loop.md](improvement_loop.md)). | `HYPOTHESES`, `optimize` |
| `service.py`, `cli.py`, `app.py` | Shared service layer, argparse CLI (`setup`, `search`, `build-index`, `evaluate`, `probe`, `diagnose`, `optimize`, `test`), Streamlit UI with a contract selector. | `load_searcher`, `main` |

## 3. Search, step by step

For `Consider the Sponsorship Agreement between R. C. Boyd Enterprises, LLC and Cano Petroleum, Inc.; Is there an anti-assignment clause?`:

1. **Split the query.** `strip_query_context` keeps `Is there an anti-assignment clause?` for passage scoring. The full query
   (with the contract description) is kept for routing.
2. **Expand.** `lexicon.expand` sees "anti-assignment" and appends `assign assignment transfer consent written prior delegate successors`.
3. **Route.** The full query is scored with BM25 against the whole-contract index. Dividing by the score's upper bound gives
   a prior in `[0, 1)` per contract. The best value is the *contract match*. With `doc_filter_ratio: 0.8`, only contracts
   whose prior is at least 0.8 × best stay in play.
4. **Rank chunks.** BM25 and TF-IDF each rank the allowed chunks (top `candidates: 50`). RRF (`rrf_k: 60`) fuses the two
   rankings, and `doc_boost × prior` is added to the fused score.
5. **Remove duplicates.** `suppress_overlaps` drops a candidate if more than 50% of it is already covered by a kept passage, and stops at `top_k`.
6. **Confidence.** With `signal: doc_peakedness`, each passage's confidence is `(peak − median) / peak` of the literal-question
   BM25 scores over the chunks of that passage's contract, with the peak taken outside the opening (title/parties) chunk (`peak_skip_opening: true`). If one clause clearly answers the question, the score profile has a
   sharp peak. If nothing does, the profile is flat.
7. **Decide** (`abstention.decide`). Abstain if nothing matched, if the contract match is below `min_doc_prior: 0.59`, or if top-1
   confidence is below `threshold: 0.66`. Each case gives its own reason. Otherwise return passages whose `text == document[start:end]`.

If the user picks a contract (`--contract` in the CLI, dropdown in the app), routing and the contract gate are skipped. The search
is limited to that contract.

## 4. Rules the code always keeps (and the tests that enforce them)

| Rule | Where it is enforced |
|---|---|
| `document[start:end] == passage.text` for every chunking option, including Unicode, CRLF and empty documents | `chunking.py`, `tests/test_chunking.py`, `tests/test_retrieval.py` |
| All 272 gold spans match the file text read with UTF-8 in text mode | `data.validate_span`, `tests/test_data.py` (data-marked) |
| Gold spans never reach the searcher; `ContractSearcher` gets only query text and the manifest's contract list | `retrieval.allowed_contract_paths` reads only `contracts` |
| The optimizer can only load train queries | `optimize.py` imports `load_train_queries`; `tests/test_config_cli_optimize.py` |
| `--split test` refuses to run without `--final`, and every test run is appended to `experiments/final_test_runs.jsonl` | `cli.py` |
| The unanswerable set is frozen (its sha-256 is pinned by a test) | `tests/test_data.py` |
| Deterministic: same config → same results and same config hash | no randomness in the pipeline; `Config.hash` |

## 5. Making a change

* **Try a different pipeline setting:** copy `configs/final.yaml`, edit it, then run `contract-intel evaluate --split train --config <file>`.
  Every option is in `config.py`. Invalid values raise `ConfigError` with the field name.
* **Add a new option:** add the field to the right dataclass in `config.py` with a default that keeps current behaviour. Read it in
  `retrieval.py` / `ranking.py`. Add a unit test. Old configs keep their hash only if the default is a no-op.
* **Add a query-expansion rule:** append a `(triggers, terms)` tuple to `lexicon.RULES`. Write it from contract vocabulary,
  never from gold spans.
* **Add a loop hypothesis:** see [improvement_loop.md §6](improvement_loop.md#6-adding-a-hypothesis).
* **Check nothing broke:** run `.venv/Scripts/python -m pytest` (208 tests, about 7 s; 3 need `data/raw`).
