# Self-improvement loop

`contract-intel optimize --config configs/baseline.yaml --iterations 16` runs one command, in about 60 s. It starts from the
baseline, tries one hypothesis at a time on the **train split only**, keeps a change only if a rule declared in advance says it helped, logs
every attempt, and writes the winner to `configs/final.yaml`. The code is in `src/contract_intel/optimize.py`. The objective it
optimises is frozen in [optimization_objective.md](optimization_objective.md). The results of each run are in
[REPORT.md §5](../REPORT.md#5-iteration-history).

## 1. One iteration

```
             ┌───────────────────────────────────────────────────────────────┐
             ▼                                                               │
  incumbent config ─► evaluate on train (140 answerable + 33 unanswerable)   │
             │                                                               │
             ▼                                                               │
  diagnose: share of each failure category (diagnostics.py)                  │
             │                                                               │
             ▼                                                               │
  next hypothesis in HYPOTHESES ── trigger not met (< 10% of failures)? ──► log "skipped"
             │
             ▼
  evaluate every grid point (one config change each) on train
             │
             ▼
  check guardrails for each point; take the best point that passes them
             │
             ▼
  gain > minimum improvement and all guardrails pass?  ── yes ─► accept: candidate becomes the incumbent
             │ no                                                         │
             ▼                                                            │
         reject                                                           │
             └──────── append a row to experiments/iterations.jsonl ◄─────┘
```

After the last hypothesis, the abstention thresholds are fitted on train (rule R, §3), and the result is written to
`configs/final.yaml` together with a final `finalize` log row.

## 2. Diagnosis

`diagnostics.categorize` puts each train query in one category (first match wins), using gold spans **only after retrieval**:

| Category | Meaning | Typical fix to try |
|---|---|---|
| `false_abstention` | abstained on an answerable query | lower threshold, better confidence signal |
| `wrong_document` | no top-k passage from the gold contract | routing / document prior |
| `right_doc_wrong_section` | right contract, no gold character retrieved | query processing, expansion |
| `incomplete_coverage` | some gold characters, recall@5 < 0.5 | chunk size / overlap |
| `overly_broad` | recall@5 ≥ 0.5 but precision@5 < 0.25 | smaller chunks, span refinement |
| `success` | recall ≥ 0.5 and precision ≥ 0.25 | — |
| `correct_abstention` / `false_acceptance` | unanswerable queries | abstention signal |

Some hypotheses have a `trigger`: they run only if their category is at least 10% of failures. For example, span refinement only runs when
`overly_broad` is a real problem. This keeps the loop from spending iterations on problems it has not measured.

## 3. Objective, guardrails and acceptance

The loop has three phases. Each phase optimises one metric and protects the others:

| Phase | Objective (train) | Guardrails (all must pass) | Accept if gain > |
|---|---|---|---|
| A, recall | macro char Recall@5, abstention off | Precision@5 ≥ 0.9 × baseline; latency ≤ min(5 × baseline, 500 ms); chunks ≤ 10 × baseline | 0.002 |
| C, precision | macro char Precision@5, abstention off | Recall@5 and Recall@1 each drop ≤ 0.02 vs incumbent; latency; chunks | 0.002 |
| B, abstention | correct-abstention rate on the 33 train unanswerable queries at the rule-R threshold | fitted coverage ≥ 0.90; Recall@5 and Precision@5 drop ≤ 0.002; latency | 0.02 (one query) |

* **Rule R (threshold fit):** choose the highest confidence threshold that still answers ≥ 90% of train answerable queries.
* **Contract gate:** choose the largest `min_doc_prior` that rejects ≤ 1% of train answerable queries. It uses no unanswerable labels.
* **Why these rules:** recall comes first because a reviewer can skim a long passage but cannot read a clause that was never
  retrieved. The minimum improvements are about one query's worth, so the loop does not chase noise. The objective and
  every revision of it were committed *before* the run they governed (see the git history of `docs/optimization_objective.md`).

## 4. Hypotheses (proposer)

The proposer is a fixed, ordered list (`optimize.HYPOTHESES`). Each entry has a rationale and a small grid. Rows are logged with
`proposer: grid_search` or, when a diagnosis trigger gated them, `rule+grid_search`.

| Phase | Hypothesis | Grid |
|---|---|---|
| A | `chunk_strategy` structure-aware chunks | paragraph, section |
| A | `chunk_size` | 300, 500, 700, 1500, 2500 |
| A | `overlap_boundaries` | overlap 0/100/200 × snapping on/off |
| A | `query_processing` strip descriptor + document prior (+ stopwords / stem) | 4 variants, trigger `right_doc_wrong_section` |
| A | `query_expansion` legal lexicon | 2 variants, trigger `right_doc_wrong_section` |
| A | `bm25_params` | k1 ∈ {0.8, 1.2, 1.6, 2.0} × b ∈ {0.5, 0.75, 0.9} |
| A | `rrf_fusion` with TF-IDF | rrf_k 20, 60 |
| A | `stemmer` extended suffix folding (Run 6) | extended |
| C | `route_filter` | ratio 0.8, 0.9, 1.0 |
| C | `chunk_size_revisit` | 500, 700, 1000 |
| C | `score_cutoff` | 0.5, 0.7, 0.85 |
| C | `span_refinement` | ≤ 300 / 500 / 800 chars, trigger `overly_broad` |
| B | `topical_confidence` | min contracts 3–16 |
| B | `confidence_signal` peakedness | 4 variants |
| B | `opening_block_confidence` peak outside the opening chunk (Run 6) | on |
| B | `gap_policy` | min_gap 0.05–0.3 |

No LLM proposer is used: the system has no LLM, and a deterministic grid makes every run reproducible.

## 5. The log (`experiments/iterations.jsonl`)

Each line is one event: `baseline`, `iteration`, `skipped` or `finalize`. Iteration rows record:

| Field | Content |
|---|---|
| `run_id`, `timestamp`, `git_commit` | which run, when, and which code it ran (old hashes: see [commit_hashes.md](commit_hashes.md)) |
| `phase`, `hypothesis_name`, `hypothesis`, `proposer` | what was tried and why, and who proposed it |
| `diagnosis_before` / `diagnosis_after` | failure-category shares before and after |
| `changed_parameters`, `parent_config_hash`, `candidate_config_hash` | the exact change |
| `metrics_before` / `metrics_after`, `objective_before` / `objective_after` | the effect |
| `guardrails` | each guardrail's value, limit and pass/fail |
| `grid` | every grid point that was evaluated, not just the winner |
| `decision`, `decision_rationale` | accepted / rejected / skipped, with the rule that decided it |

The file has 79 rows across 6 runs. Run 6 produced the current `configs/final.yaml`: 16 iterations, 7 accepted, 9 rejected.
Each accepted or rejected candidate config is also saved in `experiments/results/<run_id>_iterNN_candidate.json`.

Quick look at a run:

```bash
python -c "import json; [print(r.get('iteration'), r.get('hypothesis_name'), r.get('decision'), r.get('objective_before'), r.get('objective_after')) for r in map(json.loads, open('experiments/iterations.jsonl'))]"
```

## 6. Adding a hypothesis

1. Add the option to `config.py` (default = current behaviour) and use it in the pipeline.
2. Append a `Hypothesis(name, phase, rationale, grid, trigger=None)` to `HYPOTHESES` in the position where it should run.
3. Run `contract-intel optimize --config configs/baseline.yaml --iterations <n>`. The new row appears in the log whether it
   is accepted or not.
4. Freeze the result (commit `configs/final.yaml`) **before** any test evaluation, then run
   `contract-intel evaluate --split test --final`, which is logged as a new test run.

## 7. Honesty controls

* The loop loads train data only (`load_train_queries`); a test checks this.
* Gold spans are used for scoring and diagnosis, never as input to the searcher.
* The unanswerable set and objective were committed before the first loop run, and the unanswerable set has not been edited since.
* Rejected and skipped attempts are logged with the same detail as accepted ones.
* Test evaluations are counted in `experiments/final_test_runs.jsonl` (4 so far), and the report discloses each one.
