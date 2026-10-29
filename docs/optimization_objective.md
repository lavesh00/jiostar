# Optimization objective and guardrails

**Revision 1.1.** Revision 1.0 (commit `f5b73f4`, committed with the frozen unanswerable set) was written before
any retrieval code existed. After the retrieval baseline could run, a smoke run on **train queries only** (no
candidate evaluation, no optimizer run) showed two flaws in v1.0, so it is corrected here **before any
optimization iteration was executed**, as the brief asks ("choose guardrails based on baseline behaviour"):

1. v1.0 required `correct abstention >= 0.50` for every candidate, but the baseline (abstention disabled) can never
   satisfy that, so *every* structural candidate would be rejected for reasons unrelated to what it changes.
   Abstention is therefore optimised in its own phase.
2. v1.0 used an absolute Precision@5 tolerance of 0.02, but baseline P@5 is only ~0.019, so that guardrail was vacuous.
   It is now relative.

The unanswerable set and all metric code are untouched. This file is not edited by the loop.

## Data used by the loop
Only the **train** split (`assign_split(query) == "train"`): 140 answerable queries (LegalBench-RAG-mini, CUAD) and the
33 train unanswerable queries. `optimize` imports only `load_train_queries`; it cannot load the test split.

## Phase A — structure and lexical retrieval (hypotheses 1-7)
* **Objective:** maximise macro character **Recall@5** on train answerable queries with abstention disabled
  (threshold 0; the only abstentions are "no term matched").
* **Guardrails** (B = baseline, evaluated on the same train examples with the same code):
  * G1: Precision@5 >= 0.90 x B_precision@5 (do not return materially more noise than the baseline).
  * G4: mean search latency <= min(5 x B_latency, 500 ms).
  * G5: number of chunks <= 10 x B_chunks (blocks degenerate tiny-chunk blow-ups).
* **Acceptance:** all guardrails pass **and** Recall@5 improves on the incumbent by more than **0.002** absolute.

## Phase B — abstention (hypotheses 8-9)
* **Threshold rule R (train only):** for a given configuration, the abstention threshold on the normalised top
  confidence is the *highest* value on the grid 0.000-0.950 (step 0.005) such that answerable **coverage >= 0.90**
  (G2). For the gap policy the same rule is applied for each `min_gap` in {0, 0.05, ..., 0.5} and the best
  correct-abstention rate wins (ties -> smaller `min_gap`).
* **Objective:** maximise the **correct-abstention rate** on train unanswerable queries at the rule-R threshold.
* **Guardrails:** G2 coverage >= 0.90 (by construction); retrieval-only Recall@5 and Precision@5 not lower than the
  incumbent's by more than 0.002 absolute; G4 as above.
* **Acceptance:** guardrails pass **and** correct-abstention improves by more than **0.02** absolute (at least one
  more of the 33 train unanswerable queries correctly refused).
* After the last iteration the incumbent's threshold is re-fitted with rule R and written to `configs/final.yaml`
  (logged as a `finalize` row).

## Diagnosis-gated hypotheses
Hypothesis 6 (document prior) is only attempted if "wrong_document" failures reach >= 10% of train answerable
queries at k=5 under the incumbent; otherwise it is logged as `skipped` (not counted as an iteration).

## Things the loop may not do
* load or score the test split, or pass gold spans/answers to retrieval, ranking or abstention;
* edit the unanswerable set, this document, or the metric implementation.

## Overfitting caveat
Train has 140 answerable queries over 29 contracts and the loop selects among a small fixed set of hypotheses with
a few dozen evaluated grid points; train gains may not transfer. The held-out test split (54 answerable, 6
unanswerable queries) is evaluated once after the configuration is frozen. The 6 test unanswerable queries make
the abstention estimate on test very coarse; raw counts are reported.

## Revision 1.2 — hypotheses list for Run 2 (objective, guardrails and thresholds above are unchanged)
Run 1 (`run_id 20261029T175554Z`, rows kept in `experiments/iterations.jsonl`, summary in `experiments/run1_superseded/`)
executed the original 9 hypotheses on train. It exposed (a) a **bug**: with the gap policy the second passage's topical
confidence could exceed the first's, giving a negative gap and spurious abstention in retrieval-only metrics (fixed by
clipping the gap at 0); and (b) that stripping the "Consider the <contract>;" prefix alone destroyed routing to the
right contract (diagnosis: 65-72% of failures were already "right contract, wrong section"). Run 2 therefore
re-runs the loop from the unchanged baseline with an updated hypothesis list, still train-only:
`chunk_strategy, chunk_size, overlap_boundaries, query_processing (strip + document prior), query_expansion
(hand-written lexicon), bm25_params, rrf_fusion` (phase A) and `topical_confidence, confidence_strip_context,
gap_policy` (phase B). Both runs are reported; Run 2 produces `configs/final.yaml`.

## Revision 1.3 — abstention-signal fixes before Run 3 (objective, guardrails, thresholds unchanged)
Run 2 (`20261029T175827Z`) showed every phase-B candidate rejected, and inspection found two defects in the
*signal*, not in the declared objective: (1) `decide()` also abstained when the top confidence was exactly 0 even
with threshold 0, so "retrieval-only" metrics were not abstention-free once topical confidence was enabled
(recall fell for reasons unrelated to chunking); (2) query terms that occur nowhere in the indexed corpus were
dropped from the confidence normaliser, so a question about an absent concept (e.g. "arbitrated") looked as well
supported as a present one. Fixes: abstain on zero candidates only, and count unseen query terms with maximal idf
in the normaliser. Run 3 re-runs the same 10 hypotheses from the unchanged baseline; Runs 1-2 stay in the log.

## Revision 1.4 — Run 4 (final run): new phase-B hypothesis (objective, guardrails, thresholds unchanged)
Run 3 (`20261029T180030Z`) crashed in the last iteration (division by zero in the gap policy when the top confidence
is 0 -- a bug, fixed) and had already shown phase-B gains of at most 2/33 queries. A train-only diagnostic of the
confidence signal (AUC of answerable vs unanswerable top-1 confidence: normalised BM25 0.21-0.56, i.e. dominated by
query form; peakedness of literal question terms within the routed contract ~0.9) motivated a new, config-gated
signal `abstention.signal: doc_peakedness`. Run 4 replaces the former `confidence_strip_context` hypothesis with
`confidence_signal` (peakedness, with topical filters 0/3/5/8); the other hypotheses are unchanged. Run 4 is the run
that produces `configs/final.yaml`. Runs 1-3 remain in `experiments/iterations.jsonl` (Run 3 has no `finalize` row
because it crashed).
Caveat: the answerable queries are templated ("Consider the <contract>; <question>") whereas the unanswerable ones
are free-form, so any signal can partly exploit form differences; the peakedness signal always strips the
"Consider ..." prefix to limit this, but the effect cannot be ruled out and is reported as a limitation.
