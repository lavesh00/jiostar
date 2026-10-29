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
