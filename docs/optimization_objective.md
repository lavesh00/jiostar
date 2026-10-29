# Optimization objective and guardrails (declared before any optimization run)

Status: **frozen**. Committed together with `data/unanswerable_queries.jsonl` before any retrieval code
or experiment existed. Any later change must be a new, separately logged revision; the loop never edits this file.

## Data used by the loop
Only the **train** split (`assign_split(query) == "train"`): 140 answerable queries (LegalBench-RAG-mini, CUAD)
and the train portion of the 39 frozen unanswerable queries. The test split is never loaded by `optimize`.

## Primary objective
Maximise **macro character Recall@5** over the train answerable queries
(official-style per-query char recall, averaged over queries; abstained queries score 0).

## Guardrails (all must hold for a candidate to be accepted)
The baseline is evaluated first; guardrails are expressed relative to it, so they are fixed *before*
looking at any candidate. B = baseline value on the same train examples.

| # | Guardrail | Rule | Why |
|---|-----------|------|-----|
| G1 | Macro char Precision@5 | candidate >= B_precision@5 - 0.02 | Precision is the cost of returning long/broad passages; allow tiny loss only. |
| G2 | Answerable coverage (1 - false-abstention rate) | candidate >= 0.90 | A system that abstains on >10% of answerable questions is unusable. |
| G3 | Correct abstention rate on train unanswerable | candidate >= max(B_correct_abstention, 0.50) | Abstention must not get worse; 0.50 is a floor so the baseline (no abstention, rate 0) is not an anchor. |
| G4 | Query latency | candidate mean search latency <= 5x baseline and <= 0.5 s | Keeps the tool interactive. |
| G5 | Index size | candidate chunk count <= 10x baseline | Prevents degenerate tiny-chunk/overlap blowups. |

The baseline ships with abstention **disabled** (threshold 0) so its metrics are pure retrieval quality;
G3 therefore requires an abstention threshold to be selected by the loop (see the threshold step).

## Acceptance rule
A candidate is **accepted** iff (a) all guardrails pass and (b) its primary objective exceeds the current
configuration's by more than **0.002** absolute (a minimum improvement to avoid accepting noise).
Otherwise it is **rejected**. Ties go to the incumbent. The accepted candidate becomes the incumbent
for later iterations. Rejected candidates are still logged with their metrics.

## Abstention threshold selection (train only)
After the structural/lexical iterations, the loop selects the abstention threshold on the normalized top score
by sweeping a fixed grid (0.00-0.60, step 0.01) on train data and picking the **largest-recall configuration that satisfies G2 and G3**;
ties are broken toward the lowest threshold with the highest correct-abstention rate. The threshold step
is treated as an ordinary iteration (proposed, evaluated, accepted/rejected under the same rule).

## Things the loop may not do
* load or score the test split, or any code path that accepts `split="test"`;
* pass gold spans/answers to retrieval, ranking or abstention;
* edit the unanswerable set, this document, or the metric implementation.

## Overfitting caveat
Train has 140 answerable queries over only 29 contracts, and the loop picks among a small fixed set of
hypotheses. Improvements on train may not transfer; the held-out test split (54 answerable queries) is the only unbiased estimate and is evaluated once after the configuration is frozen.
