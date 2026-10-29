# Contract Intelligence Assistant — evaluation report

Every number here is read from a file under `experiments/` produced by the commands in §9. Hypotheses and
measurements are labelled as such. Reading time: about 15 minutes.

## 1. Summary

I built an offline clause-retrieval system for the 29 CUAD contracts behind the 194-query LegalBench-RAG-mini set. It uses
BM25 + TF-IDF with reciprocal-rank fusion, document routing and a small legal query-expansion lexicon, with no LLM. It returns ranked passages
with exact character offsets, or abstains with a stated reason. It has a CLI and a Streamlit app that share one service
layer, a train-only self-improvement loop (5 logged runs, 61 log rows), and 203 tests. **On the held-out test split
(54 answerable + 6 unanswerable queries), the final configuration reaches character Recall@5 0.798 and Precision@5 0.061, versus
0.136 / 0.0085 for the baseline.** It answers 92.6% of answerable test queries (4 false abstentions) and correctly refuses 2 of the 6
unanswerable test queries. It also refuses all 60 questions about CUAD contracts that are *not* indexed, up from 0/60 before the
contract gate was added. The weak points are precision (passages are about 4× longer than the gold clauses) and abstention on
questions about contracts that *are* indexed (2/6 on test, 24/33 on train).

## 2. System design

```
manifest (29 contracts) -> original text (read_text, utf-8; never modified)
  -> fixed 1500-char chunks as [start, end) offsets -> BM25 + TF-IDF inverted index
query -> strip "Consider the <contract>;" -> stem/stopwords -> lexicon expansion
      -> document prior from the full query (which contract?) -> keep contracts with prior >= 0.8 x best
      -> BM25 & TF-IDF rankings fused by RRF (+ document prior) -> overlap suppression -> top-k
      -> abstain if contract match < 0.59 or peakedness confidence < 0.66 -> passages with text == doc[start:end]
```

| Decision | Final choice | Alternatives tried (all measured on train) |
|---|---|---|
| Chunking | fixed 1500-char windows, no overlap | paragraph- and section-aware packing (worse recall *and* precision); 300–2500 chars; overlap 100/200 and sentence snapping (worse); 500–1000 chars after the ranker improved (higher precision, but recall fell beyond the guardrail; §5) |
| Ranking | BM25 (k1 1.2, b 0.75) + TF-IDF fused by RRF (k = 60) | 11 k1/b settings (no gain); BM25 alone |
| Query processing | descriptor stripped for chunk scoring, kept for a document prior (weight 1.0); stemming, stopwords, expansion lexicon | stripping without the prior (recall collapsed: routing lost) |
| Routing | keep only contracts whose prior ≥ 0.8 × best | no filter (top-5 lists mixed in unrelated contracts) |
| Span narrowing | none | relative score cut-off (no effect); sentence-window refinement to ≤ 300/500/800 chars (precision 0.15–0.20, recall dropped 0.20–0.25) |
| Abstention | contract gate (doc prior ≥ 0.59) + "peakedness" ≥ 0.66: how far the best literal match of the question stands above the contract's median chunk | normalised BM25 score (dominated by query length, AUC 0.21–0.56 on train), topical-term filtering, top-1/top-2 gap |
| Contract choice | optional explicit contract (`--contract`, dropdown in the app) bypasses routing and the gate | — |

**Offsets** are exact by construction. Chunks are intervals of the untouched text; trimming only moves bounds inward. Tests check
`doc[start:end] == text` for every option, Unicode, CRLF files and empty/whitespace documents.

**LLM use: none.** Contracts are confidential, and an offline lexical system needs no API keys, has no per-query cost and needs no
cache to reproduce. It also gave large measured gains without an LLM. Next steps would be a local reranker or query rewriter
(e.g. a small model via Ollama) compared against this pipeline on train; I did not try one. Claude Code was used to develop the repository (README).

## 3. Evaluation setup

* **Data.** `mini()` and `assign_split()` are implemented verbatim. 194 queries, 272 gold spans, 29 indexed contracts; all 272 spans checked
  `document[start:end] == answer`: **0 mismatches, 0 overlapping gold pairs, 0 skipped**. Split: **140 train / 54 test**.
* **Unanswerable set.** 39 hand-written questions about the indexed contracts (`data/unanswerable_queries.jsonl`), committed in `1f25e0c` before any
  retrieval code or loop run and never edited (a test pins its sha-256). `assign_split` put **33 in train and 6 in test** (15%, not the third the
  function usually gives). I did not rebalance them, because the brief requires the same function and no edits.
* **Out-of-corpus probe.** 60 CUAD benchmark questions about contracts *outside* the index (one per contract, chosen deterministically by
  hash; 40 train / 20 test by `assign_split`), in `data/out_of_corpus_probe.jsonl`. It was built after Run 4, is used only for reporting
  and is never used by the loop.
* **Retrieval metrics.** Character precision/recall@k for k = 1, 3, 5, following the LegalBench-RAG definition and checked against the
  authors' `run_benchmark.py`. A character counts as relevant if it lies in a gold span of the same file. Precision is 0 when nothing is returned.
  Scores are macro-averaged over queries. One documented deviation: retrieved and gold intervals are merged before counting so
  overlaps are never double counted; this matches the reference whenever results don't overlap. Abstained answerable queries score 0
  and stay in the denominator. Supplementary metrics: document hit@5, span hit@5, MRR.
* **Abstention.** Coverage = answered / answerable. False-abstention rate = 1 − coverage. Correct abstention = abstained / unanswerable.
  False acceptance = answered / unanswerable. Raw counts are always shown next to the rates.
* **Test-split usage: 3 evaluations in total** (`experiments/final_test_runs.jsonl`). Run 1: Run-4 final config. Run 2: baseline. Run 3: the
  Run-5 final config reported here, evaluated after the freeze tag `config-frozen-v2`. No test number influenced a configuration
  decision. However, Run 5's precision phase was motivated by the train-split diagnosis after test runs 1–2 existed, so I disclose
  that I had seen test numbers before Run 5.

## 4. Results

| Metric (macro) | Baseline train | Final train | Baseline test | Final test |
|---|---|---|---|---|
| Answerable / unanswerable queries | 140 / 33 | 140 / 33 | 54 / 6 | 54 / 6 |
| Recall@1 · Precision@1 | 0.014 · 0.004 | 0.478 · 0.137 | 0.019 · 0.003 | **0.535 · 0.146** |
| Recall@3 · Precision@3 | 0.068 · 0.013 | 0.699 · 0.078 | 0.088 · 0.009 | **0.697 · 0.082** |
| Recall@5 · Precision@5 | 0.190 · 0.019 | 0.750 · 0.052 | 0.136 · 0.009 | **0.798 · 0.061** |
| Micro Recall@5 · Precision@5 | 0.164 · 0.019 | 0.702 · 0.058 | 0.071 · 0.009 | 0.776 · 0.066 |
| Document hit@5 · span hit@5 | 1.00 · 0.28 | 0.90 · 0.80 | 1.00 · 0.19 | 0.93 · 0.85 |
| Coverage (answered/answerable) | 140/140 | 126/140 = 0.900 | 54/54 | 50/54 = 0.926 |
| Correct abstention (unanswerable) | 0/33 | **24/33 = 0.727** | 0/6 | **2/6 = 0.333** |
| False acceptance (unanswerable) | 33/33 | 9/33 | 6/6 | 4/6 |
| Correct abstention, out-of-corpus probe | 0/60 | 40/40 (train part) | 0/60 | 20/20 (test part) |
| Mean search latency · index build | 2.9 ms · 0.2 s | 1.2 ms · 0.5 s | 3.2 ms · 0.2 s | 1.2 ms · 0.5 s |

Final-config train Recall@5 *without* abstention is 0.823, which is what the loop optimised; the 14 false abstentions bring it to 0.750.
The Run-4 configuration (the previous final, before route filter and gate) scored 0.655 / 0.049 Recall@5 / Precision@5 on test and 0/60 on the probe.

**Train–test gap.**
* **Retrieval:** test is *not* worse than train (Recall@5 0.798 vs 0.750, Precision@5 0.061 vs 0.052). The lexicon and every
  accepted change were chosen on train. With 54 test queries, one standard error of Recall@5 is roughly ±0.05, so the difference is
  within noise. The likely explanation is that the test queries use the same ~30 CUAD question templates, so the train-tuned lexicon
  transfers. That would not hold for differently worded questions.
* **Coverage:** coverage is higher on test (0.926 vs 0.900). That is expected: the threshold was set to land exactly on 0.90 on train.
* **Abstention on the frozen unanswerable set: this is where the gap is real** (0.727 on train → 0.333 on test, n = 6). Two causes are measurable:
  * 11 of the 24 correct train abstentions come from the contract gate. It fires because those free-form questions name their
    contract less fully than the benchmark's "Consider the …" descriptor; none of the 6 test questions trip it.
  * The confidence threshold (0.66) sits close to the confidence of several unanswerable questions (0.66–0.70, §6 E).

  With n = 6 the test estimate is very uncertain either way. I would not claim better than roughly 30–50% correct abstention on in-corpus
  unanswerable questions.
* **Out-of-corpus questions:** 60/60 correct on both splits, at a cost of 1 of 140 train answerable queries rejected by the gate (budget fixed at 1%).

## 5. Iteration history

**Objective (`docs/optimization_objective.md`).**
* Rev 1.0 was committed with the frozen unanswerable set. Revisions 1.1–1.5 were each committed *before* the run they governed, and
  each states its reason.
* **Phase A (recall):** maximise character Recall@5 on train answerable queries with abstention off. Guardrails: Precision@5 ≥ 0.9 × baseline,
  latency ≤ min(5 × baseline, 500 ms), chunks ≤ 10 × baseline. Accept if the gain is > 0.002.
* **Phase C (precision, rev 1.5):** maximise Precision@5 while Recall@5 and Recall@1 each lose ≤ 0.02. Accept if the gain is > 0.002.
* **Phase B (abstention):** maximise correct abstention on the 33 train unanswerable queries at the threshold that keeps coverage ≥ 0.90,
  without losing retrieval quality. Accept if the gain is > 0.02 (one query).
* **Contract gate (rev 1.5):** calibrated to reject ≤ 1% of train answerable queries; it uses no unanswerable labels.

*Why recall first:* a reviewer can skim a slightly long passage, but cannot read a clause that was never retrieved. Precision is still
guarded because flooding the reviewer with text defeats the purpose.

**How the loop works.** `contract-intel optimize` (about 50 s). Each iteration:
1. Diagnose the incumbent on train (wrong document / wrong section / partial coverage / overly broad / false abstention / false acceptance).
2. Skip hypotheses whose trigger condition is not met; skips are logged.
3. Evaluate a small grid for one hypothesis.
4. Pick the best guardrail-passing point.
5. Accept it or reject it by the rule above.
6. Append a row to `experiments/iterations.jsonl` with the change, rationale, proposer, metrics before and after, guardrails, the full grid and the decision.

Proposals come from a fixed hypothesis list (grid search), two of which are gated by diagnosis rules. The loop imports only a
train-only loader; a test asserts it never requests the test split.

**Run 5: the run that produced `configs/final.yaml`** (14 iterations; 6 accepted, 8 rejected)

| # | Phase | Hypothesis (why) | Best grid point | Objective before → after | Decision |
|---|---|---|---|---|---|
| 1 | A | structure-aware chunks (clauses are paragraph-shaped) | section | R@5 0.190 → 0.105 | rejected (worse; P@5 below guardrail) |
| 2 | A | chunk size (gold ≈ 400 chars) | 1500 | 0.190 → 0.268 | **accepted** (2500 gave 0.395 but P@5 0.0166 < 0.0167 limit) |
| 3 | A | overlap / boundary snapping (clauses cut in half) | overlap 100 | 0.268 → 0.261 | rejected |
| 4 | A | strip descriptor + document prior + stemming (67% "right contract, wrong section") | prior 1.0 + stem | 0.268 → 0.394 | **accepted** |
| 5 | A | legal query-expansion lexicon | + stopwords | 0.394 → 0.717 | **accepted** |
| 6 | A | BM25 k1/b | k1 2.0, b 0.5 | 0.7169 → 0.7170 | rejected (< 0.002) |
| 7 | A | RRF with TF-IDF | k = 60 | 0.7169 → 0.7195 | **accepted** (marginal, +0.0026) |
| 8 | C | route filter (top-5 mixed in other contracts) | ratio 0.8 | P@5 0.0498 → 0.0582 (R@5 0.72 → 0.82) | **accepted** |
| 9 | C | smaller chunks again, now that the ranker is stronger | 500 | P@5 0.058 → 0.129, but R@5 0.82 → 0.70 | rejected (recall guardrail) |
| 10 | C | relative score cut-off | 0.5 | 0.0582 → 0.0582 | rejected (no effect: top-5 scores are close) |
| 11 | C | sentence-window refinement (84% "overly broad") | 300 chars | P@5 0.058 → 0.203, but R@5 0.82 → 0.58 | rejected (recall guardrail) |
| 12 | B | topical-term confidence | min 3 contracts | correct abstention 0.333 → 0.333 | rejected |
| 13 | B | peakedness confidence | — | 0.333 → **0.727** | **accepted** |
| 14 | B | top-1/top-2 gap policy | 0.05 | 0.727 → 0.727 | rejected |

Measured precision/recall trade-off from the rejected Phase C grids (train, abstention off):

| Variant | Precision@5 | Recall@5 |
|---|---|---|
| **final (1500-char chunks + route filter)** | **0.058** | **0.823** |
| 1000-char chunks | 0.081 | 0.790 |
| 700-char chunks | 0.106 | 0.774 |
| 500-char chunks | 0.129 | 0.699 |
| refine to ≤ 800 chars | 0.155 | 0.622 |
| refine to ≤ 300 chars | 0.203 | 0.575 |

A deployment that values precision could pick 700 or 1000 chars. I kept the pre-declared rule rather than relax it after seeing these results.

**Earlier runs (in the same log; the code changed between runs, so they are not independent trials):**
* *Run 1:* first 9 hypotheses. It exposed a negative-gap bug, and showed that stripping the descriptor alone destroys routing.
* *Run 2:* the bug fixed and the document prior added; every abstention candidate failed. I traced this to two flaws in the signal:
  it abstained on zero confidence even with threshold 0, and it ignored unseen query words.
* *Run 3:* crashed on a division-by-zero bug, since fixed.
* *Run 4:* added the peakedness signal. This was the previous final config (test: R@5 0.655, P@5 0.049, 3/6 correct abstention, 0/60 out-of-corpus).
* *Run 5:* added Phase C and the gate, as above.

I changed code and hypotheses after seeing train results four times, so the train numbers carry more researcher freedom than a
single pre-registered pass would. The test split remains the honest check.

## 6. Error analysis (final config, `experiments/results/final_*_per_query.jsonl`)

Categories at k = 5:

| Split | Overly broad | Wrong section | Partial coverage | Wrong document | False abstention | Success | False acceptance |
|---|---|---|---|---|---|---|---|
| Train (140 answerable / 33 unanswerable) | 108 | 14 | 3 | 0 | 14 | 1 | 9 / 33 |
| Test (54 / 6) | 42 | 4 | 2 | 0 | 4 | 2 | 4 / 6 |

"Success" requires recall ≥ 0.5 *and* precision ≥ 0.25, which a 1500-character passage rarely meets. Routing errors ("wrong document")
dropped to zero once the route filter was in place.

**A. Right contract, wrong section (3)**
1. `cuad-3d4d98175e` (test), Dynamex: "exclusivity agreement?". The gold sentence ("Dynamex agrees not to provide sameday delivery services
   for any other provider …") contains no exclusivity vocabulary; the system returned an insurance clause. *Diagnosis:* a restrictive
   covenant phrased without the expected words; lexical matching cannot connect them.
2. `cuad-1a6adb98b6` (train), Theravance: "covenant not to sue?". The gold text concerns termination on reconstruction or amalgamation; the system returned
   a confidentiality chunk. *Diagnosis:* there is no lexical bridge between question and gold, and the annotation looks loose.
3. `cuad-4efcc74d1f` (train), PC Quote: "licenses granted?". Gold: "PCQ agrees to allow ABW to co-brand the PCQ SOFTWARE …" (permission
   wording, no "license"); the system returned the indemnity section. *Diagnosis:* the grant is phrased as "allow", which the lexicon does not map.

**B. Partial coverage (2)**
4. `cuad-5938744c36` (test), King Pharmaceuticals: "services after termination?". The gold is an audit right that survives
   termination (§ cross-reference). The system retrieved the post-termination payment article, giving R@5 0.11. *Diagnosis:* the answer is
   spread across survival clauses in different articles.
5. `cuad-03914e2684` (train), BicycleTX IP assignment. Gold 44840–45359; the chunk 43501–45000 ends mid-clause, giving R@5 0.31. *Diagnosis:* fixed
   windows cut the clause; overlap was tested in iteration 3 and lost on average.

**C. Overly broad (2)**
6. `cuad-ec6283c4e4` (test), BicycleTX "expiration date": the right 166-character sentence sits inside chunk 0–1500 (recall 1.0, P@5 0.022).
7. `cuad-712e68f706` (train), BicycleTX governing law: the gold sits inside chunk 54000–55500 (recall 1.0, P@5 0.039). *Diagnosis for both:* this is
   the dominant failure. Refinement fixes it for these two, but over the whole train set it costs recall (§5 table).

**D. False abstention (3)**
8. `cuad-f5fd1dec27` (test), Cano governing law. §11 "Governing Law" exists, but confidence 0.640 < 0.66. *Diagnosis:* in an 8.8k-character
   contract the peak over the median is modest; the threshold is a single global value.
9. `cuad-5f26f5e6ab` (train), Glu amendment "expiration date" (confidence 0.227). The text says "shall expire", and the crude stemmer maps
   "expiration" and "expire" to different stems.
10. `cuad-c3e0e0f9d2` (train), BicycleTX right-of-first-refusal: rejected by the contract gate (match < 0.59). This is the 1% budgeted
    gate loss; the gold (a share-option grant) also looks loosely annotated.

**E. False acceptance (4)**. In every case the top passage is the contract's opening block: the free-form question names the parties,
which match the title and parties block, so the confidence peak is real but about the wrong thing.

11. `unans-002` (test): liability insurance for Boyd/Cano (0.660, exactly at the threshold).
12. `unans-020` (test): ARC Group assigning sponsorship to a franchisee (0.696).
13. `unans-003` (train): Cano option to renew for 2009 (0.712). The heading "Term and Renewal" also matches "renew".
14. `unans-023` (train): Flotek arbitration (0.681).

*Fix hypothesis, not implemented:* drop party-name tokens from the peakedness question (topical filtering did this in iteration 12, but it did not help).

## 7. Limitations and next steps

* **Precision.** Five 1500-character passages against roughly 400-character gold clauses. A clause-level re-segmentation or reranker that keeps recall is
  the main open problem (the §5 table shows the trade-off curve). A cross-encoder or local-LLM reranker is the obvious stretch experiment.
* **Cross-references and definitions.** 41 of 140 train and 22 of 54 test queries have gold text that refers elsewhere ("Section …", "as defined", "hereof"); nothing
  resolves these. Some contracts incorporate documents that are not indexed (leases, ordinances, share purchase agreements).
* **Query form.** Every benchmark query names its contract and uses one of about 30 CUAD templates. The expansion lexicon and the gate rely on this. For
  free-form questions, use the contract selector. A question that names no contract still passes the gate and searches every contract
  (verified in the app); it should instead ask the user to choose a contract.
* **Abstention.**
  * A single global threshold, fitted on train at 90% coverage.
  * The unanswerable questions are free-form while the answerable ones are templated, so part of the measured abstention may reflect query
    form; the gate in particular does.
  * Only 6 test unanswerable questions.
  * The gate's reason text was reworded after the per-query files were written ("does not match any indexed contract well" → "could not be matched
    confidently …"); the decisions themselves are unchanged.
* **Unanswerable-set verification.** 17 items (Cano ×4, RGC ×3, Glu amendment ×2, Veoneer ×4, ALCO ×4) were verified by reading the whole contract and
  running regex absence checks. The other 22 were verified by regex absence checks plus reading the section structure and keyword contexts, not every line.
  Some rationales say "structural read" where that is all that was done. No lawyer has reviewed the set.
* **Stemmer.** A crude home-grown suffix stripper (failure 9); a proper lemmatiser is untested.
* **Before legal users rely on it:**
  * clause-level passages;
  * cross-reference resolution;
  * calibrated per-contract or per-question-type confidence;
  * evaluation on media-industry agreements (licensing, distribution) rather than CUAD exhibits;
  * review of results by lawyers;
  * an audit trail per answer.

  The output must stay presented as evidence to read, not as an answer.

## 8. Privacy and security

Everything runs locally; the only network access is the one-time dataset download. No contract text goes to any model or API.
Retrieved text is treated as data and never interpreted as instructions. The repository contains no secrets (`.env.example` has
placeholders only) and no raw corpus (`data/raw/` is git-ignored).

## 9. Reproducibility

```bash
pip install -e ".[data,app,dev]" && contract-intel setup                    # ~30 s download + validation
contract-intel evaluate --split train --config configs/baseline.yaml        # ~3 s each
contract-intel evaluate --split train --config configs/final.yaml
contract-intel optimize --config configs/baseline.yaml --iterations 14      # ~50 s; re-derives configs/final.yaml (hash 78a53ef0d1aa)
contract-intel evaluate --split test --final --config configs/final.yaml    # appends test run #4 to experiments/final_test_runs.jsonl
contract-intel probe --config configs/final.yaml                            # out-of-corpus probe
pytest                                                                      # 203 passed (~7 s; 3 need data/raw)
```
Environment: Python 3.12.0 on Windows 11. Core dependency: PyYAML. Hashes: manifest `e2a5024fd7e7ef0b`, final config `78a53ef0d1aa`. The git
history records the order of events: unanswerable set frozen (`1f25e0c`); each objective revision before its run; freeze tags `config-frozen`
(Run 4) and `config-frozen-v2` (Run 5); then the test runs. Experiment logs cite commit hashes from before the history's author
metadata was normalised; [docs/commit_hashes.md](docs/commit_hashes.md) maps them to the current ones.

## 10. Requirement checklist

| Requirement | Status | Evidence |
|---|---|---|
| Mini selection, split, offset check | Done | `data.py`; 194 / 140 / 54; 0 mismatches |
| Ranked passages with document and offsets; abstention with reason | Done | `retrieval.py`; tests |
| Web app and CLI | Done | `app.py` (AppTest exercised), `cli.py` (tests) |
| Configurable pipeline; ≥ 2 approaches compared | Done | `configs/*.yaml`; §5 |
| ≥ 30 unanswerable queries committed before the loop | Done | 39 queries, commit `1f25e0c` (33 train / 6 test) |
| Char P/R at k = 1, 3, 5; abstention both directions; train for development, test for reporting | Done | §3–4; 3 test runs |
| Baseline, declared objective, loop with diagnosis, ≥ 5 logged iterations | Done | §5; 14 iterations in Run 5, 61 log rows in total |
| Train–test gap discussed | Done | §4 |
| ≥ 10 failures grouped by cause | Done | 14 in §6 |
| Tests for core pieces | Done | 203 passing |
| Cross-references / clause-aware chunking (stretch) | Partly | clause-aware chunking tried and rejected; cross-references not resolved |
| LLM use (optional) | Not used | §2 |
