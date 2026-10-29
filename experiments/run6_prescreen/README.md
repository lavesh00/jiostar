# Run 6 pre-screen (train split only)

Before objective revision 1.6 was written, two ideas for the abstention signal were checked with these throw-away scripts
against the Run 5 final config (`experiments/run5/final_run5.yaml`) on the **train** split. No test data was loaded.

| Script | Output | What it checks |
|---|---|---|
| `coverage_signals.py` | `coverage_signals.txt` | idf-weighted share of question words found in the top passage(s), alone and combined with peakedness. All variants were worse than peakedness (≤ 15/33 vs 24/33), so none went into the loop. |
| `opening_block_signals.py` | `opening_block_signals.txt` | peakedness with the contract's opening chunk left out (`skip_first`: 28/33), or with question words from the first N characters removed (`drop_headN`: 17–26/33). `skip_first` became the loop hypothesis `opening_block_confidence`. |

Each line shows the threshold that keeps coverage at 0.90 on train answerable queries (with the 0.59 contract gate), and the
correct abstentions it gives on the 33 train unanswerable queries. The loop itself fits its threshold on a coarser grid, so its
result for the accepted hypothesis (27/33, `experiments/iterations.jsonl`, Run 6 iteration 15) differs slightly from this check.

Run from the repository root: `python experiments/run6_prescreen/opening_block_signals.py` (needs `data/raw`).
