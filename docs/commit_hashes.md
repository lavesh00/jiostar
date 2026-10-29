# Commit hash map

The commit metadata (author account, message trailers) was normalised before publishing. This changed every commit hash, but not the
file content, date or commit subject of any commit. The order of events (unanswerable set frozen first, each objective revision before its run,
config frozen before each test run) is the same.

Experiment logs (`experiments/iterations.jsonl`, `experiments/*/baseline*.json`) and the frozen
`docs/optimization_objective.md` were written at the time and still cite the **old** short hashes. They are left unedited on purpose.
Use this table to find the matching commit.

| Old | New | Commit |
|---|---|---|
| `6482d34` | `6f8d1d4` | Scaffold contract_intel: config, exact mini selection/split/offset validation, manifest, audit plan |
| `f5b73f4` | `1f25e0c` | Freeze unanswerable set (39 queries) and optimization objective before any tuning |
| `df7ca4a` | `058b1c0` | Objective v1.1: phase A/B split and relative precision guardrail |
| `12e8f8d` | `fbc2ee2` | Add chunking, lexical indexes, retrieval service, abstention, metrics, evaluation, diagnostics, optimizer |
| `e890270` | `0343d8b` | Fix negative-gap abstention bug; query expansion lexicon; Run 1; objective rev 1.2 |
| `6f3b9b5` | `67ee3a6` | Abstention signal fixes; objective rev 1.3; Run 2 |
| `96d458e` | `35e589b` | doc_peakedness abstention signal; objective rev 1.4; Run 3 partial |
| `3d8d621` | `48eb4f9` | Freeze final config from Run 4 (tag `config-frozen`) |
| `a9e8771` | `01b4919` | CLI, service layer, Streamlit app, tests |
| `72ece3f` | `0a83a1b` | Final test evaluation (2 logged runs), README, REPORT |
| `8872e7a` | `a60e46f` | REPORT wording fixes |
| `1215cfd` | `197431f` | Record fresh-clone test check in REPORT |
| `972c7c2` | `cf94e59` | Objective rev 1.5; route filter, gate, span refinement; out-of-corpus probe |
| `14f97ba` | `42b20c8` | Run 5: freeze final config `78a53ef0d1aa` (tag `config-frozen-v2`) |
| `6040ac9` | `ec05276` | Run 5 evaluation (test run 3), README/REPORT rewrite |
