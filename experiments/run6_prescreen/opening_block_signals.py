# ruff: noqa  -- throw-away pre-screen script, kept as run for provenance (see README.md here)
# TRAIN-ONLY prototype: peakedness variants that discount the contract's opening (title/parties) block.
import statistics

from contract_intel.config import load_config
from contract_intel.data import load_train_queries
from contract_intel.indexing import strip_query_context
from contract_intel.retrieval import ContractSearcher

cfg = load_config("experiments/run5/final_run5.yaml"); p = cfg.paths
qs = load_train_queries(p.benchmark_file, p.corpus_root, p.unanswerable_file)
s = ContractSearcher.from_config(cfg)
head_tokens = {fp: set(s.tokenize(s.docs.text(fp)[:N])) for N in [1] for fp in s.docs.paths()}

def peak(tokens, fp, skip_first=False):
    raw, _ = s.index.bm25(tokens)
    idxs = s.doc_chunks[fp][1:] if skip_first else s.doc_chunks[fp]
    vals = sorted(raw.get(i, 0.0) for i in s.doc_chunks[fp])
    pk = max(raw.get(i, 0.0) for i in idxs) if idxs else 0
    med = statistics.median(vals)
    return (pk - med) / pk if pk > 0 else 0.0

variants = {}
for N in (300, 600, 1000, 1500):
    variants[f"drop_head{N}"] = N
rows = []
for q in qs:
    raw = s.search_raw(q.text)
    if not raw.passages:
        rows.append((q.answerable, raw.doc_match, None)); continue
    fp = raw.passages[0].file_path
    toks = s.tokenize(strip_query_context(q.text))
    sig = {"peak": raw.top1, "skip_first": peak(toks, fp, True)}
    for name, N in variants.items():
        head = set(s.tokenize(s.docs.text(fp)[:N]))
        kept = [t for t in toks if t not in head] or toks
        sig[name] = peak(kept, fp)
    rows.append((q.answerable, raw.doc_match, sig))
ans = [r for r in rows if r[0]]; un = [r for r in rows if not r[0]]
for name in ["peak", "skip_first", *variants]:
    best = None
    for i in range(1001):
        th = i / 1000
        ab = lambda r: r[1] < 0.59 or r[2] is None or r[2][name] < th
        cov = sum(not ab(r) for r in ans) / len(ans)
        if cov >= 0.90: best = (th, cov, sum(ab(r) for r in un))
    print(f"{name:12s} th={best[0]:.3f} coverage={best[1]:.3f} correct_abstain={best[2]}/{len(un)}")
