# ruff: noqa  -- throw-away pre-screen script, kept as run for provenance (see README.md here)
# TRAIN-ONLY prototype of alternative abstention signals.
import math

from contract_intel.config import load_config
from contract_intel.data import load_train_queries
from contract_intel.indexing import strip_query_context
from contract_intel.retrieval import ContractSearcher

cfg = load_config("experiments/run5/final_run5.yaml")
p = cfg.paths
qs = load_train_queries(p.benchmark_file, p.corpus_root, p.unanswerable_file)
s = ContractSearcher.from_config(cfg)
rows = []
for q in qs:
    raw = s.search_raw(q.text)
    if not raw.passages:
        rows.append((q.answerable, raw.doc_match, {}, q.id)); continue
    qt = set(s.tokenize(strip_query_context(q.text)))
    w = {t: s.index.idf.get(t, max(s.index.idf.values())) for t in qt}
    tot = sum(w.values()) or 1
    sig = {"peak": raw.top1}
    for name, ps in (("cov1", raw.passages[:1]), ("cov3", raw.passages[:3])):
        best = 0
        for pa in ps:
            pt = set(s.tokenize(pa.text))
            best = max(best, sum(v for t, v in w.items() if t in pt) / tot)
        sig[name] = best
    sig["peak*cov1"] = sig["peak"] * sig["cov1"]
    sig["peak*sqrtcov1"] = sig["peak"] * math.sqrt(sig["cov1"])
    sig["peak*cov3"] = sig["peak"] * sig["cov3"]
    sig["min"] = min(sig["peak"], sig["cov1"])
    rows.append((q.answerable, raw.doc_match, sig, q.id))

ans = [r for r in rows if r[0]]; un = [r for r in rows if not r[0]]
for name in ["peak", "cov1", "cov3", "peak*cov1", "peak*sqrtcov1", "peak*cov3", "min"]:
    best = None
    for i in range(1001):
        th = i / 1000
        ab = lambda r: r[1] < 0.59 or not r[2] or r[2][name] < th
        cov = sum(not ab(r) for r in ans) / len(ans)
        if cov >= 0.90:
            best = (th, cov, sum(ab(r) for r in un))
    print(f"{name:14s} threshold={best[0]:.3f} coverage={best[1]:.3f} correct_abstain={best[2]}/{len(un)}")
