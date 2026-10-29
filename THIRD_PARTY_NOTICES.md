# Third-party notices and provenance

| Component | Source | License | How it is used |
|---|---|---|---|
| LegalBench-RAG data export (benchmark JSON + corpus) | https://huggingface.co/datasets/awinml/legalbench-rag (byte-exact copy of the LegalBench-RAG release; Pipitone & Alami, 2024) | Each part keeps its source license; the CUAD portion is **CC BY 4.0** (CUAD v1, Hendrycks et al., Zenodo 4595826) per the dataset's own `LICENSE` | Downloaded by `contract-intel setup` into git-ignored `data/raw/`. Not redistributed. The committed manifest holds only query text, split labels, file paths and offsets. Attribution: CUAD (The Atticus Project), LegalBench-RAG (ZeroEntropy). |
| Official LegalBench-RAG code | https://github.com/zeroentropy-ai/legalbenchrag (`legalbenchrag/run_benchmark.py`) | MIT, (c) 2025 ZeroEntropy | Read only, to confirm the character precision/recall convention. Metrics re-implemented from the definition; no code copied. |
| `legal-retrieval-benchmark` | https://github.com/Akshitha024/legal-retrieval-benchmark (cloned locally, git-ignored) | MIT, (c) 2026 Akshitha Reddy Lingampally | Audited only (see IMPLEMENTATION_PLAN.md); no code copied. Its BM25 / RRF ideas informed the design. |
| PyYAML | PyPI | MIT | Config parsing. |
| pytest, ruff | PyPI | MIT | Dev tooling. |
| Streamlit | PyPI | Apache-2.0 | Web UI. |
| huggingface_hub | PyPI | Apache-2.0 | Dataset download in `setup`. |

The query-expansion lexicon (`src/contract_intel/lexicon.py`) is original work, written from general contract-drafting knowledge and
public CUAD category names.
