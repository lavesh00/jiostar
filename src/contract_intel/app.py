"""Streamlit UI: ``streamlit run src/contract_intel/app.py`` (uses the same service layer as the CLI)."""

from __future__ import annotations

import os
from typing import Any

from contract_intel.config import ConfigError
from contract_intel.service import SetupError, load_searcher


def run_query(query: str, config_path: str, top_k: int, searcher: Any = None) -> dict[str, Any]:
    """UI-independent core: returns the stable result dict or ``{"error": message}``."""
    try:
        s = searcher or load_searcher(config_path)
        if not query.strip():
            return {"error": "Please enter a question."}
        return s.search(query.strip(), top_k).to_dict()
    except (ConfigError, SetupError, FileNotFoundError, ValueError) as e:
        return {"error": str(e)}


def main() -> None:  # pragma: no cover - exercised manually / via streamlit AppTest
    import streamlit as st

    st.set_page_config(page_title="Contract Intelligence Assistant", layout="wide")
    st.title("Contract Intelligence Assistant")
    st.caption("Retrieves exact answer-bearing passages from the indexed CUAD contracts. "
               "Results are evidence for a human reviewer, not legal advice or a definitive answer.")
    config_path = st.sidebar.text_input("Config", os.environ.get("CONTRACT_INTEL_CONFIG", "configs/final.yaml"))
    top_k = st.sidebar.slider("Passages to show", 1, 10, 5)

    @st.cache_resource(show_spinner="Building index...")
    def cached(path: str) -> Any:
        return load_searcher(path)

    try:
        searcher = cached(config_path)
    except (ConfigError, SetupError, FileNotFoundError) as e:
        st.error(f"Cannot start: {e}")
        return
    cfg = searcher.cfg
    st.sidebar.markdown(f"**Config:** `{cfg.name}` (`{cfg.hash()}`)  \n**Contracts:** {len(searcher.docs)}  \n"
                        f"**Chunks:** {len(searcher.chunks)}  \n**Chunking:** {cfg.chunking.strategy}/{cfg.chunking.size}  \n"
                        f"**Abstention:** {cfg.abstention.signal} >= {cfg.abstention.threshold}")

    query = st.text_input("Ask a question about the contracts", placeholder="Consider the Services Agreement ...; What is the governing law?")
    if st.button("Search", type="primary"):
        res = run_query(query, config_path, top_k, searcher)
        if "error" in res:
            st.warning(res["error"])
            return
        meta = res["meta"]
        if res["abstained"]:
            st.info(f"**Abstained** - no passage is sufficiently relevant. Reason: {res['abstain_reason']} "
                    f"(top confidence {meta['top1_confidence']:.3f}). No passages are shown.")
            return
        st.success(f"{len(res['results'])} passages - {meta['elapsed_ms']:.1f} ms - top confidence {meta['top1_confidence']:.3f}")
        for p in res["results"]:
            with st.container(border=True):
                st.markdown(f"**#{p['rank']}** `{p['file_path']}`  \nchars **[{p['start']}:{p['end']}]** - score {p['score']:.3f} - confidence {p['confidence']:.3f}")
                st.text(p["text"])


if __name__ == "__main__":  # pragma: no cover
    main()
