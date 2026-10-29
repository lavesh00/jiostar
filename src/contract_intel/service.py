"""Thin application layer shared by the CLI and the Streamlit app (so both behave identically)."""

from __future__ import annotations

from pathlib import Path

from .config import Config, ConfigError, load_config
from .retrieval import ContractSearcher, SearchResponse, allowed_contract_paths


class SetupError(RuntimeError):
    """Dataset or index prerequisites are missing; the message says how to fix it."""


def load_searcher(config_path: str | Path = "configs/final.yaml", cfg: Config | None = None) -> ContractSearcher:
    """Build a searcher from a config file; raises ConfigError / SetupError with actionable messages."""
    cfg = cfg or load_config(config_path)
    manifest = Path(cfg.paths.manifest_file)
    if not manifest.exists():
        raise SetupError(f"manifest not found: {manifest}. Run `contract-intel setup` first.")
    root = Path(cfg.paths.corpus_root)
    missing = [fp for fp in allowed_contract_paths(manifest) if not (root / fp).exists()]
    if missing:
        raise SetupError(
            f"{len(missing)} indexed contract(s) missing under {root} (e.g. {missing[0]}). "
            "Run `contract-intel setup` to download the LegalBench-RAG CUAD files."
        )
    return ContractSearcher.from_config(cfg)


def search(query: str, config_path: str | Path = "configs/final.yaml", top_k: int | None = None,
           searcher: ContractSearcher | None = None) -> SearchResponse:
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    return (searcher or load_searcher(config_path)).search(query.strip(), top_k)


__all__ = ["ConfigError", "SetupError", "load_searcher", "search"]
