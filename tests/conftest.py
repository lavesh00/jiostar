from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from contract_intel.config import Config, PathsConfig

FIX = Path(__file__).parent / "fixtures"
REAL_BENCH = ROOT / "data/raw/legalbench-rag/benchmarks/cuad.json"
REAL_CORPUS = ROOT / "data/raw/legalbench-rag/corpus"


def fixture_paths(tmp_path: Path) -> PathsConfig:
    return PathsConfig(
        corpus_root=str(FIX / "corpus"),
        benchmark_file=str(FIX / "benchmarks/cuad.json"),
        unanswerable_file=str(FIX / "unanswerable.jsonl"),
        manifest_file=str(FIX / "manifest.json"),
        cache_dir=str(tmp_path / "cache"),
        experiments_dir=str(tmp_path / "experiments"),
    )


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(name="fixture", paths=fixture_paths(tmp_path))


needs_data = pytest.mark.skipif(not REAL_BENCH.exists(), reason="LegalBench-RAG CUAD data not downloaded")
