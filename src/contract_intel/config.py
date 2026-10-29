"""Typed configuration: load from YAML, validate, hash."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

CHUNK_STRATEGIES = ("fixed", "paragraph", "section")
FUSIONS = ("none", "rrf")


class ConfigError(ValueError):
    """Raised for malformed or invalid configuration."""


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "fixed"
    size: int = 1000  # max chunk length in characters
    overlap: int = 0  # used by fixed windows and when splitting oversize blocks
    snap_boundaries: bool = False  # move window ends back to a whitespace/sentence boundary


@dataclass(frozen=True)
class RetrievalConfig:
    k1: float = 1.2
    b: float = 0.75
    lowercase: bool = True
    remove_stopwords: bool = False
    stem: bool = False  # light suffix stripping
    strip_query_context: bool = False  # drop a leading "Consider the <contract>;" clause
    expand_query: bool = False  # append hand-written legal lexicon terms (see lexicon.py)
    fusion: str = "none"  # "rrf": fuse BM25 with TF-IDF cosine
    rrf_k: int = 60
    doc_boost: float = 0.0  # weight of a document-level BM25 prior added to the chunk score
    candidates: int = 50  # chunks kept before overlap suppression
    doc_filter_ratio: float = 0.0  # >0: keep only contracts whose document prior >= ratio * best prior
    min_rel_score: float = 0.0  # >0: drop passages scoring below ratio * top passage score
    refine_max_chars: int = 0  # >0: shrink each passage to its best-matching sentence window of <= N chars
    max_overlap_frac: float = 0.5  # drop a result overlapping a higher-ranked one by more than this


@dataclass(frozen=True)
class AbstentionConfig:
    policy: str = "top_score"  # "top_score" | "top_score_gap"
    threshold: float = 0.0  # on the normalized top score in [0, 1]; 0 disables abstention
    min_gap: float = 0.0  # top_score_gap only: lower bound on (top1 - top2) / top1
    signal: str = "normalized_score"  # "normalized_score" | "doc_peakedness"
    strip_context: bool = False  # confidence uses the query without its 'Consider the <contract>;' prefix
    min_doc_prior: float = 0.0  # >0: abstain when the best contract match is weaker (named contract not indexed)
    topical_min_docs: int = 0  # >0: confidence ignores query terms found in fewer than N contracts (names)


@dataclass(frozen=True)
class EvalConfig:
    k_values: tuple[int, ...] = (1, 3, 5)
    top_k: int = 5


@dataclass(frozen=True)
class PathsConfig:
    corpus_root: str = "data/raw/legalbench-rag/corpus"
    benchmark_file: str = "data/raw/legalbench-rag/benchmarks/cuad.json"
    unanswerable_file: str = "data/unanswerable_queries.jsonl"
    manifest_file: str = "data/manifests/cuad_mini.json"
    cache_dir: str = "cache"
    experiments_dir: str = "experiments"


@dataclass(frozen=True)
class Config:
    name: str = "baseline"
    seed: int = 0
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    abstention: AbstentionConfig = field(default_factory=AbstentionConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["eval"]["k_values"] = list(d["eval"]["k_values"])
        return d

    def hash(self) -> str:
        """Hash of the pipeline-relevant settings (name and paths excluded)."""
        d = self.to_dict()
        d.pop("name")
        d.pop("paths")
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def with_changes(self, changes: dict[str, Any]) -> Config:
        """Apply dotted-key changes, e.g. {"chunking.size": 600}, and validate."""
        cfg = self
        for dotted, value in changes.items():
            section, key = dotted.split(".")
            sub = getattr(cfg, section)
            if key not in {f.name for f in fields(sub)}:
                raise ConfigError(f"unknown config key {dotted!r}")
            cfg = replace(cfg, **{section: replace(sub, **{key: value})})
        validate(cfg)
        return cfg


_SECTIONS = {
    "chunking": ChunkingConfig,
    "retrieval": RetrievalConfig,
    "abstention": AbstentionConfig,
    "eval": EvalConfig,
    "paths": PathsConfig,
}


def validate(cfg: Config) -> None:
    c, r, a, e = cfg.chunking, cfg.retrieval, cfg.abstention, cfg.eval
    if c.strategy not in CHUNK_STRATEGIES:
        raise ConfigError(f"chunking.strategy must be one of {CHUNK_STRATEGIES}, got {c.strategy!r}")
    if c.size < 50:
        raise ConfigError("chunking.size must be >= 50")
    if not 0 <= c.overlap < c.size:
        raise ConfigError("chunking.overlap must satisfy 0 <= overlap < size")
    if r.k1 <= 0 or not 0 <= r.b <= 1:
        raise ConfigError("retrieval.k1 must be > 0 and retrieval.b in [0, 1]")
    if r.fusion not in FUSIONS:
        raise ConfigError(f"retrieval.fusion must be one of {FUSIONS}")
    if not 0 <= r.doc_filter_ratio <= 1 or not 0 <= r.min_rel_score <= 1 or r.refine_max_chars < 0:
        raise ConfigError("invalid retrieval.doc_filter_ratio / min_rel_score / refine_max_chars")
    if not 0 <= a.min_doc_prior <= 1:
        raise ConfigError("abstention.min_doc_prior must be in [0, 1]")
    if r.candidates < 1 or not 0 <= r.max_overlap_frac <= 1 or r.doc_boost < 0:
        raise ConfigError("invalid retrieval.candidates / max_overlap_frac / doc_boost")
    if a.policy not in ("top_score", "top_score_gap"):
        raise ConfigError("abstention.policy must be 'top_score' or 'top_score_gap'")
    if a.signal not in ("normalized_score", "doc_peakedness"):
        raise ConfigError("abstention.signal must be 'normalized_score' or 'doc_peakedness'")
    if a.topical_min_docs < 0:
        raise ConfigError("abstention.topical_min_docs must be >= 0")
    if not 0 <= a.threshold <= 1 or not 0 <= a.min_gap <= 1:
        raise ConfigError("abstention.threshold and min_gap must be in [0, 1]")
    if not e.k_values or any(k < 1 for k in e.k_values) or e.top_k < 1:
        raise ConfigError("eval.k_values / eval.top_k must be positive")
    if max(e.k_values) > e.top_k:
        raise ConfigError("eval.top_k must be >= max(eval.k_values)")


def from_dict(raw: dict[str, Any]) -> Config:
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    unknown = set(raw) - set(_SECTIONS) - {"name", "seed"}
    if unknown:
        raise ConfigError(f"unknown top-level config keys: {sorted(unknown)}")
    kwargs: dict[str, Any] = {k: raw[k] for k in ("name", "seed") if k in raw}
    for name, cls in _SECTIONS.items():
        section = raw.get(name) or {}
        if not isinstance(section, dict):
            raise ConfigError(f"section {name!r} must be a mapping")
        bad = set(section) - {f.name for f in fields(cls)}
        if bad:
            raise ConfigError(f"unknown keys in {name!r}: {sorted(bad)}")
        if name == "eval" and "k_values" in section:
            section = {**section, "k_values": tuple(section["k_values"])}
        kwargs[name] = cls(**section)
    cfg = Config(**kwargs)
    validate(cfg)
    return cfg


def load_config(path: str | Path) -> Config:
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ConfigError(f"config file not found: {p}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"malformed YAML in {p}: {e}") from e
    try:
        return from_dict(raw or {})
    except TypeError as e:
        raise ConfigError(str(e)) from e


def dump_config(cfg: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
