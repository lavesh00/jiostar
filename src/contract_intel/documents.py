"""Source-document loading with the original text preserved untouched."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .data import read_text


class DocumentStore:
    """Maps a dataset-style ``file_path`` (e.g. ``cuad/x.txt``) to its original text."""

    def __init__(self, texts: dict[str, str]) -> None:
        self._texts = dict(texts)

    @classmethod
    def from_files(cls, corpus_root: str | Path, file_paths: Iterable[str]) -> DocumentStore:
        root = Path(corpus_root)
        texts = {}
        for fp in sorted(set(file_paths)):
            p = root / fp
            if not p.exists():
                raise FileNotFoundError(f"indexed contract missing on disk: {p}")
            texts[fp] = read_text(p)
        return cls(texts)

    def __contains__(self, file_path: str) -> bool:
        return file_path in self._texts

    def __len__(self) -> int:
        return len(self._texts)

    def paths(self) -> list[str]:
        return sorted(self._texts)

    def text(self, file_path: str) -> str:
        return self._texts[file_path]

    def slice(self, file_path: str, start: int, end: int) -> str:
        return self._texts[file_path][start:end]
