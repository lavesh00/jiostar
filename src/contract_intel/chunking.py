"""Offset-preserving chunking.

Every chunk is a half-open interval ``[start, end)`` of the *original* document text; the chunk text
is always ``text[start:end]`` (never re-assembled), so offsets are exact by construction.
Chunk boundaries are trimmed of surrounding whitespace; empty chunks are dropped and duplicates removed.

Strategies
  fixed      sliding character windows of ``size`` with ``overlap`` (optionally snapped to boundaries)
  paragraph  blank-line separated blocks; blocks shorter than size/3 are merged forward (up to ``size``);
             blocks longer than ``size`` are split with fixed windows
  section    split at heading-like lines (ARTICLE/Section/"1.2 Title"), then pack paragraphs inside a
             section greedily up to ``size``; oversize blocks are split with fixed windows
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import ChunkingConfig


@dataclass(frozen=True, order=True)
class Chunk:
    file_path: str
    start: int  # inclusive
    end: int  # exclusive


_BLANK = re.compile(r"\n[ \t\r\f\v]*\n")
_HEADING = re.compile(
    r"^[ \t]*(?:(?:ARTICLE|Article|SECTION|Section)\s+[0-9IVXivx]+\b|\d{1,3}(?:\.\d{1,3}){0,3}\.?[ \t]+[A-Z][^\n]{0,80})",
    re.M,
)
_SENT_END = re.compile(r"[.;:]\s")
_MIN_FILL = 1 / 3


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def _snap_end(text: str, pos: int, end: int, size: int) -> int:
    """Move ``end`` back to a sentence end or whitespace within the last 25% of the window."""
    lo = pos + int(size * 0.75)
    window = text[lo:end]
    last = None
    for m in _SENT_END.finditer(window):
        last = m
    if last is not None:
        return lo + last.start() + 1
    ws = max(window.rfind(" "), window.rfind("\n"))
    return lo + ws if ws > 0 else end


def _snap_start(text: str, pos: int, limit: int = 40) -> int:
    """Advance a window start off a mid-word position to the next whitespace boundary."""
    if pos <= 0 or pos >= len(text) or text[pos - 1].isspace() or text[pos].isspace():
        return pos
    for i in range(pos, min(pos + limit, len(text))):
        if text[i].isspace():
            return i
    return pos


def _fixed(text: str, lo: int, hi: int, size: int, overlap: int, snap: bool) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    pos = lo
    while pos < hi:
        end = min(pos + size, hi)
        if snap and end < hi:
            end = max(_snap_end(text, pos, end, size), pos + 1)
        s, e = _trim(text, pos, end)
        if e > s:
            out.append((s, e))
        if end >= hi:
            break
        nxt = max(end - overlap, pos + 1)
        pos = _snap_start(text, nxt) if snap and overlap else nxt
    return out


def _blocks(text: str, lo: int, hi: int) -> list[tuple[int, int]]:
    """Blank-line separated, whitespace-trimmed blocks in [lo, hi)."""
    out, pos = [], lo
    for m in _BLANK.finditer(text, lo, hi):
        s, e = _trim(text, pos, m.start())
        if e > s:
            out.append((s, e))
        pos = m.end()
    s, e = _trim(text, pos, hi)
    if e > s:
        out.append((s, e))
    return out


def _pack(text: str, blocks: list[tuple[int, int]], cfg: ChunkingConfig, fill: float) -> list[tuple[int, int]]:
    """Merge consecutive blocks until a chunk reaches ``fill * size`` (never exceeding ``size``)."""
    out: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = None
    target = cfg.size * fill

    def flush() -> None:
        nonlocal cur
        if cur is not None:
            out.append(cur)
            cur = None

    for s, e in blocks:
        if e - s > cfg.size:
            flush()
            out.extend(_fixed(text, s, e, cfg.size, cfg.overlap, cfg.snap_boundaries))
            continue
        if cur is None:
            cur = (s, e)
        elif (cur[1] - cur[0]) < target and e - cur[0] <= cfg.size:
            cur = (cur[0], e)
        else:
            flush()
            cur = (s, e)
    flush()
    return out


def chunk_spans(text: str, cfg: ChunkingConfig) -> list[tuple[int, int]]:
    """Deterministic ``[(start, end), ...]`` spans, sorted, unique, non-empty, in-bounds."""
    n = len(text)
    if cfg.strategy == "fixed":
        spans = _fixed(text, 0, n, cfg.size, cfg.overlap, cfg.snap_boundaries)
    elif cfg.strategy == "paragraph":
        spans = _pack(text, _blocks(text, 0, n), cfg, _MIN_FILL)
    elif cfg.strategy == "section":
        cuts = sorted({m.start() for m in _HEADING.finditer(text)} | {0, n})
        spans = []
        for a, b in zip(cuts, cuts[1:], strict=False):
            spans.extend(_pack(text, _blocks(text, a, b), cfg, 1.0))
    else:  # pragma: no cover - validated in config
        raise ValueError(f"unknown chunking strategy {cfg.strategy!r}")
    return sorted(set(spans))


def chunk_document(file_path: str, text: str, cfg: ChunkingConfig) -> list[Chunk]:
    return [Chunk(file_path, s, e) for s, e in chunk_spans(text, cfg)]
