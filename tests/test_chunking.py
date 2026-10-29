from __future__ import annotations

import pytest

from contract_intel.chunking import chunk_document, chunk_spans
from contract_intel.config import ChunkingConfig

MESSY = (
    "ARTICLE I DEFINITIONS\n\n1.1 Term. The term is five years.\n\n\n\n   \n"
    "1.2 Fee. Pay €500 — monthly; no exceptions. Café.\n"
    + "word " * 700 + "\n\nSection 2 Termination.\nEither party may terminate on notice.\n\n"
    "(a) first (b) second\n \n" + "x" * 2500 + "\n\n3. END"
)
CASES = {
    "empty": "",
    "spaces": "   \n\n\t  ",
    "one_word": "Agreement",
    "no_newlines": "alpha beta gamma. " * 200,
    "messy": MESSY,
    "unicode": "café — €" * 300,
}
CONFIGS = [
    ChunkingConfig(strategy=s, size=size, overlap=ov, snap_boundaries=snap)
    for s in ("fixed", "paragraph", "section")
    for size, ov in ((100, 0), (300, 50), (1000, 200))
    for snap in (False, True)
]


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"{c.strategy}-{c.size}-{c.overlap}-{c.snap_boundaries}")
def test_chunk_invariants(name, cfg):
    text = CASES[name]
    spans = chunk_spans(text, cfg)
    assert spans == sorted(set(spans))  # sorted, unique (no duplicate chunks)
    for s, e in spans:
        assert 0 <= s < e <= len(text)
        piece = text[s:e]
        assert piece and piece == piece.strip()  # non-empty, trimmed
        assert e - s <= cfg.size
    if text.strip():
        assert spans, "non-blank documents must yield chunks"
    else:
        assert spans == []


@pytest.mark.parametrize("strategy", ["fixed", "paragraph", "section"])
def test_all_non_whitespace_text_is_covered(strategy):
    cfg = ChunkingConfig(strategy=strategy, size=200, overlap=40)
    covered = bytearray(len(MESSY))
    for s, e in chunk_spans(MESSY, cfg):
        covered[s:e] = b"\x01" * (e - s)
    missing = [i for i, ch in enumerate(MESSY) if not ch.isspace() and not covered[i]]
    assert not missing, f"{len(missing)} non-space chars not covered, first at {missing[:3]}"


def test_fixed_windows_exact_and_overlap():
    text = "a" * 25
    # no whitespace to trim: windows of 10 with overlap 3 -> starts 0,7,14,21
    assert chunk_spans(text, ChunkingConfig(size=50, overlap=0)) == [(0, 25)]
    cfg = ChunkingConfig(strategy="fixed", size=10, overlap=3)
    assert chunk_spans("a" * 25, ChunkingConfig(strategy="fixed", size=50, overlap=10)) == [(0, 25)]
    spans = chunk_spans("a" * 100, ChunkingConfig(strategy="fixed", size=50, overlap=10))
    assert spans[0] == (0, 50) and spans[1] == (40, 90) and spans[-1][1] == 100
    assert cfg.overlap == 3


def test_paragraph_strategy_keeps_paragraphs_whole():
    text = "First paragraph here, long enough to stand alone as a unit of text.\n\nSecond paragraph, also long enough to stand alone here."
    spans = chunk_spans(text, ChunkingConfig(strategy="paragraph", size=80, overlap=0))
    assert [text[s:e] for s, e in spans] == [
        "First paragraph here, long enough to stand alone as a unit of text.",
        "Second paragraph, also long enough to stand alone here.",
    ]


def test_deterministic_and_reconstructable():
    cfg = ChunkingConfig(strategy="section", size=300, overlap=30, snap_boundaries=True)
    a = chunk_document("f.txt", MESSY, cfg)
    assert a == chunk_document("f.txt", MESSY, cfg)
    assert all(c.file_path == "f.txt" and MESSY[c.start : c.end] for c in a)


def test_offsets_are_character_not_byte_offsets():
    text = "€€€ hello world €€"  # '€' is 3 bytes in UTF-8
    (s, e), = chunk_spans(text, ChunkingConfig(size=100))
    assert (s, e) == (0, len(text)) and len(text.encode("utf-8")) > len(text)
