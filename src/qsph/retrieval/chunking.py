"""Document chunking for retrieval.

Splits a document's text into overlapping passages, each carrying enough
provenance (source id, passage index, char span) that a downstream claim can be
traced back to exactly the text that grounded it. Provenance is not optional
here: it is what lets the physics-critic (and a human reviewer) verify that a
generator's reasoning rests on real retrieved text rather than a hallucination.

Design choices:
  * Overlapping windows so a fact spanning a chunk boundary is not lost.
  * Sentence-aware splitting where possible (split on sentence enders) so chunks
    are semantically coherent, falling back to hard character windows for text
    without clean sentence structure (common in PDF-extracted text).
  * We chunk on already-extracted text. PDF -> text extraction is a separate
    concern handled by the ingestion layer (see docs), because robust PDF
    parsing has its own heavy dependencies and failure modes; keeping chunking
    text-only makes it fast, deterministic, and testable offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Split on sentence enders followed by whitespace. Deliberately simple and
# auditable; scientific text has many edge cases (abbreviations, decimals) but
# over-splitting is harmless here because we re-join into overlapping windows.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage with full provenance back to its source."""

    source_id: str  # DOI / filename of the source document
    index: int  # 0-based passage index within the source
    text: str
    char_start: int  # start offset in the (normalised) source text
    char_end: int  # end offset

    def citation(self) -> str:
        """Compact human-readable provenance string for display/audit."""
        return f"{self.source_id}#p{self.index}"


def _normalise(text: str) -> str:
    """Collapse whitespace and de-hyphenate line breaks from PDF extraction.

    PDF text extraction routinely inserts hyphen+newline inside words and hard
    line breaks mid-sentence; normalising here means chunk offsets are stable
    and grounding checks downstream match cleanly.
    """
    text = text.replace("-\n", "")  # de-hyphenate across line breaks
    return " ".join(text.split())


def chunk_text(
    source_id: str,
    text: str,
    *,
    target_chars: int = 800,
    overlap_chars: int = 150,
) -> list[Chunk]:
    """Split source text into overlapping, sentence-aware chunks.

    Args:
        source_id: identifier stored on every chunk for provenance.
        text: the document text (already extracted from PDF/XML).
        target_chars: approximate chunk size. ~800 chars is a few sentences --
            large enough to carry a self-contained fact (a material + its DOS
            statement), small enough to keep retrieval precise.
        overlap_chars: how much consecutive chunks overlap, so a fact on a
            boundary appears whole in at least one chunk.

    Returns:
        List of Chunks in document order. Empty input yields an empty list
        (not an error) so callers can handle empty/failed extractions gracefully.
    """
    if overlap_chars >= target_chars:
        raise ValueError("overlap_chars must be smaller than target_chars")

    norm = _normalise(text)
    if not norm:
        return []

    sentences = _SENTENCE_SPLIT.split(norm)

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_len = 0
    cursor = 0  # char offset into norm of the start of the current buffer
    index = 0

    def flush(next_start: int) -> None:
        nonlocal buffer, buffer_len, cursor, index
        if not buffer:
            return
        chunk_str = " ".join(buffer)
        char_start = cursor
        char_end = cursor + len(chunk_str)
        chunks.append(
            Chunk(
                source_id=source_id,
                index=index,
                text=chunk_str,
                char_start=char_start,
                char_end=char_end,
            )
        )
        index += 1
        cursor = next_start

    running_offset = 0
    for sentence in sentences:
        sent_len = len(sentence) + 1  # +1 for the join space
        if buffer_len + sent_len > target_chars and buffer:
            # Flush current buffer, then start a new one with tail overlap.
            flush_start = running_offset
            flush(flush_start)
            # Build overlap: keep trailing sentences up to overlap_chars.
            overlap: list[str] = []
            olen = 0
            for s in reversed(buffer):
                if olen + len(s) > overlap_chars:
                    break
                overlap.insert(0, s)
                olen += len(s) + 1
            buffer = overlap
            buffer_len = olen
            cursor = running_offset - olen
        buffer.append(sentence)
        buffer_len += sent_len
        running_offset += sent_len

    flush(running_offset)
    return chunks
