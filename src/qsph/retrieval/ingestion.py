"""PDF (and plain-text) ingestion into the retrieval store.

This is the layer that turns an actual uploaded file into chunked, embedded,
provenance-carrying passages in a DocumentStore. It is kept deliberately
separate from chunking/retrieval because robust PDF text extraction has its own
heavy dependencies and failure modes; isolating it here means the rest of the
retrieval stack stays fast, pure, and testable without any PDF library.

Design:
  * A small `PDFExtractor` protocol so the extraction backend is swappable
    (pdfplumber by default; a publisher TDM-API or OCR backend could implement
    the same interface for licensed or scanned content).
  * `ingest_pdf` / `ingest_text` convert a file into DocumentStore entries,
    tagging every resulting chunk with a stable source id (the DOI if known,
    else the filename) so downstream provenance/citation works.
  * Failures are surfaced as `IngestionResult` with an error message, never as
    silent empty success -- a paper that failed to extract must be visibly
    distinguishable from a paper with no relevant content.

Scientific-PDF reality this accounts for: multi-column layouts, hyphenated line
breaks, and ligatures. pdfplumber handles columns reasonably; the chunker's
normalisation step de-hyphenates. We do NOT attempt figure/table OCR here --
that is a separate (VLM/OCR) concern and out of scope for the text path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from qsph.retrieval.retriever import DocumentStore


class PDFExtractor(Protocol):
    """Protocol for a PDF-to-text backend. Implement to swap extraction engines."""

    def extract(self, path: Path) -> str:
        """Return the concatenated text of the PDF at `path`."""
        ...


class PdfplumberExtractor:
    """Default PDF text extractor using pdfplumber.

    pdfplumber is chosen over pypdf for scientific papers because it handles
    multi-column layouts and preserves reading order better. It is an optional
    dependency: importing this class without pdfplumber installed raises a clear
    install hint rather than failing obscurely later.
    """

    def __init__(self) -> None:
        try:
            import pdfplumber  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "PdfplumberExtractor requires pdfplumber: "
                "pip install 'qsph[ingest]'"
            ) from exc
        self._pdfplumber = pdfplumber

    def extract(self, path: Path) -> str:  # pragma: no cover - needs real PDF I/O
        pages: list[str] = []
        with self._pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        return "\n".join(pages)


@dataclass
class IngestionResult:
    """Outcome of ingesting one document."""

    source_id: str
    n_chunks: int
    ok: bool
    error: str | None = None

    @property
    def failed(self) -> bool:
        return not self.ok


def ingest_text(
    store: DocumentStore,
    source_id: str,
    text: str,
    *,
    target_chars: int = 800,
    overlap_chars: int = 150,
) -> IngestionResult:
    """Ingest already-extracted text into the store.

    Useful when text comes from a source other than a local PDF -- e.g. a
    publisher TDM API returning XML/text, or a paste-in box on the website. An
    empty extraction yields ok=True with n_chunks=0 (a valid "nothing to index"
    outcome), distinct from an extraction *error*.
    """
    try:
        n = store.add_document(
            source_id, text, target_chars=target_chars, overlap_chars=overlap_chars
        )
        return IngestionResult(source_id=source_id, n_chunks=n, ok=True)
    except Exception as exc:  # noqa: BLE001 - report any indexing failure honestly
        return IngestionResult(
            source_id=source_id, n_chunks=0, ok=False, error=str(exc)
        )


def ingest_pdf(
    store: DocumentStore,
    path: str | Path,
    *,
    source_id: str | None = None,
    extractor: PDFExtractor | None = None,
    target_chars: int = 800,
    overlap_chars: int = 150,
) -> IngestionResult:
    """Extract text from a PDF file and ingest it into the store.

    Args:
        store: the DocumentStore to add chunks to.
        path: path to the PDF file.
        source_id: provenance id for all chunks from this file. Defaults to the
            filename; pass the DOI when known so citations are meaningful.
        extractor: PDF backend; defaults to PdfplumberExtractor. Injectable so
            tests can supply a fake extractor and so licensed/OCR backends can
            be swapped in.

    Returns:
        IngestionResult. A missing file or an extraction error is reported as
        ok=False with a message, never as silent empty success.
    """
    p = Path(path)
    sid = source_id or p.name

    if not p.exists():
        return IngestionResult(
            source_id=sid, n_chunks=0, ok=False, error=f"file not found: {p}"
        )

    backend = extractor or PdfplumberExtractor()
    try:
        text = backend.extract(p)
    except Exception as exc:  # noqa: BLE001 - extraction can fail many ways
        return IngestionResult(
            source_id=sid, n_chunks=0, ok=False, error=f"extraction failed: {exc}"
        )

    if not text.strip():
        # Extraction "succeeded" but produced nothing -- likely a scanned/image
        # PDF needing OCR. Report it distinctly so the user knows to try OCR,
        # rather than silently indexing an empty document.
        return IngestionResult(
            source_id=sid,
            n_chunks=0,
            ok=False,
            error="no extractable text (scanned/image PDF? OCR not applied)",
        )

    return ingest_text(
        store,
        sid,
        text,
        target_chars=target_chars,
        overlap_chars=overlap_chars,
    )
