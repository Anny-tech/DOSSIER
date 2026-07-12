"""PDF ingestion tests.

Covers the real extraction path (against a fixture PDF built at test time), the
injectable-extractor path (offline, no PDF library), and the error cases that
must be reported distinctly rather than as silent empty success: missing file,
extraction failure, and scanned/empty PDFs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qsph.retrieval.ingestion import (
    IngestionResult,
    ingest_pdf,
    ingest_text,
)
from qsph.retrieval.retriever import DocumentStore

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


def test_ingest_text_indexes_chunks():
    store = DocumentStore()
    result = ingest_text(
        store,
        "doi:test",
        "Bi2Te3 shows a density of states peak below the Fermi level. "
        "The Seebeck coefficient is large. Antimony doping shifts E_F.",
        target_chars=80,
        overlap_chars=20,
    )
    assert result.ok
    assert result.n_chunks == store.n_chunks
    assert result.n_chunks >= 1


def test_ingest_pdf_real_extraction():
    """Extract from an actual PDF and confirm the content is indexed and
    retrievable with provenance."""
    if not FIXTURE_PDF.exists():
        pytest.skip("fixture PDF not present")
    store = DocumentStore()
    result = ingest_pdf(store, FIXTURE_PDF, source_id="doi:bi2te3")
    assert result.ok
    assert result.n_chunks >= 1
    # The extracted content is retrievable and carries the right source id.
    passages = store.retrieve("density of states Fermi level Seebeck", top_k=3)
    assert passages
    assert passages[0].chunk.source_id == "doi:bi2te3"
    # A distinctive phrase from the PDF made it through extraction+chunking.
    all_text = " ".join(p.chunk.text for p in passages).lower()
    assert "bi2te3" in all_text or "seebeck" in all_text


def test_ingest_pdf_with_injected_fake_extractor():
    """The extractor is swappable: a fake backend lets ingestion be tested with
    no PDF library and no real file content."""
    store = DocumentStore()

    class FakeExtractor:
        def extract(self, path):
            return (
                "GeBi2Te4 exhibits topological surface states with a Dirac cone "
                "inside the bulk gap, observed by ARPES."
            )

    # Point at any existing file; the fake ignores its contents.
    result = ingest_pdf(
        store,
        FIXTURE_PDF if FIXTURE_PDF.exists() else __file__,
        source_id="doi:fake",
        extractor=FakeExtractor(),
    )
    assert result.ok
    assert result.n_chunks >= 1
    passages = store.retrieve("topological surface states ARPES")
    assert passages[0].chunk.source_id == "doi:fake"


def test_missing_file_reported_as_error():
    store = DocumentStore()
    result = ingest_pdf(store, "/nonexistent/path/paper.pdf")
    assert result.failed
    assert "not found" in result.error


def test_extraction_failure_reported():
    store = DocumentStore()

    class BrokenExtractor:
        def extract(self, path):
            raise RuntimeError("corrupt PDF")

    result = ingest_pdf(
        store,
        FIXTURE_PDF if FIXTURE_PDF.exists() else __file__,
        extractor=BrokenExtractor(),
    )
    assert result.failed
    assert "extraction failed" in result.error
    assert store.n_chunks == 0  # nothing indexed on failure


def test_empty_pdf_reported_as_no_text():
    """A scanned/image PDF that yields no text is reported distinctly, so the
    user knows to try OCR rather than believing the paper was indexed."""
    store = DocumentStore()

    class EmptyExtractor:
        def extract(self, path):
            return "   \n  "

    result = ingest_pdf(
        store,
        FIXTURE_PDF if FIXTURE_PDF.exists() else __file__,
        extractor=EmptyExtractor(),
    )
    assert result.failed
    assert "no extractable text" in result.error


def test_ingestion_result_failed_property():
    r = IngestionResult(source_id="s", n_chunks=0, ok=False, error="x")
    assert r.failed
    r2 = IngestionResult(source_id="s", n_chunks=3, ok=True)
    assert not r2.failed
