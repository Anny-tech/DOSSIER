"""Tiered PDF extraction: cheap rule-based first, LLM fallback when it fails.

This composes the pieces into the extractor the ingestion layer actually uses:

    primary rule-based extractor  ->  quality gate  ->  LLM fallback (if poor)

It implements the same `extract(path) -> str` interface as any PDFExtractor, so
the ingestion layer and everything downstream are unchanged. The design goals:

  * Cheap by default: the LLM reader runs ONLY when the quality gate judges the
    rule-based text too broken to trust. Most born-digital papers pass the gate
    and never incur LLM cost.
  * Honest provenance: text produced by the LLM fallback is a less-reliable
    source than a verbatim parser span (the model can paraphrase). The tiered
    extractor tracks WHICH backend produced the text so the ingestion layer can
    tag those chunks, and the physics-critic downstream still has the final say
    on any value regardless of how the text was read.
  * Swappable backends: GROBID (best for scholarly PDFs), PyMuPDF (fast, better
    than pdfplumber), pdfplumber (fallback), and an LLM reader all implement the
    same protocol.

Backends that need external services/keys (GROBID server, LLM API, a multimodal
model) are gated: they raise a clear error if unavailable rather than failing
obscurely, and are not exercised in CI. The tiering logic and quality gate ARE
tested offline with fake backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from qsph.retrieval.quality import QualityReport, QualityThresholds, assess_extraction


class TextExtractor(Protocol):
    """Any backend that turns a PDF path into text."""

    def extract(self, path: Path) -> str:
        ...


# --------------------------------------------------------------------------
# Rule-based backends (upgrade path from pdfplumber)
# --------------------------------------------------------------------------

class PyMuPDFExtractor:
    """Fast rule-based extractor. Generally better than pdfplumber on text.

    PyMuPDF (fitz) was found in comparative studies to outperform pdfplumber and
    pypdf for text extraction, while staying light and GPU-free. A good default
    primary backend.
    """

    def __init__(self) -> None:
        try:
            import fitz  # noqa: PLC0415  (PyMuPDF)
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PyMuPDFExtractor requires PyMuPDF: pip install 'qsph[ingest]'"
            ) from exc
        self._fitz = fitz

    def extract(self, path: Path) -> str:  # pragma: no cover - needs real I/O
        doc = self._fitz.open(str(path))
        try:
            return "\n".join(page.get_text() for page in doc)
        finally:
            doc.close()


class GrobidExtractor:
    """Layout-aware extractor for scholarly PDFs via a running GROBID service.

    GROBID is the domain standard for scientific articles (used in S2ORC); it
    handles multi-column layout and section structure far better than rule-based
    parsers, and its section tagging can enrich provenance. It runs as a separate
    service (Docker); this client posts the PDF to its full-text endpoint.

    Requires a reachable GROBID server. Raises if the service or `requests` is
    unavailable rather than silently degrading.
    """

    def __init__(self, server_url: str = "http://localhost:8070") -> None:
        try:
            import requests  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "GrobidExtractor requires requests: pip install 'qsph[ingest]'"
            ) from exc
        self._requests = requests
        self._url = server_url.rstrip("/")

    def extract(self, path: Path) -> str:  # pragma: no cover - needs service
        endpoint = f"{self._url}/api/processFulltextDocument"
        with open(path, "rb") as fh:
            resp = self._requests.post(
                endpoint, files={"input": fh}, timeout=120
            )
        resp.raise_for_status()
        # GROBID returns TEI XML; extract text from the body. We keep this
        # dependency-light by stripping tags; a fuller impl would parse sections
        # for richer provenance.
        return _tei_to_text(resp.text)


def _tei_to_text(tei_xml: str) -> str:
    """Extract readable text from GROBID TEI XML, tags stripped.

    Minimal and dependency-free. A production version would map <div>/<head>
    elements to section-tagged chunks for section-level provenance; here we keep
    it simple and let chunking handle the flat text.
    """
    import re

    # Drop the header/references-heavy front matter tags but keep their text.
    text = re.sub(r"<[^>]+>", " ", tei_xml)
    return " ".join(text.split())


# --------------------------------------------------------------------------
# LLM fallback backend
# --------------------------------------------------------------------------

class LLMReaderExtractor:
    """Fallback that uses a multimodal LLM to read pages the parser mangled.

    IMPORTANT provenance note: this returns text the MODEL transcribed, which is
    less reliable than a verbatim parser span (the model can paraphrase or err).
    We prompt it to transcribe verbatim, but the tiered extractor still marks
    output from this backend as lower-confidence so downstream grounding/critic
    checks treat it accordingly. This is the honest handling of the tradeoff we
    accepted when adding an LLM to the reading path.

    Requires a multimodal LLM client and page-rendering deps; gated accordingly.
    """

    def __init__(self, client, *, page_renderer=None):
        # client: object with a .read_pages(images)->str or .complete(...) method
        # page_renderer: callable(path)->list of page image bytes. Injected so
        # the rendering backend (pdf2image, PyMuPDF pixmaps) is swappable and so
        # tests can supply a fake.
        self._client = client
        self._render = page_renderer

    def extract(self, path: Path) -> str:  # pragma: no cover - needs model+render
        if self._render is None:
            raise RuntimeError(
                "LLMReaderExtractor needs a page_renderer to rasterise pages."
            )
        images = self._render(path)
        return self._client.read_pages(images)


# --------------------------------------------------------------------------
# The tiered extractor
# --------------------------------------------------------------------------

@dataclass
class ExtractionOutcome:
    """Records how a document was extracted, for provenance and diagnostics."""

    text: str
    backend_used: str  # 'primary' or 'llm_fallback'
    primary_quality: QualityReport
    escalated: bool

    @property
    def low_confidence_source(self) -> bool:
        """True when the LLM fallback produced the text (weaker provenance)."""
        return self.backend_used == "llm_fallback"


class TieredExtractor:
    """Primary rule-based extraction with a quality-gated LLM fallback.

    Implements `extract(path) -> str` so it drops into the ingestion layer as any
    other PDFExtractor. Use `extract_with_outcome` when you want the provenance
    metadata (which backend, quality report) to tag chunks appropriately.
    """

    def __init__(
        self,
        primary: TextExtractor,
        *,
        fallback: TextExtractor | None = None,
        thresholds: QualityThresholds | None = None,
        expected_terms: list[str] | None = None,
    ):
        self.primary = primary
        self.fallback = fallback
        self.thresholds = thresholds
        self.expected_terms = expected_terms

    def extract(self, path: Path) -> str:
        return self.extract_with_outcome(path).text

    def extract_with_outcome(self, path: Path) -> ExtractionOutcome:
        """Extract, escalating to the LLM fallback only if the gate says so."""
        primary_text = self.primary.extract(path)
        quality = assess_extraction(
            primary_text,
            thresholds=self.thresholds,
            expected_terms=self.expected_terms,
        )

        if quality.acceptable or self.fallback is None:
            # Either the primary text is good, or we have no fallback to escalate
            # to (in which case we return the best we have and let the quality
            # report flag it, rather than pretending it is good).
            return ExtractionOutcome(
                text=primary_text,
                backend_used="primary",
                primary_quality=quality,
                escalated=False,
            )

        # Escalate: the primary extraction was judged too poor to trust.
        fallback_text = self.fallback.extract(path)
        return ExtractionOutcome(
            text=fallback_text,
            backend_used="llm_fallback",
            primary_quality=quality,
            escalated=True,
        )
