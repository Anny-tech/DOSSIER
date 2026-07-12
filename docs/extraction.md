# Tiered PDF extraction

## Why

Rule-based PDF parsers (pdfplumber, pypdf, even PyMuPDF) work on born-digital
text but struggle with real scientific journal articles: multi-column layouts
get spliced, equations and ligatures mangle, tables collapse. Comparative
studies confirm *all* rule-based parsers degrade on scientific documents. So the
extractor is tiered: a cheap parser first, an LLM reader only when the parser
demonstrably failed.

## Pipeline

```
   PDF ──> primary rule-based extractor ──> quality gate ──> good? ──> use it
                                                   │
                                                   └─ poor ──> LLM fallback reader
```

- **Primary backends** (`tiered_extractor.py`): `GrobidExtractor` (best for
  scholarly PDFs; the S2ORC standard; needs a GROBID Docker service),
  `PyMuPDFExtractor` (fast, better than pdfplumber, no GPU), and the original
  pdfplumber extractor. Any implements the `extract(path) -> str` protocol.
- **Quality gate** (`quality.py`): deterministic assessment of whether the
  primary text is good enough. See below.
- **LLM fallback** (`LLMReaderExtractor`): a multimodal LLM transcribes the
  pages the parser mangled. Used ONLY when the gate escalates, so most papers
  never incur LLM cost.

## The quality gate — scope and honesty

The gate detects extraction **failure**, not extraction **correctness**. It can
tell that text is garbled/empty/mangled; it cannot verify that an extracted
N(E_F) value is the *right* value — that is the physics-critic's job downstream.
This honest scoping matters: the gate's only decision is "is this text broken
enough to pay for an LLM re-read?"

Signals (all tunable via `QualityThresholds`, all auditable):
- near-empty output (hard veto) — scanned/image PDF or parser crash
- low alphabetic density — ligature/math/symbol mangling
- high broken-word rate — column-splicing (words split into fragments)
- too few well-formed sentences — reading order destroyed
- low average word length — column-splice garbage
- optional: expected topic terms absent — the relevant content was dropped

Thresholds are conservative defaults and **should be validated on a labelled set
of good/bad extractions** before being trusted in production. Build such a set by
running the primary extractor over a sample of your target journals and hand-
labelling which outputs are usable; tune the thresholds to that.

## Provenance: the LLM fallback is lower-confidence

A parser returns verbatim source spans; an LLM *transcribes* and can paraphrase
or err. So `ExtractionOutcome.low_confidence_source` is `True` when the LLM
fallback produced the text. Downstream, chunks from a low-confidence source
should be tagged so the grounding check and the physics-critic treat them with
appropriate caution. The critic still has the final say on any value regardless
of how it was read — which is exactly why extraction imperfection degrades
gracefully into a rejected hypothesis rather than a silently-wrong result.

## Cost control

The LLM reader runs only on escalation, so cost scales with the *failure rate*
of the primary parser, not the document count. Choosing a strong primary backend
(GROBID) minimises escalations. This is the cheapest-effort cost lever discussed
in the project notes: don't call the LLM when a good parser suffices.
