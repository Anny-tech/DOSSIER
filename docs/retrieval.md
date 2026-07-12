# Retrieval over uploaded papers (RAG, not fine-tuning)

## What this does

Lets a user upload papers and have the QSPH generator reason from their content,
while keeping every claim traceable to a source passage.

Flow: PDF text (extracted by the ingestion layer) -> `chunk_text` splits it into
overlapping, provenance-carrying passages -> `DocumentStore.add_document`
embeds and stores them -> at query time `DocumentStore.retrieve` ranks passages
by relevance -> `build_evidence_block` formats them (with `[source#p]` citations)
-> `make_evidence_grounded_generator` hands the evidence to the generator.

## Why RAG and explicitly NOT LoRA / fine-tuning

- **There is no training signal in a handful of uploaded PDFs.** LoRA is
  supervised fine-tuning; it needs many labeled input->output pairs. Uploads
  have none.
- **Fine-tuning destroys provenance.** It dissolves source text into weight
  deltas, so you can no longer point at the passage that grounded a claim. The
  physics-critic depends on that traceability; RAG preserves it.
- **Cost.** Retrieval is milliseconds per query; per-session fine-tuning is
  minutes-to-hours of GPU time per upload.
- **It is the wrong tool for "read this document."** Retrieval / long-context is
  the field-standard solution; fine-tuning-to-memorize-a-document does not work.

The one legitimate LoRA use in this project is unrelated to uploads: fine-tuning
a cheap local model as the *generator* on a LARGE LABELED dataset, to avoid
per-call API cost. That is future work and needs the labeled dataset first.

## Embedders

`retrieval/embedders.py` provides a swappable `BaseEmbedder`:
- `HuggingFaceEmbedder` (default PhysBERT; MatSciBERT as the ablation arm) —
  needs `pip install 'qsph[retrieval]'` (torch + weights).
- `HashingEmbedder` — dependency-free, offline, for testing the plumbing only.
  Carries lexical-overlap signal only; never use for real retrieval-quality
  claims.

## PDF -> text

Chunking operates on already-extracted text. Robust PDF/XML extraction (and, for
licensed content, publisher TDM APIs) is a separate ingestion concern with its
own dependencies; keeping chunking text-only makes it fast and testable. Wire an
extractor (e.g. a PDF-to-text tool, or a publisher TDM API where you hold a
license) to feed `add_document`.
