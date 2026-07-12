"""RAG pipeline tests: chunking, retrieval ranking, provenance, evidence block.

These prove the retrieval-as-evidence path works and, crucially, that provenance
survives end to end -- a retrieved passage can always be traced to its source,
which is what keeps the critic's grounding meaningful. Uses the offline
HashingEmbedder so the whole pipeline runs without downloading models.
"""

from __future__ import annotations

from qsph.retrieval.chunking import chunk_text
from qsph.retrieval.retriever import (
    DocumentStore,
    build_evidence_block,
)


def test_chunking_produces_provenance():
    text = (
        "Bi2Te3 is a well-known thermoelectric. Its density of states shows a "
        "sharp feature near the Fermi level. The Seebeck coefficient is large. "
        "Sb2Te3 behaves similarly. Doping shifts the Fermi level significantly."
    )
    chunks = chunk_text("doi:test", text, target_chars=120, overlap_chars=30)
    assert len(chunks) >= 2
    # Every chunk carries source id and a monotone index.
    assert all(c.source_id == "doi:test" for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # Citations are well-formed and unique.
    cites = [c.citation() for c in chunks]
    assert cites[0] == "doi:test#p0"
    assert len(set(cites)) == len(cites)


def test_chunking_empty_text():
    assert chunk_text("s", "") == []
    assert chunk_text("s", "    \n  ") == []


def test_chunking_dehyphenates_pdf_linebreaks():
    text = "The lat-\ntice constant and the den-\nsity of states were computed."
    chunks = chunk_text("s", text, target_chars=200, overlap_chars=20)
    joined = " ".join(c.text for c in chunks)
    assert "lattice" in joined
    assert "density" in joined


def test_retriever_ranks_relevant_passage_first():
    store = DocumentStore()
    store.add_document(
        "paperA",
        "Bi2Te3 has a large density of states peak just below the Fermi level, "
        "which drives its high Seebeck coefficient and thermoelectric response.",
        target_chars=200,
        overlap_chars=20,
    )
    store.add_document(
        "paperB",
        "The mechanical hardness of tungsten carbide makes it useful for cutting "
        "tools and industrial abrasives in manufacturing applications.",
        target_chars=200,
        overlap_chars=20,
    )
    results = store.retrieve("density of states Fermi level Seebeck", top_k=3)
    assert results
    # The thermoelectric-DOS passage should outrank the tungsten-carbide one.
    assert results[0].chunk.source_id == "paperA"
    assert results[0].score >= results[-1].score  # sorted descending


def test_retrieve_on_empty_store_returns_nothing():
    store = DocumentStore()
    assert store.retrieve("anything") == []


def test_evidence_block_carries_citations():
    store = DocumentStore()
    store.add_document(
        "doi:xyz",
        "GeBi2Te4 exhibits topological surface states observed by ARPES with a "
        "Dirac cone inside the bulk gap.",
        target_chars=200,
        overlap_chars=20,
    )
    passages = store.retrieve("topological surface states ARPES Dirac cone")
    block = build_evidence_block(passages)
    assert "doi:xyz#p0" in block  # provenance present in the evidence
    assert "cite" in block.lower()  # generator is instructed to attribute


def test_empty_evidence_block_is_empty_string():
    assert build_evidence_block([]) == ""


def test_add_document_reports_chunk_count():
    store = DocumentStore()
    n = store.add_document(
        "s",
        "One sentence. Two sentence. Three sentence. Four sentence. Five.",
        target_chars=40,
        overlap_chars=10,
    )
    assert n == store.n_chunks
    assert n >= 2
