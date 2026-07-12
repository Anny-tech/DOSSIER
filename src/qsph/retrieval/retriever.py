"""Retrieval over uploaded documents, feeding grounded evidence to the generator.

This is the honest version of "let the user upload papers and have the system
reason from them": documents are chunked and embedded once at upload, then at
query time the most relevant passages are retrieved and handed to the generator
AS EVIDENCE WITH PROVENANCE. The generator reasons from retrieved text; it does
not absorb the documents into weights. Every retrieved passage keeps its source
id and offsets, so any claim the generator makes can be traced back and checked
-- which is exactly what keeps the physics-critic's grounding meaningful.

Contrast with fine-tuning (which this deliberately is NOT): retrieval preserves
provenance and updates nothing; the model's weights are untouched. That is the
property the whole verification story depends on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qsph.retrieval.chunking import Chunk, chunk_text
from qsph.retrieval.embedders import BaseEmbedder, HashingEmbedder


@dataclass
class RetrievedPassage:
    """A retrieved chunk plus its relevance score, for evidence + provenance."""

    chunk: Chunk
    score: float

    def as_evidence(self) -> str:
        """Formatted for injection into the generator prompt, WITH citation.

        The citation travels with the text so the generator is prompted to
        attribute claims, and so the critic/user can verify them.
        """
        return f"[{self.chunk.citation()}] {self.chunk.text}"


@dataclass
class DocumentStore:
    """An in-memory store of chunked, embedded uploaded documents.

    Deliberately in-memory and simple: for a per-session website upload of a
    handful of papers this is sufficient and dependency-free. For a large
    persistent corpus, swap in a vector database (ChromaDB) behind the same
    interface -- the retriever contract does not change.
    """

    embedder: BaseEmbedder = field(default_factory=HashingEmbedder)
    _chunks: list[Chunk] = field(default_factory=list)
    _vectors: list[list[float]] = field(default_factory=list)

    def add_document(
        self,
        source_id: str,
        text: str,
        *,
        target_chars: int = 800,
        overlap_chars: int = 150,
    ) -> int:
        """Chunk, embed, and store one document. Returns number of chunks added.

        Embedding happens once here, at upload, not per query -- so query-time
        retrieval is cheap. This is the architectural opposite of per-session
        fine-tuning, which would pay a large cost on every upload and still lose
        provenance.
        """
        chunks = chunk_text(
            source_id, text, target_chars=target_chars, overlap_chars=overlap_chars
        )
        if not chunks:
            return 0
        vectors = self.embedder.embed([c.text for c in chunks])
        self._chunks.extend(chunks)
        self._vectors.extend(vectors)
        return len(chunks)

    @property
    def n_chunks(self) -> int:
        return len(self._chunks)

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedPassage]:
        """Return the top_k passages most relevant to the query.

        Ranking is cosine similarity in the embedder's space. Ties and empty
        stores are handled gracefully (empty store -> empty result), so a query
        before any upload does not error -- it simply returns no evidence, and
        the generator proceeds with database-only reasoning.
        """
        if not self._chunks:
            return []
        query_vec = self.embedder.embed([query])[0]
        scored = [
            RetrievedPassage(chunk=chunk, score=BaseEmbedder.cosine(query_vec, vec))
            for chunk, vec in zip(self._chunks, self._vectors, strict=True)
        ]
        scored.sort(key=lambda p: p.score, reverse=True)
        return scored[:top_k]


def build_evidence_block(passages: list[RetrievedPassage]) -> str:
    """Assemble retrieved passages into a single evidence block for the prompt.

    Returns an empty string when there is no evidence, so the generator's prompt
    template can conditionally include an evidence section only when uploads
    exist. Each passage is on its own line with its citation, so the generator
    is structurally encouraged to ground claims in specific sources.
    """
    if not passages:
        return ""
    lines = ["Retrieved evidence from uploaded documents:"]
    lines.extend(p.as_evidence() for p in passages)
    lines.append(
        "\nWhen you use a fact from this evidence, cite its [source#p] tag so "
        "the claim can be verified against the source."
    )
    return "\n".join(lines)
