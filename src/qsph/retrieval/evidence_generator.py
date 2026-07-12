"""Wiring retrieved evidence into hypothesis generation.

This connects the RAG evidence source to the generator-critic loop. The key
design point: retrieval produces an *evidence block* (passages + citations) that
is prepended to whatever prompt the generator builds. The generator reasons FROM
that evidence; provenance is preserved so the critic and the user can trace each
claim back to a source passage.

We expose `make_evidence_grounded_generator`, which wraps a base generator
callable so that, on each attempt, it first retrieves query-relevant passages
from the DocumentStore and supplies them as context. The base generator is still
injectable (a real LLM adapter, or a scripted one for tests), so this stays
testable offline and provider-agnostic.

This is emphatically retrieval, not fine-tuning: the DocumentStore is consulted
at query time, the model's weights are never modified, and every passage keeps
its citation.
"""

from __future__ import annotations

from collections.abc import Callable

from qsph.critic.hypothesis import Hypothesis
from qsph.retrieval.retriever import DocumentStore, build_evidence_block

# A base generator receives (material, prior_feedback, evidence_block) and
# returns a Hypothesis. The evidence_block is "" when there are no uploads, so
# the generator degrades gracefully to database-only reasoning.
EvidenceAwareGeneratorFn = Callable[[str, str | None, str], Hypothesis]


def make_evidence_grounded_generator(
    base_generator: EvidenceAwareGeneratorFn,
    store: DocumentStore,
    *,
    top_k: int = 5,
    query_template: str = "{material} density of states electronic structure",
):
    """Wrap a base generator so it retrieves evidence before generating.

    Returns a plain GeneratorFn (material, feedback) -> Hypothesis, so it drops
    straight into GeneratorCriticLoop.run() with no loop changes.

    The retrieval query is derived from the material plus DOS-relevant terms so
    the retrieved passages are topically aligned with what the generator needs
    to reason about. Callers can override query_template for other properties.
    """

    def generator(material: str, feedback: str | None) -> Hypothesis:
        query = query_template.format(material=material)
        passages = store.retrieve(query, top_k=top_k)
        evidence_block = build_evidence_block(passages)
        return base_generator(material, feedback, evidence_block)

    return generator
