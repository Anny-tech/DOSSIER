"""Integration: uploaded PDF text -> RAG retrieval -> evidence-grounded
generator -> physics-critic loop, with provenance preserved throughout.

This proves capability (2) end to end: a user uploads a paper, and the generator
reasons from its retrieved content while the critic still gates the result. The
base generator here is scripted (no LLM), but it demonstrates that the evidence
block reaches the generator and that a generator using the evidence produces a
hypothesis the critic can verify.
"""

from __future__ import annotations

import numpy as np

from qsph.agents.loop import GeneratorCriticLoop
from qsph.critic.hypothesis import Favorability, Hypothesis, PropertyArm
from qsph.critic.physics_critic import PhysicsCritic
from qsph.features.dos_features import compute_dos_features
from qsph.retrieval.evidence_generator import make_evidence_grounded_generator
from qsph.retrieval.retriever import DocumentStore


def test_uploaded_paper_evidence_reaches_generator():
    # A user uploads a paper describing a material's DOS.
    store = DocumentStore()
    store.add_document(
        "doi:uploaded",
        "The compound XYZ shows a pronounced density of states peak exactly at "
        "the Fermi level, indicating a van Hove singularity favorable for "
        "electron-phonon superconductivity.",
        target_chars=250,
        overlap_chars=30,
    )

    captured_evidence = {}

    def base_generator(material, feedback, evidence_block):
        # Record what evidence the generator received, to prove it flowed in.
        captured_evidence["block"] = evidence_block
        # A faithful generator reasons from the evidence: the passage says a
        # van Hove peak sits at E_F, so it hypothesises accordingly.
        van_hove = "van hove" in evidence_block.lower()
        return Hypothesis(
            material=material,
            arm=PropertyArm.SUPERCONDUCTIVITY,
            van_hove_near_ef=van_hove,
            favorability=Favorability.FAVORABLE if van_hove else Favorability.UNCERTAIN,
            reasoning="Grounded in retrieved passage reporting a van Hove peak at E_F.",
            analog_materials=["XYZ"],
        )

    generator = make_evidence_grounded_generator(base_generator, store, top_k=3)

    # Ground-truth DOS for the material DOES have a peak at E_F (evidence is
    # correct), so the critic should accept.
    e = np.linspace(-3, 3, 601)
    d = 0.5 + 3.0 * np.exp(-(e**2) / (2 * 0.05**2))
    feats = compute_dos_features(e, d)

    loop = GeneratorCriticLoop(PhysicsCritic(), max_revisions=1)
    result = loop.run("XYZ", generator, feats)

    # Evidence with provenance reached the generator.
    assert "doi:uploaded#p0" in captured_evidence["block"]
    # The generator used it and the critic accepted the grounded hypothesis.
    assert result.succeeded
    assert result.final.hypothesis.van_hove_near_ef is True


def test_critic_still_catches_evidence_the_dos_contradicts():
    """Even when a paper CLAIMS a van Hove at E_F, if the material's computed
    DOS contradicts it, the critic rejects -- retrieval does not get to override
    physics. This is the safety property: evidence informs reasoning, but the
    deterministic critic still has the final say."""
    store = DocumentStore()
    store.add_document(
        "doi:overclaim",
        "We assert a van Hove singularity at the Fermi level for compound QRS.",
        target_chars=200,
        overlap_chars=20,
    )

    def base_generator(material, feedback, evidence_block):
        van_hove = "van hove" in evidence_block.lower()
        return Hypothesis(
            material=material,
            arm=PropertyArm.SUPERCONDUCTIVITY,
            van_hove_near_ef=van_hove,
            favorability=Favorability.FAVORABLE,
        )

    generator = make_evidence_grounded_generator(base_generator, store)

    # But the ACTUAL computed DOS has its peak far from E_F (0.8 eV away).
    e = np.linspace(-3, 3, 601)
    d = 0.15 + 3.0 * np.exp(-((e - 0.8) ** 2) / (2 * 0.05**2))
    feats = compute_dos_features(e, d)

    loop = GeneratorCriticLoop(PhysicsCritic(), max_revisions=0)
    result = loop.run("QRS", generator, feats)

    # The paper claimed it; the generator repeated it; the critic caught it.
    assert not result.succeeded
    assert "van_hove" in result.final.verdict.revision_feedback().lower()


def test_no_uploads_generator_degrades_gracefully():
    """With no uploaded documents, retrieval returns nothing and the generator
    still works from database reasoning alone (empty evidence block)."""
    store = DocumentStore()  # empty

    def base_generator(material, feedback, evidence_block):
        assert evidence_block == ""  # no evidence, as expected
        return Hypothesis(
            material=material,
            arm=PropertyArm.SUPERCONDUCTIVITY,
            n_ef_claim="high",
            favorability=Favorability.FAVORABLE,
        )

    generator = make_evidence_grounded_generator(base_generator, store)
    e = np.linspace(-3, 3, 601)
    feats = compute_dos_features(e, np.full_like(e, 1.5))
    loop = GeneratorCriticLoop(PhysicsCritic(), max_revisions=0)
    result = loop.run("M", generator, feats)
    assert result.succeeded
