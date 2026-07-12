"""Physics-critic tests.

Each test encodes a specific reasoning-failure mode the critic must catch. The
headline is `test_catches_shifted_van_hove_analogy_error`: a hypothesis that
inherits 'van Hove at E_F' from a parent compound, when the target's peak has
shifted away, must be rejected. That is the concrete case proving the LLM's
analogy reasoning needs a physics gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from qsph.critic.hypothesis import (
    EvidenceTier,
    Favorability,
    Hypothesis,
    MetalClass,
    PropertyArm,
)
from qsph.critic.physics_critic import CriticThresholds, PhysicsCritic
from qsph.features.dos_features import compute_dos_features


def _grid(lo=-3.0, hi=3.0, n=601):
    return np.linspace(lo, hi, n)


def _peak(e, center, height, width):
    return height * np.exp(-((e - center) ** 2) / (2 * width**2))


@pytest.fixture
def critic() -> PhysicsCritic:
    return PhysicsCritic(CriticThresholds())


# --------------------------------------------------------------------------
# Metal/insulator classification
# --------------------------------------------------------------------------
def test_metal_claim_matches_metallic_dos(critic):
    e = _grid()
    feats = compute_dos_features(e, np.full_like(e, 0.8))
    hyp = Hypothesis(
        material="X", arm=PropertyArm.SUPERCONDUCTIVITY, metal_class=MetalClass.METAL
    )
    v = critic.review(hyp, feats)
    assert v.accepted


def test_metal_claim_rejected_for_insulator(critic):
    e = _grid()
    d = np.where(np.abs(e) <= 0.5, 0.0, 1.0)  # gap at E_F
    feats = compute_dos_features(e, d)
    hyp = Hypothesis(
        material="X", arm=PropertyArm.SUPERCONDUCTIVITY, metal_class=MetalClass.METAL
    )
    v = critic.review(hyp, feats)
    assert not v.accepted
    assert any("metal_class" in c.name for c in v.gating_checks)


# --------------------------------------------------------------------------
# HEADLINE: the shifted-van-Hove analogy error
# --------------------------------------------------------------------------
def test_catches_shifted_van_hove_analogy_error(critic):
    """The generator reasons: 'like parent compound P, this has a van Hove at
    E_F, so it is favorable for superconductivity.' But in the target the peak
    sits 0.5 eV above E_F. The critic must reject the van_hove_near_ef claim."""
    e = _grid()
    # Target DOS: prominent peak shifted OFF E_F by 0.5 eV, low N(E_F).
    d = 0.15 + _peak(e, center=0.5, height=3.0, width=0.05)
    feats = compute_dos_features(e, d)

    hyp = Hypothesis(
        material="doped-variant",
        arm=PropertyArm.SUPERCONDUCTIVITY,
        van_hove_near_ef=True,  # inherited from parent, WRONG for target
        favorability=Favorability.FAVORABLE,
        reasoning="By analogy to parent P, a van Hove sits at E_F.",
        analog_materials=["parent-P"],
    )
    v = critic.review(hyp, feats)
    assert not v.accepted
    reasons = v.revision_feedback()
    assert "van_hove" in reasons.lower()


def test_correct_van_hove_claim_accepted(critic):
    e = _grid()
    d = 0.5 + _peak(e, center=0.0, height=3.0, width=0.05)  # peak AT E_F
    feats = compute_dos_features(e, d)
    hyp = Hypothesis(
        material="X",
        arm=PropertyArm.SUPERCONDUCTIVITY,
        van_hove_near_ef=True,
        favorability=Favorability.FAVORABLE,
    )
    v = critic.review(hyp, feats)
    assert v.accepted


# --------------------------------------------------------------------------
# Superconductivity favorability requires a DOS basis
# --------------------------------------------------------------------------
def test_favorable_sc_rejected_without_dos_basis(critic):
    """'favorable' with low N(E_F) and no near-E_F peak is unsupported."""
    e = _grid()
    d = np.full_like(e, 0.1)  # very low DOS everywhere, no peak
    feats = compute_dos_features(e, d)
    hyp = Hypothesis(
        material="X",
        arm=PropertyArm.SUPERCONDUCTIVITY,
        favorability=Favorability.FAVORABLE,
        reasoning="claimed favorable with no DOS support",
    )
    v = critic.review(hyp, feats)
    assert not v.accepted
    assert any(c.name == "sc_favorability_basis" for c in v.gating_checks)


def test_favorable_sc_accepted_with_high_n_ef(critic):
    e = _grid()
    d = np.full_like(e, 1.5)  # high N(E_F)
    feats = compute_dos_features(e, d)
    hyp = Hypothesis(
        material="X",
        arm=PropertyArm.SUPERCONDUCTIVITY,
        n_ef_claim="high",
        favorability=Favorability.FAVORABLE,
    )
    v = critic.review(hyp, feats)
    assert v.accepted


def test_n_ef_claim_mismatch_rejected(critic):
    e = _grid()
    d = np.full_like(e, 0.1)  # low
    feats = compute_dos_features(e, d)
    hyp = Hypothesis(
        material="X", arm=PropertyArm.SUPERCONDUCTIVITY, n_ef_claim="high"
    )
    v = critic.review(hyp, feats)
    assert not v.accepted


# --------------------------------------------------------------------------
# Thermoelectric asymmetry
# --------------------------------------------------------------------------
def test_thermoelectric_asymmetry_match(critic):
    e = _grid()
    d = np.where(e < 0, 1.5, 0.5)  # valence-heavy
    feats = compute_dos_features(e, d)
    hyp = Hypothesis(
        material="Bi2Te3",
        arm=PropertyArm.THERMOELECTRIC,
        asymmetry_claim="valence-heavy",
    )
    v = critic.review(hyp, feats)
    assert v.accepted


def test_thermoelectric_asymmetry_mismatch_rejected(critic):
    e = _grid()
    d = np.where(e < 0, 1.5, 0.5)  # actually valence-heavy
    feats = compute_dos_features(e, d)
    hyp = Hypothesis(
        material="Bi2Te3",
        arm=PropertyArm.THERMOELECTRIC,
        asymmetry_claim="conduction-heavy",  # wrong sign
    )
    v = critic.review(hyp, feats)
    assert not v.accepted


# --------------------------------------------------------------------------
# Topological scope honesty
# --------------------------------------------------------------------------
def test_topological_rejects_database_only_evidence(critic):
    """A topological claim backed only by bulk database DOS is out of scope:
    bulk DOS cannot certify topology. Critic must refuse."""
    hyp = Hypothesis(
        material="Bi2Te3",
        arm=PropertyArm.TOPOLOGICAL,
        surface_metallic_in_bulk_gap=True,
        evidence_tier=EvidenceTier.DATABASE_COMPUTED,  # wrong evidence type
    )
    v = critic.review(hyp, None)
    assert not v.accepted
    assert any(c.name == "topological_scope" for c in v.gating_checks)


def test_topological_accepts_literature_evidence(critic):
    hyp = Hypothesis(
        material="Bi2Te3",
        arm=PropertyArm.TOPOLOGICAL,
        surface_metallic_in_bulk_gap=True,
        evidence_tier=EvidenceTier.LITERATURE_TIER1,
    )
    v = critic.review(hyp, None)
    assert v.accepted


def test_quantitative_arm_without_features_cannot_pass(critic):
    hyp = Hypothesis(
        material="X",
        arm=PropertyArm.SUPERCONDUCTIVITY,
        n_ef_claim="high",
        favorability=Favorability.FAVORABLE,
    )
    v = critic.review(hyp, None)  # no features supplied
    assert not v.accepted
    assert any(c.name == "features_available" for c in v.gating_checks)


def test_revision_feedback_is_actionable(critic):
    """Rejected hypotheses must yield specific feedback for the generator loop."""
    e = _grid()
    d = np.full_like(e, 0.1)
    feats = compute_dos_features(e, d)
    hyp = Hypothesis(
        material="X",
        arm=PropertyArm.SUPERCONDUCTIVITY,
        n_ef_claim="high",
        van_hove_near_ef=True,
        favorability=Favorability.FAVORABLE,
    )
    v = critic.review(hyp, feats)
    fb = v.revision_feedback()
    # Feedback should name the failing checks so the generator can correct them.
    assert "n_ef" in fb.lower() or "van_hove" in fb.lower()
    assert len(fb) > 0
