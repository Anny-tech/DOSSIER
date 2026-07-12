"""Generator-critic loop tests.

The important behaviour to prove: when the critic rejects a hypothesis, its
feedback reaches the generator, and a generator that uses that feedback can
converge to an accepted hypothesis. We test this with scripted generators (no
LLM) so the loop logic is verified deterministically.
"""

from __future__ import annotations

import numpy as np

from qsph.agents.loop import GeneratorCriticLoop
from qsph.critic.hypothesis import Favorability, Hypothesis, PropertyArm
from qsph.critic.physics_critic import PhysicsCritic
from qsph.features.dos_features import compute_dos_features


def _grid(lo=-3.0, hi=3.0, n=601):
    return np.linspace(lo, hi, n)


def test_loop_accepts_correct_first_hypothesis():
    e = _grid()
    feats = compute_dos_features(e, np.full_like(e, 1.5))  # high N(E_F)

    def generator(material: str, feedback: str | None) -> Hypothesis:
        return Hypothesis(
            material=material,
            arm=PropertyArm.SUPERCONDUCTIVITY,
            n_ef_claim="high",
            favorability=Favorability.FAVORABLE,
        )

    loop = GeneratorCriticLoop(PhysicsCritic(), max_revisions=2)
    result = loop.run("X", generator, feats)
    assert result.succeeded
    assert result.n_attempts == 1


def test_loop_revises_after_rejection():
    """First hypothesis is wrong (claims high N(E_F) on low-DOS material); the
    generator uses the feedback to correct on the second attempt."""
    e = _grid()
    feats = compute_dos_features(e, np.full_like(e, 0.15))  # low N(E_F)

    calls = {"n": 0}

    def generator(material: str, feedback: str | None) -> Hypothesis:
        calls["n"] += 1
        if feedback is None:
            # First attempt: wrong claim.
            return Hypothesis(
                material=material,
                arm=PropertyArm.SUPERCONDUCTIVITY,
                n_ef_claim="high",
                favorability=Favorability.FAVORABLE,
            )
        # Revision: the feedback told us N(E_F) is low; correct the claim and
        # abstain from favorability (honest response to low DOS).
        return Hypothesis(
            material=material,
            arm=PropertyArm.SUPERCONDUCTIVITY,
            n_ef_claim="low",
            favorability=Favorability.UNFAVORABLE,
        )

    loop = GeneratorCriticLoop(PhysicsCritic(), max_revisions=2)
    result = loop.run("X", generator, feats)
    assert result.succeeded
    assert result.n_attempts == 2
    assert calls["n"] == 2
    # The feedback actually reached the generator on the second call.
    assert result.attempts[0].verdict.accepted is False


def test_loop_gives_up_after_budget():
    """A generator that never corrects exhausts the budget and returns the
    final rejected verdict rather than forcing a false accept."""
    e = _grid()
    feats = compute_dos_features(e, np.full_like(e, 0.15))

    def stubborn_generator(material: str, feedback: str | None) -> Hypothesis:
        return Hypothesis(
            material=material,
            arm=PropertyArm.SUPERCONDUCTIVITY,
            n_ef_claim="high",  # always wrong for this low-DOS material
            favorability=Favorability.FAVORABLE,
        )

    loop = GeneratorCriticLoop(PhysicsCritic(), max_revisions=2)
    result = loop.run("X", stubborn_generator, feats)
    assert not result.succeeded
    assert result.n_attempts == 3  # initial + 2 revisions
    assert not result.final.accepted


def test_trace_is_populated():
    e = _grid()
    feats = compute_dos_features(e, np.full_like(e, 0.15))

    def generator(material: str, feedback: str | None) -> Hypothesis:
        return Hypothesis(
            material=material,
            arm=PropertyArm.SUPERCONDUCTIVITY,
            n_ef_claim="high",
            favorability=Favorability.FAVORABLE,
            reasoning="analogy to parent",
            analog_materials=["parent"],
        )

    loop = GeneratorCriticLoop(PhysicsCritic(), max_revisions=1)
    result = loop.run("X", generator, feats)
    trace = result.trace()
    assert "Material: X" in trace
    assert "REJECTED" in trace
    assert "analogy to parent" in trace
