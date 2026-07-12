"""End-to-end integration: fixture DOS -> generator-critic loop -> enrichment.

This test runs the WHOLE pipeline offline with synthetic DOS and scripted
generators, proving the pieces connect and that the critic-gated strategy
produces higher enrichment than an ungated one on a controlled set where we know
the ground truth. It is the executable skeleton of the paper's headline
experiment; swapping the fixture provider for real MP DOS and the scripted
generator for an LLM turns it into the real study.
"""

from __future__ import annotations

from qsph.agents.loop import GeneratorCriticLoop
from qsph.critic.hypothesis import Favorability, Hypothesis, PropertyArm
from qsph.critic.physics_critic import PhysicsCritic
from qsph.data.loaders import FixtureDOSProvider
from qsph.evaluate.enrichment import LabeledPrediction, compare_strategies


def test_full_pipeline_gated_beats_ungated():
    provider = FixtureDOSProvider()

    # Build a controlled material set:
    #   - "sc_*"  : real superconductors, high N(E_F) (favorable DOS)
    #   - "non_*" : non-superconductors, low N(E_F) (unfavorable DOS)
    # Ground truth is therefore correlated with N(E_F), as physics expects for
    # this conventional-SC toy setting.
    materials = []
    labels = {}
    for i in range(5):
        m = f"sc_{i}"
        materials.append(provider.make(m, n_ef=1.5))  # high DOS
        labels[m] = True
    for i in range(5):
        m = f"non_{i}"
        materials.append(provider.make(m, n_ef=0.1))  # low DOS
        labels[m] = False

    critic = PhysicsCritic()
    loop = GeneratorCriticLoop(critic, max_revisions=1)

    # Ungated LLM strategy: naively calls EVERYTHING favorable (the failure mode
    # -- confident over-prediction without physical grounding).
    def ungated_generator(material, feedback):
        return Hypothesis(
            material=material,
            arm=PropertyArm.SUPERCONDUCTIVITY,
            n_ef_claim="high",  # claims high regardless of truth
            favorability=Favorability.FAVORABLE,
        )

    # Critic-gated strategy: same naive generator, but the critic rejects the
    # 'favorable + high N(E_F)' claim when the DOS doesn't support it. On
    # rejection the (stubborn) generator can't fix it, so those end unaccepted
    # -> not counted as favorable. The gate is what filters the false claims.
    gated_predictions = []
    ungated_predictions = []
    for mdos in materials:
        feats = mdos.features()
        is_sc = labels[mdos.material]

        # Ungated: whatever the generator says, taken at face value.
        ungated_hyp = ungated_generator(mdos.material, None)
        ungated_fav = ungated_hyp.favorability is Favorability.FAVORABLE
        ungated_predictions.append(
            LabeledPrediction(mdos.material, is_sc, ungated_fav)
        )

        # Gated: favorable only if the loop ACCEPTS a favorable hypothesis.
        result = loop.run(mdos.material, ungated_generator, feats)
        gated_fav = (
            result.succeeded
            and result.final.hypothesis.favorability is Favorability.FAVORABLE
        )
        gated_predictions.append(
            LabeledPrediction(mdos.material, is_sc, gated_fav)
        )

    report = compare_strategies(
        {"critic_gated": gated_predictions, "ungated_llm": ungated_predictions}
    )

    # The ungated strategy calls everything favorable -> enrichment ~1 (no
    # discrimination). The gated strategy rejects the low-DOS false claims ->
    # its favorable set is SC-enriched.
    assert report.gated_beats("critic_gated", "ungated_llm")
    assert report.results["critic_gated"].precision == 1.0  # only true SCs pass
    assert report.results["ungated_llm"].enrichment < (
        report.results["critic_gated"].enrichment
    )
