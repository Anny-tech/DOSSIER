"""LLM generator adapter tests, using a fake client (no API key, no network).

These cover the parts most likely to break in production: prompt construction
(does analog + evidence + feedback all reach the model?), defensive JSON parsing
(does a malformed response degrade to UNCERTAIN instead of crashing?), and full
integration with the generator-critic loop (does an LLM-produced hypothesis get
gated by the physics-critic like any other?).
"""

from __future__ import annotations

import json

import numpy as np

from qsph.agents.llm_generator import (
    build_generation_prompt,
    make_llm_generator,
    parse_hypothesis,
)
from qsph.agents.loop import GeneratorCriticLoop
from qsph.critic.hypothesis import Favorability, PropertyArm
from qsph.critic.physics_critic import PhysicsCritic
from qsph.features.dos_features import compute_dos_features
from qsph.retrieval.retriever import DocumentStore


class FakeClient:
    """Returns a scripted response; records the prompts it received."""

    def __init__(self, response: str):
        self.response = response
        self.last_system = None
        self.last_user = None

    def complete(self, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        return self.response


# -- prompt construction ----------------------------------------------------
def test_prompt_includes_analogs_evidence_and_feedback():
    system, user = build_generation_prompt(
        "NewMaterial",
        PropertyArm.SUPERCONDUCTIVITY,
        analog_block="AnalogA: N(E_F) high, van Hove at E_F",
        evidence_block="[doi:x#p0] some retrieved passage",
        critic_feedback="[van_hove_near_ef] peak is 0.5 eV off E_F",
    )
    assert "NewMaterial" in user
    assert "AnalogA" in user
    assert "doi:x#p0" in user
    assert "REJECTED" in user  # feedback framing present
    assert "0.5 eV off E_F" in user
    assert "JSON" in system


def test_prompt_omits_absent_sections():
    _system, user = build_generation_prompt(
        "M", PropertyArm.THERMOELECTRIC
    )
    assert "Known analog materials" not in user
    assert "REJECTED" not in user


# -- defensive parsing ------------------------------------------------------
def test_parse_valid_hypothesis():
    raw = json.dumps(
        {
            "material": "XYZ",
            "arm": "superconductivity",
            "n_ef_claim": "high",
            "van_hove_near_ef": True,
            "favorability": "favorable",
            "reasoning": "analogous to a known high-N(E_F) superconductor",
            "analog_materials": ["NbTi"],
        }
    )
    h = parse_hypothesis(raw, "XYZ", PropertyArm.SUPERCONDUCTIVITY)
    assert h.material == "XYZ"
    assert h.n_ef_claim == "high"
    assert h.van_hove_near_ef is True
    assert h.favorability is Favorability.FAVORABLE
    assert h.analog_materials == ["NbTi"]


def test_parse_strips_code_fences():
    raw = '```json\n{"material":"M","arm":"thermoelectric",' \
          '"favorability":"uncertain"}\n```'
    h = parse_hypothesis(raw, "M", PropertyArm.THERMOELECTRIC)
    assert h.material == "M"
    assert h.arm is PropertyArm.THERMOELECTRIC


def test_parse_malformed_returns_uncertain():
    h = parse_hypothesis("this is not json", "M", PropertyArm.SUPERCONDUCTIVITY)
    assert h.favorability is Favorability.UNCERTAIN
    assert h.material == "M"
    assert "could not be parsed" in h.reasoning


def test_parse_invalid_enum_falls_back_gracefully():
    raw = json.dumps(
        {"material": "M", "arm": "superconductivity", "favorability": "nonsense"}
    )
    h = parse_hypothesis(raw, "M", PropertyArm.SUPERCONDUCTIVITY)
    # Invalid favorability -> UNCERTAIN, not a crash.
    assert h.favorability is Favorability.UNCERTAIN


# -- full loop integration --------------------------------------------------
def test_llm_generator_in_loop_accepted():
    response = json.dumps(
        {
            "material": "M",
            "arm": "superconductivity",
            "n_ef_claim": "high",
            "favorability": "favorable",
            "reasoning": "high N(E_F) by analogy",
        }
    )
    client = FakeClient(response)
    generator = make_llm_generator(client, PropertyArm.SUPERCONDUCTIVITY)

    e = np.linspace(-3, 3, 601)
    feats = compute_dos_features(e, np.full_like(e, 1.5))  # high N(E_F) -> consistent
    loop = GeneratorCriticLoop(PhysicsCritic(), max_revisions=1)
    result = loop.run("M", generator, feats)
    assert result.succeeded


def test_llm_generator_wrong_claim_rejected_by_critic():
    """An LLM hypothesis is gated exactly like any other: a 'high N(E_F)' claim
    on a low-DOS material is rejected by the critic."""
    response = json.dumps(
        {
            "material": "M",
            "arm": "superconductivity",
            "n_ef_claim": "high",
            "favorability": "favorable",
        }
    )
    client = FakeClient(response)
    generator = make_llm_generator(client, PropertyArm.SUPERCONDUCTIVITY)

    e = np.linspace(-3, 3, 601)
    feats = compute_dos_features(e, np.full_like(e, 0.1))  # low N(E_F)
    loop = GeneratorCriticLoop(PhysicsCritic(), max_revisions=0)
    result = loop.run("M", generator, feats)
    assert not result.succeeded


def test_llm_generator_receives_evidence_from_store():
    """When an evidence store is supplied, retrieved passages reach the model."""
    store = DocumentStore()
    store.add_document(
        "doi:ev",
        "Material M shows a density of states peak at the Fermi level.",
        target_chars=200,
        overlap_chars=20,
    )
    response = json.dumps(
        {"material": "M", "arm": "superconductivity", "favorability": "uncertain"}
    )
    client = FakeClient(response)
    generator = make_llm_generator(
        client, PropertyArm.SUPERCONDUCTIVITY, evidence_store=store
    )
    e = np.linspace(-3, 3, 601)
    feats = compute_dos_features(e, np.full_like(e, 0.5))
    GeneratorCriticLoop(PhysicsCritic(), max_revisions=0).run("M", generator, feats)
    # The retrieved evidence (with provenance) was placed in the user prompt.
    assert "doi:ev#p0" in client.last_user


def test_llm_generator_uses_analog_provider():
    captured = {}

    def analog_provider(material):
        captured["material"] = material
        return "AnalogX: N(E_F) high, van Hove near E_F"

    response = json.dumps(
        {"material": "M", "arm": "superconductivity", "favorability": "uncertain"}
    )
    client = FakeClient(response)
    generator = make_llm_generator(
        client, PropertyArm.SUPERCONDUCTIVITY, analog_provider=analog_provider
    )
    e = np.linspace(-3, 3, 601)
    feats = compute_dos_features(e, np.full_like(e, 0.5))
    GeneratorCriticLoop(PhysicsCritic(), max_revisions=0).run("M", generator, feats)
    assert captured["material"] == "M"
    assert "AnalogX" in client.last_user
