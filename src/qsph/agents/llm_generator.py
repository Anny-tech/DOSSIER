"""LLM-backed hypothesis generator.

This is the component that turns the generator-critic loop from a scripted test
harness into a real reasoning system. It prompts an LLM to produce a STRUCTURED
Hypothesis (not free text) from:

  * the target material,
  * retrieved analog materials and their DOS features (the Direction-1 core:
    reason by analogy from known materials to a target outside the computed set),
  * optionally, retrieved literature evidence passages (RAG), and
  * on revisions, the physics-critic's feedback on the previous attempt.

Design decisions:
  * Output is constrained to a JSON schema mirroring the Hypothesis model, so the
    result is machine-checkable by the critic. Free-text reasoning is captured in
    a dedicated field, not mixed into the claims.
  * The adapter is provider-agnostic behind a tiny `LLMClient` protocol. An
    Anthropic implementation is provided; it requires the `anthropic` package and
    an API key, and is not exercised in CI. All prompt-building and response-
    parsing logic lives in provider-agnostic functions that ARE tested offline
    with a fake client, so the parsing (the part most likely to break) is covered
    without network.
  * Parsing is defensive: a malformed LLM response yields an explicit low-content
    Hypothesis with favorability=UNCERTAIN rather than crashing the loop. A
    hypothesis the model failed to express cleanly must not masquerade as a
    confident claim.

Nothing here fabricates model output. The fake client used in tests only returns
strings the test supplies.
"""

from __future__ import annotations

import json
from typing import Protocol

from qsph.critic.hypothesis import (
    EvidenceTier,
    Favorability,
    Hypothesis,
    MetalClass,
    PropertyArm,
)


class LLMClient(Protocol):
    """Minimal chat interface an LLM backend must provide."""

    def complete(self, system: str, user: str) -> str:
        """Return the model's text response to a system+user prompt."""
        ...


class AnthropicClient:  # pragma: no cover - needs anthropic + API key
    """LLMClient backed by the Anthropic Messages API.

    Temperature defaults low: hypothesis generation should be stable so that the
    critic's judgement, not sampling noise, drives the loop. The generator's job
    is careful reasoning, not creative variety.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        api_key: str | None = None,
    ):
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "AnthropicClient requires: pip install 'qsph[llm]'"
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def complete(self, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")


# --------------------------------------------------------------------------
# Prompt construction (provider-agnostic, testable)
# --------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a materials-physics reasoning assistant. Given a target material, you \
propose a QUALITATIVE, mechanistically-justified hypothesis about a \
density-of-states (DOS) linked property, reasoning BY ANALOGY from known \
reference materials. You do not have the target's computed DOS; you infer likely \
DOS character from the analogs and any provided literature evidence.

You must return ONLY a JSON object with these fields (omit fields you cannot \
justify):
  material: string
  arm: one of "superconductivity", "thermoelectric", "topological"
  metal_class: one of "metal", "insulator", "semimetal" (optional)
  n_ef_claim: one of "high", "moderate", "low" (optional)
  van_hove_near_ef: boolean (optional)
  asymmetry_claim: "valence-heavy" | "conduction-heavy" | "symmetric" (optional)
  surface_metallic_in_bulk_gap: boolean (optional; topological arm only)
  favorability: "favorable" | "unfavorable" | "uncertain"
  reasoning: string (your mechanistic justification, citing analogs/evidence)
  analog_materials: list of strings (the known materials you reasoned from)

Be honest: if the evidence does not support a favorable call, say "uncertain" or \
"unfavorable". Do not claim a van Hove singularity at the Fermi level unless the \
analogs/evidence specifically support it for THIS material.\
"""


def build_generation_prompt(
    material: str,
    arm: PropertyArm,
    *,
    analog_block: str = "",
    evidence_block: str = "",
    critic_feedback: str | None = None,
) -> tuple[str, str]:
    """Build (system, user) prompts for a generation attempt.

    analog_block: formatted description of retrieved analog materials and their
        known DOS features (built by the retrieval/analog layer).
    evidence_block: RAG passages from uploaded papers (already citation-tagged).
    critic_feedback: on a revision, the previous verdict's revision_feedback().
        Including it is what makes the loop corrective rather than repetitive.
    """
    parts = [f"Target material: {material}", f"Property arm: {arm.value}"]
    if analog_block:
        parts += ["", "Known analog materials and their DOS features:", analog_block]
    if evidence_block:
        parts += ["", evidence_block]
    if critic_feedback:
        parts += [
            "",
            "A previous hypothesis was REJECTED by the physics critic for these "
            "reasons. Revise your hypothesis to be consistent with the physics:",
            critic_feedback,
        ]
    parts += ["", "Return the JSON hypothesis now."]
    return _SYSTEM_PROMPT, "\n".join(parts)


# --------------------------------------------------------------------------
# Response parsing (provider-agnostic, testable)
# --------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        if t.endswith("```"):
            t = t[: t.rfind("```")]
    return t.strip()


def parse_hypothesis(
    raw: str, material: str, arm: PropertyArm
) -> Hypothesis:
    """Parse an LLM JSON response into a Hypothesis, defensively.

    On any parse failure or invalid enum, returns a minimal UNCERTAIN hypothesis
    for the material rather than raising -- an unparseable model answer is an
    honest "no confident claim", not a crash. material/arm are passed in so the
    fallback is still well-formed and attributable.
    """
    fallback = Hypothesis(
        material=material,
        arm=arm,
        favorability=Favorability.UNCERTAIN,
        reasoning="(model response could not be parsed into a structured "
        "hypothesis; treated as uncertain)",
    )

    try:
        data = json.loads(_strip_fences(raw))
    except (json.JSONDecodeError, TypeError):
        return fallback
    if not isinstance(data, dict):
        return fallback

    def _enum(value, enum_cls, default=None):
        if value is None:
            return default
        try:
            return enum_cls(value)
        except ValueError:
            return default

    try:
        return Hypothesis(
            material=str(data.get("material", material)),
            arm=_enum(data.get("arm"), PropertyArm, arm) or arm,
            metal_class=_enum(data.get("metal_class"), MetalClass),
            n_ef_claim=data.get("n_ef_claim"),
            van_hove_near_ef=data.get("van_hove_near_ef"),
            asymmetry_claim=data.get("asymmetry_claim"),
            surface_metallic_in_bulk_gap=data.get("surface_metallic_in_bulk_gap"),
            favorability=_enum(
                data.get("favorability"), Favorability, Favorability.UNCERTAIN
            )
            or Favorability.UNCERTAIN,
            reasoning=str(data.get("reasoning", "")),
            analog_materials=list(data.get("analog_materials", []) or []),
            evidence_tier=_enum(
                data.get("evidence_tier"),
                EvidenceTier,
                EvidenceTier.DATABASE_COMPUTED,
            )
            or EvidenceTier.DATABASE_COMPUTED,
        )
    except Exception:  # noqa: BLE001 - any construction error -> safe fallback
        return fallback


# --------------------------------------------------------------------------
# The generator factory
# --------------------------------------------------------------------------

def make_llm_generator(
    client: LLMClient,
    arm: PropertyArm,
    *,
    analog_provider=None,
    evidence_store=None,
    top_k_evidence: int = 5,
):
    """Build a GeneratorFn(material, feedback) -> Hypothesis backed by an LLM.

    analog_provider: optional callable(material) -> analog_block string, i.e. the
        retrieved known materials and their DOS features. This is the
        Direction-1 mechanism; without it the model reasons from parametric
        knowledge alone (weaker, but still functional).
    evidence_store: optional DocumentStore for RAG over uploaded papers. When
        present, relevant passages are retrieved and included as evidence.

    The returned callable matches the loop's GeneratorFn signature exactly, so it
    drops into GeneratorCriticLoop.run() unchanged.
    """

    def generator(material: str, feedback: str | None) -> Hypothesis:
        analog_block = analog_provider(material) if analog_provider else ""

        evidence_block = ""
        if evidence_store is not None:
            from qsph.retrieval.retriever import build_evidence_block

            query = f"{material} density of states electronic structure"
            passages = evidence_store.retrieve(query, top_k=top_k_evidence)
            evidence_block = build_evidence_block(passages)

        system, user = build_generation_prompt(
            material,
            arm,
            analog_block=analog_block,
            evidence_block=evidence_block,
            critic_feedback=feedback,
        )
        raw = client.complete(system, user)
        return parse_hypothesis(raw, material, arm)

    return generator
