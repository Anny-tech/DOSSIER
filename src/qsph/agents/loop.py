"""The generator-critic loop.

This is the control structure that turns the physics-critic from a filter into a
reasoning loop: the generator proposes a hypothesis, the critic reviews it, and
if rejected, the critic's specific physical feedback is fed back for a revised
attempt. This mirrors the QSPHAgents generator-critic idea but replaces the
LLM-only critic with the deterministic physics-critic, which is the core
methodological change.

The generator is an injected callable, not a hard-coded LLM. This is deliberate:
  * it lets the loop be tested offline with a scripted generator (no API, no
    cost, fully deterministic), and
  * it makes the LLM provider swappable -- the real generator is an adapter that
    calls an LLM with the retrieved analogs and the revision feedback.

The loop records every attempt, so the full reasoning trace (proposal ->
critique -> revision) is available for the website UI, the paper's qualitative
examples, and the hackathon demo.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from qsph.critic.hypothesis import Hypothesis, Verdict
from qsph.critic.physics_critic import PhysicsCritic
from qsph.features.dos_features import DOSFeatures

# A generator takes the material, optional prior feedback, and returns a
# Hypothesis. Prior feedback is None on the first attempt, then the critic's
# revision_feedback() string on subsequent attempts.
GeneratorFn = Callable[[str, str | None], Hypothesis]


@dataclass
class LoopAttempt:
    """One proposal-critique cycle, retained for the reasoning trace."""

    attempt_index: int
    hypothesis: Hypothesis
    verdict: Verdict

    @property
    def accepted(self) -> bool:
        return self.verdict.accepted


@dataclass
class LoopResult:
    """Outcome of running the generator-critic loop for one material."""

    material: str
    attempts: list[LoopAttempt] = field(default_factory=list)

    @property
    def final(self) -> LoopAttempt | None:
        return self.attempts[-1] if self.attempts else None

    @property
    def succeeded(self) -> bool:
        return bool(self.final and self.final.accepted)

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)

    def trace(self) -> str:
        """Human-readable reasoning trace for the UI / demo / paper examples."""
        lines = [f"Material: {self.material}"]
        for a in self.attempts:
            status = "ACCEPTED" if a.accepted else "REJECTED"
            lines.append(f"\n-- Attempt {a.attempt_index} [{status}] --")
            lines.append(f"  favorability: {a.hypothesis.favorability.value}")
            if a.hypothesis.reasoning:
                lines.append(f"  reasoning: {a.hypothesis.reasoning}")
            if a.hypothesis.analog_materials:
                lines.append(
                    f"  analogs: {', '.join(a.hypothesis.analog_materials)}"
                )
            if not a.accepted:
                lines.append(f"  critic: {a.verdict.revision_feedback()}")
        return "\n".join(lines)


class GeneratorCriticLoop:
    """Runs generate -> critique -> revise until acceptance or budget exhausted.

    max_revisions bounds cost: with a real LLM generator, each attempt is an API
    call, so we cap attempts. A hypothesis that cannot be made physically
    consistent within the budget is returned as-is with its final (rejected)
    verdict -- an honest "could not produce a physically-supported hypothesis"
    outcome, which is itself informative (and better than forcing a false
    accept).
    """

    def __init__(self, critic: PhysicsCritic, max_revisions: int = 2):
        self.critic = critic
        self.max_revisions = max_revisions

    def run(
        self,
        material: str,
        generator: GeneratorFn,
        features: DOSFeatures | None,
    ) -> LoopResult:
        result = LoopResult(material=material)
        feedback: str | None = None

        for attempt_index in range(self.max_revisions + 1):
            hypothesis = generator(material, feedback)
            verdict = self.critic.review(hypothesis, features)
            result.attempts.append(
                LoopAttempt(attempt_index, hypothesis, verdict)
            )
            if verdict.accepted:
                break
            # Prepare feedback for the next revision.
            feedback = verdict.revision_feedback()

        return result
