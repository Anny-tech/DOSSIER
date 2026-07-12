"""Hypothesis and verdict data models.

A Hypothesis is the qualitative, mechanistically-justified claim the generator
produces about a material's DOS-linked property. The physics-critic turns each
claim into a checkable proposition and returns a Verdict. Keeping these as
explicit typed models (not free-text) is what makes the critic possible: you
cannot deterministically check "the DOS looks favorable" prose, but you can
check the structured claims that prose decomposes into.

We separate three property arms, matching the agreed scope:
  * SUPERCONDUCTIVITY -> claims about N(E_F) magnitude and van Hove proximity
  * THERMOELECTRIC    -> claims about near-E_F DOS asymmetry (Seebeck/Mott)
  * TOPOLOGICAL       -> claims about surface-DOS metallicity-in-a-bulk-gap;
                         validated qualitatively against literature evidence,
                         NOT from bulk database DOS.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PropertyArm(str, Enum):
    SUPERCONDUCTIVITY = "superconductivity"
    THERMOELECTRIC = "thermoelectric"
    TOPOLOGICAL = "topological"


class MetalClass(str, Enum):
    METAL = "metal"
    INSULATOR = "insulator"
    SEMIMETAL = "semimetal"


class Favorability(str, Enum):
    """Binary-with-uncertain favorability for the property in question.

    We keep an explicit UNCERTAIN so the generator can decline rather than being
    forced into a false binary -- and so the critic can distinguish 'claimed
    favorable and wrong' from 'honestly abstained'.
    """

    FAVORABLE = "favorable"
    UNFAVORABLE = "unfavorable"
    UNCERTAIN = "uncertain"


class EvidenceTier(str, Enum):
    """Quality of the evidence a hypothesis rests on.

    Central to the topological arm (see project scope): a claim grounded in
    clean Tier-1 ARPES surface-DOS evidence is trusted more than one leaning on
    a Tier-3 compound known mainly for bulk topology. The critic reports this
    tier rather than flattening all confidence to one number.
    """

    DATABASE_COMPUTED = "database_computed"  # MP/3DSC bulk DOS
    LITERATURE_TIER1 = "literature_tier1"  # canonical, clean surface-DOS
    LITERATURE_TIER2 = "literature_tier2"  # well-validated, some noise
    LITERATURE_TIER3 = "literature_tier3"  # adjacent, weak surface-DOS evidence


class Hypothesis(BaseModel):
    """A structured, checkable claim about one material's DOS-linked property."""

    material: str
    arm: PropertyArm

    # Qualitative DOS claims the generator commits to. All optional because a
    # given arm only uses some; the critic checks whichever are present and
    # applicable.
    metal_class: MetalClass | None = None
    n_ef_claim: str | None = Field(
        default=None,
        description="Qualitative N(E_F): one of 'high', 'moderate', 'low'.",
    )
    van_hove_near_ef: bool | None = Field(
        default=None,
        description="Claim that a van Hove peak sits near E_F.",
    )
    asymmetry_claim: str | None = Field(
        default=None,
        description="Thermoelectric: 'valence-heavy', 'conduction-heavy', "
        "or 'symmetric' near E_F.",
    )
    surface_metallic_in_bulk_gap: bool | None = Field(
        default=None,
        description="Topological: claim of metallic surface states inside a "
        "bulk gap. Checked against literature evidence, not bulk DOS.",
    )

    favorability: Favorability = Favorability.UNCERTAIN
    reasoning: str = Field(
        default="",
        description="The mechanistic justification. Not machine-checked, but "
        "surfaced to the user and to expert evaluation.",
    )
    analog_materials: list[str] = Field(
        default_factory=list,
        description="Known materials the generator reasoned by analogy from. "
        "Central to Direction 1: the hypothesis is for a material outside the "
        "computed set, grounded in these retrieved analogs.",
    )
    evidence_tier: EvidenceTier = EvidenceTier.DATABASE_COMPUTED


class CheckResult(BaseModel):
    """Outcome of one physics check against computed/retrieved DOS features."""

    name: str
    passed: bool
    detail: str
    # Some checks are advisory (they annotate) vs gating (a fail rejects the
    # hypothesis). The critic decides overall verdict from the gating ones.
    gating: bool = True


class Verdict(BaseModel):
    """The critic's overall judgement on a hypothesis."""

    hypothesis: Hypothesis
    checks: list[CheckResult] = Field(default_factory=list)

    @property
    def gating_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if c.gating]

    @property
    def accepted(self) -> bool:
        """Accepted iff it was checked and every gating check passed.

        Empty gating set -> not accepted: an unchecked hypothesis is not a
        verified one (same conservative stance as the extraction verifier).
        """
        gating = self.gating_checks
        return bool(gating) and all(c.passed for c in gating)

    @property
    def rejection_reasons(self) -> list[str]:
        return [c.detail for c in self.gating_checks if not c.passed]

    def revision_feedback(self) -> str:
        """Textual feedback the generator uses to revise a rejected hypothesis.

        This is what turns the critic into a *loop* rather than a filter: the
        specific physical reason a claim failed is fed back so the next
        generation can correct it, mirroring how a human referee's comment
        drives a revision.
        """
        if self.accepted:
            return "Hypothesis is physically consistent with the DOS evidence."
        return " ".join(
            f"[{c.name}] {c.detail}" for c in self.gating_checks if not c.passed
        )
