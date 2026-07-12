"""Enrichment evaluation: the paper's headline validation.

The central empirical claim of Direction 1 is:

    Physics-critic-gated 'favorable' hypotheses are enriched for real
    superconductors, MORE so than (a) ungated LLM hypotheses and (b) the trivial
    parent-inheritance heuristic.

This module computes that enrichment against 3DSC ground-truth labels. Given a
set of materials each with a known label (is_superconductor) and each processed
by one or more *strategies* (gated / ungated / parent-inheritance), it computes,
per strategy, the precision and enrichment of the 'favorable'-predicted set.

Enrichment is the key statistic: it is the superconductor rate among predicted-
favorable materials divided by the base rate in the whole set. Enrichment > 1
means the strategy concentrates real superconductors; higher is better. Reporting
enrichment (not just precision) is what makes the comparison fair across sets
with different base rates.

This harness is offline and strategy-agnostic: it consumes pre-computed
predictions, so it can be tested with synthetic labels and used for real once
predictions are produced by the loop over 3DSC materials.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LabeledPrediction:
    """One material's ground-truth label and a strategy's favorability call."""

    material: str
    is_superconductor: bool  # 3DSC ground truth
    predicted_favorable: bool  # did this strategy call it favorable?


@dataclass
class EnrichmentResult:
    """Enrichment statistics for one strategy over a material set."""

    strategy: str
    n_total: int
    n_predicted_favorable: int
    n_true_sc_in_favorable: int
    base_rate: float  # superconductor fraction in the whole set
    precision: float  # SC fraction among predicted-favorable
    enrichment: float  # precision / base_rate
    recall: float  # fraction of all SCs captured in predicted-favorable

    def summary(self) -> str:
        return (
            f"{self.strategy}: enrichment={self.enrichment:.2f}x  "
            f"precision={self.precision:.3f}  recall={self.recall:.3f}  "
            f"(favorable={self.n_predicted_favorable}/{self.n_total}, "
            f"base_rate={self.base_rate:.3f})"
        )


def compute_enrichment(
    strategy: str, predictions: list[LabeledPrediction]
) -> EnrichmentResult:
    """Compute enrichment of the predicted-favorable set for one strategy."""
    n_total = len(predictions)
    if n_total == 0:
        raise ValueError("no predictions supplied")

    n_sc_total = sum(1 for p in predictions if p.is_superconductor)
    base_rate = n_sc_total / n_total

    favorable = [p for p in predictions if p.predicted_favorable]
    n_fav = len(favorable)
    n_sc_in_fav = sum(1 for p in favorable if p.is_superconductor)

    precision = n_sc_in_fav / n_fav if n_fav else 0.0
    enrichment = precision / base_rate if base_rate > 0 else 0.0
    recall = n_sc_in_fav / n_sc_total if n_sc_total else 0.0

    return EnrichmentResult(
        strategy=strategy,
        n_total=n_total,
        n_predicted_favorable=n_fav,
        n_true_sc_in_favorable=n_sc_in_fav,
        base_rate=base_rate,
        precision=precision,
        enrichment=enrichment,
        recall=recall,
    )


@dataclass
class ComparisonReport:
    """Head-to-head enrichment across strategies -- the paper's headline table."""

    results: dict[str, EnrichmentResult]

    def best_by_enrichment(self) -> str:
        return max(self.results, key=lambda s: self.results[s].enrichment)

    def summary(self) -> str:
        lines = ["Enrichment comparison (higher enrichment = better):"]
        for r in sorted(
            self.results.values(), key=lambda x: x.enrichment, reverse=True
        ):
            lines.append(f"  {r.summary()}")
        best = self.best_by_enrichment()
        lines.append(f"\nBest strategy by enrichment: {best}")
        return "\n".join(lines)

    def gated_beats(self, gated: str, baseline: str) -> bool:
        """Does the gated strategy achieve higher enrichment than a baseline?

        This is the literal test of the paper's claim. Returns True iff the
        gated strategy's enrichment strictly exceeds the baseline's.
        """
        if gated not in self.results or baseline not in self.results:
            raise KeyError("strategy not present in report")
        return self.results[gated].enrichment > self.results[baseline].enrichment


def compare_strategies(
    predictions_by_strategy: dict[str, list[LabeledPrediction]],
) -> ComparisonReport:
    """Compute enrichment for each strategy and bundle into a comparison.

    Typical strategies: 'critic_gated', 'ungated_llm', 'parent_inheritance'.
    The paper's claim holds iff critic_gated enrichment exceeds BOTH baselines.
    """
    results = {
        strat: compute_enrichment(strat, preds)
        for strat, preds in predictions_by_strategy.items()
    }
    return ComparisonReport(results=results)
