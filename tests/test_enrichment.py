"""Enrichment harness tests.

Proves the headline statistic behaves correctly: a strategy that concentrates
real superconductors in its 'favorable' set gets enrichment > 1, and the
head-to-head comparison correctly identifies when the gated strategy beats a
baseline. Uses synthetic labels; the same code runs on real 3DSC predictions.
"""

from __future__ import annotations

import pytest

from qsph.evaluate.enrichment import (
    LabeledPrediction,
    compare_strategies,
    compute_enrichment,
)


def _make(materials_flags):
    """materials_flags: list of (is_sc, predicted_favorable)."""
    return [
        LabeledPrediction(f"m{i}", is_sc, fav)
        for i, (is_sc, fav) in enumerate(materials_flags)
    ]


def test_perfect_strategy_high_enrichment():
    # 4 SCs, 6 non-SCs; strategy calls favorable exactly the 4 SCs.
    preds = _make(
        [(True, True)] * 4 + [(False, False)] * 6
    )
    r = compute_enrichment("perfect", preds)
    assert r.base_rate == pytest.approx(0.4)
    assert r.precision == pytest.approx(1.0)
    assert r.enrichment == pytest.approx(2.5)  # 1.0 / 0.4
    assert r.recall == pytest.approx(1.0)


def test_random_strategy_enrichment_near_one():
    # Strategy calls favorable a set with the same SC rate as the whole -> ~1x.
    preds = _make(
        [(True, True), (False, True)] * 3 + [(True, False), (False, False)] * 3
    )
    r = compute_enrichment("random", preds)
    assert r.enrichment == pytest.approx(1.0, abs=1e-9)


def test_gated_beats_parent_inheritance():
    """The paper's headline test: gated enrichment > parent-inheritance."""
    # Gated: favorable set is SC-pure (enrichment high).
    gated = _make([(True, True)] * 3 + [(False, False)] * 7)
    # Parent-inheritance: over-calls favorable, dragging in non-SCs.
    parent = _make(
        [(True, True)] * 3 + [(False, True)] * 4 + [(False, False)] * 3
    )
    # Ungated LLM: even worse, calls almost everything favorable.
    ungated = _make([(True, True)] * 3 + [(False, True)] * 7)

    report = compare_strategies(
        {
            "critic_gated": gated,
            "parent_inheritance": parent,
            "ungated_llm": ungated,
        }
    )
    assert report.gated_beats("critic_gated", "parent_inheritance")
    assert report.gated_beats("critic_gated", "ungated_llm")
    assert report.best_by_enrichment() == "critic_gated"


def test_empty_predictions_raise():
    with pytest.raises(ValueError):
        compute_enrichment("x", [])


def test_no_favorable_predictions_zero_precision():
    preds = _make([(True, False), (False, False)])
    r = compute_enrichment("cautious", preds)
    assert r.n_predicted_favorable == 0
    assert r.precision == 0.0
    assert r.enrichment == 0.0
