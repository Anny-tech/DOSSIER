"""Tests for DOS feature computation.

We build synthetic DOS curves whose features we know analytically, so the tests
assert the extractor recovers the right physics. These synthetic curves stand in
for real Materials Project / literature DOS during offline testing; the feature
math is identical regardless of source.
"""

from __future__ import annotations

import numpy as np
import pytest

from qsph.features.dos_features import compute_dos_features


def _grid(lo=-3.0, hi=3.0, n=601):
    return np.linspace(lo, hi, n)


def _gaussian_peak(e, center, height, width):
    return height * np.exp(-((e - center) ** 2) / (2 * width**2))


def test_metallic_flat_dos_has_finite_n_ef():
    e = _grid()
    d = np.full_like(e, 0.8)  # flat metallic DOS
    f = compute_dos_features(e, d)
    assert f.n_ef == pytest.approx(0.8, abs=1e-6)
    assert not f.gap_present


def test_insulating_dos_detects_gap():
    e = _grid()
    # DOS zero in [-0.5, 0.5], finite outside -> gap straddling E_F.
    d = np.where(np.abs(e) <= 0.5, 0.0, 1.0)
    f = compute_dos_features(e, d)
    assert f.gap_present
    assert f.n_ef == pytest.approx(0.0, abs=1e-6)
    assert f.gap_width_ev > 0.8  # ~1.0 eV gap


def test_van_hove_peak_at_ef_detected():
    e = _grid()
    # Sharp peak centered exactly at E_F.
    d = 0.2 + _gaussian_peak(e, center=0.0, height=3.0, width=0.05)
    f = compute_dos_features(e, d)
    assert f.nearest_peak_offset_ev < 0.05  # peak essentially at E_F
    assert f.peak_prominence > 1.0


def test_van_hove_peak_shifted_off_ef():
    e = _grid()
    # Peak shifted 0.5 eV above E_F -- the canonical 'shifted analogy' case.
    d = 0.2 + _gaussian_peak(e, center=0.5, height=3.0, width=0.05)
    f = compute_dos_features(e, d)
    assert f.nearest_peak_offset_ev == pytest.approx(0.5, abs=0.05)


def test_valence_heavy_asymmetry():
    e = _grid()
    # More DOS below E_F than above -> valence-heavy, asymmetry > 0.
    d = np.where(e < 0, 1.5, 0.5)
    f = compute_dos_features(e, d)
    assert f.asymmetry > 0.1


def test_conduction_heavy_asymmetry():
    e = _grid()
    d = np.where(e > 0, 1.5, 0.5)
    f = compute_dos_features(e, d)
    assert f.asymmetry < -0.1


def test_symmetric_dos_low_asymmetry():
    e = _grid()
    d = np.full_like(e, 1.0)
    f = compute_dos_features(e, d)
    assert abs(f.asymmetry) < 0.1


def test_malformed_input_raises():
    e = _grid()
    with pytest.raises(ValueError):
        compute_dos_features(e, np.ones(len(e) - 1))  # length mismatch
    with pytest.raises(ValueError):
        compute_dos_features(e[::-1], np.ones_like(e))  # descending grid
    with pytest.raises(ValueError):
        compute_dos_features(e, -np.ones_like(e))  # negative DOS
