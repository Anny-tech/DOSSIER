"""Density-of-states feature computation.

This module turns a raw DOS curve into the small set of physically-meaningful
scalars that (a) the generator is asked to reason about qualitatively and (b)
the physics-critic checks those qualitative claims against. Keeping this layer
pure -- a DOS array in, physical scalars out, no LLM, no I/O -- is deliberate:
it is the deterministic ground the critic stands on, and it is where the
"self-consistent LLM error" is caught, because these numbers do not care what
the LLM believes.

Design decision: features are computed from a DOS given as energies RELATIVE TO
THE FERMI LEVEL (E - E_F, in eV) plus total DOS (states/eV/cell). This makes the
layer indifferent to the DOS *source* -- Materials Project, a slab calculation,
or a value parsed from a paper -- which is exactly the evidence-source-agnostic
design the project requires. The critic never needs to know where the DOS came
from, only that it is expressed on this common (E - E_F) grid.

Physics notes (kept explicit so a domain reader can audit the choices):
  * N(E_F): DOS at E - E_F = 0. Interpolated, since the grid rarely lands
    exactly on 0. This is the single most important scalar for conventional
    (BCS/electron-phonon) superconductivity -- higher N(E_F) generally means
    stronger coupling, all else equal.
  * van Hove proximity: the energy distance from E_F to the nearest prominent
    DOS peak. A van Hove singularity sitting AT E_F is a well-known booster of
    the electron-phonon coupling; "is there a peak within a small window of
    E_F" is a checkable, superconductivity-relevant claim.
  * spectral asymmetry near E_F: whether DOS weight just below E_F (valence
    side) exceeds that just above (conduction side). This is the DOS-slope
    quantity tied to the Seebeck coefficient via the Mott relation, so it is
    what the *thermoelectric* arm's critic will check.
  * gap character: whether N(E) is ~0 over a window straddling E_F (insulating)
    vs finite (metallic). Used for the metal/insulator classification the
    critic verifies, and as the bulk-side screen for topological candidates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# np.trapz was removed in NumPy 2.0 in favour of np.trapezoid. Support both so
# the code runs across NumPy versions without forcing a pin.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


@dataclass(frozen=True)
class DOSFeatures:
    """The computed physical scalars a hypothesis is checked against.

    Every field is a deterministic function of the input DOS. `n_ef` and the
    peak/gap quantities are what the critic's checks consume. We keep the raw
    numbers here and let the critic apply thresholds, so thresholds live in one
    place (the critic config) rather than being baked into feature extraction.
    """

    n_ef: float  # DOS at E_F, states/eV/cell
    nearest_peak_offset_ev: float  # |E_peak - E_F| for the nearest prominent peak
    nearest_peak_height: float  # DOS value at that peak
    peak_prominence: float  # how much that peak rises above local baseline
    valence_weight: float  # integrated DOS in a window just below E_F
    conduction_weight: float  # integrated DOS in a window just above E_F
    gap_present: bool  # True if DOS ~ 0 across a window straddling E_F
    gap_width_ev: float  # estimated gap width if gap_present, else 0.0

    @property
    def asymmetry(self) -> float:
        """Signed valence-vs-conduction imbalance in [-1, 1].

        Positive => more weight below E_F (valence-heavy). This is the sign the
        thermoelectric critic compares against a hypothesis's asymmetry claim.
        Returns 0.0 when both windows are empty (degenerate), the conservative
        choice.
        """
        total = self.valence_weight + self.conduction_weight
        if total <= 0.0:
            return 0.0
        return (self.valence_weight - self.conduction_weight) / total


def _interp_at(energies: np.ndarray, dos: np.ndarray, x: float) -> float:
    """Linearly interpolate the DOS at energy x (relative to E_F).

    np.interp handles the out-of-range case by clamping to the endpoints, which
    is acceptable here because we only ever query at or near E_F = 0, which is
    interior to any reasonable DOS window.
    """
    return float(np.interp(x, energies, dos))


def compute_dos_features(
    energies_rel_ef: np.ndarray,
    total_dos: np.ndarray,
    *,
    asymmetry_window_ev: float = 1.0,
    peak_search_window_ev: float = 2.0,
    gap_threshold: float = 0.05,
    gap_probe_window_ev: float = 0.3,
    min_peak_prominence: float = 0.1,
) -> DOSFeatures:
    """Compute DOS features from a DOS curve on an (E - E_F) grid.

    Args:
        energies_rel_ef: 1D array of energies relative to E_F (eV), ascending.
        total_dos: 1D array of DOS values (states/eV/cell), same length.
        asymmetry_window_ev: half-window each side of E_F for valence/conduction
            weight integration (thermoelectric-relevant).
        peak_search_window_ev: how far from E_F to look for the nearest peak
            (van-Hove-relevant). Peaks outside this window are irrelevant to E_F.
        gap_threshold: DOS below this counts as "zero" for gap detection.
        gap_probe_window_ev: half-window straddling E_F used to test for a gap.
        min_peak_prominence: minimum rise above local baseline to count a peak,
            filtering numerical noise wiggles.

    Returns:
        DOSFeatures with all scalars populated.

    Raises:
        ValueError on malformed input (mismatched lengths, non-ascending grid,
        empty arrays), because silently proceeding would corrupt every check
        downstream.
    """
    e = np.asarray(energies_rel_ef, dtype=float)
    d = np.asarray(total_dos, dtype=float)

    if e.ndim != 1 or d.ndim != 1:
        raise ValueError("energies and dos must be 1D arrays")
    if e.shape != d.shape:
        raise ValueError(f"length mismatch: {e.shape} vs {d.shape}")
    if e.size < 3:
        raise ValueError("need at least 3 DOS points")
    if not np.all(np.diff(e) > 0):
        raise ValueError("energies_rel_ef must be strictly ascending")
    if np.any(d < 0):
        raise ValueError("DOS values must be non-negative")

    # --- N(E_F) --------------------------------------------------------------
    n_ef = _interp_at(e, d, 0.0)

    # --- gap detection straddling E_F ---------------------------------------
    gap_mask = np.abs(e) <= gap_probe_window_ev
    gap_present = bool(np.all(d[gap_mask] < gap_threshold)) if gap_mask.any() else False
    gap_width_ev = _estimate_gap_width(e, d, gap_threshold) if gap_present else 0.0

    # --- nearest prominent peak to E_F (van Hove proximity) ------------------
    (
        nearest_peak_offset,
        nearest_peak_height,
        peak_prominence,
    ) = _nearest_peak_to_ef(
        e, d, window=peak_search_window_ev, min_prominence=min_peak_prominence
    )

    # --- valence / conduction weight (asymmetry, thermoelectric) -------------
    val_mask = (e < 0) & (e >= -asymmetry_window_ev)
    con_mask = (e > 0) & (e <= asymmetry_window_ev)
    valence_weight = (
        float(_trapezoid(d[val_mask], e[val_mask])) if val_mask.sum() > 1 else 0.0
    )
    conduction_weight = (
        float(_trapezoid(d[con_mask], e[con_mask])) if con_mask.sum() > 1 else 0.0
    )
    # trapz over an ascending-then-masked region can be negative in sign only if
    # the grid direction flips; we take magnitude since these are weights.
    valence_weight = abs(valence_weight)
    conduction_weight = abs(conduction_weight)

    return DOSFeatures(
        n_ef=n_ef,
        nearest_peak_offset_ev=nearest_peak_offset,
        nearest_peak_height=nearest_peak_height,
        peak_prominence=peak_prominence,
        valence_weight=valence_weight,
        conduction_weight=conduction_weight,
        gap_present=gap_present,
        gap_width_ev=gap_width_ev,
    )


def _nearest_peak_to_ef(
    e: np.ndarray, d: np.ndarray, *, window: float, min_prominence: float
) -> tuple[float, float, float]:
    """Find the prominent local maximum nearest to E_F within +/- window.

    Returns (|E_peak - E_F|, DOS_at_peak, prominence). If no prominent peak
    exists in the window, returns (inf, 0.0, 0.0) so the critic reads it as
    "no van Hove structure near E_F" rather than crashing.

    Prominence here is a lightweight local measure: peak height minus the
    higher of the two adjacent local minima within the window. This avoids a
    scipy dependency and is transparent to audit; it is sufficient for the
    coarse "is there a real peak near E_F" question the critic asks.
    """
    in_win = np.abs(e) <= window
    if in_win.sum() < 3:
        return float("inf"), 0.0, 0.0

    ew = e[in_win]
    dw = d[in_win]

    # Local maxima: strictly greater than both neighbours.
    best_offset = float("inf")
    best_height = 0.0
    best_prom = 0.0
    for i in range(1, len(dw) - 1):
        if dw[i] > dw[i - 1] and dw[i] > dw[i + 1]:
            # local baseline = min DOS on each side within the window
            left_min = dw[: i + 1].min()
            right_min = dw[i:].min()
            prominence = dw[i] - max(left_min, right_min)
            if prominence < min_prominence:
                continue
            offset = abs(ew[i])
            if offset < best_offset:
                best_offset = offset
                best_height = float(dw[i])
                best_prom = float(prominence)
    return best_offset, best_height, best_prom


def _estimate_gap_width(e: np.ndarray, d: np.ndarray, threshold: float) -> float:
    """Estimate the width of the gap straddling E_F.

    Walks outward from E_F in both directions until DOS rises above threshold;
    the gap width is the sum of the two distances. Coarse but adequate for the
    metal/insulator distinction the critic needs.
    """
    # index nearest to E_F
    i0 = int(np.argmin(np.abs(e)))
    # walk up
    hi = i0
    while hi < len(d) - 1 and d[hi] < threshold:
        hi += 1
    # walk down
    lo = i0
    while lo > 0 and d[lo] < threshold:
        lo -= 1
    return float(e[hi] - e[lo])
