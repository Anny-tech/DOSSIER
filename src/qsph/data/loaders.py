"""Data loaders for DOS features and superconductivity labels.

Two real sources back the quantitative superconductivity arm:
  * Materials Project (via mp-api) -- bulk DOS for computing DOSFeatures.
  * 3DSC (Sommer et al., Sci Data 2023) -- SuperCon-derived T_c labels mapped to
    MP structures, including tested NON-superconductors. This is the ground
    truth for the enrichment validation.

Design stance (same as the rest of the project): the real loaders require
network/credentials/downloaded data and are therefore thin and clearly-gated;
they raise a helpful error if the resource is absent rather than fabricating
data. A `FixtureDOSProvider` supplies synthetic-but-well-formed DOS for offline
tests of everything downstream, so the pipeline is fully exercisable without
network. No synthetic value is ever presented as real.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from qsph.features.dos_features import DOSFeatures, compute_dos_features


@dataclass
class MaterialDOS:
    """A material's DOS on the common (E - E_F) grid, source-tagged."""

    material: str
    energies_rel_ef: np.ndarray
    total_dos: np.ndarray
    source: str  # 'materials_project', 'literature', 'fixture'

    def features(self) -> DOSFeatures:
        return compute_dos_features(self.energies_rel_ef, self.total_dos)


@dataclass
class SuperconductorLabel:
    """3DSC ground-truth label for one material."""

    material: str
    is_superconductor: bool
    critical_temperature_k: float | None = None  # T_c if superconducting


class MaterialsProjectDOSLoader:
    """Loads bulk DOS from the Materials Project. Requires mp-api + API key."""

    def __init__(self, api_key: str | None = None):
        try:
            from mp_api.client import MPRester  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "MaterialsProjectDOSLoader requires mp-api: "
                "pip install 'qsph[data]'"
            ) from exc
        self._MPRester = MPRester
        self._api_key = api_key

    def load(self, material_id: str) -> MaterialDOS:  # pragma: no cover
        """Fetch DOS for an MP material id (e.g. 'mp-149').

        Not covered in CI (needs network + key). Converts the MP DOS onto the
        (E - E_F) grid the feature layer expects. The energy shift by the Fermi
        level is the critical step -- get it wrong and every feature is wrong.
        """
        with self._MPRester(self._api_key) as mpr:
            dos = mpr.get_dos_by_material_id(material_id)
        energies_rel_ef = np.asarray(dos.energies) - dos.efermi
        total = np.asarray(dos.densities[next(iter(dos.densities))])
        # Ensure ascending energy grid for the feature extractor.
        order = np.argsort(energies_rel_ef)
        return MaterialDOS(
            material=material_id,
            energies_rel_ef=energies_rel_ef[order],
            total_dos=total[order],
            source="materials_project",
        )


class ThreeDSCLabelLoader:
    """Loads superconductivity labels from a local 3DSC CSV export.

    3DSC is distributed as tabular data (composition, T_c, matched MP id, and
    tested non-superconductors). The user downloads it (see docs/data.md); this
    loader parses it. It raises if the file is absent rather than inventing
    labels.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load(self) -> list[SuperconductorLabel]:  # pragma: no cover - needs file
        import csv
        from pathlib import Path

        path = Path(self.csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"3DSC file not found at {self.csv_path}. See docs/data.md for "
                "how to obtain the 3DSC dataset."
            )
        labels: list[SuperconductorLabel] = []
        with path.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                tc = row.get("tc") or row.get("critical_temp")
                tc_val = float(tc) if tc not in (None, "", "None") else None
                # In 3DSC, tested non-superconductors carry T_c = 0.
                is_sc = tc_val is not None and tc_val > 0.0
                labels.append(
                    SuperconductorLabel(
                        material=row.get("material") or row.get("formula", ""),
                        is_superconductor=is_sc,
                        critical_temperature_k=tc_val,
                    )
                )
        return labels


class FixtureDOSProvider:
    """Offline provider of synthetic, well-formed DOS for testing the pipeline.

    Emphatically synthetic: it generates DOS curves with controllable N(E_F) and
    peak placement so tests can exercise the loop/enrichment code without
    network. Never use for real results; it carries no real materials data.
    """

    def __init__(self, grid=(-3.0, 3.0, 601)):
        self._grid = np.linspace(*grid)

    def make(
        self,
        material: str,
        *,
        n_ef: float = 1.0,
        peak_center: float | None = None,
        peak_height: float = 2.0,
        peak_width: float = 0.05,
        gap_half_width: float | None = None,
    ) -> MaterialDOS:
        e = self._grid
        d = np.full_like(e, n_ef)
        if gap_half_width is not None:
            d = np.where(np.abs(e) <= gap_half_width, 0.0, np.maximum(n_ef, 0.5))
        if peak_center is not None:
            d = d + peak_height * np.exp(
                -((e - peak_center) ** 2) / (2 * peak_width**2)
            )
        return MaterialDOS(
            material=material,
            energies_rel_ef=e,
            total_dos=d,
            source="fixture",
        )

    def batch(self, specs: Iterable[dict]) -> list[MaterialDOS]:
        return [self.make(**spec) for spec in specs]
