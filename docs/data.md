# Data acquisition

The quantitative arms need two external data sources. Neither is committed to the
repo (licensing + size); this doc explains how to obtain each and wire it in.

## Materials Project (bulk DOS)

1. Get a free API key at https://materialsproject.org (Dashboard → API).
2. `pip install -e ".[data]"` (installs `mp-api`, `pymatgen`).
3. Set `MP_API_KEY` in your environment, or pass `api_key=` to
   `MaterialsProjectDOSLoader`.
4. The loader fetches DOS by material id and shifts energies by the Fermi level
   onto the (E − E_F) grid the feature extractor expects. **Getting the Fermi
   shift right is critical** — every downstream feature depends on it. Sanity-
   check against a known metal (finite N(E_F)) and a known insulator (gap at
   E_F) before trusting a batch.

## 3DSC (superconductivity labels)

3DSC (Sommer et al., *Sci. Data* 2023) maps SuperCon T_c values — **including
tested non-superconductors** — onto Materials Project structures. It is the
ground truth for the enrichment validation.

1. Download from the paper's repository / archive (see
   [doi:10.1038/s41597-023-02721-y](https://doi.org/10.1038/s41597-023-02721-y)
   and the linked GitHub/Zenodo). Use the `3DSC_MP` variant to align with the MP
   DOS source.
2. Place the CSV under `data/` (git-ignored).
3. `ThreeDSCLabelLoader(csv_path=...).load()` parses it into
   `SuperconductorLabel` records. Non-superconductors carry T_c = 0 → labelled
   `is_superconductor=False`, which is exactly the negative class the enrichment
   metric needs.

## The evidence-vs-validation rule (do not break this)

- **Evidence** (what the generator may reason from): MP DOS **and** literature
  DOS (via RAG). Retrieve freely.
- **Validation ground truth** (what enrichment is scored against): **database /
  computed only** for the superconductivity and thermoelectric arms. Never let
  literature-extracted DOS values become validation labels — that would mean
  grading the pipeline against values the same pipeline extracted (circular).

For the **topological arm** this rule is relaxed by necessity: there is no
database ground truth for surface DOS, so validation is qualitative against the
experimental record, and the arm is reported as a demonstration, not a
throughput claim.

## Offline development

`qsph.data.loaders.FixtureDOSProvider` generates synthetic, well-formed DOS with
controllable N(E_F) / peak placement so the entire pipeline (loop + enrichment)
runs in tests without network. It is clearly synthetic and never presented as
real data.
