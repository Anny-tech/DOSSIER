# QSPH — physics critic gated structure-property hypothesis generation

A generator physics critic agent that produces physically-falsifiable hypotheses about DOS-linked material properties, gating LLM analogy-based reasoning against computed and literature DOS evidence.**

This project extends **[QSPHAgents](https://github.com/sbanik2/QSPHAgents)**, the 2025 LLM-hackathon prototype by Suvo Banik, Ankita Biswas, Huanhuan Zhao, and Collin Kovacs. That prototype introduced the generator–critic idea for qualitative DOS reasoning; QSPH turns it into a rigorous, evaluable method by replacing the LLM-only critic with a **deterministic physics-critic** and sharpening the target to **property-relevant DOS features**.

> **Status:** research build. The DOS-feature layer, physics-critic, generator–critic loop, and enrichment-evaluation harness are implemented and tested (31 tests, all passing, offline). The LLM generator, Materials Project / 3DSC data access, RAG retrieval, and web app are structured but require credentials / external packages to run — they are **not** faked; stubs raise clear errors rather than returning fabricated data.

## The contribution, and the referee question it answers

**Q: If you already compute N(E_F) and the van Hove structure to run the critic, what does the LLM add?**

**A:** The generator reasons *by analogy from retrieved known materials* to a hypothesis about a material **outside the computed database** — a hypothetical substitution, a doped variant, or a literature-only compound whose DOS was never computed. Analogy-based reasoning is prone to confident, physically-impossible claims (e.g. inheriting a parent compound's "van Hove at E_F" when a substitution has shifted E_F off the peak). The physics-critic catches exactly those, and can **force a revision** rather than merely flagging. The contribution is *physically-falsifiable hypothesis generation for materials outside the computed set* — not property prediction, where graph networks already excel.

**Headline validation:** critic-gated "favorable-for-superconductivity" hypotheses are enriched for real superconductors (3DSC labels) more than (a) ungated LLM hypotheses and (b) the trivial parent-inheritance heuristic. The enrichment harness that computes this is implemented and tested.

## Three arms, honest about evidence and validation

| Arm | Evidence (reason FROM) | Ground truth (validate AGAINST) | Strength |
|---|---|---|---|
| **Superconductivity** | MP/3DSC bulk DOS + literature DOS | 3DSC T_c labels + computed N(E_F) | Quantitative (enrichment) — **headline** |
| **Thermoelectrics** | MP bulk DOS + literature DOS | MP-computed DOS features + TE datasets | Quantitative |
| **Topological insulators** | **Literature surface DOS only** | Experimental record + expert judgment | Qualitative demonstration |

The critic enforces **scope honesty in code**: it screens topological *candidates* from surface-DOS evidence but **refuses to certify a topological invariant from bulk DOS** — because bulk DOS cannot determine topology. That refusal is a feature, not a gap.

## The demo moment (works now)

```
Generator (reasoning by analogy from a parent compound):
  claims van Hove near E_F : True
  claims N(E_F)            : high
  claims favorability      : favorable

Critic verdict: REJECTED
  [FAIL] n_ef_claim: claimed 'high'; computed 0.150
  [FAIL] van_hove_near_ef: claimed True; nearest peak offset 0.500 eV (window 0.15)
  [FAIL] sc_favorability_basis: 'favorable' requires DOS basis; N(E_F)=0.150 ...
→ feedback fed back to generator for revision
```

Run it: `PYTHONPATH=src python -c "..."` (see `docs/` for the full script), or `pytest -q`.

## Install & test

```bash
pip install -e ".[dev]"     # tiny footprint, no torch/LLM needed
pytest -q                   # 31 tests, all offline
# optional capability groups:
pip install -e ".[data,llm,retrieval,web]"
```

## Reasoning from uploaded papers (using RAG)

Users can upload papers and have the generator reason from their content. This
is **retrieval-augmented**, not finetuned: documents are chunked, embedded, and
stored at upload; at query time the most relevant passages are retrieved and
handed to the generator **as evidence with provenance**. The model's weights are
never modified, so every claim remains traceable to a source passage — which is
exactly what keeps the physics-critic's grounding meaningful.

The critic still has the final say: if an uploaded paper *claims* a van Hove
singularity at E_F but the material's computed DOS contradicts it, the critic
rejects the hypothesis. Evidence informs reasoning; physics decides. (This is
why RAG, not LoRA: fine-tuning would dissolve the source text into weights,
destroying the provenance the critic depends on.)

## Layout

```
src/qsph/
  features/   DOS -> physical scalars (N(E_F), van Hove offset, asymmetry, gap)
  critic/     Hypothesis schema + the deterministic physics-critic
  agents/     generator-critic loop, LLM generator adapter (structured output)
  data/       Materials Project + 3DSC loaders (+ offline fixture provider)
  evaluate/   enrichment harness (the headline validation)
  retrieval/  RAG over uploaded papers: tiered PDF extraction (rule-based +
              quality-gated LLM fallback), chunking, embedders, retriever,
              evidence-grounded generator (evidence, not fine-tuning)
configs/      per-arm thresholds (documented, auditable)
tests/        59 tests: analogy-error case, RAG provenance, PDF ingestion,
              LLM-generator parsing, full-pipeline integration
docs/         data acquisition, retrieval notes
```

## Running the full pipeline

PDF upload → RAG retrieval → LLM generator → physics-critic, all wired:

```python
from qsph.retrieval.retriever import DocumentStore
from qsph.retrieval.ingestion import ingest_pdf
from qsph.agents.llm_generator import make_llm_generator, AnthropicClient
from qsph.agents.loop import GeneratorCriticLoop
from qsph.critic.hypothesis import PropertyArm
from qsph.critic.physics_critic import PhysicsCritic

store = DocumentStore()
ingest_pdf(store, "paper.pdf", source_id="doi:...")        # real PDF -> chunks
client = AnthropicClient()                                  # needs API key
generator = make_llm_generator(client, PropertyArm.SUPERCONDUCTIVITY,
                               evidence_store=store)         # RAG-grounded
loop = GeneratorCriticLoop(PhysicsCritic(), max_revisions=2)
result = loop.run("MyMaterial", generator, computed_dos_features)
print(result.trace())                                       # reasoning + critique
```

## Credits

Extends QSPHAgents (Banik, Biswas, Zhao, Kovacs, 2025). Validation data: 3DSC (Sommer et al., *Sci. Data* 2023, [doi:10.1038/s41597-023-02721-y](https://doi.org/10.1038/s41597-023-02721-y)); Materials Project.

## License

MIT.
