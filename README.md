# growth_review

Compiled constraints on the growth of structure — **measurements**, **forecasts**
and **theory curves** — with the loaders and plotting layer that turn them into
review figures.

The package exists to keep three things apart that growth plots routinely mix:

| kind | what it is | how it is drawn |
|---|---|---|
| `measurement` | a published central value with an uncertainty | outlined marker at the published value |
| `forecast` | a projected *precision*, with no central value of its own | thin bar around the fiducial curve, no marker face |
| `theory` | a tabulated model prediction | curve |

A forecast bar sits exactly on the model curve by construction. Styled like a
measurement, it tells the reader the model has been confirmed at that precision.

---

## Install

```bash
cd Packages/growth_review
uv venv && uv pip install -e ".[dev,notebook]"
# or: pip install -e ".[dev]"
```

The notebook expects a Jupyter kernel pointing at that environment:

```bash
.venv/bin/python -m ipykernel install --user --name growth_review \
    --display-name growth_review
```

## Use

```python
import growth_review as gr

gr.use_style()
print(gr.summary_table())                     # the registry

df = gr.load_fsigma8(kind="measurement")      # PV + RSD in one tidy schema
fig, meta = gr.figures.fig_pv(cited_only=True)
print(meta["caption"])                        # every plotted point cited
```

Build every figure:

```bash
growth-review-figures --outdir figures                 # full compilation
growth-review-figures --outdir figures --cited-only    # only what the HDR cites
growth-review-figures --only pv rsd --format png
```

It exits non-zero if any plotted measurement could not be cited.

`python -m growth_review` prints the registry.

## Layout

```
growth_review/
├── cosmology.py   flat-LCDM background + integrated linear growth (no CAMB/CLASS)
├── datasets.py    the registry: every file, its kind, probe, columns and caveats
├── io.py          readers, and the tidy fsigma8 view over all of them
├── methods.py     PV method taxonomy + the paper sentence behind each assignment
├── citations.py   BibTeX keys and the caption builders
├── style.py       palette, reserved colours, and the redshift x-scales
├── plotting.py    primitives over the tidy schema
├── figures.py     the five composed figures + CLI
└── data/
    ├── measurements/   fsigma8 (PV, RSD), BAO, S8, SN Hubble diagram
    ├── forecasts/      Euclid, DESI design, 4MOST, ZTF/LSST SN-PV
    └── theory/         COLA growth histories: GR, f(R), nDGP
```

`notebooks/growth_review.ipynb` walks through all of it.

## The redshift axis

The compilation runs from z = 0 (peculiar velocities, most below z = 0.08) to
z = 1.5 (clustering). No single scale serves both, so four are available and
each figure picks one:

- `linear` — right for a PV-only or RSD-only panel; useless for a mixed one,
  where every PV point stacks into the leftmost 5%.
- `symlog` — linear below z = 0.1, log above. **The default for mixed panels.**
- `log1p` — log(1+z). Also keeps z = 0, but 1+z only runs 1 → 1.1 across the PV
  range, so it barely expands the low-z end.
- `log` — silently drops the six z = 0 rows. Never on a panel with PV data.

Section 4 of the notebook plots the same data on all four.

## Peculiar-velocity method families

PV results are grouped by *what the estimator does to the data before fitting*,
not by what optimiser it runs afterwards:

| family | what it does | rows |
|---|---|---|
| `field_level` | the likelihood of the observed field values is written through a model covariance and maximised or sampled; no summary statistic first | 6 |
| `two_point` | a correlation function, power spectrum or momentum spectrum is measured, then fitted | 16 |
| `vd_linear` | the velocity field is predicted from a redshift-survey density field by **linear theory** — directly, or through a Wiener filter / constrained realisation, which is still linear dynamics — and compared with the measured velocities | 12 |
| `vd_dynamical` | the same comparison, but the field is evolved from initial conditions by a **gravity solver on a particle field** (LPT, COLA, particle-mesh, N-body), reaching the mildly non-linear regime | **0** |
| `consensus` | correlated combination of several of the above (DESI DR1) | 1 |

**`vd_dynamical` is empty, and that is the result, not an oversight.** Every
published fσ8 from a velocity–density comparison predicts the velocity field with
linear dynamics. The analyses that do evolve a particle field publish
reconstructions, Bayesian evidences and cosmographies — not a growth rate.
Checked against Graziani+2019 (linear regime), Boruah–Lavaux–Hudson 2022,
Valade+2026 (*"a particle-mesh gravity solver"* on CF4) and Manticore-Local;
none quotes fσ8. `methods.EMPTY_FAMILY_EVIDENCE` records the check, and a test
asserts it, so the day one appears the suite fails and says so.

Beware the word "forward model": in Boubel 2024, Stiskalek 2026, Boruah 2020 and
Stahl 2021 it refers to forward-modelling the **distance indicator** (Tully-Fisher
scatter, Malmquist bias), not the dynamics of the density field — Stiskalek's own
abstract says "a *linear theory* reconstruction". Those four stay in `vd_linear`;
`methods.FORWARD_MODEL_IS_THE_INDICATOR` records why, and a test pins it.

The fitting technique (`forward_likelihood`, `forward_model`,
`hierarchical_bayes`, `mcmc`, `chi2`) is a separate column, so nothing is lost.
`growth_review.methods.EVIDENCE` holds, for every row, the sentence in its own
paper that fixes its family — `gr.evidence_report()` prints the audit.

## Caveats that travel with the data

Each registry entry carries its own; the ones that bite most often:

- The PV rows are **not independent**. 6dFGSv, SDSS PV, SFI++, 2MTF and 2M++
  recur across most of them, and CF3/CF4 *contain* several. Never average.
- The four `DESI_DR1_*` rows are three correlated estimators on one dataset plus
  their consensus. `load_fsigma8` keeps only the consensus unless asked.
- Six PV rows normalise to the **linear** σ₈. Carrick et al. quote both for the
  same data: 0.401 ± 0.024 linear vs 0.427 ± 0.026, a 6.5% offset — larger than
  several error bars in the table.
- Three rows (`provenance='der'`) never quote fσ₈ at all; calling their result
  fσ₈ assumes f = Ωₘ^0.55, i.e. assumes GR. Pass `drop_derived=True` for any
  growth-index or modified-gravity fit.

## Provenance

The peculiar-velocity compilation was built by tracing every value to its own
paper; no row is second-hand. Its header block documents all 18 columns and 12
warnings, including six values where Turner (2024) Table 1 disagrees with the
source. It is kept in sync with
`~/.claude/skills/observational-cosmology/references/` (`pv-surveys.md`,
`fsigma8-pv-compilation.md`).
