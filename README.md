# growth_review

Compiled constraints on the growth of structure — **measurements**, **forecasts**
and **theory curves** — with the loaders and plotting layer that turn them into
review figures.

The package exists to keep three things apart that growth plots routinely mix:

| kind | what it is | how it is drawn |
|---|---|---|
| `measurement` | a published central value with an uncertainty | outlined marker at the published value |
| `forecast` | a projected *precision*, with no central value of its own | thin bar around the fiducial curve, no marker face |
| `theory` | a model prediction: the fiducial ΛCDM curve, or an EFTCAMB / simulation table | curve |

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
fig, meta = gr.figures.fig_pv()
print(meta["caption"])
```

Build every figure:

```bash
growth-review-figures --outdir figures
growth-review-figures --only pv rsd --format png
```

`python -m growth_review` prints both registries: the data files and the theory
models.

## Theory curves

Two sources, and the split is deliberate.

**The fiducial reference** — flat ΛCDM (or w₀wₐ) with GR growth — is integrated
here in numpy, because every figure needs it and it costs nothing:

```python
z = np.linspace(0, 2, 200)
gr.theory.fsigma8("GR", z)                      # the reference curve
gr.theory.fsigma8("gamma_0.68", z)              # f = Ωₘ(z)^γ, a diagnostic
ax.plot(*gr.theory.get("GR").curve(zmax=1.5))
gr.plotting.plot_theory(ax, "GR", "cola_fr")
```

The growth equation is integrated by RK4 in ln a; the solver reproduces the
closed-form ΛCDM growth integral to 1e-6 and f matches Ωₘ^0.55 to 0.5% (both
asserted in the test suite).

**Modified gravity comes from EFTCAMB — nothing here approximates it.**
`growth_review.theory.eftcamb` is the port of
`Science/Peculiar_Vel/theory_plots/modified_gravity/EFTCAMB_fsigma8.ipynb`: the
model registry (designer f(R), pure EFT, RPH, Hořava, ADE, K-mouflage,
quintessence, beyond Horndeski, scaling cubic Galileon, Jordan–Brans–Dicke), the
B₀↔f_R0 inversion, the JBD shooting path, `compute()` with CAMB's
highest-z-first ordering undone, the H(z)=H₀ consistency check, and the export
writer. It needs a compiled H-EFTCAMB (`$EFTCAMB_PATH`).

```bash
# where the build lives (NERSC)
growth-review-eftcamb-export \
    --models GR Kmouflage ScalingCubicGalileon Horava JBD_wBD100 \
    --fR0 -1e-4 -1e-5 -1e-6
# -> data/theory/eftcamb_<model>.ecsv, flags + cosmology + stability in the header
```

```python
# anywhere: copy the ECSVs into data/theory/ and they become ordinary models,
# with the source notebook's own colours, linestyles and labels
gr.theory.register_exports()
gr.plotting.plot_theory(ax, "GR", *gr.theory.list_models(family="eftcamb"))
```

`notebooks/eftcamb_theory.ipynb` runs the whole port and writes the exports.

**nDGP is the one model in that module that is not an EFTCAMB run** — it is not one
in the source notebook either. DGP has no EFTCAMB mapping, and the covariant
embedding of its decoupling limit is ghost-unstable on one branch and
gradient-unstable on the other once coupled to gravity (§3.5 there). What the
notebook uses instead, and what `eftcamb._ndgp_growth_ode` / `_ndgp_results`
transcribe — integrator, tolerances and σ8 anchoring included — is the standard
quasi-static treatment of the nDGP simulation and RSD literature: μ(a) = 1 +
1/(3β), β = 1 + 2Hr_c(1 + Ḣ/3H²) (Koyama & Maartens 2006; Schmidt 2009; Barreira
et al. 2016), on ΛCDM's expansion history, with σ8 anchored to the module's own
CAMB GR run. It still needs a build for that anchor, and scipy for the ODE. Its
z = 0 fσ8 enhancement reproduces the archived figure: +14.0% for H₀r_c = 1 and
+3.5% for H₀r_c = 5, against +13.5% and +3.5% read off the original.

An earlier version of this layer computed nDGP *and* Hu-Sawicki f(R) itself, from
a quasi-static μ(k,a) of my own writing. That was removed: an approximation
written here is not the model EFTCAMB defines, and shipping both invited reading
one as the other. `test_no_modified_gravity_model_is_computed_in_this_package`
pins the decision, and nDGP came back only as the source notebook's own code, in
the EFTCAMB module, reached the same way as everything else there.

**σ8 normalisation** is an explicit argument, because it changes the numbers:
`norm="early"` (default) means a shared primordial amplitude, what a Boltzmann
code does at fixed Aₛ, so a faster-growing model predicts a larger σ8 today;
`norm="today"` rescales to a common σ8(0) and leaves only the redshift shape. An
EFTCAMB export carries its own amplitude and is used unchanged; the COLA tables
carry growth only, so they are shape × fiducial σ8.

## Citations live outside the package

BibTeX keys belong to a manuscript, not to a data compilation, so **none are
hardcoded here**. Every `fig_*` takes an optional `bibkey` mapping
(`{row key: "key1,key2"}`); with it the captions gain `\cite{...}` clauses and a
`meta["missing_citations"]` audit, and `cited_only=True` restricts the panel to
what the mapping covers.

Section 10 of the notebook builds that mapping **by matching the compilation's
`arxiv` column against a `.bib` file**, so it self-updates as the bibliography
grows and cannot drift out of date. Only two things need a human: bib entries
that carry no arXiv identifier, and duplicate entries sharing one — both are
reported rather than guessed at.

## Layout

```
growth_review/
├── theory/        fsigma8(z) curves, callable anywhere
│   ├── background.py  flat CPL background: E(a), Omega_m(a), dlnH/dlna
│   ├── growth.py      RK4 integration of the GR growth equation + growth index
│   ├── model.py       GrowthModel / TableModel / Cosmology
│   ├── registry.py    named models, their styles, the one-liner accessors
│   ├── tables.py      tabulated curves (COLA runs, EFTCAMB exports)
│   └── eftcamb.py     THE modified-gravity backend: port of EFTCAMB_fsigma8.ipynb
├── datasets.py    the registry: every file, its kind, probe, columns and caveats
├── io.py          readers, and the tidy fsigma8 view over all of them
├── methods.py     PV method taxonomy + the paper sentence behind each assignment
├── style.py       palette, reserved colours, and the redshift x-scales
├── plotting.py    primitives over the tidy schema + the theory-curve drawing
├── figures.py     the five composed figures + CLI
└── data/
    ├── measurements/   fsigma8 (PV, RSD), BAO, S8, SN Hubble diagram
    ├── forecasts/      Euclid, DESI design, 4MOST, ZTF/LSST SN-PV
    └── theory/         COLA growth histories, and eftcamb_*.ecsv exports
```

## Notebooks

| notebook | what it covers | needs EFTCAMB |
|---|---|---|
| `notebooks/growth_review.ipynb` | the compilation itself: registry, loaders, PV method families, the five data figures, citation plumbing | no |
| `notebooks/eftcamb_theory.ipynb` | the port of `EFTCAMB_fsigma8.ipynb`: model registry, f_R0-specified designer f(R), fσ8 figure, B₀ scan, f(k,z) scale dependence, H(z) comparison, the 0.1% and 1% background-unmodified figures, and the export step | **yes** — shipped unexecuted |
| `notebooks/forecasts_vs_theory.ipynb` | ZTF / LSST peculiar-velocity and Euclid / DESI forecasts on a log-z axis: precision vs model deviation, σ per bin, the \|f_R0\| each programme reaches at 3σ | only for the model curves; the forecast panels build without it |

They run on the `growth_review` kernel:

```bash
.venv/bin/python -m ipykernel install --user --name growth_review \
    --display-name growth_review
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=growth_review notebooks/theory_curves.ipynb
```

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

Points at nearly coincident redshifts are dodged horizontally. Two guards on
that, because a dodge is a lie told for legibility and must not become a lie
about the data: `floor=0` stops the six rows quoted at exactly z_eff = 0 from
being spread to *negative* redshift, and `min_gap` enforces a separation
between all neighbours rather than only within a tied block — without it two
adjacent blocks can land a thousandth apart, invisible for markers and fatal
for per-point labels. Every figure that dodges says so in its caption.

## Peculiar-velocity method families

PV results are grouped by *what the estimator does to the data before fitting*,
not by what optimiser it runs afterwards:

| family | what it does | rows |
|---|---|---|
| `field_level` | the likelihood of the observed field values is written through a model covariance and maximised or sampled; no summary statistic first | 6 |
| `two_point` | a correlation function, power spectrum or momentum spectrum is measured, then fitted | 16 |
| `vd_linear` | the velocity field is predicted from a redshift-survey density field by **linear theory** — directly, or through a Wiener filter / constrained realisation, which is still linear dynamics — and compared with the measured velocities | 13 |
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

The PV figure labels every point with its source (`Author et al. Year`, rotated,
on a common baseline above the tallest error bar). Sections 12 and 13 of the
notebook print the matching caption citation lists and a full account of what
the review leaves out.

## Caveats that travel with the data

Each registry entry carries its own; the ones that bite most often:

- The PV rows are **not independent**. 6dFGSv, SDSS PV, SFI++, 2MTF and 2M++
  recur across most of them, and CF3/CF4 *contain* several. Never average.
- The four `DESI_DR1_*` rows are three correlated estimators on one dataset plus
  their consensus. `load_fsigma8` keeps only the consensus unless asked.
- Seven PV rows normalise to the **linear** σ₈. Carrick et al. quote both for the
  same data: 0.401 ± 0.024 linear vs 0.427 ± 0.026, a 6.5% offset — larger than
  several error bars in the table.
- Four rows (`provenance='der'`) never quote fσ₈ at all; calling their result
  fσ₈ assumes f = Ωₘ^0.55, i.e. assumes GR. Pass `drop_derived=True` for any
  growth-index or modified-gravity fit.
- The `probe` column marks whether the galaxy **clustering** is modelled too, not
  whether a density field is used. Every `vd_linear` row uses one (2M++, 2MRS,
  PSCz) while being labelled `PV`, because there the density field is an input
  rather than part of the fitted statistic.

## Provenance

The peculiar-velocity compilation was built by tracing every value to its own
paper; no row is second-hand. Its header block documents all 18 columns and 15
warnings, including six values where Turner (2024) Table 1 disagrees with the
source. This package is the **only** copy: the mirror that used to sit in
`~/.claude/skills/observational-cosmology/references/` was deleted on
2026-08-14. The discussion companions (`pv-surveys.md`,
`fsigma8-pv-compilation.md`) remain there and point here for the data.
