# Theory curves

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
