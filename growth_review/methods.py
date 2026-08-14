"""Peculiar-velocity method taxonomy: labels, ordering, and the audit trail.

The family assignment itself lives in the ``method_family`` column of
``data/measurements/fsigma8_pv.csv``, so the data file is self-describing. What
lives here is the EVIDENCE: for each row, the sentence in its own paper that
fixes the assignment. Keeping it means the next disagreement about the taxonomy
is settled by reading this file, not by re-downloading 35 PDFs.

The axis of the classification is *what the estimator does to the data before
fitting*, not what optimiser it then runs. Almost every analysis maximises a
likelihood somewhere; that is a fitting detail and is recorded separately in the
``fit_technique`` column. Two earlier versions of this taxonomy were wrong in
opposite directions -- one used "MLE" as a family alongside "2pt" (mixing the
fitting step with the observable), the other over-corrected and folded the
field-level analyses into the two-point bucket. Maximum-likelihood *fields* is a
distinct compression regime, and it is the one Ravoux et al. (2025) is built on.
"""

FAMILY_LABEL = {
    "field_level":  "Field-level likelihood-based inference",
    "two_point":    "Two-point statistics",
    "vd_linear":    "Velocity--density comparison",
    "vd_dynamical": "Velocity--density comparison (particle forward model)",
    "consensus":    "Consensus of three estimators (DESI DR1)",
}

# Legend / caption order: increasing compression of the data, then the two
# velocity-density families side by side so the reader sees them as siblings.
FAMILY_ORDER = ["field_level", "two_point", "vd_linear", "vd_dynamical",
                "consensus"]

FAMILY_DEFINITION = {
    "field_level":
        "The likelihood of the observed density and velocity FIELD VALUES is "
        "written through a model covariance and maximised or sampled. No "
        "intermediate summary statistic is measured.",
    "two_point":
        "A two-point statistic -- correlation function, power spectrum, momentum "
        "spectrum -- is MEASURED from the data, then fitted with a model.",
    "vd_linear":
        "The velocity field is PREDICTED from a redshift-survey density field by "
        "linear theory -- directly, or through a Wiener filter / constrained "
        "realisation, which is still linear dynamics -- and compared with the "
        "measured velocities. Depends on an external density field, its bias, "
        "and linear theory.",
    "vd_dynamical":
        "The same comparison, but the field is evolved from initial conditions "
        "by a GRAVITY SOLVER acting on a particle field (LPT, COLA, "
        "particle-mesh, N-body), so the prediction reaches the mildly "
        "non-linear regime. Depends on the solver, its resolution, and the "
        "initial-condition prior instead of on linear theory.",
    "consensus":
        "A correlated combination of results from several of the above applied "
        "to the same data. Never group it with a family.",
}

# vd_dynamical is declared and empty, and that is the finding, not an oversight.
# Every published fsigma8 value from a velocity-density comparison predicts the
# velocity field with LINEAR dynamics. The analyses that do evolve a particle
# field publish reconstructions, Bayesian evidences and cosmographies -- not a
# growth rate. Checked 2026-08-14 against the four strongest candidates:
EMPTY_FAMILY_EVIDENCE = {
    "vd_dynamical": [
        ("Graziani et al. 2019", "1901.01818",
         "hierarchical Bayesian forward model of CF3, but 'assumes the LCDM "
         "model within the linear regime'; no fsigma8 quoted"),
        ("Boruah, Lavaux & Hudson 2022", "2111.15535",
         "'a forward-modelled velocity field reconstruction algorithm'; the "
         "forward model is of the distance data and Malmquist bias, and no "
         "fsigma8 value is reported"),
        ("Valade et al. 2026", "2602.03699",
         "'The dynamics are integrated with a particle-mesh gravity solver, "
         "thus probing the mildly non-linear regime' on CF4 -- the one clear "
         "particle-evolving analysis, and it reports no fsigma8"),
        ("Manticore-Local", "2505.10682",
         "BORG with a nonlinear gravitational solver on 2M++; reports Bayesian "
         "evidence comparisons, not a growth rate"),
    ],
}

# Rows whose paper says "forward model" but means the DISTANCE INDICATOR --
# Tully-Fisher scatter, Malmquist bias, the standardisation -- not the dynamics
# of the density field. They stay in vd_linear. Listed because the word alone
# would otherwise pull them into vd_dynamical on a careless reading.
FORWARD_MODEL_IS_THE_INDICATOR = {
    "Boubel2024": "forward-models the Tully-Fisher relation; the velocity field "
                  "is the 2M++ linear-theory prediction",
    "Stiskalek2026": "'jointly calibrates each distance indicator and a LINEAR "
                     "THEORY reconstruction'",
    "Boruah2020": "'forward likelihood' is the likelihood in distance-modulus "
                  "space; the field is Carrick et al.'s 2M++ reconstruction",
    "Stahl2021": "same framework as Boruah 2020, against the 2M++ reconstruction",
}

# key -> sentence from the paper that fixes the family. Read 2026-08-14.
EVIDENCE = {
    "Johnson2014": "'The covariance matrix of the data'; 'P(m|d) ~ L'; the velocity "
                   "field is modelled as a Gaussian random field.",
    "AdamsBlake2017": "Lai et al. 2023: 'In the development of the maximum likelihood "
                      "fields method, Adams & Blake (2017) were the first to "
                      "simultaneously model...'",
    "Huterer2017": "Section headings 'Peculiar velocity covariance' / 'Likelihood "
                   "Analysis'; fits the amplitude of the signal covariance.",
    "AdamsBlake2020": "'auto-covariance models of both probes in a fully "
                      "self-consistent, maximum-likelihood method'",
    "Lai2023": "'we focus on the maximum likelihood fields method to constrain the "
               "growth rate of structure'",
    "DESI_DR1_maxlike": "'using the maximum likelihood fields method'",

    "Howlett2017": "'suitable for measuring the velocity power spectrum' -- an "
                   "estimator is applied, then fitted. (Turner 2024 Table 1 groups "
                   "this with the maximum-likelihood analyses; the paper does not.)",
    "Qin2019": "momentum power spectrum estimator on 'the combined density and "
               "velocity fields'",
    "Qin2024": "'auto- and cross- power spectrum of the galaxy density and momentum "
               "fields'",
    "Turner2023": "'peculiar velocities and galaxy clustering correlations'",
    "Lyall2024": "'measuring peculiar velocity and galaxy clustering two-point "
                 "correlations'",
    "Dupuy2019": "velocity correlation function from Cosmicflows-3",
    "Courtois2023_ungrouped": "'the pairwise correlation of radial peculiar velocities'",
    "Courtois2023_grouped": "same estimator, grouped CF4",
    "Courtois2023_snia": "same estimator, CF4 SN Ia subset",
    "Wang2026_group": "'the parallel peculiar velocity correlation function'",
    "Wang2026_local": "same paper, local growth rate",
    "Nusser2017": "'the correlation between radial peculiar velocities ... and the "
                  "dipole moment of the 2MRS galaxy distribution'",
    "Franco2026": "'Modeling the enhancement of the correlation function within the "
                  "Kaiser formalism' (arXiv:2605.00450)",
    "Nguyen2025": "'auto- and cross-correlation function measurements'",
    "DESI_DR1_corrfunc": "galaxy and momentum correlation functions",
    "DESI_DR1_powerspec": "galaxy density and momentum power spectra",

    "Carrick2015": "'We compare the predicted peculiar velocities from 2M++ to "
                   "Tully-Fisher and SNe peculiar velocities'",
    "Turnbull2012": "'We have compared the peculiar velocities to the predictions "
                    "from the IRAS PSCz and have found Om^0.55 sigma8,lin of "
                    "0.40 +/- 0.07'",
    "Boruah2020": "'a simple chi2 minimization technique and a forward likelihood "
                  "method'",
    "Said2020": "'by comparing observed Fundamental Plane peculiar velocities ... "
                "with predicted velocities and densities'",
    "HollingerHudson2024": "'measured by comparing peculiar velocities with those "
                           "predicted from a galaxy density field'",
    "Stahl2021": "'we compare against the corresponding reconstruction from the 2M++ "
                 "galaxy redshift survey'",
    "LilowNusser2021": "'comparing our peculiar velocity CRs with the observed "
                       "velocities from Cosmicflows-3'; the CRs come from 'a Wiener "
                       "filter estimator in spherical Fourier-Bessel space with random "
                       "realizations of log-normally distributed density fields' -- a "
                       "reconstruction, but with linear dynamics, hence vd_linear",
    "Boubel2024": "'Comparing peculiar velocities predicted from the density field ... "
                  "with peculiar velocities measured using a distance indicator'",
    "Stiskalek2026": "'A unified hierarchical forward model jointly calibrates each "
                     "distance indicator and a linear theory reconstruction'",
    "Davis2011": "paper title: 'Local Gravity versus Local Velocity'",
    "MaBranchiniScott2012": "'A comparison of the galaxy peculiar velocity field with "
                            "the PSCz gravity field'",
    "Feix2015": "'Comparing these variations with the peculiar velocities inferred "
                "from galaxy redshift surveys'",
    "Feix2017": "same luminosity-fluctuation comparison, SDSS DR13",

    "DESI_DR1_consensus": "correlated combination of the three DESI DR1 estimators",
}


def evidence_report(df=None):
    """Printable audit: every row, its family, and the sentence behind it."""
    if df is None:
        from .io import load_fsigma8
        df = load_fsigma8(kind="measurement", method="pv", require_z=False,
                          desi_estimators=True)
    lines = []
    for fam in FAMILY_ORDER:
        sub = df[df["method_family"] == fam]
        if not len(sub):
            continue
        lines.append(f"\n{FAMILY_LABEL[fam]}  ({len(sub)})")
        lines.append("  " + FAMILY_DEFINITION[fam])
        for k in sub["key"]:
            lines.append(f"    {k:<24} {EVIDENCE.get(k, '** NO EVIDENCE RECORDED **')}")
    return "\n".join(lines)
