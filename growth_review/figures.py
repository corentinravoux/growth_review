"""Composed review figures.

Each ``fig_*`` returns ``(fig, meta)``. ``meta`` carries what the figure cannot
show: the LaTeX caption, the rows plotted, and what had to be dropped.

**Citations are not part of this package.** BibTeX keys belong to a manuscript,
not to a data compilation, so nothing here hardcodes them. Pass a ``bibkey``
mapping (``{row key: "bibkey1,bibkey2"}``) and the captions gain ``\\cite{...}``
clauses and a ``meta["missing_citations"]`` audit; pass nothing and you get the
same figures with plain descriptive captions. The notebook builds that mapping
automatically by matching the compilation's ``arxiv`` column against a ``.bib``
file, which is both less brittle and less work than maintaining a dict by hand.

Run ``growth-review-figures --outdir figures`` to write all of them.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import io
from .cosmology import FlatLCDM
from .methods import FAMILY_LABEL, FAMILY_ORDER
from .plotting import (annotate_points, annotate_provenance,
                       fit_ylim_to_labels, plot_by_family, plot_by_survey,
                       plot_forecasts, plot_measurements, plot_reference,
                       plot_theory, residual_panel)
from .style import DESI_COLOR, PALETTE, dodge_x, style_axes, use_style

# Per-tracer nudges for the DESI labels, in points. LRG2 and QSO sit almost on
# top of an eBOSS DR16 point, so they are pushed sideways rather than up.
DESI_RSD_OFFSETS = {"LRG1": (-4, 12), "LRG2": (-6, 14), "LRG3": (7, 12),
                    "ELG2": (0, -18), "QSO": (20, 4), "BGS": (0, -18)}

# Surveys the RSD panel leaves out by default. The compilation keeps them all --
# this is a figure decision, reversible with exclude_surveys=().
#
# The SDSS family contributed four generations of RSD measurement (MGS, BOSS
# DR12, eBOSS DR14, eBOSS DR16). Plotting all four buries the point the figure
# is making: showing only the first and the last makes the twenty-year gain in
# precision legible at a glance instead of turning that stretch of the panel
# into a thicket.
RSD_EXCLUDED_SURVEYS = (
    "BOSS DR12",       # intermediate SDSS-family release
    "eBOSS DR14 QSO",  # intermediate SDSS-family release
    "eBOSS DR14 LRG",  # intermediate SDSS-family release
)


# ------------------------------------------------------- citation plumbing
# Generic: these take the mapping as an argument and know nothing about any
# particular bibliography.
def _cite(keys):
    return r"\cite{" + keys + "}" if keys else r"\textbf{[NO CITATION]}"


def _cite_rows(df, bibkey):
    """(\\cite{...}, uncitable keys) for every row of `df`."""
    keys, missing = [], []
    for k in df["key"]:
        bk = bibkey.get(k)
        if bk is None:
            missing.append(k)
        else:
            keys.extend(bk.split(","))
    return _cite(",".join(dict.fromkeys(k for k in keys if k))), sorted(set(missing))


def _cite_by_family(df, bibkey):
    """Semicolon-separated "<family> \\cite{...}" clauses, consensus separate."""
    parts, missing, consensus = [], [], ""
    for fam in FAMILY_ORDER:
        sub = df[df["method_family"] == fam]
        if not len(sub):
            continue
        cite, miss = _cite_rows(sub, bibkey)
        missing.extend(miss)
        if fam == "consensus":
            consensus = cite
        else:
            parts.append(f"{FAMILY_LABEL[fam].lower()}~{cite}")
    other = df[~df["method_family"].isin(FAMILY_ORDER)]
    if len(other):
        cite, miss = _cite_rows(other, bibkey)
        missing.extend(miss)
        parts.append(f"other~{cite}")
    return "; ".join(parts), consensus, sorted(set(missing))


def _restrict(df, bibkey, cited_only):
    """(kept rows, keys dropped for having no bibliography entry)."""
    if not cited_only:
        return df, []
    if bibkey is None:
        raise ValueError("cited_only=True needs a bibkey mapping")
    dropped = sorted(set(df["key"]) - set(bibkey))
    return df[df["key"].isin(bibkey)].copy(), dropped


# ------------------------------------------------------------------- figure 1
def fig_overview(cosmo=None, bibkey=None, cited_only=False, scale="symlog"):
    """Every fsigma8 measurement in the review, PV and RSD, on one axis.

    This is the figure the symlog scale exists for. PV occupies z < 0.08 and RSD
    runs to z = 1.5; on a linear axis the entire peculiar-velocity literature
    collapses into the first few percent of the panel, and a pure log axis
    cannot draw the six z = 0 rows at all. Linear below z = 0.1 and log above
    gives each regime about half the panel and keeps z = 0 on it.
    """
    cosmo = cosmo or FlatLCDM()
    df, dropped = _restrict(io.load_fsigma8(kind="measurement"), bibkey, cited_only)

    pv = df[df["method"] == "pv"]
    rsd = df[df["method"] == "rsd"]
    is_desi = rsd["ref"].astype(str).str.startswith("DESI").to_numpy()
    xpv = dodge_x(pv["z"].to_numpy(), min_sep=0.006, step=0.014, scale=scale,
                  floor=0.0)
    xrsd = dodge_x(rsd["z"].to_numpy(), min_sep=0.04, step=0.016, scale=scale)

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(11, 7.5), sharex=True,
        gridspec_kw=dict(height_ratios=[2.7, 1], hspace=0.07))

    plot_reference(ax, cosmo, zmin=0.0, zmax=1.75)
    plot_by_family(ax, pv, x=xpv, ms=6.5)
    plot_measurements(ax, rsd[~is_desi], x=xrsd[~is_desi], color=PALETTE["violet"],
                      marker="^", ms=6.5,
                      label=f"Galaxy clustering / RSD ({(~is_desi).sum()})")
    plot_measurements(ax, rsd[is_desi], x=xrsd[is_desi], color=DESI_COLOR,
                      marker="^", ms=8, zorder=5,
                      label=f"DESI DR1 full shape ({is_desi.sum()})")

    style_axes(ax, xlabel="", ylim=(0.20, 0.78), scale=scale,
               ticks=[0, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5],
               legend_kw=dict(loc="upper center", bbox_to_anchor=(0.5, 1.30),
                              fontsize=9.5, ncol=3))
    ax.set_xlim(-0.004, 1.75)

    # The residual panel needs the same dodged abscissa as the top panel, so
    # rebuild the frame in the order the two x arrays were computed in.
    residual_panel(axr, pd.concat([pv, rsd], ignore_index=True), cosmo, by="method",
                   color_map={"pv": PALETTE["blue"], "rsd": PALETTE["violet"]},
                   marker_map={"pv": "o", "rsd": "^"},
                   x=np.concatenate([xpv, xrsd]), ms=5)
    axr.set_xlabel("Redshift $z$")
    axr.set_ylim(-45, 45)

    caption = (
        r"\caption{Compiled measurements of $f\sigma_8(z)$ from peculiar "
        r"velocities and from galaxy clustering, against the $\Lambda$CDM "
        r"prediction for a Planck~2018 fiducial cosmology (black). The abscissa "
        r"is linear below $z=0.1$ and logarithmic above, which keeps the $z=0$ "
        r"peculiar-velocity results on the panel while still showing the "
        r"clustering measurements out to $z\simeq1.5$. Peculiar-velocity results "
        r"are grouped by the method used to extract $f\sigma_8${PV}. Clustering "
        r"measurements{RSD}. Points at nearly coincident redshifts are displaced "
        r"horizontally for legibility. The lower panel shows the fractional "
        r"deviation from the fiducial prediction.}")
    missing = []
    if bibkey is not None:
        body, consensus, miss_pv = _cite_by_family(pv, bibkey)
        rsd_cite, miss_rsd = _cite_rows(rsd.sort_values("z"), bibkey)
        missing = sorted(set(miss_pv + miss_rsd))
        caption = caption.replace(
            "{PV}", f": {body}" + (f"; the DESI~DR1 consensus {consensus}"
                                   if consensus else ""))
        caption = caption.replace("{RSD}", f" are taken from {rsd_cite}")
    else:
        caption = caption.replace("{PV}", "").replace("{RSD}", " are shown in violet and red")

    return fig, dict(caption=caption, missing_citations=missing, dropped=dropped,
                     plotted=df, n_pv=len(pv), n_rsd=len(rsd))


# ------------------------------------------------------------------- figure 2
def fig_rsd(cosmo=None, bibkey=None, cited_only=False, scale="linear",
            exclude_surveys=RSD_EXCLUDED_SURVEYS):
    """Galaxy-clustering fsigma8: one colour per survey, DESI DR1 in red."""
    cosmo = cosmo or FlatLCDM()
    # Survey exclusion first, citation filter second, so `dropped` reports only
    # rows that were wanted on the panel and could not be cited.
    df = io.load_fsigma8(kind="measurement", method="rsd")
    excluded = sorted(set(df.loc[df["survey"].isin(exclude_surveys), "key"]))
    df, dropped = _restrict(df[~df["survey"].isin(exclude_surveys)].copy(),
                            bibkey, cited_only)
    is_desi = df["ref"].astype(str).str.startswith("DESI")
    pre, desi = df[~is_desi], df[is_desi]

    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    plot_reference(ax, cosmo, zmax=1.65)
    xpre = dodge_x(pre["z"].to_numpy(), min_sep=0.035, step=0.028, scale=scale)
    # No per-point labels on the pre-DESI set: 19 points in 1.5 units of z means
    # the annotations collide with each other and with the error bars. The
    # legend carries the survey; the tracer name is kept only for DESI, where
    # the six samples are the point of the figure.
    plot_by_survey(ax, pre, x=xpre, ms=6.5, annotate=False)
    plot_measurements(ax, desi, color=DESI_COLOR, marker="o", ms=8,
                      label="DESI DR1 (full shape)", zorder=5)
    annotate_points(ax, desi, offsets=DESI_RSD_OFFSETS, color=DESI_COLOR)

    # Top raised so the two-column legend sits above every error bar instead of
    # over the SDSS MGS one, whose 0.19 uncertainty reaches 0.72.
    style_axes(ax, xlim=(-0.02, 1.64), ylim=(0.20, 0.85), scale=scale,
               legend_kw=dict(loc="upper right", fontsize=13, ncol=2,
                              handletextpad=0.5, columnspacing=1.1))
    fig.tight_layout()

    caption = (
        r"\caption{Measurements of $f\sigma_8(z)$ from galaxy clustering "
        r"(redshift-space distortions), compared with the $\Lambda$CDM "
        r"prediction for a Planck~2018 fiducial cosmology (black line). Points "
        r"are coloured by survey; DESI~DR1 full-shape results are shown in red "
        r"and labelled by tracer sample. Of the SDSS-family measurements only "
        r"the first (SDSS~MGS) and the last (eBOSS~DR16) are shown, so that the "
        r"gain in precision over two decades stays legible. Points at nearly "
        r"coincident redshifts are displaced horizontally for legibility"
        r"{CITE}.}")
    missing = []
    if bibkey is not None:
        cite, missing = _cite_rows(df.sort_values("z"), bibkey)
        caption = caption.replace("{CITE}", f". Measurements are taken from {cite}")
    else:
        caption = caption.replace("{CITE}", "")

    return fig, dict(caption=caption, missing_citations=missing, dropped=dropped,
                     excluded_surveys=excluded, plotted=df, n=len(df))


# ------------------------------------------------------------------- figure 3
def fig_pv(cosmo=None, bibkey=None, cited_only=False, scale="linear",
           desi_estimators=False, provenance=True):
    """Peculiar-velocity fsigma8, grouped by method family.

    Grouped by method rather than by survey on purpose: the compilation is
    inhomogeneous -- the entries share data, and 6dFGSv, SDSS PV, SFI++, 2MTF
    and 2M++ recur across most of them -- so a per-survey legend would suggest
    an independence that is not there. What does vary meaningfully between rows
    is how the velocity field was compressed before fitting.
    """
    cosmo = cosmo or FlatLCDM()
    df, dropped = _restrict(
        io.load_fsigma8(kind="measurement", method="pv",
                        desi_estimators=desi_estimators), bibkey, cited_only)
    dropped_no_z = io.dropped_without_z()["key"].tolist()

    # floor=0: six rows sit at exactly z_eff = 0, and a symmetric dodge would
    # put half of them at negative redshift -- an artefact a reader takes for data
    # min_gap keeps the rotated per-point labels from overlapping: at 6 pt a
    # rotated line is ~7 pt wide, which is ~0.0015 in z on this panel.
    # min_gap keeps the rotated per-point labels from overlapping: at 7.5 pt a
    # rotated line is ~9 pt wide, which is ~0.0019 in z on this panel.
    # min_gap keeps the rotated per-point labels from overlapping: at 9 pt a
    # rotated line is ~11 pt wide, which is ~0.0023 in z on this panel.
    x = dodge_x(df["z"].to_numpy(), min_sep=0.005, step=0.0022, scale=scale,
                floor=0.0, min_gap=0.0023 if provenance else None)
    zmax = float(np.nanmax(df["z"])) if len(df) else 0.08

    # Low enough that the two-column legend sits under every error bar rather
    # than over one: the lowest whisker reaches 0.239, and a three-row legend at
    # this font is ~0.07 tall in data units.
    bottom = 0.12
    fig, ax = plt.subplots(figsize=(11, 7))
    plot_reference(ax, cosmo, zmax=zmax * 1.15)
    plot_by_family(ax, df, x=x, ms=7.5)
    texts = anchors = None
    if provenance:
        texts, anchors = annotate_provenance(ax, df, x=x)
    style_axes(ax, xlim=(-0.004, zmax * 1.22), ylim=(bottom, 0.74), scale=scale,
               legend_kw=dict(loc="lower left", fontsize=13, ncol=2))
    if provenance:
        # measured with the real renderer, not estimated -- see the docstring
        fit_ylim_to_labels(ax, texts, anchors, bottom)
    fig.tight_layout()

    caption = (
        r"\caption{Measurements of $f\sigma_8$ from peculiar velocities at "
        r"$z<0.1$, compared with the $\Lambda$CDM prediction for a Planck~2018 "
        r"fiducial cosmology (black line). The compilation is inhomogeneous --- "
        r"the measurements share data and differ in normalisation --- so points "
        r"are grouped by the method used to extract $f\sigma_8$ rather than by "
        r"survey{PV}. The DESI~DR1 consensus (red star){DESI} combines the three "
        r"estimators applied to that survey, accounting for their correlation. "
        r"Points at nearly coincident redshifts are displaced horizontally for "
        r"legibility. Error bars include the systematic contribution where the "
        r"authors quote one separately.}")
    missing = []
    if bibkey is not None:
        body, consensus, missing = _cite_by_family(df, bibkey)
        caption = caption.replace("{PV}", f": {body}").replace(
            "{DESI}", f" {consensus}" if consensus else "")
    else:
        caption = caption.replace("{PV}", "").replace("{DESI}", "")

    return fig, dict(caption=caption, missing_citations=missing, dropped=dropped,
                     dropped_no_z=dropped_no_z, plotted=df, n=len(df))


# ------------------------------------------------------------------- figure 4
def fig_forecasts(cosmo=None, scale="symlog"):
    """Forecast precisions on fsigma8, drawn around the fiducial prediction.

    No measurement appears here. Mixing the two on one panel is the standard way
    a growth figure misleads: a forecast bar sits exactly on the model curve by
    construction, so a reader who takes it for data concludes that the model is
    confirmed at that precision.
    """
    cosmo = cosmo or FlatLCDM()
    fc = io.load_fsigma8(kind="forecast")

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_reference(ax, cosmo, zmin=0.0, zmax=2.0)

    # LSST's nested redshift ranges are reduced to the widest bin of each
    # configuration so the panel does not show the same supernovae three times.
    sn = fc[fc["dataset"] == "fsigma8_sn_pv"]
    sn = sn[sn["label"].str.contains(r"\[0.02, 0.14\]")]
    rest = fc[fc["dataset"] != "fsigma8_sn_pv"]

    series = [(rest[rest["dataset"] == d], lab, col, mk) for d, lab, col, mk in [
        ("fsigma8_euclid", "Euclid (Amendola+16)", PALETTE["aqua"], "s"),
        ("bao_rsd_desi", "DESI design, ELG/LRG/QSO", PALETTE["violet"], "D"),
        ("bao_rsd_desi_bgs", "DESI design, BGS", PALETTE["magenta"], "^"),
        ("fsigma8_sn_pv_howlett2017", "SN-PV (Howlett+17)", PALETTE["green"], "v"),
        ("fsigma8_4most_bgs_howlett2017", "4MOST BGS (Howlett+17)", PALETTE["yellow"], "P"),
    ]]
    series.append((sn, "ZTF / LSST SN-PV (Carreres+23, Rosselli+25)",
                   PALETTE["orange"], "o"))

    for sub, label, color, marker in series:
        plot_forecasts(ax, sub, cosmo, color=color, marker=marker, ms=6,
                       label=f"{label} ({len(sub)})")

    style_axes(ax, xlim=(-0.004, 2.05), ylim=(0.30, 0.62), scale=scale,
               legend_kw=dict(loc="lower left", fontsize=9, ncol=1))
    fig.tight_layout()

    caption = (
        r"\caption{Forecast precision on $f\sigma_8(z)$ for current and planned "
        r"surveys, shown as error bars around the Planck~2018 $\Lambda$CDM "
        r"prediction (black line). These are projected uncertainties, not "
        r"measurements: the central values carry no information and are the "
        r"fiducial model's by construction. The abscissa is linear below $z=0.1$ "
        r"and logarithmic above. For the LSST supernova peculiar-velocity "
        r"forecasts only the widest redshift range of each configuration is "
        r"shown, since the narrower bins are drawn from the same supernovae.}")
    return fig, dict(caption=caption, missing_citations=[], dropped=[],
                     n=sum(len(s[0]) for s in series))


# ------------------------------------------------------------------- figure 5
def fig_theory(cosmo=None, bibkey=None, cited_only=False, k=0.1, scale="symlog"):
    """Tabulated COLA growth histories against the compiled measurements."""
    cosmo = cosmo or FlatLCDM()
    df, dropped = _restrict(
        io.load_fsigma8(kind="measurement", drop_derived=True), bibkey, cited_only)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_reference(ax, cosmo, zmin=0.0, zmax=2.0)
    plot_reference(ax, cosmo, zmin=0.0, zmax=2.0, gamma=0.68,
                   color=PALETTE["grey"], ls="-.", lw=1.6)
    for name, label, color in [
            ("cola_growth_gr", "COLA GR", PALETTE["blue"]),
            ("cola_growth_fr", "COLA $f(R)$", PALETTE["orange"]),
            ("cola_growth_dgp", "COLA nDGP", PALETTE["aqua"])]:
        plot_theory(ax, name, sigma8=cosmo.sigma8, k=k, zmax=2.0,
                    color=color, label=f"{label} ($k={k:g}$ h/Mpc)")

    x = dodge_x(df["z"].to_numpy(), min_sep=0.006, step=0.010, scale=scale,
                floor=0.0)
    plot_measurements(ax, df, x=x, color="0.35", marker="o", ms=5,
                      label=f"measurements ({len(df)})", alpha=0.75)

    style_axes(ax, xlim=(-0.004, 2.05), ylim=(0.20, 0.75), scale=scale,
               legend_kw=dict(loc="lower left", fontsize=9, ncol=2))
    fig.tight_layout()

    caption = (
        r"\caption{Tabulated growth histories from COLA simulations of GR, "
        rf"$f(R)$ and nDGP at $k={k:g}\,h\,$Mpc$^{{-1}}$, together with a "
        r"$\gamma=0.68$ growth-index curve, compared with the compiled "
        r"measurements. In $f(R)$ the growth rate is scale dependent, so the "
        r"curve shown is an effective $f\sigma_8$ at that wavenumber and not the "
        r"model's growth rate. Measurements whose published quantity is not "
        r"$f\sigma_8$ (those that quote $\Omega_m^{0.55}\sigma_8$ or $\beta$) are "
        r"excluded here, since identifying their result with $f\sigma_8$ assumes "
        r"the GR growth index and would make the comparison circular.}")
    return fig, dict(caption=caption, missing_citations=[], dropped=dropped,
                     plotted=df, n=len(df))


# ----------------------------------------------------------------------- driver
FIGURES = {
    "overview": fig_overview,
    "rsd": fig_rsd,
    "pv": fig_pv,
    "forecasts": fig_forecasts,
    "theory": fig_theory,
}


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Build the growth-review figures. Captions here carry no "
                    "citations: build those in the notebook, which derives the "
                    "BibTeX keys from a .bib file by arXiv matching.")
    p.add_argument("--outdir", default="figures", type=Path)
    p.add_argument("--only", nargs="*", choices=sorted(FIGURES), default=None)
    p.add_argument("--format", default="pdf")
    args = p.parse_args(argv)

    use_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cosmo = FlatLCDM()

    for name in (args.only or sorted(FIGURES)):
        fig, meta = FIGURES[name](cosmo=cosmo)
        path = args.outdir / f"growth_{name}.{args.format}"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

        print(f"\n{'=' * 78}\n{name}  ->  {path}\n{'=' * 78}")
        print(f"points plotted: {meta.get('n', meta.get('n_pv', 0) + meta.get('n_rsd', 0))}")
        if meta.get("dropped_no_z"):
            print(f"dropped, no author-assigned z_eff: {', '.join(meta['dropped_no_z'])}")
        print(meta["caption"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
