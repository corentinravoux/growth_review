"""Composed review figures.

Each ``fig_*`` returns ``(fig, meta)``. ``meta`` carries what the figure cannot
show: the LaTeX caption with every plotted point cited, the list of points that
had to be dropped, and any point that could not be cited. A figure whose
``meta["missing_citations"]`` is non-empty is not finished.

Run ``python -m growth_review.figures --outdir figures`` (or the installed
``growth-review-figures``) to write all of them.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import citations as cit
from . import io
from .cosmology import FlatLCDM
from .methods import FAMILY_LABEL, FAMILY_ORDER
from .plotting import (annotate_points, plot_by_family, plot_by_survey,
                       plot_forecasts, plot_measurements, plot_reference,
                       plot_theory, residual_panel)
from .style import (DESI_COLOR, FAMILY_COLOR, FAMILY_MARKER, MARKERS, PALETTE,
                    dodge_x, style_axes, use_style)

DESI_RSD_OFFSETS = {"LRG1": (-4, 12), "LRG2": (0, 12), "LRG3": (7, 12),
                    "ELG2": (0, -18), "QSO": (0, 12), "BGS": (0, -18)}


# ------------------------------------------------------------------- figure 1
def fig_overview(cosmo=None, cited_only=False, scale="symlog"):
    """Every fsigma8 measurement in the review, PV and RSD, on one axis.

    This is the figure the symlog scale exists for. PV occupies z < 0.08 and RSD
    runs to z = 1.5; on a linear axis the entire peculiar-velocity literature
    collapses into the first few percent of the panel, and a pure log axis
    cannot draw the six z = 0 rows at all. Linear below z = 0.1 and log above
    gives each regime about half the panel and keeps z = 0 on it.
    """
    cosmo = cosmo or FlatLCDM()
    df = io.load_fsigma8(kind="measurement")
    if cited_only:
        df = cit.cited_only(df)

    pv = df[df["method"] == "pv"]
    rsd = df[df["method"] == "rsd"]
    xpv = dodge_x(pv["z"].to_numpy(), min_sep=0.006, step=0.014, scale=scale)
    xrsd = dodge_x(rsd["z"].to_numpy(), min_sep=0.04, step=0.016, scale=scale)

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(11, 7.5), sharex=True,
        gridspec_kw=dict(height_ratios=[2.7, 1], hspace=0.07))

    plot_reference(ax, cosmo, zmin=0.0, zmax=1.75)
    plot_by_family(ax, pv, x=xpv, ms=6.5)
    plot_measurements(ax, rsd[~rsd["ref"].astype(str).str.startswith("DESI")],
                      x=xrsd[~rsd["ref"].astype(str).str.startswith("DESI")],
                      color=PALETTE["violet"], marker="^", ms=6.5,
                      label="Galaxy clustering / RSD "
                            f"({(~rsd['ref'].astype(str).str.startswith('DESI')).sum()})")
    desi_rsd = rsd[rsd["ref"].astype(str).str.startswith("DESI")]
    plot_measurements(ax, desi_rsd, x=xrsd[rsd["ref"].astype(str).str.startswith("DESI")],
                      color=DESI_COLOR, marker="^", ms=8,
                      label=f"DESI DR1 full shape ({len(desi_rsd)})", zorder=5)

    style_axes(ax, xlabel="", ylim=(0.20, 0.78), scale=scale,
               ticks=[0, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5],
               legend_kw=dict(loc="upper center", bbox_to_anchor=(0.5, 1.30),
                              fontsize=9.5, ncol=3))
    ax.set_xlim(-0.004, 1.75)

    # The residual panel needs the same dodged abscissa as the top panel, so
    # rebuild the frame in the order the two x arrays were computed in.
    all_x = np.concatenate([xpv, xrsd])
    stacked = pd.concat([pv, rsd], ignore_index=True)
    residual_panel(axr, stacked, cosmo, by="method",
                   color_map={"pv": PALETTE["blue"], "rsd": PALETTE["violet"]},
                   marker_map={"pv": "o", "rsd": "^"}, x=all_x, ms=5)
    axr.set_xlabel("Redshift $z$")
    axr.set_ylim(-45, 45)

    pv_body, pv_consensus, pv_missing = cit.caption_pv(pv)
    rsd_cite, rsd_missing = cit.caption_rsd(rsd)
    caption = (
        r"\caption{Compiled measurements of $f\sigma_8(z)$ from peculiar "
        r"velocities and from galaxy clustering, against the $\Lambda$CDM "
        r"prediction for a Planck~2018 fiducial cosmology (black). The abscissa "
        r"is linear below $z=0.1$ and logarithmic above, which keeps the $z=0$ "
        r"peculiar-velocity results on the panel while still showing the "
        r"clustering measurements out to $z\simeq1.5$. Peculiar-velocity results "
        r"are grouped by the method used "
        rf"to extract $f\sigma_8$: {pv_body}"
        + (rf"; the DESI~DR1 consensus {pv_consensus}" if pv_consensus else "")
        + rf". Clustering measurements are taken from {rsd_cite}. Points at nearly "
        r"coincident redshifts are displaced horizontally for legibility. The "
        r"lower panel shows the fractional deviation from the fiducial "
        r"prediction.}")
    return fig, dict(caption=caption,
                     missing_citations=sorted(set(pv_missing + rsd_missing)),
                     n_pv=len(pv), n_rsd=len(rsd))


# ------------------------------------------------------------------- figure 2
def fig_rsd(cosmo=None, cited_only=False, scale="linear"):
    """Galaxy-clustering fsigma8: one colour per survey, DESI DR1 in red."""
    cosmo = cosmo or FlatLCDM()
    df = io.load_fsigma8(kind="measurement", method="rsd")
    dropped = []
    if cited_only:
        dropped = cit.uncited(df)
        df = cit.cited_only(df)

    pre = df[~df["ref"].astype(str).str.startswith("DESI")]
    desi = df[df["ref"].astype(str).str.startswith("DESI")]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    plot_reference(ax, cosmo, zmax=1.65)
    xpre = dodge_x(pre["z"].to_numpy(), min_sep=0.035, step=0.028, scale=scale)
    # No per-point labels on the pre-DESI set: 19 points in 1.5 units of z means
    # the annotations collide with each other and with the error bars. The
    # legend carries the survey and the caption carries the citation; the tracer
    # name is kept only for DESI, where the six samples are the point of the
    # figure.
    plot_by_survey(ax, pre, x=xpre, ms=6.5, annotate=False)
    plot_measurements(ax, desi, color=DESI_COLOR, marker="o", ms=8,
                      label="DESI DR1 (full shape)", zorder=5)
    annotate_points(ax, desi, offsets=DESI_RSD_OFFSETS, color=DESI_COLOR)

    style_axes(ax, xlim=(-0.02, 1.64), ylim=(0.20, 0.70), scale=scale,
               legend_kw=dict(loc="upper left", bbox_to_anchor=(1.01, 1.02),
                              fontsize=9, ncol=1, handletextpad=0.5))
    fig.tight_layout()

    cite, missing = cit.caption_rsd(df)
    caption = (
        r"\caption{Measurements of $f\sigma_8(z)$ from galaxy clustering "
        r"(redshift-space distortions), compared with the $\Lambda$CDM "
        r"prediction for a Planck~2018 fiducial cosmology (black line). Points "
        r"are coloured by survey; DESI~DR1 full-shape results are shown in red "
        r"and labelled by tracer sample. Points at nearly coincident "
        r"redshifts are displaced horizontally for legibility. Measurements are "
        rf"taken from {cite}.}}")
    return fig, dict(caption=caption, missing_citations=missing, dropped=dropped,
                     n=len(df))


# ------------------------------------------------------------------- figure 3
def fig_pv(cosmo=None, cited_only=False, scale="linear", desi_estimators=False):
    """Peculiar-velocity fsigma8, grouped by method family.

    Grouped by method rather than by survey on purpose: the compilation is
    inhomogeneous -- the entries share data, and 6dFGSv, SDSS PV, SFI++, 2MTF
    and 2M++ recur across most of them -- so a per-survey legend would suggest
    an independence that is not there. What does vary meaningfully between rows
    is how the velocity field was compressed before fitting.
    """
    cosmo = cosmo or FlatLCDM()
    df = io.load_fsigma8(kind="measurement", method="pv",
                         desi_estimators=desi_estimators)
    dropped_z = io.dropped_without_z()["key"].tolist()
    dropped = []
    if cited_only:
        dropped = cit.uncited(df)
        df = cit.cited_only(df)

    x = dodge_x(df["z"].to_numpy(), min_sep=0.004, step=0.0032, scale=scale)
    zmax = float(np.nanmax(df["z"])) if len(df) else 0.08

    fig, ax = plt.subplots(figsize=(9.5, 6))
    plot_reference(ax, cosmo, zmax=zmax * 1.15)
    plot_by_family(ax, df, x=x, ms=7.5)

    style_axes(ax, xlim=(-0.006, zmax * 1.2), ylim=(0.22, 0.74), scale=scale,
               legend_kw=dict(loc="upper left", fontsize=9.5, ncol=2))
    fig.tight_layout()

    body, consensus, missing = cit.caption_pv(df)
    caption = (
        r"\caption{Measurements of $f\sigma_8$ from peculiar velocities at "
        r"$z<0.1$, compared with the $\Lambda$CDM prediction for a Planck~2018 "
        r"fiducial cosmology (black line). The compilation is inhomogeneous --- "
        r"the measurements share data and differ in normalisation --- so points "
        r"are grouped by the method used to extract $f\sigma_8$ rather than by "
        rf"survey: {body}"
        + (rf". The DESI~DR1 consensus (red star) {consensus} combines the three "
           r"estimators applied to that survey, accounting for their correlation"
           if consensus else "")
        + r". Points at nearly coincident redshifts are displaced horizontally "
          r"for legibility. Error bars include the systematic contribution where "
          r"the authors quote one separately.}")
    return fig, dict(caption=caption, missing_citations=missing, dropped=dropped,
                     dropped_no_z=dropped_z, n=len(df))


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

    # One series per source dataset; LSST's nested redshift ranges are reduced to
    # the widest bin of each configuration so the panel does not show the same
    # supernovae three times.
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
        r"fiducial model's by construction. The abscissa is linear below $z=0.1$ and "
        r"logarithmic above. "
        r"For the LSST supernova peculiar-velocity forecasts only the widest "
        r"redshift range of each configuration is shown, since the narrower bins "
        r"are drawn from the same supernovae.}")
    return fig, dict(caption=caption, missing_citations=[], n=sum(len(s[0]) for s in series))


# ------------------------------------------------------------------- figure 5
def fig_theory(cosmo=None, k=0.1, scale="symlog", cited_only=False):
    """Tabulated COLA growth histories against the compiled measurements."""
    cosmo = cosmo or FlatLCDM()
    df = io.load_fsigma8(kind="measurement", drop_derived=True)
    if cited_only:
        df = cit.cited_only(df)

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

    x = dodge_x(df["z"].to_numpy(), min_sep=0.006, step=0.010, scale=scale)
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
    return fig, dict(caption=caption, missing_citations=[], n=len(df))


# ----------------------------------------------------------------------- driver
FIGURES = {
    "overview": fig_overview,
    "rsd": fig_rsd,
    "pv": fig_pv,
    "forecasts": fig_forecasts,
    "theory": fig_theory,
}


def main(argv=None):
    p = argparse.ArgumentParser(description="Build the growth-review figures.")
    p.add_argument("--outdir", default="figures", type=Path)
    p.add_argument("--only", nargs="*", choices=sorted(FIGURES), default=None)
    p.add_argument("--cited-only", action="store_true",
                   help="restrict to measurements with a bibliography entry")
    p.add_argument("--format", default="pdf")
    args = p.parse_args(argv)

    use_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cosmo = FlatLCDM()

    problems = []
    for name in (args.only or sorted(FIGURES)):
        fn = FIGURES[name]
        kw = dict(cosmo=cosmo)
        if "cited_only" in fn.__code__.co_varnames:
            kw["cited_only"] = args.cited_only
        fig, meta = fn(**kw)
        path = args.outdir / f"growth_{name}.{args.format}"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

        print(f"\n{'=' * 78}\n{name}  ->  {path}\n{'=' * 78}")
        print(f"points plotted: {meta.get('n', meta.get('n_pv', 0) + meta.get('n_rsd', 0))}")
        if meta.get("dropped"):
            print(f"dropped, no bibliography entry ({len(meta['dropped'])}): "
                  f"{', '.join(meta['dropped'])}")
        if meta.get("dropped_no_z"):
            print(f"dropped, no author-assigned z_eff: {', '.join(meta['dropped_no_z'])}")
        print(meta["caption"])
        if meta["missing_citations"]:
            problems.append((name, meta["missing_citations"]))

    if problems:
        print("\n** PLOTTED BUT NOT CITED -- fix before use:")
        for name, keys in problems:
            print(f"   {name}: {', '.join(keys)}")
        return 1
    print("\nEvery plotted measurement is cited in the captions above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
