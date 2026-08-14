"""Plotting primitives over the tidy fsigma8 schema.

Every function here takes a frame with the columns produced by
``io.load_fsigma8`` -- ``z``, ``value``, ``err_lo``, ``err_hi``, ``frac_err``,
``kind`` -- so none of them needs to know which file a row came from. Composed
figures live in ``figures.py``.

Three visual rules the whole package keeps, because breaking them is how growth
plots mislead:

1. Measurements and forecasts never share a style. A measurement is a filled-
   outline marker at its published value; a forecast is a thin capped bar drawn
   around the fiducial curve, with no marker face, because it has no measured
   central value.
2. DESI keeps ``PALETTE["red"]`` in every panel and no other series may use it.
3. Any horizontal dodge applied for legibility is returned to the caller so it
   can be declared in the caption.
"""
import numpy as np
import pandas as pd

from .style import (PALETTE, MARKERS, FAMILY_COLOR, FAMILY_MARKER, DESI_COLOR,
                    dodge_x, set_z_scale)
from .methods import FAMILY_LABEL, FAMILY_ORDER


# ---------------------------------------------------------------- theory curves
def plot_reference(ax, cosmo, zmin=0.0, zmax=1.7, n=400, gamma=None, **kw):
    """The fiducial LCDM/GR fsigma8(z) curve. Drawn first, at zorder 0."""
    z = np.linspace(zmin, zmax, n)
    y = cosmo.fsigma8(z) if gamma is None else cosmo.fsigma8_gamma(z, gamma)
    label = (r"$\Lambda$CDM (GR)" if gamma is None
             else rf"$f=\Omega_m(z)^{{{gamma:g}}}$")
    style = dict(color="k", lw=2.0, zorder=0, label=label)
    style.update(kw)
    return ax.plot(z, y, **style)


def plot_theory(ax, name, sigma8, k=0.01, zmax=2.0, **kw):
    """A tabulated COLA growth curve, rescaled to `sigma8`.

    `k` picks the wavenumber column. For f(R) this matters -- the growth is
    scale-dependent and a single curve is an effective quantity at one k, not
    the model's growth rate. State the k in the caption.
    """
    from .io import load_theory
    t = load_theory(name, k=k)
    t = t[t["z"] <= zmax].sort_values("z")
    style = dict(lw=1.8, ls="--", zorder=1, label=f"{name} (k={k:g} h/Mpc)")
    style.update(kw)
    return ax.plot(t["z"], sigma8 * t["fD"], **style)


# ------------------------------------------------------------------ data points
def _errorbar(ax, x, y, lo, hi, color, marker, ms, label, **kw):
    style = dict(fmt=marker, ms=ms, mec=color, ecolor=color, mfc="white",
                 elinewidth=1.3, mew=1.5, capsize=3, capthick=1.2,
                 label=label, zorder=3, ls="none")
    style.update(kw)
    return ax.errorbar(x, y, yerr=np.vstack([lo, hi]), **style)


def plot_measurements(ax, df, color=PALETTE["blue"], marker="o", ms=7,
                      label=None, x=None, **kw):
    """Measurements from a tidy frame, with asymmetric error bars.

    `x` overrides the plotted abscissa (pass the output of `dodge_x`); `df["z"]`
    stays the physical redshift either way.
    """
    if not len(df):
        return None
    x = df["z"].to_numpy() if x is None else np.asarray(x, float)
    return _errorbar(ax, x, df["value"].to_numpy(), df["err_lo"].to_numpy(),
                     df["err_hi"].to_numpy(), color, marker, ms, label, **kw)


def plot_forecasts(ax, df, cosmo, color=PALETTE["violet"], marker="_", ms=9,
                   label=None, **kw):
    """Forecast precisions, drawn as bars around the fiducial prediction.

    Deliberately unlike `plot_measurements`: no marker face, thinner bars, lower
    zorder. A forecast has no central value -- the position on the y axis is the
    fiducial cosmology's, not a measurement's, and the figure must not suggest
    otherwise.
    """
    if not len(df):
        return None
    z = df["z"].to_numpy()
    y = cosmo.fsigma8(z)
    err = df["frac_err"].to_numpy() * y
    style = dict(fmt=marker, ms=ms, mec=color, ecolor=color, mfc="none",
                 elinewidth=1.1, mew=1.4, capsize=0, alpha=0.9,
                 label=label, zorder=2, ls="none")
    style.update(kw)
    return ax.errorbar(z, y, yerr=np.vstack([err, err]), **style)


def plot_grouped(ax, df, by="method_family", cosmo=None, color_map=None,
                 marker_map=None, label_map=None, order=None, ms=7,
                 count_in_label=True, x=None, **kw):
    """One styled series per unique value of `by`.

    Handles measurements and forecasts in the same call by dispatching on the
    `kind` column, so a mixed frame does not need splitting first (a forecast
    subset requires `cosmo`). Returns the list of groups actually drawn, in
    legend order.
    """
    color_map = color_map or {}
    marker_map = marker_map or {}
    label_map = label_map or {}
    groups = list(order) if order else list(dict.fromkeys(df[by].dropna()))
    x = None if x is None else np.asarray(x, float)

    drawn = []
    for i, g in enumerate(groups):
        mask = (df[by] == g).to_numpy()
        sub = df[mask]
        if not len(sub):
            continue
        color = color_map.get(g, PALETTE[list(PALETTE)[i % len(PALETTE)]])
        marker = marker_map.get(g, MARKERS[i % len(MARKERS)])
        name = label_map.get(g, str(g))
        if count_in_label:
            name = f"{name} ({len(sub)})"
        xs = None if x is None else x[mask]
        meas = sub[sub["kind"] == "measurement"]
        fc = sub[sub["kind"] == "forecast"]
        if len(meas):
            plot_measurements(ax, meas, color=color, marker=marker, ms=ms,
                              label=name,
                              x=None if xs is None else xs[sub["kind"].to_numpy() == "measurement"],
                              **kw)
        if len(fc):
            if cosmo is None:
                raise ValueError("plotting forecasts requires `cosmo`")
            plot_forecasts(ax, fc, cosmo, color=color,
                           label=None if len(meas) else name)
        drawn.append(g)
    return drawn


def plot_by_family(ax, df, cosmo=None, ms=7, x=None, **kw):
    """PV measurements grouped by method family, with the reserved styles.

    The consensus series is drawn separately and larger: a star at the same
    point size as a circle reads as smaller, and the DESI consensus is the one
    point in these panels a reader is most likely to be looking for.
    """
    fam = df["method_family"]
    drawn = plot_grouped(ax, df[fam != "consensus"], by="method_family", cosmo=cosmo,
                         color_map=FAMILY_COLOR, marker_map=FAMILY_MARKER,
                         label_map=FAMILY_LABEL, order=FAMILY_ORDER, ms=ms,
                         x=None if x is None else np.asarray(x, float)[(fam != "consensus").to_numpy()],
                         **kw)
    cons = df[fam == "consensus"]
    if len(cons):
        plot_measurements(ax, cons, color=FAMILY_COLOR["consensus"],
                          marker=FAMILY_MARKER["consensus"], ms=ms * 2.4,
                          mfc=FAMILY_COLOR["consensus"], mew=1.0,
                          label=FAMILY_LABEL["consensus"],
                          x=None if x is None else np.asarray(x, float)[(fam == "consensus").to_numpy()],
                          zorder=6)
        drawn.append("consensus")
    return drawn


# Collaboration names that appear in the `ref` column in place of a first
# author. A survey led by one of these has many companion papers, so its legend
# entry gets no "(Author et al. Year)" -- there is no single article to name.
COLLABORATIONS = {"eBOSS", "SDSS", "DESI-Y1", "DESI"}


def survey_legend_label(sub):
    """Legend text for one survey group: "SURVEY (Author et al. Year)".

    The parenthetical is added only when the group really is one article -- a
    single non-collaboration first author and a single year. eBOSS DR16 and
    DESI DR1 each have two companion papers per tracer, so they are named by
    survey alone rather than by an arbitrary one of them.
    """
    survey = str(sub["survey"].iloc[0])
    refs, years = set(sub["ref"].astype(str)), set(sub["year"].dropna())
    if len(refs) == 1 and len(years) == 1 and not (refs & COLLABORATIONS):
        return f"{survey} ({refs.pop()} et al. {int(years.pop())})"
    return survey


# tab20 slots, all eight saturated shades first and only then the pale ones.
# Taking tab20 in its natural order alternates dark/light of the SAME hue, so
# consecutive legend entries came out as two blues and two oranges. Reds (6, 7)
# are excluded because red is reserved for DESI in every panel, and greys
# (14, 15) because grey means "de-emphasised" elsewhere in the package. tab10
# alone was one hue short: the RSD compilation has ten pre-DESI surveys.
# Pink (12) is held back behind olive and cyan: it is the one tab20 hue close
# enough to DESI's red in lightness to be mistaken for it at a glance.
_SURVEY_SLOTS = [0, 2, 4, 8, 10, 18, 16, 12, 1, 3, 5, 9, 11, 13, 17, 19]


def plot_by_survey(ax, df, cmap_name="tab20", ms=6, annotate=False, x=None,
                   fontsize=8, by="survey", **kw):
    """One colour and legend entry per survey, labelled "SURVEY (Article)"."""
    import matplotlib.pyplot as plt
    groups = list(dict.fromkeys(df[by]))
    cmap = plt.get_cmap(cmap_name)
    slots = _SURVEY_SLOTS
    color_map = {g: (DESI_COLOR if str(g).startswith("DESI")
                     else cmap(slots[i % len(slots)]))
                 for i, g in enumerate(groups)}
    label_map = {g: survey_legend_label(df[df[by] == g]) for g in groups}
    drawn = plot_grouped(ax, df, by=by, color_map=color_map,
                         marker_map={g: "o" for g in groups},
                         label_map=label_map, order=groups,
                         ms=ms, count_in_label=False, x=x, **kw)
    if annotate:
        xs = df["z"].to_numpy() if x is None else np.asarray(x, float)
        for n, (_, row) in enumerate(df.iterrows()):
            ax.annotate(str(row["label"]), (xs[n], row["value"]),
                        textcoords="offset points",
                        xytext=(0, 10 if n % 2 == 0 else -16),
                        fontsize=fontsize, color="0.35", ha="center")
    return drawn


def annotate_provenance(ax, df, x=None, rotation=90, fontsize=9.0, pad=5,
                        color="0.25", baseline=None, quantile=0.72):
    """Rotated "Author et al. Year" above every point, on a shared baseline.

    The baseline sits at the `quantile`-th percentile of the error-bar tops
    rather than above the tallest of them: anchoring on the maximum pushes every
    label to the top of the panel because of two or three outliers, leaving a
    band of white space no reader benefits from. The few points whose bars reach
    past the baseline lift their own label clear of it, so nothing is written
    over a measurement.

    All the rest share one height, which is what stops long names from running
    up through their neighbours. Horizontal separation is the dodge's job --
    pass `min_gap` to `dodge_x` sized for this `fontsize`.

    Returns (texts, anchors) for `fit_ylim_to_labels`.
    """
    xs = df["z"].to_numpy() if x is None else np.asarray(x, float)
    tops = (df["value"] + df["err_hi"]).to_numpy()
    if baseline is None:
        baseline = float(np.nanquantile(tops, quantile))
    ys = np.maximum(baseline, tops)
    texts = []
    for n, (_, row) in enumerate(df.iterrows()):
        year = "" if pd.isna(row["year"]) else f" {int(row['year'])}"
        texts.append(ax.annotate(
            f"{row['ref']}{year}", (xs[n], ys[n]),
            textcoords="offset points", xytext=(0, pad),
            rotation=rotation, rotation_mode="anchor",
            fontsize=fontsize, color=color, ha="left", va="center"))
    return texts, ys


def fit_ylim_to_labels(ax, texts, anchors, bottom, pad_px=3.0):
    """Set the smallest upper ylim that still fits every rotated label.

    Labels live in pixels and the axis in data units, and the two are coupled:
    raising the limit to make room also lowers every anchor, which frees room
    again. Estimating that with a nominal axes height and the longest string
    over-reserves badly -- the longest name is rarely the one sitting highest,
    so the panel ends up with a band of white space at the top.

    Here the text is measured once with the real renderer, then the fixed point
    is solved per label. For anchor A_i needing E_i pixels above itself in an
    axes H pixels tall, the limit must satisfy
    T >= bottom + (A_i - bottom) * H / (H - E_i); the answer is the largest.
    """
    fig = ax.figure
    fig.canvas.draw()
    height = ax.get_window_extent().height
    top = ax.get_ylim()[1]
    need = bottom
    for text, anchor in zip(texts, anchors):
        anchor_px = ax.transData.transform((0.0, anchor))[1]
        extent = text.get_window_extent().y1 - anchor_px + pad_px
        if extent >= height:                       # cannot fit; leave as is
            return top
        need = max(need, bottom + (anchor - bottom) * height / (height - extent))
    ax.set_ylim(bottom, need)
    return need


def annotate_points(ax, df, x=None, offsets=None, color="0.3", fontsize=8.5):
    """Direct horizontal labels from the `label` column; `offsets` overrides
    the (dx, dy) point offset per label value."""
    offsets = offsets or {}
    xs = df["z"].to_numpy() if x is None else np.asarray(x, float)
    for n, (_, row) in enumerate(df.iterrows()):
        dx, dy = offsets.get(str(row["label"]), (0, 11))
        ax.annotate(str(row["label"]), (xs[n], row["value"]),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=fontsize, color=color, ha="center")


# --------------------------------------------------------------- residual panel
def residual_panel(ax, df, cosmo, by=None, color_map=None, marker_map=None,
                   order=None, x=None, ms=6, percent=True):
    """Lower panel: deviation of each measurement from the fiducial prediction.

    Percent by default; set percent=False for the deviation in units of the
    point's own error bar (a pull plot), which is the honest way to look for a
    systematic offset when the errors span a factor of ten as they do here.
    """
    d = df[df["kind"] == "measurement"]
    z = d["z"].to_numpy()
    xs = z if x is None else np.asarray(x, float)
    ref = cosmo.fsigma8(z)
    if percent:
        y = 100.0 * (d["value"].to_numpy() - ref) / ref
        lo = 100.0 * d["err_lo"].to_numpy() / ref
        hi = 100.0 * d["err_hi"].to_numpy() / ref
        ax.set_ylabel(r"$\Delta f\sigma_8/f\sigma_8^{\rm fid}$ [\%]".replace("\\%", "%"),
                      fontsize=11)
    else:
        resid = d["value"].to_numpy() - ref
        err = np.where(resid < 0, d["err_hi"].to_numpy(), d["err_lo"].to_numpy())
        y = resid / err
        lo = hi = np.ones_like(y)
        ax.set_ylabel(r"pull $[\sigma]$", fontsize=11)

    if by is None:
        _errorbar(ax, xs, y, lo, hi, PALETTE["blue"], "o", ms, None)
    else:
        groups = list(order) if order else list(dict.fromkeys(d[by].dropna()))
        for i, g in enumerate(groups):
            m = (d[by] == g).to_numpy()
            if not m.any():
                continue
            _errorbar(ax, xs[m], y[m], lo[m], hi[m],
                      (color_map or {}).get(g, PALETTE["blue"]),
                      (marker_map or {}).get(g, MARKERS[i % len(MARKERS)]),
                      ms, None)
    ax.axhline(0.0, color="k", lw=1.3, zorder=1)
    ax.grid(alpha=0.3, lw=0.7, color="0.7", ls=":")
    ax.set_axisbelow(True)
    return ax
