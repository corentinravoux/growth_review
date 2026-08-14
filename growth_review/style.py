"""Matplotlib style, colour/marker identity channels, and redshift axis scales.

The base stylesheet is a vendored copy of ``desi_cr/style_cr.mplstyle`` (serif,
cm mathtext, dotted grid, 15pt labels), so figures from this package match the
rest of the workspace without importing ``desi_cr``. Set ``$PLT_STYLE`` to
override it.

The redshift axis is the one design decision this package cannot dodge. The
compilation spans z = 0 (peculiar velocities, most of them below 0.08) to
z = 2.4 (Lyman-alpha BAO). On a linear axis every PV measurement lands in the
leftmost 3% of the panel and the method comparison is unreadable; on a pure log
axis the six z = 0 rows cannot be drawn at all. Four scales, chosen per figure:

    "linear"  ordinary z axis. Right for a PV-only or an RSD-only panel.
    "symlog"  linear below `linthresh` (default 0.1), log10 above. The default
              for any panel spanning both regimes: it gives the peculiar-velocity
              range a fixed fraction of the panel width while still reaching
              z = 2, and z = 0 sits at 0.
    "log1p"   log10(1 + z). Also handles z = 0, but 1+z only runs from 1 to 1.1
              across the whole PV range, so it barely expands the low-z end --
              use it when a strictly monotone smooth transform is wanted.
    "log"     log10(z). Only for panels with no z = 0 row; it will drop them.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STYLE_FILE = Path(__file__).parent / "growth_review.mplstyle"

# Colourblind-safe categorical palette in a fixed hue order, never cycled: the
# same concept keeps the same colour across every figure in the review.
PALETTE = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948",
    "grey": "#6f6f6f",
}
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]

# Reserved assignments, so a legend never has to be re-read between figures.
PROBE_COLOR = {"pv": PALETTE["blue"], "rsd": PALETTE["orange"]}
# The two velocity-density families are given adjacent hues and a filled/open
# marker pair, so they read as siblings that differ in one thing (linear vs
# particle dynamics) rather than as two unrelated methods.
FAMILY_COLOR = {
    "field_level":  PALETTE["blue"],
    "two_point":    PALETTE["orange"],
    "vd_linear":    PALETTE["aqua"],
    "vd_dynamical": PALETTE["green"],
    "consensus":    PALETTE["red"],
    "other":        PALETTE["grey"],
}
FAMILY_MARKER = {
    "field_level": "o", "two_point": "s", "vd_linear": "D",
    "vd_dynamical": "P", "consensus": "*", "other": "v",
}
# DESI is red in every panel; no survey series may take that hue.
DESI_COLOR = PALETTE["red"]


def use_style():
    """Apply the package stylesheet ($PLT_STYLE wins if set)."""
    plt.style.use(os.environ.get("PLT_STYLE", str(STYLE_FILE)))


# ----------------------------------------------------------- redshift x-scales
SCALES = ("linear", "symlog", "log1p", "log")

# Default ticks for the two mixed-range scales. Chosen so the peculiar-velocity
# decade and the clustering decade each get labelled points, and so no two
# labels collide at the default figure width.
MIXED_TICKS = [0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]


def set_z_scale(ax, scale="linear", ticks=None, linthresh=0.1, linscale=1.0):
    """Set the redshift axis scale, with readable ticks in every case.

    ``linthresh`` / ``linscale`` apply to ``symlog``: below linthresh the axis is
    linear, above it log10, and linscale sets how many decades' worth of width
    the linear stretch is given (1.0 -> the z < 0.1 range takes about as much
    room as one decade of the log part).

    ``log1p`` is a matplotlib FuncScale over z -> log10(1+z). Both mixed scales
    keep their tick labels in z: a reader should never have to undo a transform
    in their head to read a redshift off the axis.
    """
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}, got {scale!r}")
    if scale == "linear":
        return
    if scale == "log":
        ax.set_xscale("log")
        return

    if scale == "symlog":
        ax.set_xscale("symlog", linthresh=linthresh, linscale=linscale)
    else:
        ax.set_xscale("function",
                      functions=(lambda z: np.log10(1.0 + np.asarray(z, float)),
                                 lambda y: 10.0 ** np.asarray(y, float) - 1.0))
    ax.set_xticks(MIXED_TICKS if ticks is None else ticks)
    ax.set_xticklabels([f"{t:g}" for t in (MIXED_TICKS if ticks is None else ticks)])
    ax.minorticks_off()


def dodge_x(z, min_sep, step, scale="linear", linthresh=0.1, floor=None,
            min_gap=None):
    """Spread points sharing (nearly) the same redshift symmetrically about it.

    Returns shifted x values only -- the underlying z is unchanged, and the
    shift MUST be declared in the figure caption. The offset is applied in
    whatever coordinate the axis is linear in (z, log(1+z), or symlog's two
    branches), so the visual separation stays uniform across the panel instead
    of vanishing at one end.

    `floor` shifts a whole block up rather than letting it cross that value.
    Set floor=0 on any peculiar-velocity panel: six rows sit at exactly
    z_eff = 0, and a symmetric spread would put half of them at negative
    redshift -- a plotting artefact a reader will read as data.

    `min_gap` then enforces a minimum separation between ALL neighbours, not
    just within a tied block. The block pass alone leaves two adjacent blocks
    free to end up a thousandth apart, which is invisible for markers and fatal
    for per-point labels. The pass is monotone and left-to-right, so ordering is
    preserved and nothing moves below `floor`.
    """
    z = np.asarray(z, dtype=float)
    out = z.copy()
    order = np.argsort(z, kind="stable")
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and z[order[j + 1]] - z[order[i]] < min_sep:
            j += 1
        n = j - i + 1
        if n > 1:
            offs = (np.arange(n) - (n - 1) / 2.0) * step
            block = z[order[i:j + 1]]
            if scale == "linear":
                out[order[i:j + 1]] = block + offs
            elif scale == "symlog":
                # additive inside the linear branch, multiplicative outside it --
                # the two things the symlog axis is respectively linear in
                out[order[i:j + 1]] = np.where(block < linthresh,
                                               block + offs * linthresh,
                                               block * 10.0 ** offs)
            else:
                out[order[i:j + 1]] = (1.0 + block) * 10.0 ** offs - 1.0
            if floor is not None:
                shift = floor - out[order[i:j + 1]].min()
                if shift > 0:
                    out[order[i:j + 1]] += shift
        i = j + 1

    if min_gap:
        for a, b in zip(np.argsort(out, kind="stable")[:-1],
                        np.argsort(out, kind="stable")[1:]):
            if out[b] - out[a] < min_gap:
                out[b] = out[a] + min_gap
    return out


def style_axes(ax, xlabel="Redshift $z$", ylabel=r"$f\sigma_8(z)$",
               xlim=None, ylim=None, title=None, legend_kw=None, scale="linear",
               ticks=None, linthresh=0.1, linscale=1.0):
    """Common axis furniture: labels, limits, scale, grid below the data, legend."""
    set_z_scale(ax, scale, ticks=ticks, linthresh=linthresh, linscale=linscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3, lw=0.7, color="0.7", ls=":")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if legend_kw is not False:
        kw = dict(fontsize=10, ncol=1, loc="best", handletextpad=0.6,
                  frameon=True, framealpha=0.9)
        kw.update(legend_kw or {})
        ax.legend(**kw)
    if title:
        ax.set_title(title, fontsize=13, pad=10)
    return ax
