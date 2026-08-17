"""Theory curves as objects: anything with an ``fsigma8(z)`` you can plot.

Two backends implement one interface:

``GrowthModel``
    the fiducial LCDM (or w0-wa) cosmology, GR growth, integrated in numpy. This
    is the reference curve of every figure, and the growth-index diagnostic.
``TableModel``
    a tabulated prediction read from disk: a COLA run, or an EFTCAMB curve
    exported by ``theory/eftcamb.py``. **Every modified-gravity curve in this
    package comes this way**, from the code the model was actually defined with,
    not from an approximation written here.

Both expose ``fsigma8``, ``sigma8_z``, ``growth_rate``, ``growth_factor`` and
``E``, plus the style (colour, linestyle, label) they must be drawn with, so a
figure never has to know which backend it is looking at.

**sigma8 normalisation** (``norm``) is an explicit argument because it changes the
numbers. ``"early"`` anchors a curve to the fiducial sigma8 through the GR growth
factor, i.e. a shared primordial amplitude -- what a Boltzmann code does at fixed
A_s, and what the EFTCAMB notebook this layer serves used. ``"today"`` rescales to
a common sigma8(0) instead, which hides an amplitude difference and leaves only
the redshift shape. An EFTCAMB export carries its own sigma8 and is used as it
is; a COLA table carries only growth, so it is shape-only.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .background import PLANCK18, Background
from .growth import growth_from_index, solve_growth

# GR reference solutions, keyed by (background, a_ini, n_steps): the "early"
# normalisation anchors to one of these, and re-solving per model would be both
# wasteful and a chance for two anchors to disagree.
_REFERENCE_CACHE = {}

NORMS = ("early", "today")


def _reference_solution(background, a_ini, n_steps):
    key = (background.key, a_ini, n_steps)
    if key not in _REFERENCE_CACHE:
        _REFERENCE_CACHE[key] = solve_growth(background, a_ini=a_ini,
                                             n_steps=n_steps)
    return _REFERENCE_CACHE[key]


# --------------------------------------------------------------------- style
@dataclass(frozen=True)
class Style:
    """How a curve is drawn. Fixed per model so panels stay consistent."""
    color: str = "k"
    ls: str = "-"
    lw: float = 2.0


class TheoryCurve:
    """Shared surface of every theory curve. Subclasses fill in the physics."""

    name = None
    label = ""
    family = ""
    source = ""
    caveats: Tuple[str, ...] = ()
    style = Style()

    # ------------------------------------------------------------ the curve
    def fsigma8(self, z):
        raise NotImplementedError

    def sigma8_z(self, z):
        raise NotImplementedError

    def growth_rate(self, z):
        raise NotImplementedError

    def growth_factor(self, z):
        raise NotImplementedError

    def E(self, z):
        raise NotImplementedError

    # --------------------------------------------------------------- helpers
    def curve(self, z=None, zmin=0.0, zmax=2.0, n=300):
        """(z, fsigma8(z)) on a grid -- what a plotting call actually wants."""
        z = np.linspace(zmin, zmax, n) if z is None else np.asarray(z, float)
        return z, self.fsigma8(z)

    def ratio(self, other, z=None, zmin=0.0, zmax=2.0, n=300, percent=False):
        """fsigma8 relative to another curve (a ratio, or a deviation in %)."""
        z = np.linspace(zmin, zmax, n) if z is None else np.asarray(z, float)
        r = self.fsigma8(z) / other.fsigma8(z)
        return z, (100.0 * (r - 1.0) if percent else r)

    def __repr__(self):
        return f"<{type(self).__name__} {self.name or self.label!r}>"


# ------------------------------------------------------- fiducial GR growth
class GrowthModel(TheoryCurve):
    """Background + GR linear growth, computed on demand.

    background   a ``Background``; defaults to the Planck-2018 fiducial LCDM.
    gamma        use the f = Omega_m(a)^gamma prescription instead of solving the
                 growth equation. A diagnostic (see ``growth.growth_from_index``).
    sigma8       the fiducial sigma8 the normalisation is anchored to.
    norm         "early" (shared primordial amplitude) or "today".
    reference    background of the GR anchor for norm="early"; defaults to the
                 fiducial LCDM one, so a model that changes the expansion history
                 is still anchored to the same primordial amplitude.
    """

    def __init__(self, background=None, gamma=None,
                 sigma8=PLANCK18["sigma8"], norm="early", reference=None,
                 name=None, label=None, family="", source="", caveats=(),
                 color="k", ls="-", lw=2.0, a_ini=1e-3, n_steps=2000):
        if norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")
        self.bg = background or Background()
        self.gamma = gamma
        self.sigma8 = float(sigma8)
        self.norm = norm
        self.reference_bg = reference or Background()
        self.name = name
        self.label = label if label is not None else (name or "model")
        self.family = family
        self.source = source
        self.caveats = tuple(caveats)
        self.style = Style(color=color, ls=ls, lw=lw)
        self._a_ini, self._n_steps = a_ini, n_steps
        self._solution = None
        self._sigma8_grid = None
        self._fsigma8_grid = None
        self._gamma_twin = None

    # ------------------------------------------------------------ internals
    def solution(self):
        """The growth solution, solved once and kept."""
        if self._solution is None:
            if self.gamma is not None:
                self._solution = growth_from_index(
                    self.bg, gamma=self.gamma, a_ini=self._a_ini,
                    n_steps=self._n_steps)
            else:
                self._solution = solve_growth(self.bg, a_ini=self._a_ini,
                                              n_steps=self._n_steps)
        return self._solution

    def _grids(self):
        """sigma8(a) and fsigma8(a) = dsigma8/dlna on the solver's grid."""
        if self._sigma8_grid is None:
            sol = self.solution()
            ref = _reference_solution(self.reference_bg, self._a_ini, self._n_steps)
            if self.norm == "early":
                s8 = self.sigma8 * sol.D / ref.D[-1]
            else:
                s8 = self.sigma8 * sol.D / sol.D[-1]
            self._sigma8_grid = s8
            self._fsigma8_grid = np.gradient(s8, np.log(sol.a), edge_order=2)
        return self.solution().a, self._sigma8_grid, self._fsigma8_grid

    def _interp(self, z, grid):
        a, _, _ = self._grids()
        z = np.asarray(z, dtype=float)
        return np.interp(np.log(1.0 / (1.0 + z)), np.log(a), grid)

    # ------------------------------------------------------------- interface
    def E(self, z):
        return self.bg.E(z)

    def omega_m_z(self, z):
        return self.bg.omega_m_z(z)

    def sigma8_z(self, z):
        _, s8, _ = self._grids()
        return self._interp(z, s8)

    def fsigma8(self, z):
        _, _, fs8 = self._grids()
        return self._interp(z, fs8)

    def growth_factor(self, z):
        """D(z), normalised to D(0) = 1."""
        sol = self.solution()
        return sol.D_at(1.0 / (1.0 + np.asarray(z, dtype=float))) / sol.D[-1]

    def growth_rate(self, z):
        """f(z) = dlnD/dlna."""
        return self.solution().f_at(1.0 / (1.0 + np.asarray(z, dtype=float)))

    # --------------------------------------------------------- gamma sibling
    def fsigma8_gamma(self, z, gamma=0.55):
        """fsigma8 of the f = Omega_m(z)^gamma prescription on this background.

        Kept as a method on the fiducial cosmology because that is how it is
        usually quoted -- "the same cosmology, but with this growth index" -- and
        because the package exposed it before the theory layer existed.
        """
        if self._gamma_twin is None or self._gamma_twin.gamma != gamma:
            self._gamma_twin = GrowthModel(
                background=self.bg, gamma=gamma, sigma8=self.sigma8,
                norm=self.norm, reference=self.reference_bg,
                name=f"gamma_{gamma:g}", label=rf"$f=\Omega_m(z)^{{{gamma:g}}}$",
                family="growth index", a_ini=self._a_ini, n_steps=self._n_steps)
        return self._gamma_twin.fsigma8(z)


class Cosmology(GrowthModel):
    """The fiducial flat-LCDM (or w0-wa) cosmology, GR growth.

    Kept as its own name and signature because it is what the figures pass around
    as ``cosmo``: a background whose ``fsigma8(z)`` is the reference curve.
    ``FlatLCDM`` is an alias.
    """

    def __init__(self, omega_m=PLANCK18["omega_m"], h=PLANCK18["h"],
                 sigma8=PLANCK18["sigma8"], omega_b=PLANCK18["omega_b"],
                 n_s=PLANCK18["n_s"], w0=-1.0, wa=0.0, **kw):
        bg = Background(omega_m=omega_m, h=h, omega_b=omega_b, n_s=n_s,
                        w0=w0, wa=wa)
        kw.setdefault("name", "GR")
        kw.setdefault("label", r"$\Lambda$CDM (GR)")
        kw.setdefault("family", "GR")
        kw.setdefault("lw", 2.2)
        super().__init__(background=bg, sigma8=sigma8, **kw)


FlatLCDM = Cosmology


# ---------------------------------------------------------------- table model
class TableModel(TheoryCurve):
    """A tabulated prediction: (z, D, f) or (z, sigma8, fsigma8) from disk.

    This is how every modified-gravity curve enters the package. Two sources:
    an EFTCAMB export (``theory/eftcamb.py`` writes z, E, sigma8, fsigma8 with
    the model's flags and cosmology in the header), which carries its own
    amplitude in the shared-A_s convention and is used unchanged; or a COLA
    growth table, which carries growth only, so its curve is a shape multiplied
    by the fiducial sigma8 -- the shared-sigma8(0) convention.
    """

    def __init__(self, table, name=None, label=None, sigma8=PLANCK18["sigma8"],
                 norm="today", family="", source="", caveats=(),
                 color="k", ls="--", lw=1.8, k=None):
        self.table = table
        self.name = name
        self.label = label if label is not None else (name or "table")
        self.sigma8 = float(sigma8)
        self.norm = norm
        self.family = family
        self.source = source
        self.caveats = tuple(caveats)
        self.style = Style(color=color, ls=ls, lw=lw)
        self.k = k

        z = np.asarray(table["z"], dtype=float)
        order = np.argsort(z)
        self._z = z[order]
        self._D = np.asarray(table["D"], dtype=float)[order]
        self._f = np.asarray(table["f"], dtype=float)[order]
        self._E = (np.asarray(table["E"], dtype=float)[order]
                   if "E" in table else None)
        if "sigma8" in table:
            self._s8 = np.asarray(table["sigma8"], dtype=float)[order]
        else:
            # D is normalised to D(z=0) = 1 by the reader, so this is the
            # "today" convention: the run's own amplitude is not used.
            self._s8 = self.sigma8 * self._D
        self._fs8 = (np.asarray(table["fsigma8"], dtype=float)[order]
                     if "fsigma8" in table else self._f * self._s8)

    def _interp(self, z, grid):
        return np.interp(np.asarray(z, dtype=float), self._z, grid)

    def E(self, z):
        if self._E is None:
            raise ValueError(f"{self.name!r} carries no expansion history column")
        return self._interp(z, self._E)

    def fsigma8(self, z):
        return self._interp(z, self._fs8)

    def sigma8_z(self, z):
        return self._interp(z, self._s8)

    def growth_rate(self, z):
        return self._interp(z, self._f)

    def growth_factor(self, z):
        return self._interp(z, self._D)

    @property
    def zmax(self):
        return float(self._z.max())
