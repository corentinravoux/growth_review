"""Flat-LCDM background and linear growth, in pure numpy.

Deliberately dependency-free (no CAMB/CLASS/CCL): the review figures only need
E(z), D(z), f(z) and fsigma8(z) at sub-percent accuracy over 0 < z < 3, and a
heavyweight Boltzmann dependency would make the package hard to install on a
laptop. If you need a scale-dependent growth (f(R), nDGP, any Horndeski model),
do not extend this class -- read a tabulated curve instead, see `io.load_theory`.
"""
import numpy as np

# Planck 2018 TT,TE,EE+lowE+lensing+BAO (Aghanim et al. 2020, arXiv:1807.06209,
# Table 2 last column). Kept as the single fiducial for every figure so the
# reference curve is identical across panels.
PLANCK18 = dict(omega_m=0.3138, h=0.6736, sigma8=0.8111, n_s=0.9649)


class FlatLCDM:
    """Flat LCDM background + scale-independent linear growth.

    Growth is obtained from the exact integral solution of the growing mode,

        D(a) ∝ H(a) ∫_0^a da' / (a' H(a'))^3 ,

    evaluated on a log-spaced grid and normalised to D(a=1) = 1. f = dlnD/dlna
    is differentiated from the same grid rather than approximated by
    Omega_m(a)^0.55 -- the approximation is good to ~0.5% at z=0 but the whole
    point of a growth review is to not silently assume the GR growth index.
    """

    def __init__(self, omega_m=PLANCK18["omega_m"], h=PLANCK18["h"],
                 sigma8=PLANCK18["sigma8"], n_grid=20000):
        self.omega_m = float(omega_m)
        self.omega_de = 1.0 - self.omega_m
        self.h = float(h)
        self.sigma8 = float(sigma8)

        a = 10.0 ** np.linspace(-6, 0, n_grid)
        e_a = self.E(1.0 / a - 1.0)
        integrand = 1.0 / (a * e_a) ** 3
        # cumulative trapezoid, vectorised
        d = np.zeros_like(a)
        d[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(a))
        d *= 2.5 * self.omega_m * e_a
        # The integral is identically 0 at the first grid point, which would make
        # f = a/D dD/da divide by zero there. a[0] = 1e-6 is deep in matter
        # domination where D is proportional to a, so extrapolate rather than
        # patching with d[1] (which would give f[0] = 0 instead of 1).
        d[0] = d[1] * a[0] / a[1]
        d /= d[-1]

        self._a = a
        self._D = d
        self._f = a / d * np.gradient(d, a, edge_order=2)

    # ------------------------------------------------------------- background
    def E(self, z):
        """H(z)/H0."""
        z = np.asarray(z, dtype=float)
        return np.sqrt(self.omega_m * (1 + z) ** 3 + self.omega_de)

    def omega_m_z(self, z):
        z = np.asarray(z, dtype=float)
        return self.omega_m * (1 + z) ** 3 / self.E(z) ** 2

    # ----------------------------------------------------------------- growth
    def growth_factor(self, z):
        """D(z), normalised to D(0) = 1."""
        return np.interp(1.0 / (1.0 + np.asarray(z, dtype=float)), self._a, self._D)

    def growth_rate(self, z):
        """f(z) = dlnD/dlna."""
        return np.interp(1.0 / (1.0 + np.asarray(z, dtype=float)), self._a, self._f)

    def fsigma8(self, z):
        return self.growth_rate(z) * self.sigma8 * self.growth_factor(z)

    def sigma8_z(self, z):
        return self.sigma8 * self.growth_factor(z)

    def fsigma8_gamma(self, z, gamma=0.55):
        """fsigma8 with f = Omega_m(z)^gamma instead of the exact solution.

        gamma = 0.55 recovers GR to ~0.1%; gamma is the standard one-parameter
        handle on modified growth (Linder 2005, arXiv:astro-ph/0507263). Note
        that D(z) is still the LCDM one here -- this varies f only, so it is a
        diagnostic, not a self-consistent modified-gravity prediction.
        """
        return self.omega_m_z(z) ** gamma * self.sigma8 * self.growth_factor(z)
