"""Flat background expansion histories, in pure numpy.

Everything the growth equation needs from the background is here: E(a),
Omega_m(a) and dlnE/dlna. Dark energy is CPL, w(a) = w0 + wa (1 - a), which
covers the LCDM case (w0 = -1, wa = 0) and the DESI w0-wa point without a
second class.

Radiation is deliberately absent. The growth ODE in `growth.py` starts at
a_ini = 1e-3, where radiation is still ~30% of the matter density, so its
neglect does bias the growing mode's *amplitude* there -- but every quantity
this package exposes is either normalised at a = 1 or taken as a ratio to a
reference model integrated on the same background, and a common amplitude
error cancels in both. What would NOT cancel is a background difference
between two models, which is exactly why models that modify the expansion
history (w0-wa) carry their own `Background` instead of a rescaled LCDM one.
"""
import numpy as np

# Planck 2018 TT,TE,EE+lowE+lensing+BAO (Aghanim et al. 2020, arXiv:1807.06209,
# Table 2 last column). One fiducial for the whole package, so the reference
# curve is identical in every panel. omega_b is only used by the P(k) shape in
# `power.py` (Omega_b h^2 = 0.02237 at this h).
PLANCK18 = dict(omega_m=0.3138, omega_b=0.04930, h=0.6736,
                sigma8=0.8111, n_s=0.9649)


class Background:
    """Flat FLRW background with CPL dark energy.

    Parameters are the ones a growth calculation actually uses; H0 itself never
    enters (every expression below is in units of H0), so `h` is carried only
    for the P(k) shape and for the nDGP crossover scale, both of which are
    quoted per h.
    """

    def __init__(self, omega_m=PLANCK18["omega_m"], h=PLANCK18["h"],
                 omega_b=PLANCK18["omega_b"], n_s=PLANCK18["n_s"],
                 w0=-1.0, wa=0.0):
        self.omega_m = float(omega_m)
        self.omega_de = 1.0 - self.omega_m
        self.omega_b = float(omega_b)
        self.h = float(h)
        self.n_s = float(n_s)
        self.w0 = float(w0)
        self.wa = float(wa)

    # ------------------------------------------------------------------ w(a)
    def w(self, a):
        """CPL equation of state w(a) = w0 + wa (1 - a)."""
        a = np.asarray(a, dtype=float)
        return self.w0 + self.wa * (1.0 - a)

    def de_density(self, a):
        """rho_DE(a) / rho_DE(1), the CPL integral in closed form."""
        a = np.asarray(a, dtype=float)
        return a ** (-3.0 * (1.0 + self.w0 + self.wa)) * np.exp(-3.0 * self.wa * (1.0 - a))

    # ------------------------------------------------------------ expansion
    def E_a(self, a):
        """H(a)/H0."""
        a = np.asarray(a, dtype=float)
        return np.sqrt(self.omega_m * a ** -3 + self.omega_de * self.de_density(a))

    def E(self, z):
        """H(z)/H0."""
        return self.E_a(1.0 / (1.0 + np.asarray(z, dtype=float)))

    def omega_m_a(self, a):
        """Omega_m(a) = 8 pi G rho_m / 3H^2, the ODE's source coefficient."""
        a = np.asarray(a, dtype=float)
        return self.omega_m * a ** -3 / self.E_a(a) ** 2

    def omega_m_z(self, z):
        return self.omega_m_a(1.0 / (1.0 + np.asarray(z, dtype=float)))

    def omega_de_a(self, a):
        a = np.asarray(a, dtype=float)
        return self.omega_de * self.de_density(a) / self.E_a(a) ** 2

    def dlnE_dlna(self, a):
        """dlnH/dln a, analytic rather than differenced.

        Both the friction term of the growth ODE and nDGP's beta(a) need it, and
        a finite difference of E(a) on the ODE's own grid would be the least
        accurate ingredient in either.
        """
        a = np.asarray(a, dtype=float)
        e2 = self.E_a(a) ** 2
        d_matter = -3.0 * self.omega_m * a ** -3
        d_de = -3.0 * (1.0 + self.w(a)) * self.omega_de * self.de_density(a)
        return 0.5 * (d_matter + d_de) / e2

    # ---------------------------------------------------------------- extras
    def replace(self, **kw):
        """A copy with some parameters changed (models differing only in w0/wa)."""
        base = dict(omega_m=self.omega_m, h=self.h, omega_b=self.omega_b,
                    n_s=self.n_s, w0=self.w0, wa=self.wa)
        base.update(kw)
        return Background(**base)

    @property
    def key(self):
        """Hashable identity, used to cache the GR reference solution."""
        return (self.omega_m, self.omega_b, self.h, self.n_s, self.w0, self.wa)

    def __repr__(self):
        return (f"Background(omega_m={self.omega_m:g}, h={self.h:g}, "
                f"w0={self.w0:g}, wa={self.wa:g})")
