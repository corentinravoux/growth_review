"""Linear growth of the fiducial LCDM (or w0-wa) cosmology, by direct integration.

    D'' + (2 + dlnH/dlna) D' - (3/2) Omega_m(a) D = 0 ,   ' = d/dlna

integrated with a fixed-step RK4 in N = ln a from deep in matter domination to
today. GR only -- this exists to draw the reference curve every figure compares
against, which is what the package's old ``cosmology.py`` did.

**Modified gravity is not computed here.** It comes from EFTCAMB
(``theory/eftcamb.py``, or a table it exported) or from a simulation table
(``theory/tables.py``). A quasi-static mu(k,a) approximation was tried in this
layer and removed: substituting an approximation of one's own for the Boltzmann
solve the models were defined with is not a reproduction of them.

Initial conditions are the growing mode in matter domination, D = D' = a at
a_ini = 1e-3. Radiation is absent from the background, which biases the growing
mode's amplitude there; D is normalised at a = 1, so a common amplitude error
divides out.

RK4 with 2000 steps rather than a scipy integrator keeps the package
dependency-free; the step is uniform in ln a, the solution is smooth, and
halving the step changes D(a=1) by 2e-11. The result agrees with the closed-form
LCDM growth integral to 1e-6 (asserted in the test suite).
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GrowthSolution:
    """D and f on the solver's own grid, with a as the independent variable."""
    a: np.ndarray            # (n_a,) ascending, ending at 1.0
    D: np.ndarray            # (n_a,) unnormalised (D -> a as a -> a_ini)
    f: np.ndarray            # (n_a,) dlnD/dlna

    def D_at(self, a):
        """D interpolated in ln a."""
        return np.interp(np.log(np.asarray(a, dtype=float)), np.log(self.a), self.D)

    def f_at(self, a):
        return np.interp(np.log(np.asarray(a, dtype=float)), np.log(self.a), self.f)


def solve_growth(background, a_ini=1e-3, n_steps=2000):
    """Integrate the GR growth equation. Returns a `GrowthSolution`."""
    lna = np.linspace(np.log(a_ini), 0.0, n_steps + 1)
    h = lna[1] - lna[0]

    def rhs(n, y):
        a = np.exp(n)
        om = background.omega_m_a(a)
        friction = 2.0 + background.dlnE_dlna(a)
        d, dp = y
        return np.array([dp, -friction * dp + 1.5 * om * d])

    y = np.array([a_ini, a_ini])
    out = np.empty((n_steps + 1, 2))
    out[0] = y
    for i in range(n_steps):
        n0 = lna[i]
        k1 = rhs(n0, y)
        k2 = rhs(n0 + 0.5 * h, y + 0.5 * h * k1)
        k3 = rhs(n0 + 0.5 * h, y + 0.5 * h * k2)
        k4 = rhs(n0 + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[i + 1] = y

    D = out[:, 0]
    # f = dlnD/dlna comes straight out of the state vector -- no differencing of
    # the solution, which would be the least accurate step in the chain.
    return GrowthSolution(a=np.exp(lna), D=D, f=out[:, 1] / D)


def growth_from_index(background, gamma=0.55, a_ini=1e-3, n_steps=2000):
    """The f = Omega_m(a)^gamma prescription, as a `GrowthSolution`.

    Linder's growth index (2005, astro-ph/0507263) fixes f directly, so D follows
    by integrating dlnD = f dlna rather than by solving the second-order
    equation. gamma = 0.55 reproduces GR to ~0.1%. A one-parameter diagnostic,
    not a theory: no field equation produces it, and it predicts nothing else
    that could be cross-checked. This was already in the package before the
    theory layer existed (``cosmology.fsigma8_gamma``) and is kept as it was.
    """
    lna = np.linspace(np.log(a_ini), 0.0, n_steps + 1)
    a = np.exp(lna)
    f = background.omega_m_a(a) ** gamma
    integral = np.zeros_like(lna)
    integral[1:] = np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(lna))
    D = a_ini * np.exp(integral)
    return GrowthSolution(a=a, D=D, f=f)
