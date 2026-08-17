"""EFTCAMB backend: full Boltzmann fsigma8(z) for models no analytic mu covers.

Imported lazily -- ``import growth_review`` never touches it. It needs a compiled
H-EFTCAMB build (``$EFTCAMB_PATH``), which in practice means the machine the build
was made on. **This is the only source of modified-gravity curves in the
package**: nothing here is approximated elsewhere, and on a machine without a
build the models are simply unavailable until an export is copied in.

Two ways to use it:

    from growth_review.theory import eftcamb
    z, fs8, s8 = eftcamb.compute("Kmouflage")            # in-process

    growth-review-eftcamb-export --models GR Horava JBD_wBD100
    # -> data/theory/eftcamb_<name>.ecsv, readable everywhere by
    #    growth_review.theory.table_model(path)

The export route is the intended one: run it once where EFTCAMB lives, copy the
ECSV files, and the curves become ordinary registry models on any machine.

Ported from ``Science/Peculiar_Vel/theory_plots/modified_gravity/EFTCAMB_fsigma8.ipynb``
(the flag trees were checked there against this build's own Fortran source, not
only against ``find_your_model/``), with two deliberate changes: root-finding is a
local bisection rather than ``scipy.optimize.brentq``, to keep the package
scipy-free; and failures are raised rather than printed, since a stability
rejection is a physical statement a caller should see.

One model here is not an EFTCAMB run, in the source notebook either: **nDGP**.
DGP has no EFTCAMB mapping, and the covariant embedding of its decoupling limit is
pathological on both branches once coupled to gravity (notebook S3.5), so the
notebook computes it from the standard quasi-static growth ODE with
mu(a) = 1 + 1/(3 beta(a)) and anchors sigma8 to its own CAMB GR run. That path is
transcribed here as ``_ndgp_growth_ode`` / ``_ndgp_results``, integrator and
tolerances included, and dispatched on the ``_ndgp_H0rc`` sentinel key. It still
needs a build for the GR anchor.

Known issue carried over from the notebook, worth keeping in view: this build's
``BeyondHorndeski`` background tabulation (``hubble_parameter()``) disagrees with
its own shooting solver's H0 by ~44%, while its perturbation output (fsigma8,
sigma8) is physically sane. ``background_ez`` raises on it; ``compute`` does not.
"""
import os
import sys
import warnings

import numpy as np

from . import tables

# The compiled H-EFTCAMB python wrapper lives in the `camb` folder at the root
# of the EFTCAMB tree; putting it first on sys.path makes `import camb` pick up
# EFTCAMB rather than any stock CAMB in the environment.
EFTCAMB_PATH = os.environ.get(
    "EFTCAMB_PATH", "/global/homes/r/ravouxco/2_Software/EFTCAMB")

# Cosmological parameters held fixed across all models, so every difference in a
# figure comes from the gravity sector alone. Planck 2018 TT,TE,EE+lowE.
# YHe is pinned everywhere: left automatic, CAMB's BBN predictor can return a
# 0-d array instead of a float on recent numpy, which set_cosmology rejects.
COSMO = dict(H0=67.36, ombh2=0.02237, omch2=0.1200, ns=0.9649, As=2.100e-9,
             tau=0.0544, mnu=0.06, num_massive_neutrinos=1)
YHE = 0.25

# kmax only has to be large enough for a converged sigma8.
PK_SETTINGS = dict(kmax=2.0, WantTransfer=True)

# Stability checks, as in the shipped example notebooks: the mathematical
# ghost/mass conditions are off there, and the physical ghost and gradient
# conditions are the ones that actually reject unviable models.
STABILITY = {
    "EFT_ghost_math_stability": False,
    "EFT_mass_math_stability": False,
    "EFT_ghost_stability": True,
    "EFT_gradient_stability": True,
    "EFT_mass_stability": False,
    "EFT_additional_priors": False,
}

Z_GRID = np.linspace(0.0, 2.0, 41)

_camb = None
_has_eftcamb = None

_MISSING_MESSAGE = (
    f"no EFTCAMB build available (EFTCAMB_PATH={EFTCAMB_PATH}, and `import camb` "
    "failed there). Modified-gravity curves are not computed anywhere else in "
    "this package: run growth-review-eftcamb-export where a build exists and copy "
    "the ECSV files into data/theory/, then theory.register_exports() picks them "
    "up.")


# ------------------------------------------------------------------ the build
def camb_module():
    """The EFTCAMB-flavoured ``camb``, imported once.

    A missing camb is reported as the actionable RuntimeError every other entry
    point in this module raises, not as a bare ImportError from three frames
    down: on most machines this module is simply not usable, and that is a
    configuration fact rather than a bug.
    """
    global _camb
    if _camb is None:
        sys.path.insert(0, os.path.realpath(EFTCAMB_PATH))
        try:
            import camb                               # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(_MISSING_MESSAGE) from exc
        _camb = camb
    return _camb


def available():
    """True if the imported camb is an EFTCAMB build.

    A stock CAMB has no ``EFTCAMB`` attribute on ``CAMBparams``; checking now
    beats discovering it halfway through a run.
    """
    global _has_eftcamb
    if _has_eftcamb is None:
        try:
            camb = camb_module()
        except (ImportError, RuntimeError):
            _has_eftcamb = False
            return False
        probe = camb.set_params(H0=67.3, YHe=YHE)
        _has_eftcamb = hasattr(probe, "EFTCAMB")
        if not _has_eftcamb:
            warnings.warn(
                f"camb at {camb.__file__} is not an EFTCAMB build: only the GR "
                f"model will run. Check EFTCAMB_PATH ({EFTCAMB_PATH}) and that "
                "`make python` has been run there.", stacklevel=2)
    return _has_eftcamb


def _require(name):
    if not available():
        raise RuntimeError(f"{name!r} needs an EFTCAMB build; " + _MISSING_MESSAGE)


# ---------------------------------------------------------------- root finding
def bisect(f, lo, hi, xtol=1e-10, rtol=1e-10, maxiter=200):
    """Plain bisection. Replaces ``scipy.optimize.brentq``.

    Slower in iteration count and entirely adequate here: both roots this module
    needs are monotone in their bracket and each residual evaluation is a
    background solve, so the wall-clock cost is the solve, not the search.
    """
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0:
        raise ValueError(f"f({lo:g})={flo:g} and f({hi:g})={fhi:g} have the same "
                         "sign: the root is not inside the bracket")
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if fmid == 0.0 or (hi - lo) < xtol + rtol * abs(mid):
            return mid
        if flo * fmid < 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------- model registry
def _fr(B0):
    """Designer f(R): EFTflag=3, DesignerEFTmodel=1, one parameter B0.

    B0 is the present-day Compton-wavelength parameter; B0 -> 0 recovers LCDM,
    and larger B0 means a longer-range fifth force, hence enhanced growth.
    """
    return {"EFTflag": 3, "DesignerEFTmodel": 1, "EFTB0": B0, **STABILITY}


MODELS = {
    "GR": dict(params={"EFTflag": 0}, label=r"$\Lambda$CDM (GR)"),

    # ------------------------------------------------------ designer f(R)
    "fR_B0_1e-1": dict(params=_fr(1e-1), label=r"$f(R)$, $B_0=10^{-1}$"),
    "fR_B0_1e-2": dict(params=_fr(1e-2), label=r"$f(R)$, $B_0=10^{-2}$"),
    "fR_B0_1e-3": dict(params=_fr(1e-3), label=r"$f(R)$, $B_0=10^{-3}$"),
    "fR_B0_1e-5": dict(params=_fr(1e-5), label=r"$f(R)$, $B_0=10^{-5}$"),

    # ----------------------------------------------------------- pure EFT
    "pureEFT_Om0.1": dict(
        params={"EFTflag": 1, "PureEFTmodel": 1, "EFTwDE": 11,
                "PureEFTmodelOmega": 1, "EFTOmega0": 0.1,
                **{f"PureEFTmodelGamma{i}": 0 for i in range(1, 7)},
                **STABILITY},
        label=r"pure EFT, $\Omega_0=0.1$"),

    # Pure EFT carrying the DESI DR2 BAO+CMB+DESY5 w0-wa background. Omega0=0.1
    # is the smallest constant non-minimal coupling found (scanned in the source
    # notebook, S3.4) that regularises the phantom crossing at z ~ 0.35: a
    # minimally-coupled single fluid cannot be integrated through w = -1 (Kunz &
    # Sapone 2006) and the ODE solver fails outright. Read this curve as "the
    # DESI expansion history in the smallest EFT extension that runs", not as a
    # DESI-preferred modification of gravity -- Omega0 is not a DESI parameter.
    "pureEFT_w0waCDM_DESI": dict(
        params={"EFTflag": 1, "PureEFTmodel": 1,
                "PureEFTmodelOmega": 1, "EFTOmega0": 0.1,
                **{f"PureEFTmodelGamma{i}": 0 for i in range(1, 7)},
                "EFTwDE": 2, "EFTw0": -0.733, "EFTwa": -1.010,
                **STABILITY},
        label=r"$w_0w_a$CDM (DESI DR2+CMB+DESY5), $\Omega_0=0.1$"),

    # ------------------------------------ alternative parametrisation (RPH)
    "RPH_mass_kin": dict(
        params={"EFTflag": 2, "AltParEFTmodel": 1,
                "RPHmassPmodel": 1, "RPHmassP0": 0.2,
                "RPHkineticitymodel": 1, "RPHkineticity0": 1.5,
                "RPHbraidingmodel": 0, "RPHtensormodel": 0, **STABILITY},
        label=r"RPH, $M_*^2{=}0.2$, $\alpha_K{=}1.5$"),

    # ------------------------ designer minimally-coupled quintessence. Note
    # "designer" names the reconstruction technique, not a promise about the
    # background: this entry targets w0=-0.85, wa=0.3, so its H(z) is NOT
    # LCDM's (unlike DesignerEFTmodel=1, which reconstructs Omega(a) to
    # reproduce a prescribed LCDM expansion history).
    "QuintessenceDesigner": dict(
        params={"EFTflag": 3, "DesignerEFTmodel": 2,
                "EFTwDE": 2, "EFTw0": -0.85, "EFTwa": 0.3, **STABILITY},
        label=r"quintessence (designer, $w_0{=}-0.85$, $w_a{=}0.3$)"),

    # ------------------------------------------------------- full mapping
    "Horava": dict(
        params={"EFTflag": 4, "FullMappingEFTmodel": 1,
                "Horava_eta": 1e-5, "Horava_lambda": 1e-3, "Horava_xi": 1e-5,
                **{**STABILITY, "EFT_ghost_stability": False,
                   "EFT_gradient_stability": False}},   # off, as in example 04
        label=r"Ho\v{r}ava"),
    "ADE": dict(
        params={"EFTflag": 4, "FullMappingEFTmodel": 2,
                "cs2": 2, "Log_ac": -3, "f_ac": 0.07, "p": 1.5, "wf": 1.2,
                **STABILITY},
        label=r"acoustic dark energy"),
    # K-mouflage: this build's background solver rejects massive neutrinos, so
    # the model's own params override COSMO's mnu -- the dict merge in
    # _call_kwargs lets a model win, precisely so the override lives with the
    # model. It is a real crack in "only the gravity sector differs".
    "Kmouflage": dict(
        params={"EFTflag": 4, "FullMappingEFTmodel": 3,
                "alphaU": 0.2, "gammaU": 1, "m": 3.0, "eps2_0": -0.01,
                "gammaA": 0.2, "mnu": 0.0, "num_massive_neutrinos": 0,
                **STABILITY},
        label=r"K-mouflage"),
    "Quintessence": dict(
        params={"EFTflag": 4, "FullMappingEFTmodel": 4, "potential_model": 2,
                "phidot_ini": 1, "V0": 1, "p": 1.5, **STABILITY},
        label=r"quintessence (potential)"),
    "BeyondHorndeski": dict(
        params={"EFTflag": 4, "FullMappingEFTmodel": 5,
                "Beyond_Horndeski_x10": -1.3, "Beyond_Horndeski_x30": 0.4,
                "Beyond_Horndeski_x40": 2.0, **STABILITY},
        label=r"beyond Horndeski"),
    "ScalingCubicGalileon": dict(
        params={"EFTflag": 4, "FullMappingEFTmodel": 6,
                "Scaling_Cubic_A": -0.2, "Scaling_Cubic_beta1": 99,
                "Scaling_Cubic_beta2": 0.8, "Scaling_Cubic_lambda": 155,
                **STABILITY},
        label=r"scaling cubic Galileon"),
    # FullMappingEFTmodel=7 (Extended Galileon) is deliberately absent: several
    # parameter points background-solve in under a second and then hang at the
    # perturbation stage (50+ min of CPU with no result), a property of that
    # module in this build rather than of the parameters.

    # ------------------------------- raw Horndeski (EFTflag=5), see _jbd_params.
    # "_jbd_wBD" is a sentinel key, not an EFTCAMB flag: compute() dispatches on
    # it to a shoot-then-solve path, since JBD needs a per-call root find.
    "JBD_wBD100": dict(params={"_jbd_wBD": 100},
                       label=r"Jordan--Brans--Dicke, $\omega_{BD}=100$"),

    # ------------------------------------------------- nDGP, see _ndgp_results.
    # "_ndgp_H0rc" is a sentinel key, not an EFTCAMB flag: compute(),
    # model_name(), background_ez() and growth_rate_kz() dispatch on it to a
    # standalone quasi-static growth-ODE solve, bypassing camb/EFTCAMB entirely.
    # H0rc = H0*r_c is the nDGP simulation suite's crossover-scale
    # parametrisation; 1 and 5 are its canonical "N1"/"N5" benchmarks
    # (Omega_rc = 0.25 / 0.01).
    "nDGP_H0rc1": dict(params={"_ndgp_H0rc": 1.0}, label=r"nDGP, $H_0r_c=1$"),
    "nDGP_H0rc5": dict(params={"_ndgp_H0rc": 5.0}, label=r"nDGP, $H_0r_c=5$"),
}


# --------------------------------------------------- Jordan-Brans-Dicke path
def _jbd_params(wBD, phi_i=1.0):
    """Flag scaffolding for Horndeski JBD (Horndeski_model=7: cubic G3 Taylor-
    expanded to order 2, wBD entering through the c32 coefficient and the
    phi_ini normalisation).

    Transcribed from ``example/05_Horndeski_jbd.ipynb``, with one deliberate
    change: it runs on this module's own COSMO rather than the shipped example's
    toy H0=44.31, so the JBD curve is on the same footing as every other one.
    """
    return dict(
        **COSMO, YHe=YHE,
        dark_energy_model="EFTCAMB", EFTflag=5, Horndeski_model=7,
        Horndeski_freefunc0_model=1, Horndeski_freefunc1_model=0,
        Horndeski_freefunc2_model=0, Horndeski_freefunc3_model=5,
        Horndeskic3_Taylor_order=2, Horndeskic3a0=0, Horndeskic30=-0.5,
        Horndeskic31=0, Horndeskic32=1.0 / (8 * wBD),
        Horndeski_freefunc4_model=0, Horndeski_freefunc5_model=0,
        Horndeski_freefunc6_model=0,
        Horndeski_parameter_number=0, Horndeski_model_specific_ic=False,
        Horndeski_phi_ini=np.sqrt(4 * wBD * phi_i), Horndeski_phidot_ini=0,
        Horndeski_evolve_hubble=False, Horndeski_shooting=False,
        model_background_num_points=10000, model_background_a_ini=1e-8,
        Horndeski_a_pertcutoff_before=1e-8, EFTCAMB_back_turn_on=1e-8,
        EFTCAMB_turn_on_time=1e-8, EFTCAMB_skip_stability=True,
        EFTCAMB_skip_RGR=True, EFTCAMB_use_background=True,
        EFTCAMB_evolve_delta_phi=True, EFTCAMB_evolve_metric_h=False,
        feedback_level=0,
    )


_JBD_LAMBDA_CACHE = {}
_JBD_RESULTS_CACHE = {}


def _jbd_shoot(wBD):
    """Root-find the ``Horndeskic00`` constant reproducing COSMO['H0'].

    JBD's field equations do not let the effective cosmological-constant term be
    set directly the way LCDM's Lambda can be -- same situation as B0_for_fR0
    below, a different unknown.
    """
    if wBD in _JBD_LAMBDA_CACHE:
        return _JBD_LAMBDA_CACHE[wBD]
    camb = camb_module()
    par = _jbd_params(wBD)

    def residual(lmd):
        pp = dict(par, feedback_level=0, Horndeski_shooting=True)
        pp["Horndeskic00"] = 3 * lmd * (pp["H0"] * 1e3 / 299792458) ** 2
        rl = camb.get_background(camb.set_params(**pp))
        return rl.Params.EFTCAMB_parameter_cache.h0 / pp["H0"] - 1

    _JBD_LAMBDA_CACHE[wBD] = bisect(residual, 0.01, 10.0, rtol=1e-4)
    return _JBD_LAMBDA_CACHE[wBD]


def _jbd_results(wBD, zs):
    """Shoot for lambda, then one full pipeline run for this (wBD, zs).

    Cached: compute(), growth_rate_kz() and background_ez() all want it and each
    call is a full Boltzmann solve.
    """
    key = (wBD, tuple(np.round(np.asarray(zs, dtype=float), 6)))
    if key in _JBD_RESULTS_CACHE:
        return _JBD_RESULTS_CACHE[key]
    camb = camb_module()
    par = dict(_jbd_params(wBD))
    par["Horndeskic00"] = 3 * _jbd_shoot(wBD) * (par["H0"] * 1e3 / 299792458) ** 2
    par.update(redshifts=list(zs), **PK_SETTINGS)
    _JBD_RESULTS_CACHE[key] = camb.get_results(camb.set_params(**par))
    return _JBD_RESULTS_CACHE[key]


# ------------------------------------------------------------------- compute
# ---------------------------------------------------------------- nDGP path
_NDGP_GROWTH_ODE_CACHE = {}


def _ndgp_growth_ode(H0rc):
    """Solve the standard nDGP quasi-static linear growth ODE

        D'' + (2 + dlnH/dN) D' - 1.5 Omega_m(a) mu(a) D = 0 ,   N = ln a

    on this module's own COSMO background -- the plain LCDM E(a). That is the
    "nDGP + DE" convention of the nDGP simulation suites and of the source
    notebook, NOT self-accelerating DGP: the normal branch's own Friedmann
    equation is H^2 - H/r_c = (8 pi G/3) rho, so a dark-energy component is
    implicitly tuned to restore LCDM's expansion history and leave the fifth
    force as the only signal. With

        mu(a) = 1 + 1/(3 beta(a)) ,
        beta(a) = 1 + 2 H r_c (1 + Hdot/3H^2) ,

    the standard scale-independent nDGP fifth-force factor: Koyama & Maartens
    2006 (astro-ph/0511634); Schmidt 2009 (0910.0235) eq. 2.7-2.9; Barreira,
    Sanchez & Schmidt 2016 (1605.03965) eq. 7-8, whose beta(a) is the same
    expression once 2 H r_c is written as (H/H0)/sqrt(Omega_rc) -- the form used
    below. H0rc=None gives the mu=1 (GR) baseline that ``_ndgp_results`` anchors
    the sigma8 rescaling to.

    **This is not an EFTCAMB run**, and it is the one model here that is not.
    DGP is a 5-dimensional braneworld construction with no EFTCAMB mapping, and
    the covariant embedding of its decoupling limit (Chow & Khoury 2009, mapped
    to the Bellini-Sawicki alpha basis and fed to EFTCAMB's RPH spline
    interface) is pathological on both branches once actually coupled to
    gravity: ghost on the dpi/dt<0 branch, gradient instability plus NaN
    fsigma8 on the other, over r_c = 20-100 Gpc, checked against EFTCAMB's own
    stability solver. What is used instead is the quasi-static treatment the
    nDGP simulation and RSD literature uses, exact only on sub-horizon,
    Vainshtein-screened scales -- which is the regime PV and RSD surveys probe,
    but not a Boltzmann solution, and unlike every other curve here it carries
    no stability check of its own.

    Transcribed from the source notebook (S3.5), including its integrator and
    tolerances, so the numbers are that notebook's rather than a reimplementation
    of its formula.
    """
    if H0rc in _NDGP_GROWTH_ODE_CACHE:
        return _NDGP_GROWTH_ODE_CACHE[H0rc]
    try:
        from scipy.integrate import solve_ivp
    except ImportError as exc:                       # pragma: no cover
        raise RuntimeError(
            "the nDGP path needs scipy (solve_ivp), the integrator the source "
            "notebook used; it is not swapped for another one here so the "
            "numbers stay that notebook's") from exc

    h = COSMO["H0"] / 100.0
    # As in the source notebook: Omega_m from the two CDM+baryon densities, the
    # 0.06 eV neutrino left out of it (a ~0.0006 shift in Omega_m h^2).
    Om0 = (COSMO["ombh2"] + COSMO["omch2"]) / h ** 2
    OL0 = 1.0 - Om0
    sqrt_Om_rc = None if H0rc is None else np.sqrt(1.0 / (4.0 * H0rc ** 2))

    def E(a):
        return np.sqrt(Om0 * a ** -3 + OL0)

    def rhs(N, state):
        D, Dp = state
        a = np.exp(N)
        Om_a = Om0 * a ** -3 / E(a) ** 2
        dlnH_dN = -1.5 * Om_a                        # exact for matter + Lambda
        if H0rc is None:
            mu = 1.0
        else:
            beta = 1.0 + (E(a) / sqrt_Om_rc) * (1.0 + dlnH_dN / 3.0)
            mu = 1.0 + 1.0 / (3.0 * beta)
        Dpp = -(2.0 + dlnH_dN) * Dp + 1.5 * Om_a * mu * D
        return [Dp, Dpp]

    # Deep matter domination, where beta -> infinity so mu -> 1 and nDGP is
    # indistinguishable from GR: both models start from the same amplitude.
    N_i = np.log(1e-3)
    a_i = np.exp(N_i)
    sol = solve_ivp(rhs, [N_i, 0.0], [a_i, a_i], method="RK45",
                    rtol=1e-10, atol=1e-14, dense_output=True)
    if not sol.success:
        raise RuntimeError(f"nDGP growth ODE failed for H0rc={H0rc}: {sol.message}")
    _NDGP_GROWTH_ODE_CACHE[H0rc] = sol
    return sol


def _ndgp_results(H0rc, zs):
    """(z, fsigma8, sigma8) for nDGP via the quasi-static growth ODE.

    sigma8 is anchored to this module's own CAMB-computed GR sigma8 (so the GR
    curve still comes from a Boltzmann solve) at the same early epoch the ODE
    starts from, where both models share mu = 1, and propagated forward with the
    ODE's own D_nDGP(a)/D_GR(a) ratio.

    **Normalisation choice, stated because it changes the answer:** both models
    share the primordial amplitude, not a common sigma8(0). nDGP's higher
    sigma8(0) is therefore a real prediction of the model; rescaling to a common
    sigma8(0) instead would hide most of the effect and leave only the shape
    difference in fsigma8(z). Both conventions appear in the literature; this is
    the one every EFTCAMB curve here already uses (a shared A_s).

    Calls ``compute("GR", zs)``, so a build is needed for the anchor even though
    the growth itself is solved here.
    """
    z_gr, fs8_gr, s8_gr = compute("GR", zs)
    a_grid = 1.0 / (1.0 + z_gr)

    sol_gr_ode = _ndgp_growth_ode(None)
    sol_ndgp = _ndgp_growth_ode(H0rc)

    D_gr_ode = np.array([sol_gr_ode.sol(np.log(a))[0] for a in a_grid])
    D_ndgp = np.array([sol_ndgp.sol(np.log(a))[0] for a in a_grid])
    Dp_ndgp = np.array([sol_ndgp.sol(np.log(a))[1] for a in a_grid])
    f_ndgp = Dp_ndgp / D_ndgp

    s8_ndgp = s8_gr * (D_ndgp / D_gr_ode)
    return z_gr, f_ndgp * s8_ndgp, s8_ndgp


def _call_kwargs(name, zs=None, pk=True):
    """COSMO + model params, merged so a model's own params win.

    Merged as one dict rather than two ``**`` unpacks: K-mouflage overrides mnu,
    and ``**COSMO, **params`` in a single call would TypeError on the duplicate.
    """
    params = dict(MODELS[name]["params"])
    if not available():
        if params.get("EFTflag", 0) == 0:
            # EFTflag=0 *is* GR, so a stock CAMB can still produce the reference
            # curve once the EFTCAMB-only keys are dropped.
            params = {k: v for k, v in params.items()
                      if not k.startswith(("EFT", "Horava"))}
        else:
            _require(name)
    out = dict(COSMO)
    if pk:
        out.update(PK_SETTINGS)
    if zs is not None:
        out["redshifts"] = list(zs)
    out.update(params)
    out["YHe"] = YHE
    return out


_CACHE = {}


def compute(name, zs=Z_GRID, use_cache=True):
    """(z ascending, fsigma8, sigma8) for one entry of ``MODELS``.

    CAMB re-sorts the requested redshifts internally, earliest first, and
    ``get_fsigma8()`` returns its array in that order -- highest z first. Zipping
    it against the redshift array as written silently mirrors the curve, so the
    ordering is undone explicitly here.

    A stability violation raises: that is a physical statement about the
    parameter point, not a bug, and callers should see it rather than get a
    silently missing curve.
    """
    zs = np.asarray(zs, dtype=float)
    key = (name, tuple(np.round(zs, 6)))
    if use_cache and key in _CACHE:
        return _CACHE[key]

    params = MODELS[name]["params"]
    # nDGP: the standalone growth-ODE path (see _ndgp_results), which returns
    # ascending-z arrays already -- no CAMB ordering to undo.
    if "_ndgp_H0rc" in params:
        out = _ndgp_results(params["_ndgp_H0rc"], zs)
        if use_cache:
            _CACHE[key] = out
        return out

    if "_jbd_wBD" in params:
        _require(name)
        results = _jbd_results(params["_jbd_wBD"], zs)
    else:
        # kwargs first: _call_kwargs is what raises the actionable error for a
        # model that needs EFTCAMB on a machine without one.
        kwargs = _call_kwargs(name, zs)
        camb = camb_module()
        results = camb.get_results(camb.set_params(**kwargs))

    fs8 = np.asarray(results.get_fsigma8())[::-1]      # back to ascending z
    s8 = np.asarray(results.get_sigma8())[::-1]
    out = (np.sort(zs), fs8, s8)
    if use_cache:
        _CACHE[key] = out
    return out


def model_name(name):
    """EFTCAMB's own name for the model -- confirms what actually ran."""
    params = MODELS[name]["params"]
    if "_ndgp_H0rc" in params:
        return ("nDGP (quasi-static phenomenological, not an EFTCAMB run), "
                f"H0*r_c={params['_ndgp_H0rc']}")
    if name == "GR" or not available():
        return "GR / LCDM"
    if "_jbd_wBD" in params:
        return f"Jordan-Brans-Dicke (Horndeski), wBD={params['_jbd_wBD']}"
    kwargs = _call_kwargs(name, pk=False)
    return camb_module().set_params(**kwargs).EFTCAMB.model_name()


def background_ez(name, zs=Z_GRID):
    """H(z)/H0, background only (~10x faster than the full pipeline).

    Checked against H(z=0) = H0, which is true by construction for every model
    here (H0 is a shared input, never fit). A mismatch means this build's H(z)
    tabulation disagrees with the H0 it was given -- how the BeyondHorndeski
    problem in the module docstring was found.
    """
    zs = np.asarray(zs, dtype=float)
    params = MODELS[name]["params"]
    # In the "nDGP + DE" convention this module uses (see _ndgp_growth_ode), the
    # expansion history IS LCDM's by construction -- a dark-energy component is
    # tuned to make it so, which is what leaves the fifth force as the only
    # signal. So this returns GR's own background rather than recomputing it.
    if "_ndgp_H0rc" in params:
        return background_ez("GR", zs)

    if "_jbd_wBD" in params:
        _require(name)
        results = _jbd_results(params["_jbd_wBD"], zs)
    else:
        kwargs = _call_kwargs(name, pk=False)
        camb = camb_module()
        results = camb.get_background(camb.set_params(**kwargs), no_thermo=True)

    hz = np.array([results.hubble_parameter(z) for z in zs])
    if np.any(zs == 0.0):
        h0 = hz[np.argmin(np.abs(zs))]
        rel = abs(h0 - COSMO["H0"]) / COSMO["H0"]
        if rel > 0.01:
            raise RuntimeError(
                f"{name}: hubble_parameter(z=0)={h0:.4f} disagrees with "
                f"H0={COSMO['H0']} by {rel:.1%} -- this build's background "
                "tabulation is inconsistent with the H0 it was given for this "
                "flag combination (BeyondHorndeski does this; its perturbation "
                "output is fine).")
    return zs, hz / COSMO["H0"]


def growth_rate_kz(name, zs=np.linspace(0.0, 2.0, 21),
                   k_nDGP=(0.01, 0.05, 0.1, 0.5)):
    """f(k, z) reconstructed from the linear power spectrum,
    f = -(1/2) dlnP/dln(1+z).

    The point of having it: ``get_fsigma8`` returns one number per redshift, so
    it compresses away f(R)'s scale dependence entirely. Under GR the curves for
    different k lie on top of each other; any spread is scale dependence.
    """
    zs = np.asarray(zs, dtype=float)
    params = MODELS[name]["params"]
    # nDGP: mu(a) has no k dependence, so its linear growth rate is
    # scale-independent by construction -- the opposite structural behaviour from
    # f(R), and the point of the comparison. f(z) is broadcast across a nominal k
    # array rather than reconstructed from a P(k,z) that does not exist here.
    if "_ndgp_H0rc" in params:
        z_out, fs8, s8 = _ndgp_results(params["_ndgp_H0rc"], zs)
        k = np.asarray(k_nDGP, dtype=float)
        return k, z_out, np.tile((fs8 / s8)[:, None], (1, k.size))

    if "_jbd_wBD" in params:
        _require(name)
        results = _jbd_results(params["_jbd_wBD"], zs)
    else:
        kwargs = _call_kwargs(name, zs)
        camb = camb_module()
        results = camb.get_results(camb.set_params(**kwargs))

    k, z_out, pk = results.get_linear_matter_power_spectrum(
        var1="delta_tot", var2="delta_tot", hubble_units=True, k_hunit=True,
        nonlinear=False)
    z_out = np.asarray(z_out)
    order = np.argsort(z_out)
    z_out, pk = z_out[order], pk[order, :]
    return k, z_out, -0.5 * np.gradient(np.log(pk), np.log(1.0 + z_out), axis=0)


# ------------------------------------------------- designer f(R): B0 <-> f_R0
_FR0_CACHE = {}


def fR0_from_B0(B0):
    """Present-day f_R0 of a designer f(R) model with Compton parameter B0.

    Not a closed-form B0 <-> f_R0 relation: designer f(R) fixes the *background*
    and solves for whatever f(R) reproduces it, so f_R0 only exists as the
    output of that solve. EFTCAMB sets Omega(a) = f_R(a) for this model
    (``fortran/eftcamb/07f_designer_models/007p3_Designer_fR.f90``, where
    f_sub_R is assigned to ``self%EFTOmega%y``), which is what is read back
    here at a = 1.

    Uses ``get_background(..., no_thermo=True)``: f_R0 needs the designer ODE
    solve only, not thermodynamics or transfer functions -- ~10x faster, same
    answer, and what EFTCAMB's own shooting examples do.
    """
    _require("designer f(R)")
    if B0 in _FR0_CACHE:
        return _FR0_CACHE[B0]
    camb = camb_module()
    pars = camb.set_params(**COSMO, **_fr(B0), YHe=YHE)
    results = camb.get_background(pars, no_thermo=True)
    _, vals = pars.EFTCAMB.get_eft_functions(results, np.array([1.0]))
    _FR0_CACHE[B0] = float(vals["EFTOmegaV"][0])
    return _FR0_CACHE[B0]


def B0_for_fR0(target_fR0, bracket=(1e-8, 0.2)):
    """Invert ``fR0_from_B0``: the B0 whose designer solve gives this f_R0.

    Root-found in log10(B0), since f_R0(B0) spans decades as B0 does.
    ``target_fR0`` must be negative -- EFTCAMB's sign convention for the
    growth-enhancing branch, the same one the Hu-Sawicki literature uses.
    """
    if target_fR0 >= 0:
        raise ValueError(f"target_fR0 must be negative (got {target_fR0!r}); the "
                         "designer f(R) growth-enhancing branch has f_R0 < 0.")
    log10_B0 = bisect(lambda x: fR0_from_B0(10.0 ** x) - target_fR0,
                      np.log10(bracket[0]), np.log10(bracket[1]),
                      xtol=1e-10, rtol=1e-10)
    return 10.0 ** log10_B0


def register_fR0(target_fR0, name=None, label=None):
    """Register a designer f(R) entry specified by f_R0 instead of B0.

    Most of the f(R) literature quotes |f_R0| (the 1e-4/1e-5/1e-6 tiers), while
    designer f(R) only takes B0, so specifying a model that way means inverting
    the solve. Returns (name, B0).
    """
    B0 = B0_for_fR0(target_fR0)
    name = name or f"fR_fR0_{abs(target_fR0):.0e}"
    MODELS[name] = dict(params=_fr(B0),
                        label=label or rf"$f(R)$, $|f_{{R0}}|={abs(target_fR0):.0e}$")
    return name, B0


# -------------------------------------------------------------------- export
def export(name, path, zs=Z_GRID, with_background=True):
    """Run one model and write it as an ECSV table readable by ``tables``.

    The header records the flags and the cosmology: a bare (z, fsigma8) table
    with no record of what produced it is not reusable six months later.
    """
    z, fs8, s8 = compute(name, zs)
    ez = None
    if with_background:
        try:
            _, ez = background_ez(name, z)
        except RuntimeError as exc:      # e.g. BeyondHorndeski's H(z) tabulation
            warnings.warn(f"{name}: background not exported ({exc})", stacklevel=2)
    meta = dict(model=name, eftcamb_model_name=model_name(name),
                label=MODELS[name]["label"],
                params={k: v for k, v in MODELS[name]["params"].items()},
                cosmology=dict(COSMO), stability=dict(STABILITY),
                sigma8_norm="shared A_s (EFTCAMB output, not rescaled)",
                source="growth_review.theory.eftcamb")
    return tables.write_export(path, z, s8, fs8, E=ez, meta=meta)


def main(argv=None):
    """CLI: ``growth-review-eftcamb-export``."""
    import argparse
    from pathlib import Path

    from .. import datasets

    p = argparse.ArgumentParser(
        description="Run EFTCAMB models and write theory tables the rest of "
                    "growth_review can read. Needs a compiled H-EFTCAMB "
                    "(EFTCAMB_PATH).")
    p.add_argument("--models", nargs="*", default=["GR"],
                   help=f"names from MODELS: {', '.join(MODELS)}")
    p.add_argument("--fR0", nargs="*", type=float, default=[],
                   help="also register+export designer f(R) at these f_R0 "
                        "(negative, e.g. -1e-5)")
    p.add_argument("--outdir", type=Path,
                   default=datasets.DATA_DIR / "theory")
    p.add_argument("--zmax", type=float, default=2.0)
    p.add_argument("--nz", type=int, default=41)
    p.add_argument("--prefix", default="eftcamb_")
    args = p.parse_args(argv)

    zs = np.linspace(0.0, args.zmax, args.nz)
    names = list(args.models)
    for target in args.fR0:
        name, B0 = register_fR0(target)
        print(f"registered {name}: f_R0={target:.3e} -> B0={B0:.6e}")
        names.append(name)

    args.outdir.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = args.outdir / f"{args.prefix}{name}.ecsv"
        export(name, path, zs)
        z, fs8, s8 = compute(name, zs)
        print(f"{name:24s} -> {path}   fs8(0)={fs8[0]:.4f}  s8(0)={s8[0]:.4f}  "
              f"[{model_name(name)}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
