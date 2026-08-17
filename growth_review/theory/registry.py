"""The model registry: named theory curves, ready to plot.

    import growth_review as gr
    gr.theory.fsigma8("GR", z)                     # the fiducial reference
    gr.plotting.plot_theory(ax, "GR", "cola_fr")

Registered lazily: a name maps to a factory, and nothing is solved or read from
disk until the model is asked for.

**What is registered by default is deliberately short**: the fiducial LCDM curve,
a growth-index diagnostic, and the shipped COLA tables. Modified-gravity models
are *not* computed in this package. They come from EFTCAMB, either in-process
(``theory/eftcamb.py``, needs a build) or as tables it exported --
``register_exports()`` picks up every ``data/theory/eftcamb_*.ecsv`` and makes it
a first-class model with the same interface.

An earlier version of this module registered quasi-static nDGP and Hu-Sawicki
f(R) models of its own. They were removed: an approximation written here is not
the model the EFTCAMB notebook defines, and having both invited exactly the
confusion of reading one as the other. nDGP does exist again -- but as the source
notebook's own implementation, transcribed into ``theory/eftcamb.py`` alongside
the flag-driven models and reached the same way, by running that module or reading
its export.
"""
from pathlib import Path

import numpy as np

from ..style import PALETTE
from . import tables
from .background import PLANCK18, Background
from .model import Cosmology, GrowthModel, TheoryCurve

_FACTORIES = {}
_FAMILIES = {}
_CACHE = {}


def latex_sci(x, sig=1):
    """Render a float as a LaTeX "m\\times10^{e}" mantissa/exponent string.

    Used for model labels, where "1e-05" in a legend is a tell that nobody looked
    at the figure.
    """
    x = float(x)
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = abs(x) / 10.0 ** exponent
    sign = "-" if x < 0 else ""
    if abs(mantissa - 1.0) < 10.0 ** -sig:
        return rf"{sign}10^{{{exponent}}}"
    return rf"{sign}{mantissa:.{sig}f}\times10^{{{exponent}}}"


# ------------------------------------------------------------------ registry
def register(name, factory, family="", replace=False):
    """Register a factory (a zero-argument callable returning a TheoryCurve)."""
    if name in _FACTORIES and not replace:
        raise KeyError(f"{name!r} is already registered; pass replace=True")
    _FACTORIES[name] = factory
    _FAMILIES[name] = family
    _CACHE.pop(name, None)
    return name


def get(model):
    """A ``TheoryCurve``, from a name or passed straight through."""
    if isinstance(model, TheoryCurve):
        return model
    if model not in _FACTORIES:
        raise KeyError(f"unknown theory model {model!r}; known: {list_models()}. "
                       "Modified-gravity curves are not computed here -- export "
                       "them from EFTCAMB (growth-review-eftcamb-export) and call "
                       "theory.register_exports().")
    if model not in _CACHE:
        built = _FACTORIES[model]()
        if built.name is None:
            built.name = model
        _CACHE[model] = built
    return _CACHE[model]


def list_models(family=None):
    """Registered names, in registration order, optionally by family."""
    return [n for n in _FACTORIES
            if family is None or _FAMILIES.get(n) == family]


def families():
    return list(dict.fromkeys(_FAMILIES.values()))


def summary_table():
    """One line per registered model -- name, family, backend, label."""
    rows = [f"{'name':<28} {'family':<14} {'backend':<9} label", "-" * 92]
    for name in list_models():
        m = get(name)
        rows.append(f"{name:<28} {_FAMILIES.get(name, ''):<14} "
                    f"{type(m).__name__.replace('Model', '').lower():<9} {m.label}")
    return "\n".join(rows)


# ---------------------------------------------------------------- shortcuts
def fiducial():
    """The fiducial cosmology every figure compares against."""
    return get("GR")


def fsigma8(model, z):
    """fsigma8(z) of any registered model, by name -- the one-liner form."""
    return get(model).fsigma8(np.asarray(z, dtype=float))


def sigma8(model, z):
    return get(model).sigma8_z(np.asarray(z, dtype=float))


def growth_rate(model, z):
    return get(model).growth_rate(np.asarray(z, dtype=float))


def ratio(model, reference="GR", z=None, zmin=0.0, zmax=2.0, n=300, percent=True):
    """fsigma8 of `model` relative to `reference` (deviation in % by default)."""
    return get(model).ratio(get(reference), z=z, zmin=zmin, zmax=zmax, n=n,
                            percent=percent)


def growth_index(gamma, background=None, color=PALETTE["grey"], ls="-.",
                 lw=1.6, **kw):
    """The f = Omega_m(z)^gamma prescription (Linder 2005) as a model."""
    return GrowthModel(background=background or Background(), gamma=gamma,
                       name=f"gamma_{gamma:g}",
                       label=rf"$f=\Omega_m(z)^{{{gamma:g}}}$",
                       family="growth index", color=color, ls=ls, lw=lw,
                       source="Linder 2005, astro-ph/0507263",
                       caveats=(
                           "A parametrisation of f, not a theory: no field "
                           "equation produces it, so it predicts nothing else "
                           "(lensing, background) to cross-check against.",
                       ), **kw)


# ------------------------------------------------------------ EFTCAMB exports
# Colours and linestyles of the EFTCAMB notebook's own MODELS registry
# (Science/Peculiar_Vel/theory_plots/modified_gravity/EFTCAMB_fsigma8.ipynb, S3),
# so an exported curve is drawn the way it is drawn there.
EFTCAMB_STYLE = {
    "GR":                     (r"$\Lambda$CDM (GR)", "k", "-", 2.2),
    "Kmouflage":              (r"K-mouflage", "#008d00", "-.", 2.0),
    "ScalingCubicGalileon":   (r"scaling cubic Galileon", "#f57bec", "-.", 2.0),
    "Horava":                 (r"Ho\v{r}ava", "#00cdff", "-.", 2.0),
    "ADE":                    (r"acoustic dark energy", "#6a4c93", "-.", 2.0),
    "Quintessence":           (r"quintessence (potential)", "#25d367", "-.", 2.0),
    "QuintessenceDesigner":   (r"quintessence (designer)", "#9563dd", "--", 2.0),
    "BeyondHorndeski":        (r"beyond Horndeski", "#c53637", "-.", 2.0),
    "JBD_wBD100":             (r"Jordan--Brans--Dicke, $\omega_{BD}=100$",
                               "#5a4600", ":", 2.0),
    "RPH_mass_kin":           (r"RPH, $M_*^2{=}0.2$, $\alpha_K{=}1.5$",
                               "#457b9d", "--", 2.0),
    # nDGP: not an EFTCAMB run (see theory/eftcamb.py), but exported and styled
    # the same way -- the source notebook's own colours and dashed linestyle.
    "nDGP_H0rc1":             (r"nDGP,  $H_0r_c=1$", "#560bad", "--", 2.0),
    "nDGP_H0rc5":             (r"nDGP,  $H_0r_c=5$", "#b5179e", "--", 2.0),
    "pureEFT_Om0.1":          (r"pure EFT, $\Omega_0=0.1$", "#1d3557", "--", 2.0),
    "pureEFT_w0waCDM_DESI":   (r"$w_0w_a$CDM (DESI DR2+CMB+DESY5), $\Omega_0=0.1$",
                               "#e2a700", "--", 2.0),
    # designer f(R), specified by f_R0 -- the tiers of the f(R) N-body literature
    "fR_fR0_1e-04":           (r"$f(R)$,  $f_{R0}=-1.0\times10^{-4}$",
                               "#9d0208", "-", 2.0),
    "fR_fR0_1e-05":           (r"$f(R)$,  $f_{R0}=-1.0\times10^{-5}$",
                               "#f77f00", "-", 2.0),
    "fR_fR0_1e-06":           (r"$f(R)$,  $f_{R0}=-1.0\times10^{-6}$",
                               "#ffd60a", "-", 2.0),
}


def export_style(model_name):
    """(label, color, ls, lw) for an EFTCAMB model, falling back to grey."""
    return EFTCAMB_STYLE.get(model_name, (model_name.replace("_", " "),
                                          "0.35", "-.", 2.0))


def background_deviations(reference="eftcamb_GR", family="eftcamb", z=None):
    """max_z |H_model/H_ref - 1| for every model with an expansion history.

    The quantitative version of "does this model change the background": a model
    that leaves H(z) alone cannot be separated from LCDM by BAO or supernova
    distances, so whatever it does to fsigma8 is growth information those probes
    cannot reach. Models whose table carries no E column (an export whose
    background query failed) are skipped rather than counted as unmodified.
    """
    z = np.linspace(0.0, 2.0, 41) if z is None else np.asarray(z, dtype=float)
    E_ref = get(reference).E(z)
    out = {}
    for name in list_models(family=family):
        if name == reference:
            continue
        try:
            out[name] = float(np.max(np.abs(get(name).E(z) / E_ref - 1.0)))
        except ValueError:                  # no expansion history in the table
            continue
    return out


def background_unmodified(threshold=1e-2, reference="eftcamb_GR", family="eftcamb",
                          exclude=(), z=None, include_reference=True):
    """Names passing the background cut, reference first.

    `threshold` = 1e-3 isolates models that preserve H(z) *by construction*
    (designer f(R), the nDGP phenomenology); 1e-2 also admits those a BAO/SN
    analysis at current precision would not obviously separate from LCDM.
    `exclude` drops models by hand -- the source notebook drops Horava, which
    clears the cut but whose fsigma8 is also indistinguishable from GR's, so it
    contributes no visible curve.
    """
    deviations = background_deviations(reference=reference, family=family, z=z)
    kept = [n for n, dev in deviations.items()
            if dev <= threshold and not any(x in n for x in exclude)]
    return ([reference] + kept) if include_reference else kept


def register_exports(directory=None, prefix="eftcamb_", replace=True):
    """Register every EFTCAMB export table found, and return their names.

    The tables are produced by ``growth-review-eftcamb-export`` on a machine with
    a compiled H-EFTCAMB; copying the ECSV files in is all that is needed for the
    curves to become ordinary registry models here. The model name is taken from
    the file's own header (written by the exporter) rather than guessed from the
    filename, and the style from ``EFTCAMB_STYLE``.
    """
    from ..datasets import DATA_DIR

    directory = Path(directory) if directory else DATA_DIR / "theory"
    names = []
    for path in sorted(directory.glob(f"{prefix}*.ecsv")):
        meta = tables.read_meta(path)
        model = meta.get("model", path.stem[len(prefix):])
        label, color, ls, lw = export_style(model)
        name = f"{prefix}{model}"
        register(name,
                 lambda p=path, l=label, c=color, s=ls, w=lw: tables.table_model(
                     p, label=l, color=c, ls=s, lw=w, norm="early",
                     family="eftcamb",
                     source=f"EFTCAMB export: {p.name}",
                     caveats=("Boltzmann output, shared-A_s sigma8; see the ECSV "
                              "header for the flags, cosmology and stability "
                              "conditions it was run with.",)),
                 family="eftcamb", replace=replace)
        names.append(name)
    return names


# ----------------------------------------------------------------- defaults
def _register_defaults():
    register("GR", Cosmology, family="GR")
    register("gamma_0.68", lambda: growth_index(0.68), family="growth index")

    # Tabulated COLA runs, shipped with the package. k = 0.1 h/Mpc for all three
    # so the curves are comparable; only the f(R) file has real scale dependence.
    for name, (dataset, color, label) in {
            "cola_gr": ("cola_growth_gr", PALETTE["blue"], "COLA GR"),
            "cola_fr": ("cola_growth_fr", PALETTE["orange"], r"COLA $f(R)$"),
            "cola_dgp": ("cola_growth_dgp", PALETTE["aqua"], "COLA nDGP")}.items():
        register(name,
                 lambda dataset=dataset, color=color, label=label:
                     tables.table_model(
                         dataset, k=0.1, color=color, ls=":",
                         label=label + r" ($k=0.1\,h/$Mpc)",
                         family="simulation",
                         source="COLA simulation suite",
                         caveats=("Amplitude is not the run's own: D is "
                                  "normalised to D(z=0)=1 and multiplied by the "
                                  "fiducial sigma8, i.e. the shared-sigma8(0) "
                                  "convention.",)),
                 family="simulation")

    # Any EFTCAMB export already copied into data/theory/ joins at import.
    register_exports()


_register_defaults()
