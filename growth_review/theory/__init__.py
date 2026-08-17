"""Theory curves for the growth review: fsigma8(z), callable from any plot.

    import growth_review as gr

    z = np.linspace(0, 2, 100)
    gr.theory.fsigma8("GR", z)              # the fiducial LCDM reference
    gr.plotting.plot_theory(ax, "GR", "cola_fr")

Two sources of curves, and the split is the point:

**The fiducial reference** -- flat LCDM (or w0-wa) with GR growth -- is integrated
here in numpy, because every figure needs it and it costs nothing. That is all
this layer computes.

**Modified gravity comes from EFTCAMB.** ``eftcamb.py`` drives a compiled
H-EFTCAMB (``$EFTCAMB_PATH``) for the models the notebook it ports defines --
designer f(R), K-mouflage, Hořava, Jordan-Brans-Dicke, beyond Horndeski, the
Galileons, pure-EFT and RPH parametrisations -- and writes ECSV tables that
``registry.register_exports()`` turns into ordinary models on any machine.
Nothing here approximates those models: a quasi-static mu(k,a) layer was written,
found to be a substitute rather than a reproduction, and removed.

Layout:

    background.py  flat CPL background: E(a), Omega_m(a), dlnH/dlna
    growth.py      RK4 integration of the GR growth equation, + the growth index
    model.py       GrowthModel / TableModel / Cosmology -- the fsigma8 objects
    registry.py    named models, their styles, the one-liner accessors
    tables.py      tabulated curves (COLA runs, EFTCAMB exports)
    eftcamb.py     the EFTCAMB backend + table-export CLI
"""
from . import background, growth, model, registry, tables
from .background import PLANCK18, Background
from .growth import GrowthSolution, growth_from_index, solve_growth
from .model import Cosmology, FlatLCDM, GrowthModel, Style, TableModel, TheoryCurve
from .registry import (EFTCAMB_STYLE, background_deviations,
                       background_unmodified, export_style, families, fiducial,
                       fsigma8, get, growth_index, growth_rate, latex_sci,
                       list_models, ratio, register, register_exports, sigma8,
                       summary_table)
from .tables import (load_export, load_theory, read_meta, table_model,
                     write_export)

__all__ = [
    "background", "growth", "model", "registry", "tables",
    "Background", "PLANCK18",
    "GrowthSolution", "solve_growth", "growth_from_index",
    "TheoryCurve", "GrowthModel", "TableModel", "Cosmology", "FlatLCDM", "Style",
    "get", "register", "list_models", "families", "summary_table", "fiducial",
    "fsigma8", "sigma8", "growth_rate", "ratio", "latex_sci", "growth_index",
    "register_exports", "EFTCAMB_STYLE", "export_style",
    "background_deviations", "background_unmodified",
    "load_theory", "load_export", "table_model", "write_export", "read_meta",
]


def __getattr__(name):
    # eftcamb is imported lazily: importing it eagerly would put an `import camb`
    # attempt (and its EFTCAMB path probe) in the way of every plain
    # `import growth_review`, on machines that will never have one.
    if name == "eftcamb":
        import importlib
        return importlib.import_module(".eftcamb", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
