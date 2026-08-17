"""growth_review -- compiled growth-rate constraints, forecasts and theory curves.

Quick start::

    import growth_review as gr

    gr.use_style()
    df = gr.load_fsigma8(kind="measurement")     # PV + RSD, one tidy schema
    print(gr.summary_table())                    # what is in the package
    fig, meta = gr.figures.fig_pv()              # a figure + its caption

Three kinds of entry live side by side and are never styled alike: published
``measurement``s, projected ``forecast`` precisions, and ``theory`` curves. See
``datasets.py`` for the data registry and the caveat attached to each file.

Theory is computed, not only tabulated::

    z = np.linspace(0, 2, 100)
    gr.theory.fsigma8("GR", z)               # LCDM, Planck 2018 fiducial
    gr.theory.fsigma8("nDGP_H0rc5", z)       # nDGP, quasi-static mu(a)
    gr.theory.ratio("fR_fR0_1e-05", "GR", z) # deviation from LCDM, in %
    gr.plotting.plot_theory(ax, *gr.theory.DEFAULT_SELECTION)

``gr.theory`` integrates the growth equation with a mu(k, a) factor in pure
numpy (LCDM, w0-wa, nDGP, Hu-Sawicki f(R), a DE-scaled mu, a growth index) and
reads tabulated runs for anything else; ``gr.theory.eftcamb`` drives a real
EFTCAMB build where one exists and exports its curves as tables.

**No BibTeX keys live in this package.** They belong to a manuscript, not to a
data compilation. Pass a ``bibkey`` mapping to any ``fig_*`` to get cited
captions; ``notebooks/growth_review.ipynb`` builds one automatically from a
``.bib`` file by matching arXiv identifiers.
"""
from . import datasets, io, methods, plotting, style, theory
from .datasets import DATASETS, get, list_datasets, summary_table
from .io import load_fsigma8, load_raw
from .methods import FAMILY_LABEL, FAMILY_ORDER, evidence_report
from .style import PALETTE, dodge_x, style_axes, use_style
from .theory import PLANCK18, Cosmology, FlatLCDM, load_theory

__version__ = "0.3.0"

__all__ = [
    "datasets", "figures", "io", "methods", "plotting", "style", "theory",
    "Cosmology", "FlatLCDM", "PLANCK18", "PALETTE",
    "DATASETS", "get", "list_datasets", "summary_table",
    "load_fsigma8", "load_raw", "load_theory",
    "FAMILY_LABEL", "FAMILY_ORDER", "evidence_report",
    "dodge_x", "style_axes", "use_style",
]


def __getattr__(name):
    # `figures` is imported lazily: eager-importing it here would make
    # `python -m growth_review.figures` warn about a double import. Must go
    # through importlib -- `from . import figures` would re-enter this hook and
    # recurse, since that statement resolves the submodule via getattr.
    if name == "figures":
        import importlib
        return importlib.import_module(".figures", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
