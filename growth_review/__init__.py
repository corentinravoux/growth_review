"""growth_review -- compiled growth-rate constraints, forecasts and theory curves.

Quick start::

    import growth_review as gr

    gr.use_style()
    df = gr.load_fsigma8(kind="measurement")     # PV + RSD, one tidy schema
    print(gr.summary_table())                    # what is in the package
    fig, meta = gr.figures.fig_pv()              # a figure + its cited caption

Three kinds of entry live side by side and are never styled alike: published
``measurement``s, projected ``forecast`` precisions, and tabulated ``theory``
curves. See ``datasets.py`` for the registry and the caveat attached to each
file.
"""
from . import citations, datasets, io, methods, plotting, style
from .citations import cited_only, uncited
from .cosmology import PLANCK18, FlatLCDM
from .datasets import DATASETS, get, list_datasets, summary_table
from .io import load_fsigma8, load_raw, load_theory
from .methods import FAMILY_LABEL, FAMILY_ORDER, evidence_report
from .style import PALETTE, dodge_x, style_axes, use_style

__version__ = "0.1.0"


def __getattr__(name):
    # `figures` is imported lazily: eager-importing it here would make
    # `python -m growth_review.figures` warn about a double import. Must go
    # through importlib -- `from . import figures` would re-enter this hook and
    # recurse, since that statement resolves the submodule via getattr.
    if name == "figures":
        import importlib
        return importlib.import_module(".figures", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "citations", "datasets", "figures", "io", "methods", "plotting", "style",
    "FlatLCDM", "PLANCK18", "PALETTE",
    "DATASETS", "get", "list_datasets", "summary_table",
    "load_fsigma8", "load_raw", "load_theory",
    "FAMILY_LABEL", "FAMILY_ORDER", "evidence_report",
    "cited_only", "uncited", "dodge_x", "style_axes", "use_style",
]
