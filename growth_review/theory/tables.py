"""Tabulated theory curves: COLA runs, and EFTCAMB exports.

Two on-disk layouts are read, and they are told apart by their columns rather
than by their filename:

**COLA growth tables** (``data/theory/cola_growth_*.ecsv``) -- ``a``, ``H/H0``
and a ``D_<k>``/``f_<k>`` pair per tabulated wavenumber. Scale-dependent growth
is stored as separate columns, so a single curve means picking a k, and picking
a k is a statement about which scales the comparison is about (see the f(R)
caveat in ``model.py``).

**EFTCAMB exports** (``eftcamb_*.ecsv``) -- ``z``, ``E``, ``sigma8``,
``fsigma8``, written by ``theory/eftcamb.py`` on a machine where EFTCAMB is
built. These carry their own sigma8 amplitude, in the shared-A_s convention, so
they are used as they are rather than renormalised.

Rows with a > 1 are dropped from the COLA files: the last one carries a spurious
f from the end-of-grid derivative (registry caveat, and the reason it is done
here once instead of in every caller).
"""
from pathlib import Path

import numpy as np

# Columns an export table must have to be read as one.
EXPORT_COLUMNS = ("z", "sigma8", "fsigma8")


def _read(source):
    """A registered dataset name, or a path to an ECSV file, as a DataFrame.

    Paths are accepted so that a table exported on another machine (an EFTCAMB
    run on NERSC) is usable the moment it is copied in, without first being
    declared in ``datasets.py``. Registry entries stay the right home for
    anything that ships with the package.
    """
    if isinstance(source, Path) or str(source).endswith(".ecsv"):
        from astropy.table import Table
        return Table.read(str(source), format="ascii.ecsv").to_pandas()
    from .. import io
    return io.load_raw(source)


def read_meta(path):
    """The ECSV header metadata of an export table (flags, cosmology, model name).

    The exporter writes what produced the curve into the header, so a table copied
    off another machine still says which EFTCAMB flags, which cosmology and which
    stability conditions it came from. Read separately from the data because the
    registry needs the model name before it builds anything.
    """
    from astropy.table import Table
    return dict(Table.read(str(path), format="ascii.ecsv").meta)


def wavenumbers(name):
    """The wavenumbers tabulated in a COLA-style table, in h/Mpc."""
    df = _read(name)
    return sorted({float(c.split("_", 1)[1]) for c in df.columns
                   if c.startswith("D_")})


def load_theory(name, k=0.01):
    """A COLA growth table as (z, a, D, f, fD), at wavenumber `k`.

    `k` selects the ``D_<k>``/``f_<k>`` column pair; available values are listed
    in the error message if the requested one is not tabulated. D is normalised
    to D(a=1) = 1, so ``fD`` is an fsigma8 *shape* -- multiply by a sigma8.
    """
    import pandas as pd

    df = _read(name)
    kcols = sorted({float(c.split("_", 1)[1]) for c in df.columns
                    if c.startswith("D_")})
    if not kcols:
        raise ValueError(f"{name!r} has no D_<k> columns; it is not a COLA-style "
                         "growth table")
    if not any(abs(kk - k) < 1e-12 for kk in kcols):
        raise ValueError(f"k={k} not tabulated in {name!r}; available: {kcols}")
    key = next(c.split("_", 1)[1] for c in df.columns
               if c.startswith("D_") and abs(float(c.split("_", 1)[1]) - k) < 1e-12)

    df = df[df["a"] <= 1.0]
    a = df["a"].to_numpy()
    D = df[f"D_{key}"].to_numpy()
    f = df[f"f_{key}"].to_numpy()
    D = D / D[-1]                                    # normalise D(a=1) = 1
    out = pd.DataFrame({"z": 1.0 / a - 1.0, "a": a, "D": D, "f": f, "fD": f * D})
    if "H/H0" in df.columns:
        out["E"] = df["H/H0"].to_numpy()
    return out


def load_export(name):
    """An EFTCAMB export table as (z, E, sigma8, fsigma8, D, f)."""
    import pandas as pd

    df = _read(name)
    missing = [c for c in EXPORT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{name!r} is missing export columns {missing}; "
                         f"has {list(df.columns)}")
    df = df.sort_values("z").reset_index(drop=True)
    s8 = df["sigma8"].to_numpy()
    out = pd.DataFrame({"z": df["z"].to_numpy(), "sigma8": s8,
                        "fsigma8": df["fsigma8"].to_numpy(),
                        "D": s8 / s8[np.argmin(df["z"].to_numpy())],
                        "f": df["fsigma8"].to_numpy() / s8})
    if "E" in df.columns:
        out["E"] = df["E"].to_numpy()
    return out


def load_curve(name, k=None):
    """Either layout, dispatched on the columns present."""
    df = _read(name)
    if all(c in df.columns for c in EXPORT_COLUMNS):
        return load_export(name)
    return load_theory(name, k=0.01 if k is None else k)


def table_model(dataset, k=None, label=None, **style):
    """A ``TableModel`` over a registered table or an ECSV path, ready to plot.

    The wavenumber is part of the identity of a COLA curve, so it goes into the
    default label: a reader who sees "COLA f(R)" without a k has been told less
    than the table knows.
    """
    from .model import TableModel

    curve = load_curve(dataset, k=k)
    stem = Path(str(dataset)).stem if str(dataset).endswith(".ecsv") else str(dataset)
    if label is None:
        label = stem.replace("_", " ")
        if k is not None:
            label += rf" ($k={k:g}\,h/$Mpc)"
    return TableModel(curve, name=stem if k is None else f"{stem}@k{k:g}",
                      label=label, k=k, **style)


def write_export(path, z, sigma8, fsigma8, E=None, meta=None):
    """Write an EFTCAMB-export ECSV table -- the format ``load_export`` reads.

    Used by ``theory/eftcamb.py`` after a Boltzmann run, so a curve computed
    once on a machine with EFTCAMB becomes a first-class theory model
    everywhere else. `meta` lands in the ECSV header, which is where the model's
    flags and cosmology belong: a bare (z, fsigma8) table with no record of what
    produced it is not reusable six months later.
    """
    from astropy.table import Table

    cols = {"z": np.asarray(z, dtype=float),
            "sigma8": np.asarray(sigma8, dtype=float),
            "fsigma8": np.asarray(fsigma8, dtype=float)}
    if E is not None:
        cols["E"] = np.asarray(E, dtype=float)
    t = Table(cols)
    t.meta.update(meta or {})
    t.write(path, format="ascii.ecsv", overwrite=True)
    return path
