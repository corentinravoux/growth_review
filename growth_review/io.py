"""Readers, and a tidy fsigma8 view over them.

Two levels:

``load_raw(name)``
    the file exactly as it is on disk, columns and all, for when you need a
    column the tidy view drops (fiducial cosmologies, number counts, ...).

``load_fsigma8(...)``
    every fsigma8 constraint in the registry -- PV and RSD, measurements and
    forecasts -- concatenated into ONE frame with one schema, so a plotting
    function never has to know which file a row came from. Unit conventions
    (fractional vs percent errors) are resolved here, once, from the registry
    metadata rather than at each call site.
"""
import numpy as np
import pandas as pd

from . import datasets as _ds

# Tidy schema. `value`/`err_lo`/`err_hi` are absolute and only filled for
# measurements; `frac_err` is a fractional (never percent) precision and only
# filled for forecasts -- there is no central value to attach it to until a
# fiducial cosmology is chosen, which is the plotting layer's job.
TIDY_COLUMNS = ["key", "ref", "label", "z", "value", "err_lo", "err_hi",
                "frac_err", "kind", "probe", "method", "dataset",
                "method_family", "fit_technique", "sigma8_norm", "provenance"]


# --------------------------------------------------------------------- raw I/O
def _read_ascii(path):
    """Whitespace-delimited table, with the header found rather than guessed.

    Three layouts occur in the package, and they have to be told apart:

      (a) a leading '#' preamble whose LAST line is the header
          (`# z f error_ref ...`);
      (b) a leading '#' preamble that is prose only, the header being the first
          uncommented line (the DESI design-forecast tables, whose first line is
          `#-- Table 2.3 of DESI Collaboration et al. 2016 Part I`);
      (c) no preamble at all.

    The discriminator between (a) and (b) is the token count: a commented header
    has exactly as many fields as a data row, a prose line almost never does.

    The format is chosen explicitly rather than left to astropy's guesser
    because the guesser is not safe here -- on a multi-line '#' preamble it
    picks the SExtractor reader and raises.
    """
    from astropy.table import Table
    with open(path) as f:
        lines = f.readlines()

    header_idx, first_data = None, None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            header_idx = i
        elif line.strip():
            first_data = line
            break

    commented = False
    if header_idx is not None and first_data is not None:
        n_header = len(lines[header_idx].lstrip().lstrip("#").split())
        commented = n_header == len(first_data.split())

    if commented:
        t = Table.read(path, format="ascii.commented_header",
                       header_start=header_idx, guess=False)
    else:
        # ascii.basic strips '#' lines as comments and takes the first
        # remaining line as the header.
        t = Table.read(path, format="ascii.basic", guess=False)
    return t.to_pandas()


def load_raw(name):
    """The named dataset, as a DataFrame, with no renaming or unit conversion."""
    d = _ds.get(name)
    if d.fmt == "csv":
        return pd.read_csv(d.full_path, comment="#")
    if d.fmt == "ascii":
        return _read_ascii(d.full_path)
    if d.fmt == "ecsv":
        from astropy.table import Table
        return Table.read(d.full_path, format="ascii.ecsv").to_pandas()
    raise ValueError(f"unhandled format {d.fmt!r} for {name!r}")


# ------------------------------------------------------------ tidy fsigma8 view
def _blank(n):
    return pd.DataFrame({c: pd.Series([np.nan] * n, dtype=object) for c in TIDY_COLUMNS})


def _tidy_pv(desi_estimators=False, drop_derived=False):
    df = load_raw("fsigma8_pv")
    if "kind" in df.columns:                     # older copies of the file
        df = df[df["kind"] == "measurement"]
    if not desi_estimators:
        df = df[~df["key"].isin(["DESI_DR1_maxlike", "DESI_DR1_corrfunc",
                                 "DESI_DR1_powerspec"])]
    if drop_derived:
        # provenance='der' rows assume f = Omega_m^0.55 to call their result
        # fsigma8 at all -- circular in any test of gravity. See the registry
        # caveat and WARNING 7 in the data file.
        df = df[df["provenance"] != "der"]
    df = df.reset_index(drop=True)

    out = _blank(len(df))
    out["key"] = df["key"].to_numpy()
    out["ref"] = df["authors"].to_numpy()
    out["label"] = df["dataset"].to_numpy()
    out["z"] = pd.to_numeric(df["z_eff"], errors="coerce").to_numpy()
    out["value"] = df["fsigma8"].to_numpy()
    # A separately-quoted systematic is added in quadrature; rows without one
    # get err_sys = 0, a no-op.
    sys_err = pd.to_numeric(df["err_sys"], errors="coerce").fillna(0.0)
    out["err_lo"] = np.sqrt(df["err_lo"].to_numpy() ** 2 + sys_err.to_numpy() ** 2)
    out["err_hi"] = np.sqrt(df["err_hi"].to_numpy() ** 2 + sys_err.to_numpy() ** 2)
    out["kind"] = "measurement"
    out["probe"] = "fsigma8"
    out["method"] = "pv"
    out["dataset"] = "fsigma8_pv"
    out["method_family"] = df["method_family"].to_numpy()
    out["fit_technique"] = df["fit_technique"].to_numpy()
    out["sigma8_norm"] = df["sigma8_norm"].to_numpy()
    out["provenance"] = df["provenance"].to_numpy()
    return out


def _tidy_rsd(drop_sdss_final=True):
    df = load_raw("fsigma8_rsd")
    if drop_sdss_final:
        # ref='SDSS' ("SDSS final") repeats six rows verbatim under a second name.
        df = df[df["ref"] != "SDSS"]
    df = df.reset_index(drop=True)

    out = _blank(len(df))
    out["key"] = (df["ref"].astype(str) + "/" + df["label"].astype(str)).to_numpy()
    out["ref"] = df["ref"].to_numpy()
    out["label"] = df["label"].to_numpy()
    out["z"] = df["zeff"].to_numpy()
    out["value"] = df["fs8_value"].to_numpy()
    out["err_lo"] = df["fs8_error"].to_numpy()
    out["err_hi"] = df["fs8_error"].to_numpy()
    out["kind"] = "measurement"
    out["probe"] = "fsigma8"
    out["method"] = "rsd"
    out["dataset"] = "fsigma8_rsd"
    return out


# Forecast files: (dataset, z column, error column, is-percent, ref, label column).
# The is-percent flag is the one thing that silently corrupts a forecast panel,
# so it lives here next to the file rather than in whoever plots it.
_FORECASTS = [
    ("fsigma8_euclid", "z", "error_ref", False, "Euclid", None, "rsd"),
    ("fsigma8_sn_pv_howlett2017", "z", "error_ref", True, "Howlett+17 SN-PV", None, "pv"),
    ("fsigma8_4most_bgs_howlett2017", "z", "error_ref", True, "Howlett+17 4MOST BGS", None, "pv"),
    ("bao_rsd_desi", "z", "dfs0.1", True, "DESI design (ELG/LRG/QSO)", None, "rsd"),
    ("bao_rsd_desi_bgs", "z", "dfs0.1", True, "DESI design (BGS)", None, "rsd"),
]


def _tidy_forecasts():
    frames = []
    for name, zc, ec, pct, ref, labc, method in _FORECASTS:
        df = load_raw(name)
        out = _blank(len(df))
        out["key"] = [f"{name}:{i}" for i in range(len(df))]
        out["ref"] = ref
        out["label"] = df[labc].to_numpy() if labc else ref
        out["z"] = df[zc].to_numpy()
        out["frac_err"] = df[ec].to_numpy() / (100.0 if pct else 1.0)
        out["kind"] = "forecast"
        out["probe"] = "fsigma8"
        out["method"] = method
        out["dataset"] = name
        frames.append(out)

    sn = load_raw("fsigma8_sn_pv")
    out = _blank(len(sn))
    out["key"] = [f"fsigma8_sn_pv:{i}" for i in range(len(sn))]
    out["ref"] = sn["label"].str.strip().to_numpy()
    out["label"] = (sn["label"].str.strip() + " ["
                    + sn["zmin"].map("{:.2f}".format) + ", "
                    + sn["zmax"].map("{:.2f}".format) + "]").to_numpy()
    out["z"] = sn["zeff"].to_numpy()
    out["frac_err"] = sn["fs8_err"].to_numpy()          # already fractional
    out["kind"] = "forecast"
    out["probe"] = "fsigma8"
    out["method"] = "pv"
    out["dataset"] = "fsigma8_sn_pv"
    frames.append(out)

    return pd.concat(frames, ignore_index=True)


def load_fsigma8(kind=None, method=None, require_z=True, desi_estimators=False,
                 drop_derived=False, drop_sdss_final=True):
    """Every fsigma8 constraint in the registry, in one tidy frame.

    kind             "measurement" | "forecast" | None (both)
    method           "pv" | "rsd" | None (both)
    require_z        drop rows whose z_eff the authors do not assign. Four PV
                     rows have none (Stiskalek2026 joint fit, Stahl2021,
                     Wang2026_*) -- verified in the papers, not a gap in the
                     compilation. They have no x-axis position on an fsigma8(z)
                     plot, so they are dropped by default rather than placed at
                     a redshift nobody published.
    desi_estimators  keep the three individual DESI DR1 PV estimators alongside
                     the consensus. They are correlated (three methods on one
                     dataset); useful for comparing methods, wrong for anything
                     that treats rows as independent.
    drop_derived     drop PV rows that never quote fsigma8 (provenance='der').
                     Set this for any growth-index or modified-gravity fit.
    drop_sdss_final  drop the duplicated "SDSS final" RSD rows.
    """
    parts = []
    if kind in (None, "measurement"):
        if method in (None, "pv"):
            parts.append(_tidy_pv(desi_estimators=desi_estimators,
                                  drop_derived=drop_derived))
        if method in (None, "rsd"):
            parts.append(_tidy_rsd(drop_sdss_final=drop_sdss_final))
    if kind in (None, "forecast"):
        fc = _tidy_forecasts()
        if method is not None:
            fc = fc[fc["method"] == method]
        parts.append(fc)

    df = pd.concat(parts, ignore_index=True) if parts else _blank(0)
    for c in ("z", "value", "err_lo", "err_hi", "frac_err"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if require_z:
        df = df[df["z"].notna()]
    return df.reset_index(drop=True)


def dropped_without_z():
    """PV rows load_fsigma8 excludes for having no author-assigned z_eff."""
    df = _tidy_pv(desi_estimators=True)
    return df[pd.to_numeric(df["z"], errors="coerce").isna()].reset_index(drop=True)


# ---------------------------------------------------------------------- theory
def load_theory(name, k=0.01):
    """A COLA growth table as (z, D, f, fsigma8-shape), at wavenumber `k`.

    `k` selects the D_<k>/f_<k> column pair; available values are printed in the
    error message if you ask for one that is not tabulated. fsigma8 is returned
    only up to the sigma8 normalisation the run used -- multiply by your sigma8.
    Rows with a > 1 are dropped: the last one carries a spurious f from the
    end-of-grid derivative (see the registry caveat).
    """
    df = load_raw(name)
    kcols = sorted({float(c.split("_", 1)[1]) for c in df.columns
                    if c.startswith("D_")})
    if not any(abs(kk - k) < 1e-12 for kk in kcols):
        raise ValueError(f"k={k} not tabulated in {name!r}; available: {kcols}")
    key = next(c.split("_", 1)[1] for c in df.columns
               if c.startswith("D_") and abs(float(c.split("_", 1)[1]) - k) < 1e-12)
    df = df[df["a"] <= 1.0]
    a = df["a"].to_numpy()
    D = df[f"D_{key}"].to_numpy()
    f = df[f"f_{key}"].to_numpy()
    D = D / D[-1]                                   # normalise D(a=1) = 1
    return pd.DataFrame({"z": 1.0 / a - 1.0, "a": a, "D": D, "f": f, "fD": f * D})
