"""Dataset registry.

Every file under ``growth_review/data/`` is declared here exactly once, with the
three facts that were previously only implicit in whoever wrote the plotting
code: what KIND of thing it is (a measurement, a forecast, a theory curve), what
PROBE it constrains, and what its columns mean. Nothing is guessed from the
filename at load time.

Three kinds, and the distinction is load-bearing:

``measurement``
    A published central value with an uncertainty. Plots as a point with an
    error bar.
``forecast``
    A projected *precision* with no central value of its own. Plots as an error
    bar drawn around a fiducial prediction -- it must never be styled like a
    measurement (see the observational-cosmology skill: "Forecast numbers
    presented next to measurements without labelling" is a listed red flag).
``theory``
    A tabulated model prediction, e.g. a COLA run of a modified-gravity model.
    Plots as a curve.

Four probes: ``fsigma8``, ``bao``, ``s8``, ``sn_distance``.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

DATA_DIR = Path(__file__).parent / "data"

KINDS = ("measurement", "forecast", "theory")
PROBES = ("fsigma8", "bao", "s8", "sn_distance")


@dataclass(frozen=True)
class Dataset:
    name: str
    kind: str
    probe: str
    path: Path
    fmt: str                       # csv | ascii | ecsv
    summary: str
    source: str                    # paper / collaboration the numbers come from
    columns: str = ""              # human-readable column note where non-obvious
    method: Optional[str] = None   # pv | rsd | ... within a probe, where it matters
    caveats: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def full_path(self) -> Path:
        return DATA_DIR / self.path


def _ds(name, kind, probe, relpath, fmt, summary, source, **kw):
    return Dataset(name=name, kind=kind, probe=probe, path=Path(relpath), fmt=fmt,
                   summary=summary, source=source, **kw)


DATASETS = {d.name: d for d in [
    # ------------------------------------------------------------ measurements
    _ds("fsigma8_pv", "measurement", "fsigma8", "measurements/fsigma8_pv.csv", "csv",
        "35 published fsigma8 values from peculiar-velocity data, one row per "
        "measurement, with asymmetric errors, method family and full provenance.",
        "compiled 2026-08-13/14; every row traced to its own paper",
        method="pv",
        columns="see the file's own header block -- 18 documented columns",
        caveats=(
            "Rows are NOT independent: 6dFGSv, SDSS PV, SFI++, 2MTF and 2M++ recur "
            "across most of them, and CF3/CF4 contain several. Never average.",
            "The four DESI_DR1_* rows are three correlated estimators on one dataset "
            "plus their consensus; load_fsigma8() keeps only the consensus by default.",
            "Six rows normalise to the LINEAR sigma8 (sigma8_norm='lin'); Carrick et al. "
            "quote both for the same data and they differ by 6.5%.",
            "method_family='vd_dynamical' is declared and EMPTY: every published "
            "velocity-density comparison predicts the velocity field with linear "
            "dynamics. The particle-evolving analyses (Valade+2026, Manticore, "
            "Boruah-Lavaux-Hudson 2022, Graziani+2019) publish reconstructions, not "
            "a growth rate. See methods.EMPTY_FAMILY_EVIDENCE.",
            "'Forward model' in Boubel2024, Stiskalek2026, Boruah2020 and Stahl2021 "
            "means the DISTANCE INDICATOR, not the field dynamics -- Stiskalek's "
            "abstract says 'a linear theory reconstruction'. Do not read them as "
            "non-linear.",
            "provenance='der' rows (Carrick2015, Davis2011, Nusser2017) do not quote "
            "fsigma8 at all -- identifying their result with fsigma8 assumes f=Om^0.55, "
            "i.e. assumes GR. Exclude them from any growth-index fit.",
        )),
    _ds("fsigma8_rsd", "measurement", "fsigma8", "measurements/fsigma8_rsd.txt", "ascii",
        "Galaxy-clustering (RSD) fsigma8, 2dF through DESI DR1 full shape.",
        "individual survey papers; see the ref/label columns",
        method="rsd",
        columns="year ref label zeff fs8_value fs8_error omfid hfid ombh2fid s8 n_s with_AP",
        caveats=(
            "ref='SDSS' rows ('SDSS final') are an exact numeric duplicate of six rows "
            "already present under their original survey name. load_fsigma8() drops them.",
            "Fiducial cosmologies differ row to row (omfid/hfid columns); no AP "
            "rescaling to a common fiducial is applied.",
        )),
    _ds("fsigma8_sdss_final", "measurement", "fsigma8", "measurements/fsigma8_sdss_final.txt",
        "ascii", "fsigma8 as quoted in the SDSS final cosmology paper, with z ranges.",
        "eBOSS Collaboration 2021, arXiv:2007.08991", method="rsd",
        columns="label zmin zmax fsigma8 err",
        caveats=("Duplicates six rows of fsigma8_rsd -- do not plot both.",)),
    _ds("bao_compilation", "measurement", "bao", "measurements/bao_compilation.txt", "ascii",
        "BAO distance measurements 2009-2020, 6dFGS through eBOSS DR16 + Lya.",
        "individual survey papers; see the ref/label columns",
        columns="year ref label zeff dvrs sigdv dmrs sigdm hrs sigh omfid hfid omb2fid rsEH",
        caveats=("-1 marks a missing entry.",
                 "rsEH=1 rows used the Eisenstein & Hu r_s fitting formula, not a "
                 "Boltzmann code -- a ~1-2% offset in r_d.",
                 "Some rows are commented out with '#' in the file; the reader keeps "
                 "them commented.")),
    _ds("bao_sdss_final", "measurement", "bao", "measurements/bao_sdss_final.txt", "ascii",
        "SDSS-family consensus BAO distance ratios.",
        "eBOSS Collaboration 2021, arXiv:2007.08991",
        columns="label zeff DV_over_rd sig_DV DM_over_rd sig_DM DH_over_rd sig_DH",
        caveats=("Missing entries are 0 here and -1 in bao_compilation.",)),
    _ds("s8_weak_lensing", "measurement", "s8", "measurements/s8_weak_lensing.txt", "ascii",
        "S8 from cosmic shear: DES Y1, HSC, KiDS-450.",
        "Troxel et al. 2018; Hamana et al. 2020; Hildebrandt et al. 2017",
        columns="label zmin zmax S8 err_hi err_lo",
        caveats=("S8 = sigma8 (Om/0.3)^0.5; the 0.5 exponent is a convention, and each "
                 "survey's own best-constrained direction differs slightly.",)),
    _ds("sn_hubble_pantheon", "measurement", "sn_distance", "measurements/sn_hubble_pantheon.txt",
        "ascii", "Binned Pantheon SN Ia Hubble diagram, as a ratio to the fiducial.",
        "Scolnic et al. 2018, arXiv:1710.00845",
        columns="z n_sn dl_over_dl_fid err"),

    # --------------------------------------------------------------- forecasts
    _ds("fsigma8_euclid", "forecast", "fsigma8", "forecasts/fsigma8_euclid.txt", "ascii",
        "Euclid spectroscopic growth-rate forecast, 14 redshift bins to z=2.",
        "Amendola et al. 2016 (Euclid Theory WG), arXiv:1606.00180 Table 4",
        method="rsd",
        columns="z f error_ref error_opt error_pess -- errors are FRACTIONAL on f"),
    _ds("fsigma8_sn_pv", "forecast", "fsigma8", "forecasts/fsigma8_sn_pv.csv", "csv",
        "SN-Ia peculiar-velocity fsigma8 forecasts for ZTF and LSST.",
        "Carreres et al. 2023; Rosselli et al. 2025", method="pv",
        columns="zmin zmax zeff fs8_err label paper -- fs8_err is FRACTIONAL",
        caveats=("The LSST rows are nested redshift ranges from one survey "
                 "(0.02-0.06, 0.02-0.10, 0.02-0.14, 0.06-0.10, 0.10-0.14): the wide "
                 "bins CONTAIN the narrow ones. Plot one nesting level at a time.",)),
    _ds("fsigma8_sn_pv_howlett2017", "forecast", "fsigma8",
        "forecasts/fsigma8_sn_pv_howlett2017.txt", "ascii",
        "SN-Ia peculiar-velocity growth-rate forecast in 10 bins to z=0.5.",
        "Howlett et al. 2017, arXiv:1708.08236", method="pv",
        columns="z f error_ref error_opt error_pess -- errors are PERCENT"),
    _ds("fsigma8_4most_bgs_howlett2017", "forecast", "fsigma8",
        "forecasts/fsigma8_4most_bgs_howlett2017.txt", "ascii",
        "4MOST bright-galaxy peculiar-velocity growth-rate forecast.",
        "Howlett et al. 2017, arXiv:1708.08236", method="pv",
        columns="z f error_ref -- errors are PERCENT"),
    _ds("bao_rsd_desi", "forecast", "bao", "forecasts/bao_rsd_desi.txt", "ascii",
        "DESI design forecast, ELG+LRG+QSO: BAO distances and fsigma8 precision.",
        "DESI Collaboration 2016 Part I, arXiv:1611.00036 Table 2.3", method="rsd",
        columns="z dR dDA dH nP02 nP016 V dNelg dNlrg dNqso dfs0.1 dfs0.2 -- "
                "dfs0.1/dfs0.2 are PERCENT errors on fsigma8 for k_max = 0.1 / 0.2 h/Mpc"),
    _ds("bao_rsd_desi_bgs", "forecast", "bao", "forecasts/bao_rsd_desi_bgs.txt", "ascii",
        "DESI design forecast, BGS.",
        "DESI Collaboration 2016 Part I, arXiv:1611.00036 Table 2.5", method="rsd",
        columns="z dR dDA dH nP02 nP016 V dNbgs dfs0.1 dfs0.2 -- dfs* are PERCENT"),
    _ds("bao_lya_desi", "forecast", "bao", "forecasts/bao_lya_desi.txt", "ascii",
        "DESI design forecast, Lyman-alpha forest BAO.",
        "DESI Collaboration 2016 Part I, arXiv:1611.00036 Table 2.7",
        columns="z dR dDA dH dNqso -- distance errors are PERCENT"),

    # ------------------------------------------------------------------ theory
    _ds("cola_growth_gr", "theory", "fsigma8", "theory/cola_growth_gr.ecsv", "ecsv",
        "COLA background + scale-dependent growth, GR reference run.",
        "COLA simulation suite",
        columns="a, H/H0, then D_k and f_k at k = 1e-5 ... 10 h/Mpc",
        caveats=("The GR columns are k-independent by construction -- all eight k "
                 "values carry identical D and f. Only the f(R) file has real scale "
                 "dependence.",
                 "The last row (a=2.0) has f = -0.408, an artefact of the "
                 "end-of-grid derivative. Drop a > 1 before plotting.")),
    _ds("cola_growth_fr", "theory", "fsigma8", "theory/cola_growth_fr.ecsv", "ecsv",
        "COLA background + scale-dependent growth, f(R) run.",
        "COLA simulation suite",
        columns="a, H/H0, then D_k and f_k at k = 1e-5 ... 10 h/Mpc",
        caveats=("f is scale-dependent here: a single fsigma8(z) curve is an "
                 "effective quantity at one k, not the model's growth rate.",
                 "Drop a > 1 before plotting.")),
    _ds("cola_growth_dgp", "theory", "fsigma8", "theory/cola_growth_dgp.ecsv", "ecsv",
        "COLA background + growth, nDGP run.",
        "COLA simulation suite",
        columns="a, H/H0, then D_k and f_k at k = 1e-5 ... 10 h/Mpc",
        caveats=("nDGP growth is scale-independent in the quasi-static limit, so the "
                 "k columns are again degenerate.",
                 "Drop a > 1 before plotting.")),
]}


def list_datasets(kind=None, probe=None, method=None):
    """Registry entries matching the given filters, as a list of Dataset."""
    for arg, allowed, label in ((kind, KINDS, "kind"), (probe, PROBES, "probe")):
        if arg is not None and arg not in allowed:
            raise ValueError(f"unknown {label} {arg!r}; expected one of {allowed}")
    return [d for d in DATASETS.values()
            if (kind is None or d.kind == kind)
            and (probe is None or d.probe == probe)
            and (method is None or d.method == method)]


def get(name) -> Dataset:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    return DATASETS[name]


def summary_table():
    """One text line per dataset -- what `python -m growth_review` prints."""
    rows = [f"{'name':<32} {'kind':<12} {'probe':<12} {'method':<7} summary",
            "-" * 110]
    for kind in KINDS:
        for d in sorted(list_datasets(kind=kind), key=lambda x: (x.probe, x.name)):
            rows.append(f"{d.name:<32} {d.kind:<12} {d.probe:<12} "
                        f"{(d.method or '-'):<7} {d.summary}")
    return "\n".join(rows)
