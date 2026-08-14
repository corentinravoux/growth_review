"""Integrity checks on the shipped data and the loaders over it.

These exist because two silent corruptions got through during the compilation of
the peculiar-velocity file: a data row was blanked by a header-only edit and went
unnoticed for two commits, and an append landed 16 values in 17 columns. Both
would have been caught here.
"""
import numpy as np
import pandas as pd
import pytest

import growth_review as gr
from growth_review import datasets as ds


# ------------------------------------------------------------------- registry
def test_every_registered_file_exists():
    missing = [d.name for d in ds.DATASETS.values() if not d.full_path.exists()]
    assert not missing, f"registered but absent: {missing}"


def test_every_shipped_file_is_registered():
    on_disk = {p.relative_to(ds.DATA_DIR).as_posix()
               for p in ds.DATA_DIR.rglob("*")
               if p.is_file() and p.name != "README.md"}
    registered = {d.path.as_posix() for d in ds.DATASETS.values()}
    assert on_disk == registered, f"unregistered: {on_disk - registered}"


def test_kinds_and_probes_are_from_the_controlled_vocabulary():
    for d in ds.DATASETS.values():
        assert d.kind in ds.KINDS
        assert d.probe in ds.PROBES


@pytest.mark.parametrize("name", sorted(ds.DATASETS))
def test_every_dataset_loads_non_empty(name):
    df = gr.load_raw(name)
    assert len(df) > 0


# ------------------------------------------------- the PV compilation itself
@pytest.fixture(scope="module")
def pv():
    return gr.load_raw("fsigma8_pv")


def test_pv_row_count_and_unique_keys(pv):
    assert len(pv) == 35
    assert pv["key"].is_unique


def test_pv_has_no_blank_rows(pv):
    # the exact failure mode that went unnoticed for two commits
    required = ["key", "authors", "year", "fsigma8", "err_lo", "err_hi",
                "probe", "indicator", "dataset", "estimator", "provenance",
                "method_family", "fit_technique"]
    for col in required:
        blank = pv[pv[col].isna() | (pv[col].astype(str).str.strip() == "")]
        assert not len(blank), f"{col} blank in rows: {list(blank['key'])}"


def test_pv_values_are_physical(pv):
    assert pv["fsigma8"].between(0.1, 1.0).all()
    for col in ("err_lo", "err_hi"):
        assert (pv[col] > 0).all()
    z = pd.to_numeric(pv["z_eff"], errors="coerce")
    assert z.dropna().between(0.0, 0.2).all()


def test_pv_controlled_vocabularies(pv):
    assert set(pv["method_family"]) <= {"field_level", "two_point",
                                        "vd_linear", "vd_dynamical", "consensus"}
    assert set(pv["provenance"]) <= {"src", "der", "T24"}
    assert set(pv["sigma8_norm"]) <= {"lin", "unstated"}
    assert set(pv["probe"]) <= {"PV", "PV+RSD"}


def test_exactly_one_consensus_row(pv):
    assert (pv["method_family"] == "consensus").sum() == 1


def test_every_pv_row_has_recorded_evidence(pv):
    from growth_review.methods import EVIDENCE
    missing = sorted(set(pv["key"]) - set(EVIDENCE))
    assert not missing, f"family assigned with no source quote: {missing}"


def test_no_published_fsigma8_uses_a_particle_forward_model(pv):
    """vd_dynamical is declared and empty, and that is a result.

    Every velocity-density comparison in the literature predicts the velocity
    field with linear dynamics; the analyses that evolve a particle field
    (Valade+2026, Manticore, Boruah-Lavaux-Hudson 2022, Graziani+2019) publish
    reconstructions and evidences, not a growth rate. If this test ever fails,
    a genuinely new kind of measurement has appeared -- update the skill
    reference and the README rather than deleting the test.
    """
    from growth_review.methods import EMPTY_FAMILY_EVIDENCE
    assert (pv["method_family"] == "vd_dynamical").sum() == 0
    assert len(EMPTY_FAMILY_EVIDENCE["vd_dynamical"]) >= 4


def test_forward_model_labels_that_mean_the_indicator_stay_linear(pv):
    """"Forward model" in these four papers means the distance indicator."""
    from growth_review.methods import FORWARD_MODEL_IS_THE_INDICATOR
    fam = pv.set_index("key")["method_family"]
    for key in FORWARD_MODEL_IS_THE_INDICATOR:
        assert fam[key] == "vd_linear", key


def test_derived_rows_are_the_documented_three(pv):
    # Warning 7 of the file header: these assume f = Om^0.55 and are circular
    # in a test of gravity. If the set changes, the warning must change too.
    assert set(pv.loc[pv["provenance"] == "der", "key"]) == {
        "Carrick2015", "Davis2011", "Nusser2017"}


def test_linear_sigma8_rows_are_the_documented_six(pv):
    assert set(pv.loc[pv["sigma8_norm"] == "lin", "key"]) == {
        "Carrick2015", "LilowNusser2021", "HollingerHudson2024", "Boruah2020",
        "Stahl2021", "Stiskalek2026"}


# ------------------------------------------------------------- the tidy view
def test_tidy_schema_and_types():
    df = gr.load_fsigma8()
    assert list(df.columns) == gr.io.TIDY_COLUMNS
    for c in ("z", "value", "err_lo", "err_hi", "frac_err"):
        assert np.issubdtype(df[c].dtype, np.floating)


def test_measurements_have_values_and_forecasts_have_precisions():
    df = gr.load_fsigma8()
    meas = df[df["kind"] == "measurement"]
    fc = df[df["kind"] == "forecast"]
    assert meas["value"].notna().all() and meas["frac_err"].isna().all()
    assert fc["frac_err"].notna().all() and fc["value"].isna().all()
    assert (fc["frac_err"] > 0).all() and (fc["frac_err"] < 1).all()


def test_desi_individual_estimators_are_excluded_by_default():
    keys = set(gr.load_fsigma8(kind="measurement", method="pv")["key"])
    assert "DESI_DR1_consensus" in keys
    assert not keys & {"DESI_DR1_maxlike", "DESI_DR1_corrfunc",
                       "DESI_DR1_powerspec"}
    with_all = set(gr.load_fsigma8(kind="measurement", method="pv",
                                   desi_estimators=True)["key"])
    assert len(with_all) == len(keys) + 3


def test_sdss_final_duplicates_are_dropped_by_default():
    df = gr.load_fsigma8(kind="measurement", method="rsd")
    assert "SDSS" not in set(df["ref"])
    assert len(df) == 25


def test_rows_without_z_are_the_documented_four():
    assert set(gr.io.dropped_without_z()["key"]) == {
        "Stahl2021", "Stiskalek2026", "Wang2026_group", "Wang2026_local"}


def test_systematics_are_added_in_quadrature():
    raw = gr.load_raw("fsigma8_pv").set_index("key")
    tidy = gr.load_fsigma8(kind="measurement", method="pv",
                           desi_estimators=True).set_index("key")
    row = raw.loc["AdamsBlake2020"]          # 0.052 stat, 0.061 sys
    expected = np.hypot(row["err_lo"], row["err_sys"])
    assert tidy.loc["AdamsBlake2020", "err_lo"] == pytest.approx(expected)


def test_drop_derived_removes_exactly_the_circular_rows():
    keep = set(gr.load_fsigma8(kind="measurement", method="pv")["key"])
    dropped = keep - set(gr.load_fsigma8(kind="measurement", method="pv",
                                         drop_derived=True)["key"])
    assert dropped == {"Carrick2015", "Davis2011", "Nusser2017"}


# --------------------------------------------------------------- cosmology
def test_growth_rate_matches_the_gr_growth_index():
    c = gr.FlatLCDM()
    for z in (0.0, 0.5, 1.0, 2.0):
        assert c.growth_rate(z) == pytest.approx(c.omega_m_z(z) ** 0.55, rel=0.01)


def test_growth_factor_normalisation():
    c = gr.FlatLCDM()
    assert c.growth_factor(0.0) == pytest.approx(1.0, abs=1e-6)
    assert c.growth_factor(1.0) < c.growth_factor(0.0)


def test_no_division_by_zero_in_growth_rate():
    c = gr.FlatLCDM()
    assert np.isfinite(c._f).all()


def test_cola_gr_run_reproduces_the_analytic_growth():
    c = gr.FlatLCDM()
    t = gr.load_theory("cola_growth_gr", k=0.01)
    t = t[(t["z"] > 0.05) & (t["z"] < 1.5)]
    assert np.allclose(t["D"], c.growth_factor(t["z"].to_numpy()), rtol=0.02)


def test_load_theory_rejects_an_untabulated_wavenumber():
    with pytest.raises(ValueError, match="not tabulated"):
        gr.load_theory("cola_growth_gr", k=0.037)


# --------------------------------------------------------------- citations
# The package hardcodes no BibTeX keys, so these test the plumbing with a
# synthetic mapping rather than a real bibliography.
def _fake_bibkey(keys):
    return {k: f"ref_{k.lower()}" for k in keys}


def test_captions_carry_no_citations_without_a_mapping():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for fn in (gr.figures.fig_pv, gr.figures.fig_rsd, gr.figures.fig_overview):
        fig, meta = fn()
        assert r"\cite{" not in meta["caption"], fn.__name__
        assert not meta["missing_citations"]
        plt.close(fig)


def test_a_full_mapping_cites_every_plotted_point():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for fn in (gr.figures.fig_pv, gr.figures.fig_rsd, gr.figures.fig_overview):
        fig, meta = fn()
        bib = _fake_bibkey(meta["plotted"]["key"])
        plt.close(fig)
        fig, meta = fn(bibkey=bib)
        assert r"\cite{" in meta["caption"], fn.__name__
        assert not meta["missing_citations"], (fn.__name__, meta["missing_citations"])
        assert "[NO CITATION]" not in meta["caption"]
        plt.close(fig)


def test_a_partial_mapping_reports_the_uncitable_rows():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, meta = gr.figures.fig_pv()
    keys = list(meta["plotted"]["key"])
    plt.close(fig)
    fig, meta = gr.figures.fig_pv(bibkey=_fake_bibkey(keys[:3]))
    assert set(meta["missing_citations"]) == set(keys[3:])
    plt.close(fig)


def test_cited_only_drops_exactly_the_unmapped_rows():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, meta = gr.figures.fig_pv()
    keys = list(meta["plotted"]["key"])
    plt.close(fig)
    fig, meta = gr.figures.fig_pv(bibkey=_fake_bibkey(keys[:3]), cited_only=True)
    assert set(meta["plotted"]["key"]) == set(keys[:3])
    assert set(meta["dropped"]) == set(keys[3:])
    assert not meta["missing_citations"]
    plt.close(fig)


def test_cited_only_without_a_mapping_is_an_error():
    with pytest.raises(ValueError, match="needs a bibkey mapping"):
        gr.figures.fig_pv(cited_only=True)


# ------------------------------------------------------------------ plotting
@pytest.mark.parametrize("scale", ["linear", "symlog", "log1p", "log"])
def test_dodge_preserves_ordering_and_never_moves_a_point_far(scale):
    z = np.array([0.0, 0.0, 0.02, 0.021, 0.5, 0.51, 1.5])
    x = gr.dodge_x(z, min_sep=0.006, step=0.014, scale=scale)
    assert len(x) == len(z)
    assert np.all(np.abs(x - z) < 0.35 * (1 + z))


@pytest.mark.parametrize("name", sorted(gr.figures.FIGURES))
def test_every_figure_builds(name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, meta = gr.figures.FIGURES[name]()
    assert "caption" in meta and meta["caption"].startswith(r"\caption{")
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_unknown_scale_is_rejected():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="scale must be one of"):
        gr.style.set_z_scale(ax, "loglog")
    plt.close(fig)
