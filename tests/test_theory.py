"""Tests for the theory layer.

What this layer computes is the fiducial reference curve, and the tests check it
against something that is not itself: the closed-form LCDM growth integral and
the Omega_m^0.55 limit. Modified gravity is not computed here -- it arrives as an
EFTCAMB export or a simulation table -- so the rest of these tests are about the
table path, the registry, and the EFTCAMB module's behaviour when no build exists.
"""
import numpy as np
import pytest

import growth_review as gr
from growth_review import theory as th


# ---------------------------------------------------------------- background
def test_lcdm_background_reduces_to_the_textbook_expression():
    bg = th.Background()
    z = np.array([0.0, 0.5, 1.0, 3.0])
    expected = np.sqrt(bg.omega_m * (1 + z) ** 3 + (1 - bg.omega_m))
    assert np.allclose(bg.E(z), expected)
    assert bg.E(0.0) == pytest.approx(1.0)


def test_dlnE_dlna_matches_a_finite_difference():
    # The analytic derivative is the friction term of the growth ODE; a sign or
    # factor error there is invisible in E(a) itself.
    for bg in (th.Background(), th.Background(w0=-0.733, wa=-1.010)):
        a = np.array([0.2, 0.5, 0.8, 1.0])
        eps = 1e-5
        numeric = (np.log(bg.E_a(a * np.exp(eps)))
                   - np.log(bg.E_a(a * np.exp(-eps)))) / (2 * eps)
        assert np.allclose(bg.dlnE_dlna(a), numeric, rtol=1e-6)


def test_cpl_density_is_lcdm_when_w_is_minus_one():
    bg = th.Background()
    assert np.allclose(bg.de_density(np.array([0.1, 0.5, 1.0])), 1.0)


# -------------------------------------------------------------- GR growth
def test_ode_growth_matches_the_closed_form_lcdm_integral():
    """D(a) ~ H(a) int_0^a da'/(a'H)^3, evaluated independently of the solver."""
    bg = th.Background()
    a = 10.0 ** np.linspace(-6, 0, 200000)
    integrand = 1.0 / (a * bg.E_a(a)) ** 3
    d = np.zeros_like(a)
    d[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(a))
    d *= 2.5 * bg.omega_m * bg.E_a(a)
    d /= d[-1]

    z = np.linspace(0.0, 3.0, 31)
    ode = th.fiducial().growth_factor(z)
    exact = np.interp(1.0 / (1.0 + z), a, d)
    assert np.abs(ode / exact - 1.0).max() < 1e-4


def test_growth_rate_matches_the_gr_growth_index():
    c = th.fiducial()
    for z in (0.0, 0.5, 1.0, 2.0):
        assert c.growth_rate(z) == pytest.approx(c.omega_m_z(z) ** 0.55, rel=0.01)


def test_growth_factor_normalisation():
    c = th.fiducial()
    assert c.growth_factor(0.0) == pytest.approx(1.0, abs=1e-6)
    assert c.growth_factor(1.0) < c.growth_factor(0.0)


def test_growth_is_finite_everywhere_including_the_first_grid_point():
    sol = th.solve_growth(th.Background())
    assert np.isfinite(sol.D).all() and np.isfinite(sol.f).all()
    # deep in matter domination the growing mode is D ~ a, f -> 1
    assert sol.f[0] == pytest.approx(1.0, abs=0.02)


def test_fiducial_sigma8_today_is_the_input_and_fsigma8_is_f_times_sigma8():
    c = th.fiducial()
    assert c.sigma8_z(0.0) == pytest.approx(th.PLANCK18["sigma8"], rel=1e-9)
    assert c.fsigma8(0.0) == pytest.approx(c.growth_rate(0.0) * c.sigma8_z(0.0),
                                           rel=1e-4)


def test_a_w0wa_background_changes_growth_and_stays_gr():
    z = np.linspace(0.0, 2.0, 11)
    m = th.Cosmology(w0=-0.733, wa=-1.010, name="w0wa", label="w0wa")
    assert not np.allclose(m.E(z), th.fiducial().E(z))
    assert not np.allclose(m.fsigma8(z), th.fiducial().fsigma8(z))
    # mu = 1 everywhere in this layer, so f stays near Om(z)^0.55 on its own
    # background
    assert m.growth_rate(0.5) == pytest.approx(m.omega_m_z(0.5) ** 0.55, rel=0.02)


def test_growth_index_reproduces_gr_at_gamma_0p55():
    z = np.linspace(0.0, 2.0, 21)
    fs8 = th.growth_index(0.55).fsigma8(z)
    assert np.allclose(fs8, th.fiducial().fsigma8(z), rtol=0.01)
    # a larger gamma suppresses growth
    assert np.all(th.growth_index(0.68).fsigma8(z) < fs8)


def test_fsigma8_gamma_on_the_fiducial_matches_the_standalone_gamma_model():
    z = np.linspace(0.0, 1.5, 16)
    assert np.allclose(th.fiducial().fsigma8_gamma(z, 0.68),
                       th.growth_index(0.68).fsigma8(z), rtol=1e-12)


def test_norm_today_shares_sigma8_and_norm_early_does_not():
    bg = th.Background(w0=-0.733, wa=-1.010)
    early = th.GrowthModel(background=bg, norm="early")
    today = th.GrowthModel(background=bg, norm="today")
    assert today.sigma8_z(0.0) == pytest.approx(th.PLANCK18["sigma8"], rel=1e-9)
    assert early.sigma8_z(0.0) != pytest.approx(today.sigma8_z(0.0), rel=1e-6)
    # the shapes are the same curve up to that constant
    z = np.linspace(0.0, 2.0, 21)
    ratio = early.fsigma8(z) / today.fsigma8(z)
    assert np.allclose(ratio, ratio[0], rtol=1e-10)


def test_an_unknown_norm_is_rejected():
    with pytest.raises(ValueError, match="norm must be one of"):
        th.GrowthModel(norm="matter_domination")


# ---------------------------------------------------------------- registry
def test_no_modified_gravity_model_is_computed_in_this_package():
    """The registry must not ship approximations of the EFTCAMB models.

    This is the test that pins the decision: MG curves come from EFTCAMB (or a
    simulation table), never from a mu(k,a) written here. If a future change adds
    an analytic nDGP/f(R)/Horndeski model back, this fails and says why.
    """
    families = set(th.families())
    assert families <= {"GR", "growth index", "simulation", "eftcamb"}, families
    assert not hasattr(th, "mg"), "the analytic mu(k,a) layer was removed on purpose"
    assert not hasattr(th, "ndgp") and not hasattr(th, "hu_sawicki")


def test_every_registered_model_returns_a_finite_curve():
    z = np.linspace(0.0, 2.0, 25)
    for name in th.list_models():
        fs8 = th.fsigma8(name, z)
        assert fs8.shape == z.shape, name
        assert np.isfinite(fs8).all(), name
        assert np.all(fs8 > 0), name


def test_registered_models_carry_a_label_a_style_and_their_caveats():
    for name in th.list_models():
        m = th.get(name)
        assert m.label and m.style.color
        if m.family in ("growth index", "simulation", "eftcamb"):
            assert m.caveats, f"{name} states no caveat"


def test_get_accepts_a_model_object_and_points_at_eftcamb_for_the_rest():
    m = th.growth_index(0.6)
    assert th.get(m) is m
    with pytest.raises(KeyError, match="growth-review-eftcamb-export"):
        th.get("nDGP_H0rc1")


def test_registering_the_same_name_twice_needs_replace():
    th.register("test_tmp_model", lambda: th.growth_index(0.6))
    try:
        with pytest.raises(KeyError, match="already registered"):
            th.register("test_tmp_model", lambda: th.growth_index(0.6))
        th.register("test_tmp_model", lambda: th.growth_index(0.7), replace=True)
        assert th.get("test_tmp_model").gamma == 0.7
    finally:
        th.registry._FACTORIES.pop("test_tmp_model", None)
        th.registry._CACHE.pop("test_tmp_model", None)


def test_ratio_is_a_percentage_deviation_by_default():
    z, dev = th.ratio("gamma_0.68", "GR", z=np.array([0.0, 1.0]))
    assert np.all(dev < 0)                       # a larger gamma suppresses growth
    _, r = th.ratio("gamma_0.68", "GR", z=np.array([0.0]), percent=False)
    assert r[0] == pytest.approx(1.0 + dev[0] / 100.0, rel=1e-12)


def test_latex_sci_renders_powers_of_ten_without_an_e():
    assert th.latex_sci(1e-5) == "10^{-5}"
    assert th.latex_sci(2e-5) == r"2.0\times10^{-5}"


def test_eftcamb_styles_cover_the_models_the_notebook_plots():
    """Every EFTCAMB model has a colour/linestyle, so an export is drawn the way
    the source notebook draws it."""
    from growth_review.theory import eftcamb
    for name in eftcamb.MODELS:
        if name.startswith("fR_B0"):             # B0-parametrised variants
            continue
        label, color, ls, lw = th.export_style(name)
        assert label and color and ls, name


# ------------------------------------------------------------------- tables
def test_table_models_read_the_shipped_cola_runs():
    z = np.linspace(0.0, 1.5, 16)
    for name in ("cola_gr", "cola_fr", "cola_dgp"):
        m = th.get(name)
        assert np.isfinite(m.fsigma8(z)).all()
        assert m.sigma8_z(0.0) == pytest.approx(th.PLANCK18["sigma8"], rel=1e-6)


def test_export_round_trip_and_registration(tmp_path):
    """The path every EFTCAMB curve takes: written on one machine, registered as
    an ordinary model on another."""
    z = np.linspace(0.0, 2.0, 21)
    ref = th.fiducial()
    th.write_export(tmp_path / "eftcamb_Kmouflage.ecsv", z,
                    ref.sigma8_z(z), ref.fsigma8(z), E=ref.E(z),
                    meta=dict(model="Kmouflage", label="K-mouflage"))
    names = th.register_exports(directory=tmp_path)
    try:
        assert names == ["eftcamb_Kmouflage"]
        m = th.get("eftcamb_Kmouflage")
        assert m.label == "K-mouflage"
        assert m.style.color == "#008d00"        # the source notebook's colour
        assert np.allclose(m.fsigma8(z), ref.fsigma8(z))
        assert np.allclose(m.E(z), ref.E(z))
        assert th.read_meta(tmp_path / "eftcamb_Kmouflage.ecsv")["model"] == "Kmouflage"
    finally:
        for n in names:
            th.registry._FACTORIES.pop(n, None)
            th.registry._CACHE.pop(n, None)


def test_export_table_missing_a_column_is_reported(tmp_path):
    from astropy.table import Table
    path = tmp_path / "eftcamb_broken.ecsv"
    Table({"z": [0.0, 1.0], "sigma8": [0.8, 0.5]}).write(
        path, format="ascii.ecsv", overwrite=True)
    with pytest.raises(ValueError, match="missing export columns"):
        th.load_export(path)


# ------------------------------------------------------------------ plotting
def test_plot_theory_draws_one_line_per_model_in_its_registered_colour():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    names = ["GR", "gamma_0.68", "cola_fr"]
    lines = gr.plotting.plot_theory(ax, names)
    assert len(lines) == 3
    assert [line.get_color() for line in lines] == [th.get(n).style.color
                                                    for n in names]
    plt.close(fig)


def test_plot_theory_accepts_a_raw_dataset_name_at_a_chosen_wavenumber():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    lines = gr.plotting.plot_theory(ax, "cola_growth_fr", k=1.0,
                                    sigma8=0.8111, label="table")
    assert len(lines) == 1 and lines[0].get_label() == "table"
    plt.close(fig)


def test_plot_forecasts_honours_an_abscissa_override():
    """Several survey configurations forecast the same bins; without a dodge
    their bars are drawn on top of each other."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fc = gr.load_fsigma8(kind="forecast")
    sub = fc[fc["ref"] == "ZTF"]
    fig, ax = plt.subplots()
    shifted = sub["z"].to_numpy() * 1.1
    container = gr.plotting.plot_forecasts(ax, sub, th.fiducial(), x=shifted)
    assert np.allclose(container[0].get_xdata(), shifted)
    plain = gr.plotting.plot_forecasts(ax, sub, th.fiducial())
    assert np.allclose(container.lines[2][0].get_segments()[0][:, 1],
                       plain.lines[2][0].get_segments()[0][:, 1])
    plt.close(fig)


def test_plot_theory_ratio_skips_the_reference_and_zeroes_on_it():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    lines = gr.plotting.plot_theory_ratio(ax, ["GR", "gamma_0.68"], reference="GR")
    assert len(lines) == 1
    assert np.all(lines[0].get_ydata() < 0)
    plt.close(fig)


# ---------------------------------------------------------- EFTCAMB backend
def test_the_eftcamb_backend_imports_without_a_build_and_says_what_is_missing():
    """Importing the module must not require camb; only *running* it does.

    This is the whole local test coverage of that module: everything that touches
    camb is untested here, because there is no build on this machine.
    """
    from growth_review.theory import eftcamb
    assert "GR" in eftcamb.MODELS and "JBD_wBD100" in eftcamb.MODELS
    if not eftcamb.available():
        with pytest.raises(RuntimeError, match="EFTCAMB"):
            eftcamb.compute("Kmouflage")


class _StubResults:
    """Minimal stand-in for a CAMB results object.

    CAMB returns the transfer-function output from the HIGHEST redshift down,
    whatever order the redshifts were requested in, and the source notebook flags
    that zipping it against the input array silently mirrors the curve. That
    ordering logic is real code in `compute()` and can be tested without a build,
    which is what this stub is for -- it carries no physics.
    """

    def __init__(self, zs):
        self.zs = np.sort(np.asarray(zs, dtype=float))[::-1]     # descending

    def get_fsigma8(self):
        return 0.4 + 0.1 * self.zs          # increases with z, so mirroring shows

    def get_sigma8(self):
        return 0.8 / (1.0 + self.zs)

    def hubble_parameter(self, z):
        return 67.36 * np.sqrt(0.3138 * (1 + z) ** 3 + 0.6862)


class _StubParams(dict):
    """set_params' return value: a dict that also answers .EFTCAMB.model_name()."""

    class _EFTCAMB:
        def model_name(self):
            return "stub EFTCAMB model"

    EFTCAMB = _EFTCAMB()


class _StubCamb:
    def set_params(self, **kw):
        self.kw = kw
        return _StubParams(kw)

    def get_results(self, pars):
        return _StubResults(pars["redshifts"])

    def get_background(self, pars, no_thermo=False):
        return _StubResults([0.0])


def test_compute_undoes_cambs_redshift_ordering(monkeypatch):
    from growth_review.theory import eftcamb

    monkeypatch.setattr(eftcamb, "_camb", _StubCamb())
    monkeypatch.setattr(eftcamb, "_has_eftcamb", True)
    monkeypatch.setattr(eftcamb, "_CACHE", {})

    zs = np.array([0.0, 0.5, 1.0, 2.0])
    z, fs8, s8 = eftcamb.compute("GR", zs)
    assert np.allclose(z, zs)                       # ascending, as requested
    assert np.allclose(fs8, 0.4 + 0.1 * zs)         # paired with the right z
    assert np.all(np.diff(s8) < 0)                  # sigma8 falls with z


def test_export_writes_the_flags_and_cosmology_into_the_header(monkeypatch, tmp_path):
    from growth_review.theory import eftcamb

    monkeypatch.setattr(eftcamb, "_camb", _StubCamb())
    monkeypatch.setattr(eftcamb, "_has_eftcamb", True)
    monkeypatch.setattr(eftcamb, "_CACHE", {})

    path = eftcamb.export("Kmouflage", tmp_path / "eftcamb_Kmouflage.ecsv",
                          np.linspace(0, 2, 5))
    meta = th.read_meta(path)
    assert meta["model"] == "Kmouflage"
    assert meta["params"]["FullMappingEFTmodel"] == 3
    assert meta["cosmology"]["H0"] == eftcamb.COSMO["H0"]
    assert "shared A_s" in meta["sigma8_norm"]
    # and it comes back as a model, styled as the source notebook styles it
    names = th.register_exports(directory=tmp_path)
    try:
        assert th.get(names[0]).label == "K-mouflage"
    finally:
        for n in names:
            th.registry._FACTORIES.pop(n, None)
            th.registry._CACHE.pop(n, None)


def test_ndgp_enhances_growth_and_shares_the_lcdm_background(monkeypatch):
    """The nDGP sentinel path: its own growth ODE, GR's sigma8 as the anchor.

    Transcribed from the source notebook, so what is checked here is the
    transcription: mu > 1 enhances growth, a smaller crossover scale enhances it
    more, the enhancement fades towards matter domination, and the expansion
    history is GR's exactly. The GR anchor comes from the stub, so the absolute
    numbers are the stub's -- the ratios are the ODE's.
    """
    pytest.importorskip("scipy")
    from growth_review.theory import eftcamb

    monkeypatch.setattr(eftcamb, "_camb", _StubCamb())
    monkeypatch.setattr(eftcamb, "_has_eftcamb", True)
    monkeypatch.setattr(eftcamb, "_CACHE", {})

    zs = np.linspace(0.0, 2.0, 21)
    _, _, s8_gr = eftcamb.compute("GR", zs)
    _, fs8_n1, s8_n1 = eftcamb.compute("nDGP_H0rc1", zs)
    _, fs8_n5, s8_n5 = eftcamb.compute("nDGP_H0rc5", zs)

    # The stub's GR fsigma8 is arbitrary, so the comparisons are on the ratios
    # the ODE owns: sigma8 is the GR anchor times D_nDGP/D_GR, and f comes from
    # the nDGP solution itself.
    boost_n1, boost_n5 = s8_n1 / s8_gr, s8_n5 / s8_gr
    assert np.all(boost_n5 > 1.0)                        # mu > 1 enhances growth
    assert np.all(boost_n1 > boost_n5)                   # smaller r_c, stronger
    assert boost_n1[0] > boost_n1[-1]                    # fades towards high z

    f_n1 = fs8_n1 / s8_n1
    f_gr = np.array([eftcamb._ndgp_growth_ode(None).sol(np.log(1 / (1 + z)))[1]
                     / eftcamb._ndgp_growth_ode(None).sol(np.log(1 / (1 + z)))[0]
                     for z in zs])
    assert np.all(f_n1 > f_gr)
    assert np.all(np.isfinite(s8_n1)) and np.all(s8_n1 > 0)

    # the normal branch shares LCDM's expansion history exactly
    z_gr, E_gr = eftcamb.background_ez("GR", zs)
    z_n1, E_n1 = eftcamb.background_ez("nDGP_H0rc1", zs)
    assert np.allclose(E_n1, E_gr)


def test_ndgp_mu_matches_the_published_formula(monkeypatch):
    """The mu the solved ODE actually contains is the published one.

    Recovered from the solution rather than from the rhs it was built with:
    mu = (D'' + (2 + dlnH/dN) D') / (1.5 Omega_m(a) D) at a = 1, with D'' from a
    finite difference of the dense output. Compared against
    mu = 1 + 1/(3 beta), beta = 1 + 2 H r_c (1 + dlnH/dlna / 3) written out here.
    """
    pytest.importorskip("scipy")
    from growth_review.theory import eftcamb

    h = eftcamb.COSMO["H0"] / 100.0
    om0 = (eftcamb.COSMO["ombh2"] + eftcamb.COSMO["omch2"]) / h ** 2
    dlnH_dN = -1.5 * om0                      # at a = 1, where E = 1

    for H0rc in (1.0, 5.0):
        beta = 1.0 + 2.0 * H0rc * (1.0 + dlnH_dN / 3.0)
        expected = 1.0 + 1.0 / (3.0 * beta)

        sol = eftcamb._ndgp_growth_ode(H0rc)
        eps = 1e-5
        D, Dp = sol.sol(0.0)
        _, Dp_plus = sol.sol(eps)
        _, Dp_minus = sol.sol(-eps)
        Dpp = (Dp_plus - Dp_minus) / (2 * eps)
        mu = (Dpp + (2.0 + dlnH_dN) * Dp) / (1.5 * om0 * D)

        assert mu == pytest.approx(expected, rel=1e-5)
        assert 1.0 < expected < 1.2            # a percent-level fifth force

    # and the mu = 1 baseline really is GR
    sol_gr = eftcamb._ndgp_growth_ode(None)
    D, Dp = sol_gr.sol(0.0)
    assert Dp / D == pytest.approx(om0 ** 0.55, rel=0.02)


def test_ndgp_growth_rate_kz_is_scale_independent(monkeypatch):
    """f(k,z) for nDGP must be flat in k -- the structural contrast with f(R)."""
    pytest.importorskip("scipy")
    from growth_review.theory import eftcamb

    monkeypatch.setattr(eftcamb, "_camb", _StubCamb())
    monkeypatch.setattr(eftcamb, "_has_eftcamb", True)
    monkeypatch.setattr(eftcamb, "_CACHE", {})

    k, z, f_kz = eftcamb.growth_rate_kz("nDGP_H0rc1", np.linspace(0, 2, 11))
    assert f_kz.shape == (z.size, k.size)
    assert np.allclose(f_kz, f_kz[:, :1])        # identical across wavenumbers


def test_ndgp_model_name_says_it_is_not_an_eftcamb_run():
    from growth_review.theory import eftcamb
    name = eftcamb.model_name("nDGP_H0rc5")
    assert "not an EFTCAMB run" in name and "5.0" in name


def test_the_local_bisection_finds_a_root():
    from growth_review.theory import eftcamb
    root = eftcamb.bisect(lambda x: x ** 3 - 2.0, 0.0, 4.0, xtol=1e-12)
    assert root == pytest.approx(2.0 ** (1 / 3), rel=1e-9)
    with pytest.raises(ValueError, match="same"):
        eftcamb.bisect(lambda x: x ** 2 + 1.0, -1.0, 1.0)
