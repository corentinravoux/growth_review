# Data manifest

Generated from `growth_review/datasets.py` -- that module is the source of
truth, this file is the readable rendering of it. Every file below was
copied verbatim from `Science/Peculiar_Vel/data_plots/data_compilation/`
(and `old/`) on 2026-08-14; only comment headers were added, and only to
files that had none. No numeric value was edited.

The one exception is `measurements/fsigma8_pv.csv`, which gained two
columns (`method_family`, `fit_technique`); the other 16 columns were
verified identical to the source, row by row, after the edit.


## measurements

### `measurements/bao_compilation.txt`

- **name** `bao_compilation`  |  **probe** `bao`
- BAO distance measurements 2009-2020, 6dFGS through eBOSS DR16 + Lya.
- **source**: individual survey papers; see the ref/label columns
- **columns**: year ref label zeff dvrs sigdv dmrs sigdm hrs sigh omfid hfid omb2fid rsEH
- :warning: -1 marks a missing entry.
- :warning: rsEH=1 rows used the Eisenstein & Hu r_s fitting formula, not a Boltzmann code -- a ~1-2% offset in r_d.
- :warning: Some rows are commented out with '#' in the file; the reader keeps them commented.

### `measurements/bao_sdss_final.txt`

- **name** `bao_sdss_final`  |  **probe** `bao`
- SDSS-family consensus BAO distance ratios.
- **source**: eBOSS Collaboration 2021, arXiv:2007.08991
- **columns**: label zeff DV_over_rd sig_DV DM_over_rd sig_DM DH_over_rd sig_DH
- :warning: Missing entries are 0 here and -1 in bao_compilation.

### `measurements/fsigma8_pv.csv`

- **name** `fsigma8_pv`  |  **probe** `fsigma8`  |  **method** `pv`
- 35 published fsigma8 values from peculiar-velocity data, one row per measurement, with asymmetric errors, method family and full provenance.
- **source**: compiled 2026-08-13/14; every row traced to its own paper
- **columns**: see the file's own header block -- 18 documented columns
- :warning: Rows are NOT independent: 6dFGSv, SDSS PV, SFI++, 2MTF and 2M++ recur across most of them, and CF3/CF4 contain several. Never average.
- :warning: The four DESI_DR1_* rows are three correlated estimators on one dataset plus their consensus; load_fsigma8() keeps only the consensus by default.
- :warning: Six rows normalise to the LINEAR sigma8 (sigma8_norm='lin'); Carrick et al. quote both for the same data and they differ by 6.5%.
- :warning: method_family='vd_dynamical' is declared and EMPTY: every published velocity-density comparison predicts the velocity field with linear dynamics. The particle-evolving analyses (Valade+2026, Manticore, Boruah-Lavaux-Hudson 2022, Graziani+2019) publish reconstructions, not a growth rate. See methods.EMPTY_FAMILY_EVIDENCE.
- :warning: 'Forward model' in Boubel2024, Stiskalek2026, Boruah2020 and Stahl2021 means the DISTANCE INDICATOR, not the field dynamics -- Stiskalek's abstract says 'a linear theory reconstruction'. Do not read them as non-linear.
- :warning: provenance='der' rows (Carrick2015, Davis2011, Nusser2017) do not quote fsigma8 at all -- identifying their result with fsigma8 assumes f=Om^0.55, i.e. assumes GR. Exclude them from any growth-index fit.

### `measurements/fsigma8_rsd.txt`

- **name** `fsigma8_rsd`  |  **probe** `fsigma8`  |  **method** `rsd`
- Galaxy-clustering (RSD) fsigma8, 2dF through DESI DR1 full shape.
- **source**: individual survey papers; see the ref/label columns
- **columns**: year ref label zeff fs8_value fs8_error omfid hfid ombh2fid s8 n_s with_AP
- :warning: ref='SDSS' rows ('SDSS final') are an exact numeric duplicate of six rows already present under their original survey name. load_fsigma8() drops them.
- :warning: Fiducial cosmologies differ row to row (omfid/hfid columns); no AP rescaling to a common fiducial is applied.

### `measurements/fsigma8_sdss_final.txt`

- **name** `fsigma8_sdss_final`  |  **probe** `fsigma8`  |  **method** `rsd`
- fsigma8 as quoted in the SDSS final cosmology paper, with z ranges.
- **source**: eBOSS Collaboration 2021, arXiv:2007.08991
- **columns**: label zmin zmax fsigma8 err
- :warning: Duplicates six rows of fsigma8_rsd -- do not plot both.

### `measurements/s8_weak_lensing.txt`

- **name** `s8_weak_lensing`  |  **probe** `s8`
- S8 from cosmic shear: DES Y1, HSC, KiDS-450.
- **source**: Troxel et al. 2018; Hamana et al. 2020; Hildebrandt et al. 2017
- **columns**: label zmin zmax S8 err_hi err_lo
- :warning: S8 = sigma8 (Om/0.3)^0.5; the 0.5 exponent is a convention, and each survey's own best-constrained direction differs slightly.

### `measurements/sn_hubble_pantheon.txt`

- **name** `sn_hubble_pantheon`  |  **probe** `sn_distance`
- Binned Pantheon SN Ia Hubble diagram, as a ratio to the fiducial.
- **source**: Scolnic et al. 2018, arXiv:1710.00845
- **columns**: z n_sn dl_over_dl_fid err


## forecasts

### `forecasts/bao_lya_desi.txt`

- **name** `bao_lya_desi`  |  **probe** `bao`
- DESI design forecast, Lyman-alpha forest BAO.
- **source**: DESI Collaboration 2016 Part I, arXiv:1611.00036 Table 2.7
- **columns**: z dR dDA dH dNqso -- distance errors are PERCENT

### `forecasts/bao_rsd_desi.txt`

- **name** `bao_rsd_desi`  |  **probe** `bao`  |  **method** `rsd`
- DESI design forecast, ELG+LRG+QSO: BAO distances and fsigma8 precision.
- **source**: DESI Collaboration 2016 Part I, arXiv:1611.00036 Table 2.3
- **columns**: z dR dDA dH nP02 nP016 V dNelg dNlrg dNqso dfs0.1 dfs0.2 -- dfs0.1/dfs0.2 are PERCENT errors on fsigma8 for k_max = 0.1 / 0.2 h/Mpc

### `forecasts/bao_rsd_desi_bgs.txt`

- **name** `bao_rsd_desi_bgs`  |  **probe** `bao`  |  **method** `rsd`
- DESI design forecast, BGS.
- **source**: DESI Collaboration 2016 Part I, arXiv:1611.00036 Table 2.5
- **columns**: z dR dDA dH nP02 nP016 V dNbgs dfs0.1 dfs0.2 -- dfs* are PERCENT

### `forecasts/fsigma8_4most_bgs_howlett2017.txt`

- **name** `fsigma8_4most_bgs_howlett2017`  |  **probe** `fsigma8`  |  **method** `pv`
- 4MOST bright-galaxy peculiar-velocity growth-rate forecast.
- **source**: Howlett et al. 2017, arXiv:1708.08236
- **columns**: z f error_ref -- errors are PERCENT

### `forecasts/fsigma8_euclid.txt`

- **name** `fsigma8_euclid`  |  **probe** `fsigma8`  |  **method** `rsd`
- Euclid spectroscopic growth-rate forecast, 14 redshift bins to z=2.
- **source**: Amendola et al. 2016 (Euclid Theory WG), arXiv:1606.00180 Table 4
- **columns**: z f error_ref error_opt error_pess -- errors are FRACTIONAL on f

### `forecasts/fsigma8_sn_pv.csv`

- **name** `fsigma8_sn_pv`  |  **probe** `fsigma8`  |  **method** `pv`
- SN-Ia peculiar-velocity fsigma8 forecasts for ZTF and LSST.
- **source**: Carreres et al. 2023; Rosselli et al. 2025
- **columns**: zmin zmax zeff fs8_err label paper -- fs8_err is FRACTIONAL
- :warning: The LSST rows are nested redshift ranges from one survey (0.02-0.06, 0.02-0.10, 0.02-0.14, 0.06-0.10, 0.10-0.14): the wide bins CONTAIN the narrow ones. Plot one nesting level at a time.

### `forecasts/fsigma8_sn_pv_howlett2017.txt`

- **name** `fsigma8_sn_pv_howlett2017`  |  **probe** `fsigma8`  |  **method** `pv`
- SN-Ia peculiar-velocity growth-rate forecast in 10 bins to z=0.5.
- **source**: Howlett et al. 2017, arXiv:1708.08236
- **columns**: z f error_ref error_opt error_pess -- errors are PERCENT


## theorys

### `theory/cola_growth_dgp.ecsv`

- **name** `cola_growth_dgp`  |  **probe** `fsigma8`
- COLA background + growth, nDGP run.
- **source**: COLA simulation suite
- **columns**: a, H/H0, then D_k and f_k at k = 1e-5 ... 10 h/Mpc
- :warning: nDGP growth is scale-independent in the quasi-static limit, so the k columns are again degenerate.
- :warning: Drop a > 1 before plotting.

### `theory/cola_growth_fr.ecsv`

- **name** `cola_growth_fr`  |  **probe** `fsigma8`
- COLA background + scale-dependent growth, f(R) run.
- **source**: COLA simulation suite
- **columns**: a, H/H0, then D_k and f_k at k = 1e-5 ... 10 h/Mpc
- :warning: f is scale-dependent here: a single fsigma8(z) curve is an effective quantity at one k, not the model's growth rate.
- :warning: Drop a > 1 before plotting.

### `theory/cola_growth_gr.ecsv`

- **name** `cola_growth_gr`  |  **probe** `fsigma8`
- COLA background + scale-dependent growth, GR reference run.
- **source**: COLA simulation suite
- **columns**: a, H/H0, then D_k and f_k at k = 1e-5 ... 10 h/Mpc
- :warning: The GR columns are k-independent by construction -- all eight k values carry identical D and f. Only the f(R) file has real scale dependence.
- :warning: The last row (a=2.0) has f = -0.408, an artefact of the end-of-grid derivative. Drop a > 1 before plotting.

