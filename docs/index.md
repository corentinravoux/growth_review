# growth_review

Compiled constraints on the growth of structure — **measurements**, **forecasts**
and **theory curves** — with the loaders and plotting layer that turn them into
review figures.

The package exists to keep three things apart that growth plots routinely mix:

| kind | what it is | how it is drawn |
|---|---|---|
| `measurement` | a published central value with an uncertainty | outlined marker at the published value |
| `forecast` | a projected *precision*, with no central value of its own | thin bar around the fiducial curve, no marker face |
| `theory` | a model prediction: the fiducial ΛCDM curve, or an EFTCAMB / simulation table | curve |

A forecast bar sits exactly on the model curve by construction. Styled like a
measurement, it tells the reader the model has been confirmed at that precision.

- [Getting started](getting-started.md): install, the registry, building figures.
- [Theory curves](theory.md): the fiducial reference, EFTCAMB, σ8 normalisation.
- [Redshift axis](redshift-axis.md): the four x-scales and when to use each.
- [API reference](api/index.md): generated from the docstrings.

## Layout

```
growth_review/
├── theory/        fsigma8(z) curves, callable anywhere
│   ├── background.py  flat CPL background: E(a), Omega_m(a), dlnH/dlna
│   ├── growth.py      RK4 integration of the GR growth equation + growth index
│   ├── model.py       GrowthModel / TableModel / Cosmology
│   ├── registry.py    named models, their styles, the one-liner accessors
│   ├── tables.py      tabulated curves (COLA runs, EFTCAMB exports)
│   └── eftcamb.py     THE modified-gravity backend: port of EFTCAMB_fsigma8.ipynb
├── datasets.py    the registry: every file, its kind, probe, columns and caveats
├── io.py          readers, and the tidy fsigma8 view over all of them
├── methods.py     PV method taxonomy + the paper sentence behind each assignment
├── style.py       palette, reserved colours, and the redshift x-scales
├── plotting.py    primitives over the tidy schema + the theory-curve drawing
├── figures.py     the five composed figures + CLI
└── data/
    ├── measurements/   fsigma8 (PV, RSD), BAO, S8, SN Hubble diagram
    ├── forecasts/      Euclid, DESI design, 4MOST, ZTF/LSST SN-PV
    └── theory/         COLA growth histories, and eftcamb_*.ecsv exports
```
