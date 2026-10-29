# Getting started

## Install

```bash
git clone https://github.com/corentinravoux/growth_review
cd growth_review
pip install -e ".[dev]"
```

## Use

```python
import growth_review as gr

gr.use_style()
print(gr.summary_table())                     # the registry

df = gr.load_fsigma8(kind="measurement")      # PV + RSD in one tidy schema
fig, meta = gr.figures.fig_pv()
print(meta["caption"])
```

Build every figure:

```bash
growth-review-figures --outdir figures
growth-review-figures --only pv rsd --format png
```

`python -m growth_review` prints both registries: the data files and the theory
models.

## Notebooks

| notebook | what it covers | needs EFTCAMB |
|---|---|---|
| `notebooks/growth_review.ipynb` | the compilation itself: registry, loaders, PV method families, the five data figures, citation plumbing | no |
| `notebooks/eftcamb_theory.ipynb` | the port of `EFTCAMB_fsigma8.ipynb`: model registry, f_R0-specified designer f(R), fσ8 figure, B₀ scan, f(k,z) scale dependence, H(z) comparison, the 0.1% and 1% background-unmodified figures, and the export step | **yes** — shipped unexecuted |
| `notebooks/forecasts_vs_theory.ipynb` | ZTF / LSST peculiar-velocity and Euclid / DESI forecasts on a log-z axis: precision vs model deviation, σ per bin, the \|f_R0\| each programme reaches at 3σ | only for the model curves; the forecast panels build without it |

They run on the `growth_review` kernel:

```bash
.venv/bin/python -m ipykernel install --user --name growth_review \
    --display-name growth_review
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=growth_review notebooks/theory_curves.ipynb
```
