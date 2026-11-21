# Rashomon Ensembled Dimension Reduction (REDR)

Proof of concept that builds method-specific Rashomon sets of dimension-reduction embeddings and combines them with Consensus Multidimensional Scaling.

This directory is a separate project. It does not change the Rashomon active-learning code in the rest of the repository. The upstream checkouts of [RashomonDR](https://github.com/williamsyy/RashomonDR) and [coMDS](https://github.com/tiffanymtang/coMDS) are cloned by `setup_vendor.sh` and are not edited. Pins are in `VENDOR.lock`.

Papers:

- Sun, Huang, Parikh, and Rudin. *The Rashomon Effect for Visualizing High-Dimensional Data.* arXiv:2604.00485.
- An and Tang. *Consensus dimension reduction via multi-view learning.* arXiv:2512.15802.

Findings, failures, and what was actually run are in `REPORT.md`. Design choices are in `DECISIONS.md`.

## Layout

```
redr/
  redr/                 Python integration package
  r/fit_consensus.R     Calls the installed coMDS package
  scripts/run_experiment.py
  scripts/verify_bridge.py
  vendor/               Cloned upstream repos (not committed)
  results/              CSV metrics, figures, and cached embeddings
```

## Install

Python 3.10+ and R 4.x with a C++ toolchain. On Debian/Ubuntu the R libraries used here are the `r-cran-*` packages for Rcpp, RcppArmadillo, dplyr, FNN, forcats, furrr, ggplot2, purrr, tibble, and tidyselect, plus `smacof` from CRAN.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r redr/requirements.txt
bash redr/setup_vendor.sh
```

`setup_vendor.sh` installs `contrastive-ne` and `parampacmap` editable with `--no-deps`. PyKeOps is skipped; the runs set `use_keops=False` and build the neighbor graph with Annoy. `parampacmap` imports `scanpy` and its training module imports PyTorch Lightning, so both are listed even though the ANN fits in this project do not call those code paths.

Check the CoMDS bridge without training:

```bash
PYTHONPATH=redr python redr/scripts/verify_bridge.py
```

## Run

From the repository root:

```bash
PYTHONPATH=redr python redr/scripts/run_experiment.py --datasets iris
```

The Iris defaults match the executed Iris run: methods `umap`, `infonce`, and `pacmap`; seeds `0 1 2`; PCA-alignment weights `0 0.5 5`; 100 CNE epochs; 450 PaCMAP epochs; Rashomon relative tolerance `0.05`. LoCoMDS uses `tau=0.1` and `percentile=0.5` and is not tuned on LCMC.

Other executed commands:

```bash
PYTHONPATH=redr python redr/scripts/run_experiment.py \
  --datasets swiss --seeds 0 1 --lambdas 0 5 --cne-epochs 80 --pacmap-epochs 220

PYTHONPATH=redr python redr/scripts/run_experiment.py \
  --datasets mnist --seeds 0 1 --lambdas 0 5 --cne-epochs 40 --pacmap-epochs 120
```

Outputs for each dataset:

- `results/<dataset>/metrics.csv` — candidates, consensus embeddings, and the common-knowledge UMAP
- `results/<dataset>/embeddings.png` — Procrustes-aligned display of the fits
- `results/<dataset>/consensus_metrics.png`
- `results/<dataset>/embeddings.npz` — the numeric embeddings
- `results/<dataset>/consensus/<name>/` — CoMDS `gspace.csv`, stress, and source weights
- `results/metrics_all.csv` — the three datasets stacked

Class labels are used for silhouette scores and for plot colors. Training sees a 2D PCA alignment target, not the labels.
