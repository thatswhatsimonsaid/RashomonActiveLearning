# Decisions

## Where the code lives

Integration code is under `redr/`. RashomonDR (`0ab277d`) and coMDS (`0570fe2`) are cloned into `redr/vendor/` and installed in place. No file in either checkout is patched. Import-time caches (`__pycache__`, `*.egg-info`) are created by the editable installs.

## What a "Rashomon set" means here

The continuous parameter set in Sun et al. is not enumerated. Each method contributes a finite candidate pool:

- random initialization seeds
- PCA-alignment weights `lambda` passed to the original trainers (`label_weights` / `label_weight`, `task_type="pca"`)
- one deliberately short run (2 epochs) so the loss gate has a poor embedding it can reject

Membership is per method. Let `L*` be the best probe loss in that method's pool. A candidate is kept when `L <= L* + 0.05 * max(|L*|, 1e-8)`.

The logged training scalar mixes the DR loss with the alignment penalty, so runs with different `lambda` are not comparable on that scalar. The probe recomputes a DR penalty on the finished embedding:

- UMAP and InfoNCE: the vendored `cne.cne.ContrastiveLoss`, with each point paired to its first high-dimensional neighbor. `spec_param=1`, mean aggregation. This is the original loss class, evaluated once, not the minibatch objective seen during training.
- PaCMAP: `record_loss_pacmap` in RashomonDR `evaluation.py`, summed across the NN, mid-near, and further-point terms with the schedule weights removed.

`generate_pair` writes neighbor-pair sources as `n_neighbors + i` (`sample_neighbors_pair_basis`). Training ignores that column via `convert_pairs`. The probe does the same and rebuilds `(i, neighbor)` edges. That avoids an index error at `n` and does not change the upstream function.

## Consensus

Embeddings are passed to the installed R package through `r/fit_consensus.R` (`coMDS` / `locoMDS`, INDSCAL, `itmax=300`, `eps=1e-6`). Source weights use the package `weights` argument. The edge-frequency run also passes `weightmat`.

LoCoMDS builds its own local weight matrices, so the edge-weight matrix is CoMDS-only. `tau=0.1` and `percentile=0.5` are the package example values. They are fixed in advance so adjusted LCMC stays an evaluation metric.

Aggregations that were run:

| Name | Views | Source weights |
| --- | --- | --- |
| `conventional_best` | lowest-probe embedding of each method | equal |
| `unrestricted_all` | every successful candidate, including 2-epoch runs | equal |
| `unrestricted_matched` | per method, as many candidates as that method kept, drawn with seed 0 and no regard to loss | equal |
| `rashomon_uniform` | loss-gate survivors | equal |
| `rashomon_method_balanced` | survivors | each method sums to `1/M` |
| `rashomon_gibbs` | survivors | within method, `exp(-(L-Lmin)/std(L))`, then each method sums to `1/M` |
| `rashomon_vote` | survivors | mean overlap of that view's 10-NN with the majority 10-NN vote |
| `rashomon_edge_weightmat` | survivors | equal source weights; pair weight = vote frequency plus `1e-3`, diagonal 0 |

Gibbs weights are a practical tilt of the probe loss. They are not a calibrated PAC-Bayes posterior.

Common-knowledge UMAP follows the mean-score arithmetic in RashomonDR `experiment_2.py` (robust std, median/max selection, keep 15 edges per point from a 30-NN graph). The candidate graph is a scikit-learn Euclidean kNN, not `pacmap.PaCMAP(save_pairs=True)`. The refit calls `CNE.fit_transform(..., graph=...)`. This is a reimplementation of that scoring step, not a launch of `experiment_2.py`.

## Evaluation

Reported at `k=10`, same triplet sample (4000 triplets, seed 0) for every embedding of a dataset:

- sklearn trustworthiness
- Venna–Kaski continuity
- neighborhood preservation (mean fraction of shared 10-NN)
- adjusted LCMC at that single `k` (Chen and Buja; the CoMDS paper averages over many `k`)
- triplet order accuracy
- silhouette, only when discrete labels exist

Display plots are orthogonally Procrustes-aligned to the PCA plane. Metrics use the raw coordinates. Rotation, reflection, and uniform scaling do not change these scores.

## Experiments that were not run

COIL-20 is not in RashomonDR and was not downloaded. No sweep of the Rashomon tolerance was run. LoCoMDS was not grid-searched. Epoch budgets on Swiss roll and MNIST are shorter than the papers (see `REPORT.md`).
