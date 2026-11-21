# REDR report

Question: on small data, does restricting CoMDS to a method-specific Rashomon set improve the consensus embedding relative to an unrestricted ensemble, to CoMDS on one embedding per method, and to RashomonDR's common-knowledge graph?

Answer from the runs below: the pipeline produces real consensus embeddings, and the loss gate does reject weak runs. On these sets the Rashomon-restricted fits do not consistently beat the unrestricted or one-per-method baselines. Iris local metrics are saturated. On the Swiss roll the PaCMAP probe keeps an embedding with worse trustworthiness than an embedding it rejects. The MNIST subset was not trained long enough to separate digits, so it checks the pipeline on a harder input and does not support a quality claim.

These are finite candidate pools (seeds, PCA-alignment weights, and one 2-epoch run). They are not exhaustive Rashomon sets.

## What ran

| Dataset | n | p | Seeds | PCA λ | CNE epochs | PaCMAP epochs | Wall time |
| --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| Iris | 150 | 4 | 0, 1, 2 | 0, 0.5, 5 | 100 | 450 | 119 s after UMAP/InfoNCE were cached; first full pass was 192 s without usable PaCMAP probes |
| Swiss roll | 300 | 3 | 0, 1 | 0, 5 | 80 | 220 | 271 s |
| MNIST subset | 200 (20 per class) | 784 | 0, 1 | 0, 5 | 40 | 120 | 132 s |

PaCMAP's published schedule reaches its final phase at epoch 200. Iris (450 epochs) includes that phase. Swiss roll (220) includes 20 epochs of it. MNIST (120) is still in the middle phase. CNE's package default is 200 epochs; Iris used 100, Swiss roll 80, MNIST 40. UMAP in this code is parametric CNE with `loss_mode="umap"`, InfoNCE is `loss_mode="infonce"`, and PaCMAP is parametric `ParamPaCMAP`.

COIL-20 was not run. The images are not in RashomonDR.

Every consensus call returned a finite embedding and a finite stress. `scripts/verify_bridge.py` passed before the training runs (CoMDS stress 0.320 and LoCoMDS stress 0.303 on three random 25-point views).

Labels were used for silhouette and for colors. Training used a 2D PCA target.

## Iris

Full UMAP and InfoNCE runs sit in a tight probe-loss band, so a 5% gate keeps almost all of them and rejects the 2-epoch runs (UMAP probe 0.259 vs best 0.198; InfoNCE 0.371 vs best 0.241). One full UMAP (seed 2, λ=0.5, probe 0.208) also falls outside. PaCMAP is different: only the three λ=5 runs are inside the band (probe 0.426–0.441). The λ=0 and λ=0.5 runs sit at 0.466–0.484, and the 2-epoch run at 0.583. On this probe, stronger PCA alignment lowered the PaCMAP DR terms.

Trustworthiness of every full Iris embedding, candidate or consensus, is between 0.972 and 0.979. That metric cannot separate the methods here.

Triplet accuracy, consensus fits:

| Embedding | Views | Stress | Trustworthiness | LCMC | Triplet | Silhouette | CoMDS seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unrestricted matched LoCoMDS | 20 | 0.147 | 0.975 | 0.587 | 0.930 | 0.428 | 10.5 |
| unrestricted all LoCoMDS | 30 | 0.134 | 0.976 | 0.582 | 0.924 | 0.405 | 19.6 |
| conventional best LoCoMDS | 3 | 0.079 | 0.977 | 0.580 | 0.922 | 0.442 | 1.9 |
| Rashomon Gibbs LoCoMDS | 20 | 0.082 | 0.976 | 0.581 | 0.922 | 0.448 | 3.8 |
| Rashomon uniform LoCoMDS | 20 | 0.074 | 0.974 | 0.572 | 0.913 | 0.451 | 3.1 |
| common-knowledge UMAP | 8 | — | 0.974 | 0.558 | 0.914 | 0.493 | — |
| unrestricted all CoMDS | 30 | 0.092 | 0.979 | 0.590 | 0.906 | 0.488 | 5.5 |
| Rashomon uniform CoMDS | 20 | 0.062 | 0.973 | 0.556 | 0.898 | 0.517 | 6.0 |
| Rashomon edge-weight CoMDS | 20 | 0.206 | 0.972 | 0.560 | 0.888 | 0.496 | 5.7 |

The best single PaCMAP candidate (seed 0, λ=5) has triplet accuracy 0.928, in the same range as the leading consensus fits. Gibbs source weights are uneven (coefficient of variation 1.08; one PaCMAP view receives weight 0.25 while each method still sums to 1/3), and the Gibbs embedding still matches uniform Rashomon LoCoMDS within about 0.01 triplet accuracy. Edge weighting raises stress and lowers triplet accuracy.

Several CoMDS panels pull one setosa point away from its class. The single best UMAP does not. Consensus is a real Iris embedding, and some of the panels are visually worse than the best single view.

A 100-epoch UMAP or InfoNCE fit takes about 4.6–5.1 s. A 450-epoch PaCMAP fit takes about 2.2–3.1 s.

## Swiss roll

The color in the figure is the manifold parameter, not a class label, and silhouette is not reported. No method, including the consensus fits, unrolls the roll into a clean rectangle. The consensus plots preserve a shared curved cloud with a yellow-to-purple gradient.

The loss gate and trustworthiness disagree for PaCMAP. The only PaCMAP member of the Rashomon set is seed 0, λ=5 (probe 0.396, trustworthiness 0.905). The rejected seed 0, λ=0 run has probe 0.480 and trustworthiness 0.960. Selecting on the probe therefore drops the more trustworthy PaCMAP view.

Triplet accuracy is highest for unrestricted matched CoMDS (0.843), then conventional CoMDS (0.835) and Rashomon Gibbs CoMDS (0.833). Rashomon uniform CoMDS is 0.825. The highest trustworthiness values belong to Rashomon vote LoCoMDS (0.983) and the edge-weighted CoMDS (0.980). Common-knowledge UMAP has trustworthiness 0.985 and the lowest triplet accuracy in the table (0.795).

Fifteen views of 300 points take 37 s for unrestricted CoMDS. LoCoMDS on seven views takes about 4 s.

## MNIST subset

Two hundred points, twenty from each digit, forty CNE epochs, and 120 PaCMAP epochs. Silhouette is negative for every full candidate (about −0.05 to −0.01). The scatter plots are mixed clouds. This budget does not recover digit classes, so the comparison is about neighborhood metrics on under-separated embeddings.

The best candidate trustworthiness is 0.768 (InfoNCE, seed 0, λ=0). Common-knowledge UMAP is slightly higher at 0.779. The best consensus triplet accuracy is Rashomon Gibbs LoCoMDS at 0.647, against 0.641 for the best single candidate and 0.641 for unrestricted matched LoCoMDS. Edge-weighted CoMDS is the weakest consensus fit on trustworthiness (0.735) and has the highest stress (0.373).

Default `fetch_openml(..., parser="auto")` raises `TypeError: expected string or bytes-like object, got 'int'` while renaming categorical levels. `parser="liac-arff"` reads the cached archive (70,000 × 784) in about 9 s. Torchvision is not installed. The subset is then drawn from that array.

## Failures fixed during the run

The first Iris PaCMAP fits succeeded and were then discarded because `record_loss_pacmap` indexed the offset source column (`n_neighbors + i`, so index 150 on 150 points). The probe now follows `convert_pairs`. Iris was rerun; cached finite UMAP and InfoNCE embeddings were reused and their probes recomputed.

## What this does not show

A consistent gain for Rashomon-restricted consensus over unrestricted CoMDS/LoCoMDS or over one good embedding per method. On Iris the views already agree, so reweighting them barely moves the embedding. On the Swiss roll the membership probe can prefer the less trustworthy PaCMAP run. On this MNIST subset the embeddings are not yet class-structured.

## Directions that follow from these runs

- Train MNIST, and any COIL-20 subset, at the papers' epoch budgets before comparing aggregators. The 40-epoch MNIST clouds are not a fair test.
- Vary hyperparameters that actually move the probe (neighbor size, negative-sample count, early exaggeration), because PCA λ in `{0, 0.5, 5}` barely moves UMAP and InfoNCE.
- Choose the Rashomon tolerance on one metric and report the others. The Swiss roll PaCMAP case is the reason: probe loss and trustworthiness rank those runs differently.
- Treat source-weight schemes as a second-order knob. Iris Gibbs weights were far from uniform and the embedding metrics stayed within about 0.01 of uniform Rashomon LoCoMDS.
- Keep edge `weightmat` as an experiment. With a `1e-3` floor it increased stress on every dataset and was the weakest Iris triplet fit.
