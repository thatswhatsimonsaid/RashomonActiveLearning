#!/usr/bin/env python3
"""Checks that do not require a full training run."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redr.comds_bridge import fit_consensus
from redr.metrics import evaluate_embedding
from redr.rashomon import mark_rashomon


def check_metrics():
    rng = np.random.default_rng(0)
    cloud = rng.normal(size=(40, 2))
    scores = evaluate_embedding(cloud, cloud.copy(), labels=None, k=5)
    assert scores["neighborhood_preservation"] == 1.0
    assert scores["trustworthiness"] > 0.99
    assert scores["continuity"] > 0.99
    assert scores["triplet_accuracy"] > 0.99
    labels = np.array([0] * 20 + [1] * 20)
    separated = np.vstack(
        [rng.normal(loc=-3, size=(20, 2)), rng.normal(loc=3, size=(20, 2))]
    )
    scored = evaluate_embedding(separated, separated, labels=labels, k=5)
    assert scored["silhouette"] > 0.5


def check_filter():
    frame = pd.DataFrame(
        {"method": ["umap", "umap", "umap"], "probe_loss": [1.0, 1.02, 2.0]}
    )
    marked = mark_rashomon(frame, delta=0.05)
    assert marked["in_rashomon"].tolist() == [True, True, False], marked["in_rashomon"].tolist()


def check_comds():
    rng = np.random.default_rng(1)
    embeddings = [rng.normal(size=(25, 2)) for _ in range(3)]
    weights = np.array([1.0, 0.5, 1.5])
    out = Path("/tmp/redr-bridge-check")
    fit = fit_consensus(embeddings, out, kind="comds", weights=weights)
    assert fit["embedding"].shape == (25, 2)
    assert np.isfinite(fit["embedding"]).all()
    assert np.isfinite(fit["stress"])
    assert fit["embedding"].std() > 0
    local = fit_consensus(
        embeddings, out / "loco", kind="locomds", weights=weights, tau=0.1, percentile=0.5
    )
    assert local["embedding"].shape == (25, 2)
    assert np.isfinite(local["embedding"]).all()
    print(
        f"bridge ok  comds_stress={fit['stress']:.4f} niter={fit['niter']} "
        f"loco_stress={local['stress']:.4f}"
    )


if __name__ == "__main__":
    check_metrics()
    check_filter()
    check_comds()
    print("verify_bridge passed")
