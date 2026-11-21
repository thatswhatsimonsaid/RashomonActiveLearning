"""Empirical Rashomon membership and source weights for CoMDS.

Membership is a loss threshold on the candidates that were trained. It is
not an enumeration of every embedding inside the continuous Rashomon set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def mark_rashomon(frame: pd.DataFrame, delta: float) -> pd.DataFrame:
    """Keep a candidate when its probe loss is within a relative band of the best."""
    out = frame.copy()
    out["in_rashomon"] = False
    out["loss_reference"] = np.nan
    out["loss_tolerance"] = np.nan
    for _method, group in out.groupby("method"):
        losses = group["probe_loss"].to_numpy(dtype=float)
        finite = np.isfinite(losses)
        if not finite.any():
            continue
        best = float(np.min(losses[finite]))
        tolerance = delta * max(abs(best), 1e-8)
        keep = finite & (losses <= best + tolerance)
        out.loc[group.index, "loss_reference"] = best
        out.loc[group.index, "loss_tolerance"] = tolerance
        out.loc[group.index[keep], "in_rashomon"] = True
    return out


def _positive_weights(values: np.ndarray) -> np.ndarray:
    weights = np.asarray(values, dtype=float)
    weights = np.clip(weights, 1e-6, None)
    return weights


def uniform_weights(members: pd.DataFrame) -> np.ndarray:
    return np.ones(len(members), dtype=float)


def method_balanced_weights(members: pd.DataFrame) -> np.ndarray:
    counts = members.groupby("method")["method"].transform("size").to_numpy(dtype=float)
    n_methods = members["method"].nunique()
    return _positive_weights(1.0 / (counts * n_methods))


def gibbs_weights(members: pd.DataFrame) -> np.ndarray:
    """Within-method Gibbs tilt of the probe loss, then equal method mass.

    beta is 1 / std(probe loss) inside each method. This is a practical
    analogue of Gibbs committee weights, not a calibrated PAC-Bayes posterior.
    """
    weights = np.zeros(len(members), dtype=float)
    positions = {idx: i for i, idx in enumerate(members.index)}
    n_methods = members["method"].nunique()
    for _method, group in members.groupby("method"):
        losses = group["probe_loss"].to_numpy(dtype=float)
        center = losses - np.min(losses)
        scale = float(np.std(losses))
        beta = 1.0 / scale if scale > 1e-8 else 0.0
        raw = np.exp(-beta * center)
        raw = raw / np.sum(raw)
        for idx, value in zip(group.index, raw):
            weights[positions[idx]] = value / n_methods
    return _positive_weights(weights)


def vote_weights(members: pd.DataFrame, embeddings: list[np.ndarray], k: int = 10) -> np.ndarray:
    """Upweight views whose neighbors agree with the majority Rashomon vote."""
    if len(embeddings) == 0:
        return np.ones(0, dtype=float)
    n = embeddings[0].shape[0]
    votes = np.zeros((n, n), dtype=float)
    neighborhoods = []
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    for embedding in embeddings:
        nn.fit(embedding)
        idx = nn.kneighbors(return_distance=False)[:, 1:]
        neighborhoods.append(idx)
        for i in range(n):
            votes[i, idx[i]] += 1.0
    majority = np.argsort(-votes, axis=1)[:, :k]
    scores = []
    for idx in neighborhoods:
        overlaps = [
            len(set(idx[i]).intersection(majority[i])) / k for i in range(n)
        ]
        scores.append(float(np.mean(overlaps)))
    return _positive_weights(np.asarray(scores, dtype=float))


def edge_weight_matrix(embeddings: list[np.ndarray], k: int = 10) -> np.ndarray:
    """Pair weight = fraction of Rashomon views in which the pair is a kNN edge."""
    n = embeddings[0].shape[0]
    votes = np.zeros((n, n), dtype=float)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    for embedding in embeddings:
        nn.fit(embedding)
        idx = nn.kneighbors(return_distance=False)[:, 1:]
        for i in range(n):
            votes[i, idx[i]] += 1.0
    frequency = votes / max(len(embeddings), 1)
    sym = 0.5 * (frequency + frequency.T)
    np.fill_diagonal(sym, 0.0)
    return sym


def mean_vote_entropy(embeddings: list[np.ndarray], k: int = 10) -> float:
    n = embeddings[0].shape[0]
    votes = np.zeros((n, n), dtype=float)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    for embedding in embeddings:
        nn.fit(embedding)
        idx = nn.kneighbors(return_distance=False)[:, 1:]
        for i in range(n):
            votes[i, idx[i]] += 1.0
    entropies = []
    for i in range(n):
        positive = votes[i][votes[i] > 0]
        prob = positive / positive.sum()
        entropies.append(float(-(prob * np.log(prob)).sum()))
    return float(np.mean(entropies))
