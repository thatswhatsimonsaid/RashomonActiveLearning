"""Mean-score common-neighbor extraction, following RashomonDR experiment_2.

The scoring arithmetic matches code/scripts/experiment_2.py. Candidate edges
are Euclidean kNN edges from scikit-learn. The original script builds that
graph with pacmap.PaCMAP(n_neighbors=50, save_pairs=True). This function
reimplements the score; it does not launch experiment_2.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def candidate_edges(x: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(x)
    idx = nn.kneighbors(return_distance=False)[:, 1:]
    sources = np.repeat(np.arange(x.shape[0]), k)
    return np.column_stack([sources, idx.ravel()]).astype(np.int64)


def _edge_distances(embedding: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.linalg.norm(embedding[edges[:, 0]] - embedding[edges[:, 1]], axis=1)


def mean_score_edges(
    embeddings: list[np.ndarray],
    edges: np.ndarray,
    keep_per_point: int,
) -> np.ndarray:
    """Return the keep_per_point lowest-score neighbors of each source point."""
    if len(embeddings) < 2:
        raise ValueError("Common-knowledge scoring needs at least two embeddings")
    distances = np.vstack([_edge_distances(z, edges) for z in embeddings])

    def robust_std(column: np.ndarray) -> float:
        med = np.median(column)
        mad = np.median(np.abs(column - med))
        return 1.4826 * mad

    variance = np.apply_along_axis(robust_std, 0, distances)
    maxima = distances.max(axis=0)
    medians = np.median(distances, axis=0)
    global_median = float(np.median(distances))
    selected = np.where(medians > global_median, medians, maxima)
    variance = (variance - variance.min()) / (np.ptp(variance) + 1e-10)
    selected = (selected - selected.min()) / (np.ptp(selected) + 1e-10)
    score = variance + selected
    frame = pd.DataFrame({"score": score, "x": edges[:, 0], "y": edges[:, 1]})
    top = (
        frame.sort_values(["x", "score"], kind="mergesort")
        .groupby("x", group_keys=False)
        .head(keep_per_point)
    )
    return top[["x", "y"]].to_numpy(dtype=np.int64)


def edges_to_graph(edges: np.ndarray, n: int):
    data = np.ones(len(edges), dtype=np.float64)
    return csr_matrix((data, (edges[:, 0], edges[:, 1])), shape=(n, n))
