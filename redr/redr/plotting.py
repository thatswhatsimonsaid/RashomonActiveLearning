"""Comparison figures. Orthogonal Procrustes is display-only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes


def display_align(embedding: np.ndarray, reference: np.ndarray) -> np.ndarray:
    z = np.asarray(embedding, dtype=float)
    ref = np.asarray(reference, dtype=float)
    z = z - z.mean(axis=0)
    ref = ref - ref.mean(axis=0)
    z_scale = np.sqrt(np.mean(np.sum(z**2, axis=1))) + 1e-12
    ref_scale = np.sqrt(np.mean(np.sum(ref**2, axis=1))) + 1e-12
    rotation, _ = orthogonal_procrustes(z / z_scale, ref / ref_scale)
    return (z / z_scale) @ rotation


def _scatter(ax, embedding, color, discrete: bool, title: str):
    if discrete:
        ax.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="tab10", s=18, linewidths=0)
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="viridis", s=12, linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")


def plot_gallery(
    path: Path,
    panels: list[tuple[str, np.ndarray]],
    color: np.ndarray,
    discrete: bool,
    reference: np.ndarray,
    ncols: int = 4,
):
    n = len(panels)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.1 * nrows), squeeze=False)
    for ax, (title, embedding) in zip(axes.ravel(), panels):
        shown = display_align(embedding, reference)
        _scatter(ax, shown, color, discrete, title)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_metric_bars(path: Path, frame: pd.DataFrame, metrics: list[str]):
    consensus = frame[frame["kind"] == "consensus"].copy()
    if consensus.empty:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 4.2), squeeze=False)
    for ax, metric in zip(axes.ravel(), metrics):
        order = consensus.sort_values(metric, ascending=False)
        ax.barh(order["embedding_id"], order[metric], color="#3c6e71")
        ax.set_xlabel(metric)
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
