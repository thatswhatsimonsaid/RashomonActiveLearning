"""Post-hoc DR losses used as the Rashomon criterion.

The packaged trainers add the PCA-alignment penalty into the logged scalar,
so two runs with different alignment weights are not comparable on that
scalar. These probes re-evaluate the original DR penalties on the finished
embedding and leave the alignment term out.

They are empirical scores on the candidates we actually trained. They are
not an exhaustive description of the continuous Rashomon set.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
from cne.cne import ContrastiveLoss

_EVAL = None


def _rashomon_evaluation():
    global _EVAL
    if _EVAL is not None:
        return _EVAL
    path = (
        Path(__file__).resolve().parents[1]
        / "vendor"
        / "RashomonDR"
        / "code"
        / "scripts"
        / "evaluation.py"
    )
    spec = importlib.util.spec_from_file_location("rashomondr_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _EVAL = module
    return module


def contrastive_probe(
    embedding: np.ndarray,
    neighbor_index: np.ndarray,
    loss_mode: str,
    negative_samples: int = 5,
) -> float:
    """Original ContrastiveLoss on (point, first high-d neighbor) pairs."""
    y = torch.tensor(np.asarray(embedding), dtype=torch.float32)
    positive = y[torch.as_tensor(neighbor_index[:, 0], dtype=torch.long)]
    features = torch.cat([y, positive], dim=0)
    criterion = ContrastiveLoss(
        negative_samples=negative_samples,
        loss_mode=loss_mode,
        metric="euclidean",
        temperature=0.5,
        eps=torch.tensor(1.0),
        spec_param=torch.tensor(1.0),
        clamp_high=1.0,
        clamp_low=1e-4,
        seed=0,
        loss_aggregation="mean",
    )
    loss, *_rest = criterion(
        features,
        log_Z=torch.tensor(0.0),
        log_temp=None,
        force_resample=True,
    )
    return float(loss.detach().cpu())


def _edges_from_neighbor_table(table: np.ndarray) -> np.ndarray:
    """Rebuild (source, neighbor) edges. Row i of `table` belongs to point i."""
    n_samples, width = table.shape
    sources = np.repeat(np.arange(n_samples, dtype=np.int32), width)
    return np.column_stack([sources, np.asarray(table, dtype=np.int32).reshape(-1)])


def pacmap_probe(
    embedding: np.ndarray,
    pair_neighbors: np.ndarray,
    pair_mn: np.ndarray,
    pair_fp: np.ndarray,
    n_samples: int,
) -> float:
    """Unweighted sum of the PaCMAP NN, MN, and FP terms in evaluation.py.

    `generate_pair` fills the source column of neighbor pairs with
    `n_neighbors + i` (`sample_neighbors_pair_basis`). Training ignores that
    column and uses `convert_pairs`, which keeps column 1 and assumes row
    order. The probe follows `convert_pairs` and then rebuilds a valid
    (source, neighbor) list for `record_loss_pacmap`.
    """
    from parampacmap.training import convert_pairs

    nn_table, fp_table, mn_table = convert_pairs(
        pair_neighbors, pair_fp, pair_mn, n_samples
    )
    evaluation = _rashomon_evaluation()
    loss_nn, loss_mn, loss_fp = evaluation.record_loss_pacmap(
        np.asarray(embedding, dtype=np.float64),
        _edges_from_neighbor_table(nn_table),
        _edges_from_neighbor_table(mn_table),
        _edges_from_neighbor_table(fp_table),
        n_samples,
    )
    return float(np.mean(loss_nn) + np.mean(loss_mn) + np.mean(loss_fp))
