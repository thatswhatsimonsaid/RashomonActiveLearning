"""Small datasets for the REDR smoke tests.

Labels are returned for evaluation and plotting. Training uses a PCA
alignment target and does not receive class labels.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits, load_iris, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_dataset(name: str, seed: int = 0) -> dict:
    name = name.lower()
    if name == "iris":
        bunch = load_iris()
        x_raw = np.asarray(bunch.data, dtype=np.float64)
        labels = np.asarray(bunch.target, dtype=np.int64)
        label_names = [str(v) for v in bunch.target_names]
        discrete = True
        color = labels.astype(np.float64)
    elif name == "swiss":
        x_raw, color = make_swiss_roll(
            n_samples=300, noise=0.15, random_state=seed
        )
        x_raw = np.asarray(x_raw, dtype=np.float64)
        color = np.asarray(color, dtype=np.float64)
        labels = None
        label_names = None
        discrete = False
    elif name == "digits":
        # 8x8 digits stand in only when a full MNIST download is unavailable.
        bunch = load_digits()
        rng = np.random.default_rng(seed)
        per_class = 30
        chosen = []
        for cls in np.unique(bunch.target):
            idx = np.flatnonzero(bunch.target == cls)
            chosen.append(rng.choice(idx, size=per_class, replace=False))
        take = np.sort(np.concatenate(chosen))
        x_raw = np.asarray(bunch.data[take], dtype=np.float64)
        labels = np.asarray(bunch.target[take], dtype=np.int64)
        label_names = [str(i) for i in range(10)]
        discrete = True
        color = labels.astype(np.float64)
    elif name == "mnist":
        x_raw, labels = _mnist_subset(n_per_class=20, seed=seed)
        label_names = [str(i) for i in range(10)]
        discrete = True
        color = labels.astype(np.float64)
    else:
        raise ValueError(f"Unknown dataset {name}")

    scaler = StandardScaler()
    x = scaler.fit_transform(x_raw).astype(np.float64)
    pca = PCA(n_components=2, random_state=0)
    x_pca = pca.fit_transform(x).astype(np.float32)
    return {
        "name": name,
        "X": x.astype(np.float32),
        "X_pca": x_pca,
        "labels": labels,
        "label_names": label_names,
        "discrete_labels": discrete,
        "color": color,
        "n": int(x.shape[0]),
        "p": int(x.shape[1]),
    }


def _mnist_subset(n_per_class: int, seed: int):
    """Stratified MNIST subset. Tries torchvision, then OpenML.

    sklearn's default OpenML parser fails on this MNIST file with current
    pandas (`TypeError: expected string or bytes-like object, got 'int'`).
    The liac-arff parser reads the same cached archive.
    """
    rng = np.random.default_rng(seed)
    images = None
    targets = None
    failures = []
    try:
        from torchvision.datasets import MNIST

        ds = MNIST(root="/tmp/redr-mnist", train=True, download=True)
        images = ds.data.numpy().reshape(len(ds), -1).astype(np.float64)
        targets = np.asarray(ds.targets, dtype=np.int64)
    except Exception as exc:
        failures.append(f"torchvision: {type(exc).__name__}: {exc}")
    if images is None:
        from sklearn.datasets import fetch_openml

        for parser in ("liac-arff", "auto"):
            try:
                bunch = fetch_openml(
                    "mnist_784", version=1, as_frame=False, parser=parser
                )
                images = np.asarray(bunch.data, dtype=np.float64)
                targets = np.asarray(bunch.target, dtype=np.int64)
                break
            except Exception as exc:
                failures.append(f"openml/{parser}: {type(exc).__name__}: {exc}")
    if images is None:
        raise RuntimeError("MNIST could not be loaded. " + " | ".join(failures))
    chosen = []
    for cls in range(10):
        idx = np.flatnonzero(targets == cls)
        chosen.append(rng.choice(idx, size=n_per_class, replace=False))
    take = np.sort(np.concatenate(chosen))
    return images[take], targets[take]
