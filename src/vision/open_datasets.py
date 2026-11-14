"""Ten-class image datasets for the vision active-learning study.

CIFAR-10 stays on its existing loader. These four are the same shape: 10
classes, an official train split for the pool, and an official test split.
Every image is resized to a 3-channel tensor so Fashion-MNIST and the CHW
archives (SVHN, STL-10) enter the ProtoPNet the same way CIFAR-10 does.

Imagenette is the FastAI 160px release. Torchvision 0.14, which the cluster
environment uses, does not ship that dataset, so this module downloads the
archive itself.
"""

from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import datasets, transforms


IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
# The HTTP host named in torchvision 0.14 refuses connections from the cluster.
# The same archive is served over HTTPS by the STL-10 site.
STL10_URL = "https://cs.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
STL10_MD5 = "91f7769df0f17e558f3565bffb0c7dfb"
IMAGENETTE_CLASSES = (
    "n01440764",
    "n02102040",
    "n02979186",
    "n03000684",
    "n03028079",
    "n03394916",
    "n03417042",
    "n03425413",
    "n03445777",
    "n03888257",
)
IMAGENETTE_NAMES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}

# Defaults match the finished CIFAR-10 simulation: 56px images, 100 pool
# images per class, 50 test images per class, 4 labeled seeds per class, and
# 4 queries per class.
OPEN_DATASETS: Dict[str, Dict[str, object]] = {
    "stl10": {
        "classes": 10,
        "class_names": (
            "airplane",
            "bird",
            "car",
            "cat",
            "deer",
            "dog",
            "horse",
            "monkey",
            "ship",
            "truck",
        ),
        "native_size": 96,
        "root": "data/STL10",
        "image_size": 56,
        "per_class_pool": 100,
        "per_class_test": 50,
        "initial_size": 40,
        "queries": 40,
    },
    "svhn": {
        "classes": 10,
        "class_names": tuple(str(digit) for digit in range(10)),
        "native_size": 32,
        "root": "data/SVHN",
        "image_size": 56,
        "per_class_pool": 100,
        "per_class_test": 50,
        "initial_size": 40,
        "queries": 40,
    },
    "fashion_mnist": {
        "classes": 10,
        "class_names": (
            "T-shirt/top",
            "trouser",
            "pullover",
            "dress",
            "coat",
            "sandal",
            "shirt",
            "sneaker",
            "bag",
            "ankle boot",
        ),
        "native_size": 28,
        "root": "data/FashionMNIST",
        "image_size": 56,
        "per_class_pool": 100,
        "per_class_test": 50,
        "initial_size": 40,
        "queries": 40,
    },
    "imagenette": {
        "classes": 10,
        "class_names": tuple(IMAGENETTE_NAMES[name] for name in IMAGENETTE_CLASSES),
        "native_size": 160,
        "root": "data/imagenette2-160",
        "image_size": 56,
        "per_class_pool": 100,
        "per_class_test": 50,
        "initial_size": 40,
        "queries": 40,
    },
}


def open_dataset_names() -> List[str]:
    return list(OPEN_DATASETS)


class _FolderSplit:
    """A class-folder split with the attributes the sampler reads."""

    def __init__(self, root: Path, split: str, class_names: Sequence[str]):
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"missing {split} split at {split_dir}")
        samples: List[Path] = []
        targets: List[int] = []
        for class_index, class_name in enumerate(class_names):
            class_dir = split_dir / class_name
            files = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            samples.extend(files)
            targets.extend([class_index] * len(files))
        self.samples = samples
        self.targets = np.asarray(targets, dtype=int)


def ensure_imagenette(root: Path) -> Path:
    """Return the directory that contains Imagenette ``train`` and ``val``."""
    root = Path(root)
    if (root / "train").is_dir() and (root / "val").is_dir():
        return root
    nested = root / "imagenette2-160"
    if (nested / "train").is_dir() and (nested / "val").is_dir():
        return nested
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "imagenette2-160.tgz"
    if not archive.is_file():
        urllib.request.urlretrieve(IMAGENETTE_URL, archive)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(root)
    if (nested / "train").is_dir():
        return nested
    if (root / "train").is_dir():
        return root
    raise FileNotFoundError(f"Imagenette archive did not contain a train split under {root}")


def _targets(dataset) -> np.ndarray:
    labels = dataset.labels if hasattr(dataset, "labels") else dataset.targets
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    return np.asarray(labels)


def _as_rgb_tensor(image, image_size: int) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    else:
        array = image
    if isinstance(array, np.ndarray):
        if array.ndim == 2:
            picture = Image.fromarray(array.astype(np.uint8), mode="L")
        elif array.ndim == 3 and array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3}:
            picture = Image.fromarray(np.transpose(array, (1, 2, 0)).astype(np.uint8))
        else:
            picture = Image.fromarray(array.astype(np.uint8))
    elif isinstance(array, Image.Image):
        picture = array
    else:
        raise TypeError(f"unsupported image type {type(array)}")
    picture = picture.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    return transforms.ToTensor()(picture)


def _read_array_image(dataset, index: int):
    return dataset.data[index]


def _read_folder_image(dataset, index: int):
    with Image.open(dataset.samples[index]) as picture:
        return picture.convert("RGB")


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_stl10(root: Path) -> None:
    """Make the STL-10 binaries available without torchvision's HTTP mirror."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        datasets.STL10(root=str(root), split="train", download=False)
        datasets.STL10(root=str(root), split="test", download=False)
        return
    except RuntimeError:
        pass
    archive = root / "stl10_binary.tar.gz"
    if not archive.is_file() or _md5(archive) != STL10_MD5:
        urllib.request.urlretrieve(STL10_URL, archive)
    if _md5(archive) != STL10_MD5:
        raise RuntimeError(f"STL-10 archive at {archive} does not match {STL10_MD5}")
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(root)
    datasets.STL10(root=str(root), split="train", download=False)
    datasets.STL10(root=str(root), split="test", download=False)


def _load_splits(name: str, data_dir: Path, download: bool):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if name == "stl10":
        if download:
            ensure_stl10(data_dir)
        train = datasets.STL10(root=str(data_dir), split="train", download=False)
        test = datasets.STL10(root=str(data_dir), split="test", download=False)
        return train, test, _read_array_image
    if name == "svhn":
        train = datasets.SVHN(root=str(data_dir), split="train", download=download)
        test = datasets.SVHN(root=str(data_dir), split="test", download=download)
        return train, test, _read_array_image
    if name == "fashion_mnist":
        train = datasets.FashionMNIST(root=str(data_dir), train=True, download=download)
        test = datasets.FashionMNIST(root=str(data_dir), train=False, download=download)
        return train, test, _read_array_image
    if name == "imagenette":
        root = ensure_imagenette(data_dir)
        train = _FolderSplit(root, "train", IMAGENETTE_CLASSES)
        test = _FolderSplit(root, "val", IMAGENETTE_CLASSES)
        return train, test, _read_folder_image
    raise KeyError(f"unknown open dataset {name}")


def class_counts(name: str, data_dir: Path, download: bool = False) -> Dict[str, List[int]]:
    """How many images each class has in the official train and test splits."""
    train, test, _reader = _load_splits(name, Path(data_dir), download=download)
    num_classes = int(OPEN_DATASETS[name]["classes"])

    def counts(dataset) -> List[int]:
        targets = _targets(dataset)
        return [int(np.sum(targets == class_index)) for class_index in range(num_classes)]

    return {"train": counts(train), "test": counts(test)}


def _take(
    dataset,
    reader: Callable,
    per_class: int,
    num_classes: int,
    image_size: int,
    rng: np.random.Generator,
    split_name: str,
    dataset_name: str,
) -> Tuple[List[torch.Tensor], List[int]]:
    targets = _targets(dataset)
    images: List[torch.Tensor] = []
    labels: List[int] = []
    for class_index in range(num_classes):
        class_ids = np.flatnonzero(targets == class_index)
        if len(class_ids) < per_class:
            raise ValueError(
                f"{dataset_name} {split_name} class {class_index} has {len(class_ids)} images, "
                f"needed {per_class}"
            )
        picked = rng.choice(class_ids, size=per_class, replace=False)
        for image_id in picked:
            images.append(_as_rgb_tensor(reader(dataset, int(image_id)), image_size))
        labels.extend([class_index] * per_class)
    return images, labels


def load_open_pool_and_test(
    name: str,
    data_dir: Path,
    per_class_pool: int,
    per_class_test: int,
    image_size: int,
    seed: int,
    download: bool = False,
):
    """Sample a balanced pool from train and a balanced test from the held-out split.

    The returned frame lists pool rows first, then test rows. ``n_pool`` is the
    cut between them. Images are float tensors in ``[0, 1]`` with shape
    ``(N, 3, image_size, image_size)``.
    """
    if name not in OPEN_DATASETS:
        raise KeyError(f"unknown open dataset {name}")
    num_classes = int(OPEN_DATASETS[name]["classes"])
    train, test, reader = _load_splits(name, Path(data_dir), download=download)
    rng = np.random.default_rng(seed)
    pool_images, pool_labels = _take(
        train, reader, per_class_pool, num_classes, image_size, rng, "train", name
    )
    test_images, test_labels = _take(
        test, reader, per_class_test, num_classes, image_size, rng, "test", name
    )
    images = torch.stack(pool_images + test_images)
    labels = pool_labels + test_labels
    frame = pd.DataFrame(
        {
            "image_id": np.arange(len(labels), dtype=int),
            "Y": labels,
        }
    )
    frame.attrs["task_type"] = "classification"
    return frame, images, torch.tensor(labels, dtype=torch.long), len(pool_labels)
