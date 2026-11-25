"""Load checks for the ten-class vision datasets.

These hit the real archives. Each ten-class dataset has to come back as a
balanced 3-channel batch that a ProtoPNet can score.
"""

from pathlib import Path

import pytest
import torch

from src.vision.open_datasets import (
    OPEN_DATASETS,
    _as_rgb_tensor,
    class_counts,
    load_open_pool_and_test,
)
from src.vision.reference_protopnet import build_cosine_protopnet


DATASET_ROOTS = [
    (name, Path(spec["root"])) for name, spec in OPEN_DATASETS.items()
]


OFFICIAL_SPLIT_SIZES = {
    "fashion_mnist": (60000, 10000),
    "svhn": (73257, 26032),
    "stl10": (5000, 8000),
    "imagenette": (9469, 3925),
    "mnist": (60000, 10000),
    "kmnist": (60000, 10000),
    "usps": (7291, 2007),
    "imagewoof": (9025, 3929),
}


def test_catalog_uses_the_cifar10_budget():
    assert list(OPEN_DATASETS) == [
        "stl10",
        "svhn",
        "fashion_mnist",
        "imagenette",
        "mnist",
        "kmnist",
        "usps",
        "imagewoof",
    ]
    for spec in OPEN_DATASETS.values():
        assert spec["classes"] == 10
        assert len(spec["class_names"]) == 10
        assert spec["image_size"] == 56
        assert spec["per_class_pool"] == 100
        assert spec["per_class_test"] == 50
        assert spec["initial_size"] == 40
        assert spec["queries"] == 40


def test_grayscale_and_channel_first_images_become_rgb():
    gray = torch.arange(16, dtype=torch.uint8).reshape(4, 4)
    color_first = torch.stack([gray, gray + 1, gray + 2])
    gray_tensor = _as_rgb_tensor(gray, image_size=8)
    color_tensor = _as_rgb_tensor(color_first, image_size=8)
    assert gray_tensor.shape == (3, 8, 8)
    assert color_tensor.shape == (3, 8, 8)
    assert gray_tensor[0].equal(gray_tensor[1]) and gray_tensor[1].equal(gray_tensor[2])
    assert not color_tensor[0].equal(color_tensor[1])


@pytest.mark.parametrize("name,root", DATASET_ROOTS, ids=[name for name, _root in DATASET_ROOTS])
def test_each_dataset_has_its_official_split_sizes(name, root):
    counts = class_counts(name, root, download=True)
    train_total, test_total = OFFICIAL_SPLIT_SIZES[name]
    assert sum(counts["train"]) == train_total
    assert sum(counts["test"]) == test_total
    assert all(count > 100 for count in counts["train"])
    assert all(count > 50 for count in counts["test"])
    assert len(counts["train"]) == 10
    assert len(counts["test"]) == 10


@pytest.mark.parametrize("name,root", DATASET_ROOTS, ids=[name for name, _root in DATASET_ROOTS])
def test_sampled_batch_is_balanced_repeatable_and_scores(name, root):
    frame, images, labels, n_pool = load_open_pool_and_test(
        name,
        root,
        per_class_pool=2,
        per_class_test=2,
        image_size=32,
        seed=0,
        download=True,
    )
    assert n_pool == 20
    assert images.shape == (40, 3, 32, 32)
    assert labels.shape == (40,)
    assert frame["Y"].tolist() == labels.tolist()
    assert float(images.min()) >= 0.0
    assert float(images.max()) <= 1.0
    assert torch.isfinite(images).all()
    pool_labels = labels[:n_pool]
    test_labels = labels[n_pool:]
    for class_index in range(10):
        assert int((pool_labels == class_index).sum()) == 2
        assert int((test_labels == class_index).sum()) == 2

    again, _images, _labels, _n_pool = load_open_pool_and_test(
        name,
        root,
        per_class_pool=2,
        per_class_test=2,
        image_size=32,
        seed=0,
        download=False,
    )
    assert frame["image_id"].tolist() == again["image_id"].tolist()
    assert torch.equal(images, _images)

    model = build_cosine_protopnet(
        num_classes=10,
        num_prototypes_per_class=1,
        image_size=(3, 32, 32),
        backbone_name="tiny",
        pretrained_backbone=False,
    )
    model.eval()
    with torch.no_grad():
        logits = model(images[:4])["logits"]
    assert logits.shape == (4, 10)
    assert torch.isfinite(logits).all()
