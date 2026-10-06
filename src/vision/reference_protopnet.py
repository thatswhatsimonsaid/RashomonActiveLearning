"""Reference cosine ProtoPNet used as the frozen feature map for vision REAL.

Proto-RSet enumerates near-optimal last layers on top of a fixed backbone and
prototype layer. This module builds that reference network from the proto-rset
classes (``VanillaProtoPNet`` with cosine prototype activation), projects
prototypes onto labeled images, and returns one similarity per prototype.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def _ensure_proto_rset_on_path() -> Path:
    root = Path(__file__).resolve().parents[2] / "third_party" / "proto-rset"
    if not root.is_dir():
        raise ImportError(
            f"proto-rset is not checked out at {root}. "
            "Initialize submodules so third_party/proto-rset is present."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


class SpatialPoolBackbone(nn.Module):
    """Downsample an image and keep its colors.

    Used to check prototype projection without a pretrained trunk. Each
    prototype is then a color patch, which is enough to separate images whose
    class is a distinct color.
    """

    def __init__(self, spatial_size: int = 4):
        super().__init__()
        self.spatial_size = spatial_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.adaptive_avg_pool2d(x, (self.spatial_size, self.spatial_size))


class TinyConvBackbone(nn.Module):
    """Small convolutional trunk for CPU checks of the ProtoPNet path.

    Real studies pass an ImageNet backbone from proto-rset. This trunk exists
    so the prototype layer, projection, and similarity interface can be
    exercised without downloading a pretrained network.
    """

    def __init__(self, out_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, out_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ImageLabelDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "img": self.images[index],
            "target": self.labels[index],
            "sample_id": torch.tensor(index, dtype=torch.long),
        }


def build_cosine_protopnet(
    num_classes: int,
    num_prototypes_per_class: int,
    backbone: Optional[nn.Module] = None,
    image_size: Tuple[int, int, int] = (3, 32, 32),
    num_addon_layers: int = 1,
    proto_channel_multiplier: float = 1.0,
) -> nn.Module:
    """Build the cosine Vanilla ProtoPNet from proto-rset.

    The last layer is initialized the way proto-rset initializes it: weight 1
    from each prototype to its own class, and weight 0 to every other class.
    """
    _ensure_proto_rset_on_path()
    from protopnet.activations import CosPrototypeActivation
    from protopnet.embedding import AddonLayers, EmbeddedBackbone
    from protopnet.models.vanilla_protopnet import VanillaProtoPNet

    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")
    if num_prototypes_per_class < 1:
        raise ValueError("num_prototypes_per_class must be at least 1.")

    feature_extractor = backbone if backbone is not None else TinyConvBackbone()
    embedded = EmbeddedBackbone(feature_extractor, input_channels=image_size)
    latent_channels = int(embedded.latent_dimension[0])
    add_on_layers = AddonLayers(
        num_prototypes=num_classes * num_prototypes_per_class,
        input_channels=latent_channels,
        proto_channel_multiplier=proto_channel_multiplier,
        num_addon_layers=num_addon_layers,
    )
    model = VanillaProtoPNet(
        backbone=embedded,
        add_on_layers=add_on_layers,
        activation=CosPrototypeActivation(),
        num_classes=num_classes,
        num_prototypes_per_class=num_prototypes_per_class,
    )
    return model


def prototype_similarities(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return max-pooled cosine similarity to each prototype, shape (N, M)."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model(images, return_similarity_score_to_each_prototype=True)
    if was_training:
        model.train()
    return output["similarity_score_to_each_prototype"]


def predict_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return class logits, shape (N, C)."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model(images)
    if was_training:
        model.train()
    return output["logits"]


def project_prototypes(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    class_specific: bool = True,
) -> None:
    """Push each prototype onto the nearest latent patch of the labeled images."""
    loader = DataLoader(
        _ImageLabelDataset(images.detach().cpu(), labels.detach().cpu()),
        batch_size=int(images.shape[0]),
        shuffle=False,
    )
    model.project(loader, class_specific=class_specific)


def fit_reference_protopnet(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    warm_steps: int = 15,
    last_layer_steps: int = 15,
    learning_rate: float = 1e-3,
    class_specific_projection: bool = True,
) -> Dict[str, float]:
    """Fit add-on layers and prototypes, project them, then fit the last layer.

    The backbone stays frozen. That matches the vision active-learning setup:
    the reference trunk is fixed, and later rounds only revisit the last layer.
    """
    device = next(model.parameters()).device
    images = images.to(device)
    labels = labels.to(device)
    criterion = nn.CrossEntropyLoss()

    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)

    initial_loss = _cross_entropy(model, images, labels, criterion)

    warm_parameters = list(model.add_on_layers.parameters()) + list(
        model.prototype_layer.parameters()
    )
    warm_optimizer = torch.optim.Adam(warm_parameters, lr=learning_rate)
    model.train()
    for _ in range(warm_steps):
        warm_optimizer.zero_grad()
        loss = criterion(model(images)["logits"], labels)
        loss.backward()
        warm_optimizer.step()

    project_prototypes(
        model,
        images,
        labels,
        class_specific=class_specific_projection,
    )

    for parameter in model.add_on_layers.parameters():
        parameter.requires_grad_(False)
    for parameter in model.prototype_layer.parameters():
        parameter.requires_grad_(False)
    last_layer = model.prototype_prediction_head.class_connection_layer
    for parameter in last_layer.parameters():
        parameter.requires_grad_(True)
    last_optimizer = torch.optim.Adam(last_layer.parameters(), lr=learning_rate)
    model.train()
    for _ in range(last_layer_steps):
        last_optimizer.zero_grad()
        loss = criterion(model(images)["logits"], labels)
        loss.backward()
        last_optimizer.step()

    final_loss = _cross_entropy(model, images, labels, criterion)
    predictions = predict_logits(model, images).argmax(dim=1)
    accuracy = float((predictions == labels).float().mean().item())
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "accuracy": accuracy,
    }


def _cross_entropy(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
) -> float:
    model.eval()
    with torch.no_grad():
        return float(criterion(model(images)["logits"], labels).item())


def class_colored_images(
    num_classes: int,
    images_per_class: int,
    image_size: int = 32,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Synthetic images whose class is a distinct color plus a little noise."""
    if num_classes > 3:
        raise ValueError("class_colored_images supports at most 3 classes.")
    generator = torch.Generator().manual_seed(seed)
    images = torch.zeros(num_classes * images_per_class, 3, image_size, image_size)
    labels = torch.arange(num_classes).repeat_interleave(images_per_class)
    noise = torch.rand(images.shape, generator=generator) * 0.05
    for class_index in range(num_classes):
        start = class_index * images_per_class
        stop = start + images_per_class
        images[start:stop, class_index] = 1.0
    return images + noise, labels
