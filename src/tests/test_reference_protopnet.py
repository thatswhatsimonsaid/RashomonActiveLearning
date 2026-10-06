"""Smoke test for the proto-rset cosine ProtoPNet reference network."""

import torch

from src.vision.reference_protopnet import (
    SpatialPoolBackbone,
    build_cosine_protopnet,
    class_colored_images,
    fit_reference_protopnet,
    predict_logits,
    project_prototypes,
    prototype_similarities,
)


def test_default_backbone_returns_class_logits():
    images, _ = class_colored_images(2, images_per_class=2, image_size=32, seed=0)
    model = build_cosine_protopnet(num_classes=2, num_prototypes_per_class=1, image_size=(3, 32, 32))
    logits = predict_logits(model, images)
    assert logits.shape == (images.shape[0], 2)
    assert torch.isfinite(logits).all()


def test_cosine_protopnet_forward_and_projection():
    torch.manual_seed(0)
    num_classes = 2
    prototypes_per_class = 2
    images, labels = class_colored_images(num_classes, images_per_class=4, image_size=32, seed=0)
    model = build_cosine_protopnet(
        num_classes=num_classes,
        num_prototypes_per_class=prototypes_per_class,
        backbone=SpatialPoolBackbone(),
        image_size=(3, 32, 32),
        num_addon_layers=0,
        proto_channel_multiplier=0.0,
    )

    logits = predict_logits(model, images)
    assert logits.shape == (images.shape[0], num_classes)

    project_prototypes(model, images, labels, class_specific=True)
    similarities = prototype_similarities(model, images)
    assert similarities.shape == (images.shape[0], num_classes * prototypes_per_class)
    assert torch.isfinite(similarities).all()

    own_class = similarities[:4, :prototypes_per_class].mean()
    other_class = similarities[:4, prototypes_per_class:].mean()
    assert float(own_class) > 0.9
    assert float(own_class) > float(other_class) + 0.5


def test_reference_protopnet_learns_synthetic_classes():
    torch.manual_seed(0)
    images, labels = class_colored_images(num_classes=2, images_per_class=8, image_size=32, seed=1)
    model = build_cosine_protopnet(
        num_classes=2,
        num_prototypes_per_class=2,
        backbone=SpatialPoolBackbone(),
        image_size=(3, 32, 32),
        num_addon_layers=0,
        proto_channel_multiplier=0.0,
    )
    metrics = fit_reference_protopnet(
        model,
        images,
        labels,
        warm_steps=5,
        last_layer_steps=20,
        learning_rate=1e-2,
    )
    assert metrics["final_loss"] < metrics["initial_loss"]
    assert metrics["accuracy"] >= 0.9
