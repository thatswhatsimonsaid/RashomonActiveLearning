"""Confirm that proto-rset's cosine ProtoPNet imports and can project prototypes."""

from __future__ import annotations

from src.vision.reference_protopnet import (
    SpatialPoolBackbone,
    _ensure_proto_rset_on_path,
    build_cosine_protopnet,
    class_colored_images,
    predict_logits,
    project_prototypes,
    prototype_similarities,
)

_ensure_proto_rset_on_path()

import protopnet
from protopnet.models.vanilla_protopnet import VanillaProtoPNet


def main() -> None:
    images, labels = class_colored_images(num_classes=2, images_per_class=4, image_size=32, seed=0)
    model = build_cosine_protopnet(
        num_classes=2,
        num_prototypes_per_class=2,
        backbone=SpatialPoolBackbone(),
        image_size=(3, 32, 32),
        num_addon_layers=0,
        proto_channel_multiplier=0.0,
    )
    if not isinstance(model, VanillaProtoPNet):
        raise TypeError(f"Expected VanillaProtoPNet, got {type(model)}")

    project_prototypes(model, images, labels, class_specific=True)
    logits = predict_logits(model, images)
    similarities = prototype_similarities(model, images)
    own = float(similarities[:4, :2].mean())
    other = float(similarities[:4, 2:].mean())
    print(f"protopnet: {protopnet.__file__}")
    print(f"model: {type(model).__name__}")
    print(f"logits: {tuple(logits.shape)}")
    print(f"similarities: {tuple(similarities.shape)} own={own:.3f} other={other:.3f}")
    if own <= other:
        raise RuntimeError("Projected prototypes do not prefer their own class.")
    print("ProtoPNet check passed.")


if __name__ == "__main__":
    main()
