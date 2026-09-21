"""Model "timm_classifier": an ImageNet-pretrained CNN/ViT from timm with a fresh 4-way head, the
conventional classification baseline (EfficientNet-B0 in the BRISC paper)."""
from ...registry import MODELS


@MODELS.register("timm_classifier")
def build_timm_classifier(cfg: dict, num_classes: int = 4, **kwargs):
    import timm

    return timm.create_model(cfg.get("arch", "efficientnet_b0"), pretrained=bool(cfg.get("pretrained", True)),
                             num_classes=num_classes, drop_rate=float(cfg.get("drop_rate", 0.2)))
