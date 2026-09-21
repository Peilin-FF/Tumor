"""Model "smp_unet": U-Net with an ImageNet-pretrained encoder (segmentation_models_pytorch), the
conventional supervised segmentation baseline. Returns a module mapping 3xSxS -> 1xSxS logits."""
from ...registry import MODELS


def _disable_hub_mixin():
    """smp 0.3.4 models inherit huggingface_hub's PyTorchModelHubMixin, whose __new__ expects a newer
    hub than transformers 4.31 (MedPLIB) allows. The mixin only serves pushing models to the Hub."""
    try:
        import huggingface_hub.hub_mixin as hm
        hm.PyTorchModelHubMixin.__new__ = lambda cls, *a, **k: object.__new__(cls)
    except Exception:
        pass


@MODELS.register("smp_unet")
def build_smp_unet(cfg: dict, **kwargs):
    _disable_hub_mixin()
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder_name=cfg.get("encoder", "tu-resnet34"), encoder_weights=cfg.get("encoder_weights", "imagenet"),
                    in_channels=3, classes=1, decoder_attention_type=cfg.get("decoder_attention"))
