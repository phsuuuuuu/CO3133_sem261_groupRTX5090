"""Segmentation models for Assignment 2."""


def build_model(cfg):
    name = cfg.get("model")

    if name == "deeplabv3_resnet50":
        from src.models.deeplabv3 import resnet_deeplabv3
        return resnet_deeplabv3(cfg)

    if name == "unet":
        from src.models.unet import build_unet
        return build_unet(cfg)

    if name == "resnet_layer3_4_finetune":
        from src.models.partial_resnet50_deeplabv3 import resnet_layer3_4_finetune
        return resnet_layer3_4_finetune(cfg)
    if name == "resnet50_full_deeplabv3":
        from src.models.resnet50_full_deeplabv3 import resnet50_full_deeplabv3
        return resnet50_full_deeplabv3(cfg)
    raise ValueError(f"Unknown model {name!r}. Choose 'deeplabv3_resnet50' or 'unet'.")
