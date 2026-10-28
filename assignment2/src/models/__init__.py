"""Segmentation models for Assignment 2."""


def build_model(cfg):
    name = cfg.get("model")

    if name == "deeplabv3_resnet50":
        from src.models.deeplabv3 import resnet_deeplabv3
        return resnet_deeplabv3(cfg)

    if name == "unet":
        from src.models.unet import build_unet
        return build_unet(cfg)

    raise ValueError(f"Unknown model {name!r}. Choose 'deeplabv3_resnet50' or 'unet'.")
