import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50


def resnet_deeplabv3(cfg):
    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
        num_classes=19,
        aux_loss=False,
    )
    #no train in backbone but train in deeplabv3
    for param in model.backbone.parameters():
        param.requires_grad = False

    return model