import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50


def resnet50_full_deeplabv3(cfg):
    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
        num_classes=19,
        aux_loss=False,
    )
    #nothing is freezed
    for param in model.backbone.parameters():
        param.requires_grad = True
    return model

if __name__ == "__main__":
    model = resnet50_full_deeplabv3(None)
    print(model)