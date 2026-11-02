import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50


def resnet_layer3_4_finetune(cfg):
    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
        num_classes=19,
        aux_loss=False,
    )
    # Freeze all backbone parameters first
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Unfreeze layer 3 and 4, keep layer 1 and 2 frozen
    for param in model.backbone.layer3.parameters():
        param.requires_grad = True
    for param in model.backbone.layer4.parameters():
        param.requires_grad = True
    return model


if __name__ == "__main__":
    model = resnet_layer3_4_finetune(None)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable params count: {len(trainable)}")
