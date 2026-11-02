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
    #unfreeze layer 3, 4 and head to fine tune the model, keep layer1  and 2 freezed
    for param in model.backbone.layer3.parameters():
        param.requires_grad = True
    for param in model.backbone.layer4.parameters():
            param.requires_grad = True
    return model

if __name__ == "__main__":
    model = resnet_layer3_4_finetune(None)
    print(model)