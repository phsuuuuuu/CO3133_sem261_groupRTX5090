from torchvision.models import resnet34, ResNet34_Weights
import torch.nn as nn
import torch

class Encoder(nn.Module):
    """resnet34 encoder returning 5 skip feature maps"""

    def __init__(self, in_channels, pretrained=True):
        super().__init__()
        weight = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet34(weights=weight)

        # adapt conv1 for 2 or 5 channels
        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                w = old_conv.weight
                mean_w = w.mean(dim=1, keepdim=True)  # rgb mean filter for lidar channels

                with torch.no_grad():
                    if in_channels > 3:
                        new_conv.weight[:, :3] = w
                        new_conv.weight[:, 3:] = mean_w.repeat(1, in_channels - 3, 1, 1)
                    if in_channels < 3:
                        new_conv.weight[:] = mean_w.repeat(1, in_channels, 1, 1)

            backbone.conv1 = new_conv

        # drop avgpool and fc
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        """return 5 feature maps: channels 64 64 128 256 512, strides 2 4 8 16 32"""
        # f0 (B,  64, H/2,  W/2 )
        # f1 (B,  64, H/4,  W/4 )
        # f2 (B, 128, H/8,  W/8 )
        # f3 (B, 256, H/16, W/16)
        # f4 (B, 512, H/32, W/32)
        f0 = self.relu(self.bn1(self.conv1(x)))
        f1 = self.layer1(self.maxpool(f0))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f0, f1, f2, f3, f4]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


