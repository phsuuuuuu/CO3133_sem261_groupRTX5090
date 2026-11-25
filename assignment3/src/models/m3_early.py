from src.models.encoder import Encoder
from src.models.decoder import Decoder
import torch
import torch.nn as nn
class M3Early(nn.Module):
    """model 3: early fusion, rgb + lidar stacked to 5-ch"""
    
    def __init__(self, pretrained=True, skip_channels=(64, 64, 128, 256, 512), max_depth=80.0):
        super().__init__()
        self.max_depth = float(max_depth)
        self.encoder = Encoder(in_channels=5, pretrained=pretrained)    # 3 rgb + 2 lidar (depth/80 + mask)
        self.decoder = Decoder(skip_channels=tuple(skip_channels), num_out=1)
    
    def forward(self, rgb, lidar):
        x = torch.cat([rgb, lidar], dim=1)    # (B, 3, H, W) + (B, 2, H, W) -> (B, 5, H, W)
        feats = self.encoder(x)    # (B, 5, H, W) -> [f0 ... f4]
        x = self.decoder(feats)    # (B, 1, H, W) raw
        d = torch.sigmoid(x) * self.max_depth    # (B, 1, H, W) metres 0..80
        return {"d": d}

def build_model(cfg):
    """build m3 model from yaml config"""
    pretrained = cfg["model"].get("pretrained", True)
    skip_channels = cfg["model"].get("skip_channels", (64, 64, 128, 256, 512))
    max_depth = cfg["data"].get("max_depth", 80.0)
    return M3Early(pretrained, skip_channels, max_depth)

class M3Split(nn.Module):
    """model 3 split stem: own conv1+bn1 for rgb and lidar, added before layer1"""

    def __init__(self, pretrained=True, skip_channels=(64, 64, 128, 256, 512), max_depth=80.0):
        super().__init__()
        self.max_depth = float(max_depth)
        self.encoder = Encoder(in_channels=3, pretrained=pretrained)    # rgb stem + resnet body
        lid = Encoder(in_channels=2, pretrained=pretrained)             # same init as m2, keep only its stem
        self.encoder.lid_conv1 = lid.conv1    # put inside encoder -> optimizer trains it
        self.encoder.lid_bn1 = lid.bn1        # own BatchNorm -> lidar not drowned by rgb
        self.decoder = Decoder(skip_channels=tuple(skip_channels), num_out=1)

    def forward(self, rgb, lidar):
        e = self.encoder
        f0 = e.relu(e.bn1(e.conv1(rgb)) + e.lid_bn1(e.lid_conv1(lidar)))    # (B, 64, H/2, W/2) fuse here
        f1 = e.layer1(e.maxpool(f0))    # (B, 64, H/4, W/4)
        f2 = e.layer2(f1)    # (B, 128, H/8, W/8)
        f3 = e.layer3(f2)    # (B, 256, H/16, W/16)
        f4 = e.layer4(f3)    # (B, 512, H/32, W/32)
        x = self.decoder([f0, f1, f2, f3, f4])    # (B, 1, H, W) raw
        d = torch.sigmoid(x) * self.max_depth    # (B, 1, H, W) metres 0..80
        return {"d": d}

def build_m3_split(cfg):
    """build m3 split-stem model from yaml config"""
    pretrained = cfg["model"].get("pretrained", True)
    skip_channels = cfg["model"].get("skip_channels", (64, 64, 128, 256, 512))
    max_depth = cfg["data"].get("max_depth", 80.0)
    return M3Split(pretrained, skip_channels, max_depth)

if __name__ == "__main__":
    model = M3Early(pretrained=False)
    print("encoder:", model.encoder.count_parameters())
    print("decoder:", model.decoder.count_parameters())
    print("params:", sum(p.numel() for p in model.parameters()))