from src.models.encoder import Encoder
from src.models.decoder import Decoder
import torch
import torch.nn as nn

class M4Conf(nn.Module):
    def __init__(self, pretrained=True, skip_channels=(64, 64, 128, 256, 512), max_depth=80.0):
        super().__init__()
        self.max_depth = max_depth
        self.enc1 = Encoder(in_channels=5, pretrained=pretrained)
        self.dec1 = Decoder(tuple(skip_channels), 2)

        self.enc2 = Encoder(in_channels=3, pretrained=pretrained)
        self.dec2 = Decoder(tuple(skip_channels), 2)
    
    def forward(self, rgb, lidar):
        x = torch.cat([rgb, lidar], dim=1)
        out1 = self.dec1(self.enc1(x))
        d1 = torch.sigmoid(out1[:, 0:1]) * self.max_depth
        c1 = out1[:, 1:2]

        d1_norm = d1 / self.max_depth
        y = torch.cat([lidar, d1_norm], dim=1)
        out2 = self.dec2(self.enc2(y))
        d2 = torch.sigmoid(out2[:, 0:1]) * self.max_depth
        c2 = out2[:, 1:2]

        weights = torch.softmax(torch.cat([c1, c2], dim=1), dim=1)
        w1, w2 = weights[:, 0:1], weights[:, 1:2]
        d = w1 * d1 + w2 * d2

        return {
            "d": d,
            "d1": d1,
            "d2": d2,
            "c1": c1,
            "c2": c2,
        }
    
def build_model(cfg):
    """build m4 model from yaml config"""
    pretrained = cfg["model"].get("pretrained", True)
    skip_channels = cfg["model"].get("skip_channels", (64, 64, 128, 256, 512))
    max_depth = cfg["data"].get("max_depth", 80.0)
    return M4Conf(pretrained, skip_channels, max_depth)
    