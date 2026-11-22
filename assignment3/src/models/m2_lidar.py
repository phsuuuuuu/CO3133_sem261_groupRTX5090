from src.models.encoder import Encoder
from src.models.decoder import Decoder 
import torch
import torch.nn as nn

class M2Lidar(nn.Module):
    """model 2: 2-ch lidar to depth in 0-max_depth m"""

    def __init__(self, pretrained=True, skip_channels=(64, 64, 128, 256, 512), max_depth=80.0):
        super().__init__()
        self.max_depth = float(max_depth)
        self.encoder = Encoder(in_channels=2, pretrained=pretrained)  # depth/80 + mask
        self.decoder = Decoder(skip_channels=tuple(skip_channels), num_out=1)

    def forward(self, rgb, lidar):
        # keep rgb arg per contract; sigmoid bounds depth to 0-max_depth
        return {"d": torch.sigmoid(self.decoder(self.encoder(lidar))) * self.max_depth}


def build_model(cfg):
    """build m2 model from yaml config"""
    pretrained = cfg["model"].get("pretrained", True)
    skip_channels = cfg["model"].get("skip_channels", (64, 64, 128, 256, 512))
    max_depth = cfg["data"].get("max_depth", 80.0)
    return M2Lidar(pretrained, skip_channels, max_depth)
    
