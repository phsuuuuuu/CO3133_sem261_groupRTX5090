from src.models.encoder import Encoder
from src.models.decoder import Decoder
import torch
import torch.nn as nn

class M1RGB(nn.Module):
    def __init__(self, pretrained=True, skip_channels=(64, 64, 128, 256, 512), max_depth=80.0):
        super().__init__()
        self.max_depth = float(max_depth)
        self.encoder = Encoder(in_channels=3, pretrained=pretrained)  # depth/80 + mask
        self.decoder = Decoder(skip_channels=tuple(skip_channels), num_out=1)

    def forward(self, rgb, lidar):
        return {"d": torch.sigmoid(self.decoder(self.encoder(rgb))) * self.max_depth}

def build_model(cfg):
    """build m1 model from yaml config"""
    pretrained = cfg["model"].get("pretrained", True)
    skip_channels = cfg["model"].get("skip_channels", (64, 64, 128, 256, 512))
    max_depth = cfg["data"].get("max_depth", 80.0)
    return M1RGB(pretrained, skip_channels, max_depth)

if __name__ == "__main__":
    model = M1RGB(pretrained=False)
    print("encoder:", model.encoder.count_parameters())
    print("decoder:", model.decoder.count_parameters())
    print("total  :", sum(p.numel() for p in model.parameters()))
