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

if __name__ == "__main__":
    model = M3Early(pretrained=False)
    print("encoder:", model.encoder.count_parameters())
    print("decoder:", model.decoder.count_parameters())
    print("params:", sum(p.numel() for p in model.parameters()))