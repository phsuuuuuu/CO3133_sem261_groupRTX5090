import torch 
import torch.nn as nn
import torch.optim as optim

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 19, base_channels: int = 32):
        super().__init__()
        self.base_c = base_channels

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(base_channels, base_channels * 2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(base_channels * 2, base_channels * 4)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(base_channels * 4, base_channels * 8)
        )

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(base_channels * 8, base_channels * 16)
        )

        self.up1 = nn.ConvTranspose2d(
            base_channels * 16, base_channels * 8, kernel_size=2, stride=2
        )

        self.conv_up1 = DoubleConv(base_channels * 16, base_channels * 8)

        self.up2 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2
        )

        self.conv_up2 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up3 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )

        self.conv_up3 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up4 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.conv_up4 = DoubleConv(base_channels * 2, base_channels)

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"                
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        b = self.bottleneck(x4)

        d1 = self.up1(b)
        d1 = torch.cat([d1, x4], dim=1)
        d1 = self.conv_up1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, x3], dim=1)
        d2 = self.conv_up2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.conv_up3(d3)

        d4 = self.up4(d3)
        d4 = torch.cat([d4, x1], dim=1)
        d4 = self.conv_up4(d4)

        logits = self.outc(d4)
        return logits
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
def build_unet(cfg: dict) -> UNet:
    base_channels = cfg.get("base_channels", 32)
    num_classes = cfg.get("num_classes", 19)
    in_channels = cfg.get("in_channels", 3)
    return UNet(in_channels=in_channels, out_channels=num_classes, base_channels=base_channels)