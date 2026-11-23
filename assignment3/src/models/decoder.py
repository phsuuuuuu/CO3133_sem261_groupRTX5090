import torch
import torch.nn as nn

 # (B, in,  h, w) -> (B, out, h, w)
class DoubleConv(nn.Module):
    """[Conv3x3 -> BN -> ReLU] x 2, same as A2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    "channels 64 64 128 256 512 from encoder from f0 f1 f2 f3 f4"
    # f0 (B,  64, H/2,  W/2 )
    # f1 (B,  64, H/4,  W/4 )
    # f2 (B, 128, H/8,  W/8 )
    # f3 (B, 256, H/16, W/16)
    # f4 (B, 512, H/32, W/32)

    def __init__(self, skip_channels=(64, 64, 128, 256, 512), num_out = 1):
        super().__init__()
        FINAL_LAYER = 32
        c0, c1, c2, c3, c4 = skip_channels

    #stage1 : f4 -> f3 : c4 -> c3
        self.up1  = nn.ConvTranspose2d(c4, c3, kernel_size=2,stride=2)    # (B, 512, H/32, W/32) -> (B, 256, H/16, W/16)
        self.conv_up1 = DoubleConv(c3 * 2, c3)    # (B, 512, H/16, W/16) -> (B, 256, H/16, W/16)

    #stage2 : f3 -> f2 : c3 -> c2
        self.up2  = nn.ConvTranspose2d(c3, c2, kernel_size=2,stride=2)    # (B, 256, H/16, W/16) -> (B, 128, H/8, W/8)
        self.conv_up2 = DoubleConv(c2 * 2, c2)    # (B, 256, H/8, W/8) -> (B, 128, H/8, W/8)

    #stage3 : f2 -> f1 : c2 -> c1
        self.up3  = nn.ConvTranspose2d(c2, c1, kernel_size=2,stride=2)    # (B, 128, H/8, W/8) -> (B, 64, H/4, W/4)
        self.conv_up3 = DoubleConv(c1 * 2, c1)    # (B, 128, H/4, W/4) -> (B, 64, H/4, W/4)

    #stage4 : f1 -> f0 : c1 -> c0
        self.up4  = nn.ConvTranspose2d(c1, c0, kernel_size=2,stride=2)    # (B, 64, H/4, W/4) -> (B, 64, H/2, W/2)
        self.conv_up4 = DoubleConv(c0 * 2, c0)    # (B, 128, H/2, W/2) -> (B, 64, H/2, W/2)

    #stage5 : f0 -> full size : c0 -> FINAL_LAYER
        self.up5  = nn.ConvTranspose2d(c0, FINAL_LAYER, kernel_size=2,stride=2)    # (B, 64, H/2, W/2) -> (B, 32, H, W)
        self.conv_up5 = DoubleConv(FINAL_LAYER, FINAL_LAYER)    # (B, 32, H, W) -> (B, 32, H, W)

    #head: FINAL_LAYER -> num_out
        self.head = nn.Conv2d(FINAL_LAYER, num_out, kernel_size=1)    # (B, 32, H, W) -> (B, num_out, H, W)

    def forward(self, feats):
        f0, f1, f2, f3, f4 = feats

    #stage1 f4 + f3
        x = self.up1(f4)    # (B, 256, H/16, W/16)
        x = torch.cat([x, f3], dim=1)    # (B, 512, H/16, W/16)
        x = self.conv_up1(x)    # (B, 256, H/16, W/16)

    #stage2 f4 + f3 + f2
        x = self.up2(x)    # (B, 128, H/8, W/8)
        x = torch.cat([x, f2], dim=1)    # (B, 256, H/8, W/8)
        x = self.conv_up2(x)    # (B, 128, H/8, W/8)

    #stage3 f4 + f3 + f2 + f1
        x = self.up3(x)    # (B, 64, H/4, W/4)
        x = torch.cat([x, f1], dim=1)    # (B, 128, H/4, W/4)
        x = self.conv_up3(x)    # (B, 64, H/4, W/4)

    #stage4 f4 + f3 + f2 + f1 + f0
        x = self.up4(x)    # (B, 64, H/2, W/2)
        x = torch.cat([x, f0], dim=1)    # (B, 128, H/2, W/2)
        x = self.conv_up4(x)    # (B, 64, H/2, W/2)

    #stage5 to final layer
        x = self.up5(x)    # (B, 32, H, W)
        x = self.conv_up5(x)    # (B, 32, H, W)

    #head
        return self.head(x)    # (B, num_out, H, W)
