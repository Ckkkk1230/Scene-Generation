from typing import Tuple
import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int, norm: bool = True) -> nn.Sequential:
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def deconv_block(in_ch: int, out_ch: int, dropout: bool = False) -> nn.Sequential:
    layers = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    """简化版U-Net生成器，用于条件到图像的映射。"""

    def __init__(self, in_channels: int = 1, out_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        # 下采样
        self.down1 = conv_block(in_channels, base_channels, norm=False)   # 128
        self.down2 = conv_block(base_channels, base_channels * 2)         # 64
        self.down3 = conv_block(base_channels * 2, base_channels * 4)     # 32
        self.down4 = conv_block(base_channels * 4, base_channels * 8)     # 16
        self.down5 = conv_block(base_channels * 8, base_channels * 8)     # 8
        self.down6 = conv_block(base_channels * 8, base_channels * 8)     # 4
        self.down7 = conv_block(base_channels * 8, base_channels * 8)     # 2
        self.down8 = conv_block(base_channels * 8, base_channels * 8)     # 1

        # 上采样
        self.up1 = deconv_block(base_channels * 8, base_channels * 8, dropout=True)
        self.up2 = deconv_block(base_channels * 16, base_channels * 8, dropout=True)
        self.up3 = deconv_block(base_channels * 16, base_channels * 8, dropout=True)
        self.up4 = deconv_block(base_channels * 16, base_channels * 8)
        self.up5 = deconv_block(base_channels * 16, base_channels * 4)
        self.up6 = deconv_block(base_channels * 8, base_channels * 2)
        self.up7 = deconv_block(base_channels * 4, base_channels)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))
        out = self.up8(torch.cat([u7, d1], dim=1))
        return out