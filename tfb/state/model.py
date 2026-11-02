"""Fully convolutional U-Net for (time x link) traffic-state inpainting.

Input: (B, C, T, L) feature grid; output: (B, 2, T, L) residuals (speed/100,
flow-per-lane/2000) over the linear-interpolation baseline. Pre-activation
residual blocks with GroupNorm; strided-conv downsampling on both axes; the
bottleneck uses dilated convolutions to widen the temporal receptive field.
"""

from __future__ import annotations

import torch
from torch import nn


class ResBlock(nn.Module):
    """Pre-activation residual block: x + two (GroupNorm, GELU, dilated 3x3 conv) stages, channel count unchanged."""

    def __init__(self, ch: int, dilation: int = 1, groups: int = 8):
        super().__init__()
        self.n1 = nn.GroupNorm(groups, ch)
        self.n2 = nn.GroupNorm(groups, ch)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x plus the residual branch; same shape as x."""
        h = self.c1(self.act(self.n1(x)))
        h = self.c2(self.act(self.n2(h)))
        return x + h


class Level(nn.Sequential):
    """n_blocks ResBlocks of width ch in sequence, cycling through `dilations`."""

    def __init__(self, ch: int, n_blocks: int, dilations=(1,)):
        super().__init__(*[ResBlock(ch, dilations[i % len(dilations)]) for i in range(n_blocks)])


class UNet(nn.Module):
    """U-Net over (T, L). ``widths`` per level; the last level is the bottleneck."""

    def __init__(self, in_ch: int, widths=(32, 64, 128, 256), blocks=(2, 2, 2, 2), out_ch: int = 2, bottleneck_dilations=(1, 2, 4)):
        super().__init__()
        self.widths = widths
        self.stem = nn.Conv2d(in_ch, widths[0], 3, padding=1)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i, w in enumerate(widths):
            dil = bottleneck_dilations if i == len(widths) - 1 else (1,)
            self.enc.append(Level(w, blocks[i], dil))
            if i < len(widths) - 1:
                self.down.append(nn.Conv2d(w, widths[i + 1], 3, stride=2, padding=1))
        self.up = nn.ModuleList()
        self.fuse = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(widths) - 1, 0, -1):
            self.up.append(nn.Conv2d(widths[i], widths[i - 1], 3, padding=1))
            self.fuse.append(nn.Conv2d(2 * widths[i - 1], widths[i - 1], 1))
            self.dec.append(Level(widths[i - 1], blocks[i - 1]))
        self.head_norm = nn.GroupNorm(8, widths[0])
        self.head = nn.Conv2d(widths[0], out_ch, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.n_down = len(widths) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_ch, T, L) -> (B, out_ch, T, L): encoder with skips, decoder resized to each skip, norm-GELU-conv head."""
        h = self.stem(x)
        skips = []
        for i, level in enumerate(self.enc):
            h = level(h)
            if i < self.n_down:
                skips.append(h)
                h = self.down[i](h)
        for up, fuse, dec, skip in zip(self.up, self.fuse, self.dec, reversed(skips)):
            h = nn.functional.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = up(h)
            h = dec(fuse(torch.cat([h, skip], dim=1)))
        return self.head(nn.functional.gelu(self.head_norm(h)))


def build_model(cfg: dict, in_ch: int) -> nn.Module:
    """UNet from a model config (widths, blocks, bottleneck_dilations, out_ch) with net.res_scale set (default (100, 2000))."""
    net = UNet(
        in_ch,
        widths=tuple(cfg.get("widths", (32, 64, 128, 256))),
        blocks=tuple(cfg.get("blocks", (2, 2, 2, 2))),
        bottleneck_dilations=tuple(cfg.get("bottleneck_dilations", (1, 2, 4))),
        out_ch=int(cfg.get("out_ch", 2)),
    )  # 3 = speed, flow, queue-probability logit
    net.res_scale = tuple(cfg.get("res_scale", (100.0, 2000.0)))  # physical units per unit head output
    return net
