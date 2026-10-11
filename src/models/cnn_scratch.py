"""Modelo 1: CNN 2D propia que codifica un corte 2.5D (3 cortes contiguos como canales).

Bloques conv-BN-ReLU con atajo residual, cuatro etapas con stride 2 y GAP.
Implementada desde cero (sin torchvision) y entrenada sin pesos previos.
"""
import torch
from torch import nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, c_in, c_out, k=3, stride=1):
        super().__init__(
            nn.Conv2d(c_in, c_out, k, stride, k // 2, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


class ResBlock(nn.Module):
    """Dos convoluciones 3×3 con atajo; el atajo usa conv 1×1 cuando cambia la forma."""

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = ConvBNReLU(c_in, c_out, 3, stride)
        self.conv2 = nn.Sequential(nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out))
        self.short = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.short = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + self.short(x))


class ScratchCNN(nn.Module):
    """(N, 3, H, W) → (N, out_dim). Con 256×256 la última etapa queda en 8×8."""

    def __init__(self, in_ch=3, widths=(32, 64, 128, 256), blocks_per_stage=2):
        super().__init__()
        self.stem = nn.Sequential(ConvBNReLU(in_ch, widths[0], 5, 2), ConvBNReLU(widths[0], widths[0]))
        stages, c = [], widths[0]
        for w in widths:
            blocks = [ResBlock(c, w, stride=2)] + [ResBlock(w, w) for _ in range(blocks_per_stage - 1)]
            stages.append(nn.Sequential(*blocks))
            c = w
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = c
        self.cam_layer = self.stages[-1]  # capa objetivo para Grad-CAM
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return torch.flatten(self.pool(self.stages(self.stem(x))), 1)
