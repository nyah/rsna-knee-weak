"""Modelo 2: backbone de torchvision preentrenado en ImageNet, sin su capa de clasificación.

Solo se toman los pesos del extractor. La agregación por cortes y por plano y la cabeza
de 12 salidas están en `mil.py` y son propias.
"""
import torch
from torch import nn
from torchvision import models as tvm

BACKBONES = {
    "resnet18": (tvm.resnet18, 512),
    "resnet34": (tvm.resnet34, 512),
    "resnet50": (tvm.resnet50, 2048),
    "efficientnet_b0": (tvm.efficientnet_b0, 1280),
}


class TVBackbone(nn.Module):
    """(N, 3, H, W) → (N, out_dim)."""

    def __init__(self, name="resnet18", pretrained=True):
        super().__init__()
        if name not in BACKBONES:
            raise ValueError(f"backbone desconocido {name}; opciones {list(BACKBONES)}")
        fn, self.out_dim = BACKBONES[name]
        net = fn(weights="DEFAULT" if pretrained else None)
        if name.startswith("resnet"):
            self.features = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                          net.layer1, net.layer2, net.layer3, net.layer4)
        else:
            self.features = net.features
        self.cam_layer = self.features[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return torch.flatten(self.pool(self.features(x)), 1)
