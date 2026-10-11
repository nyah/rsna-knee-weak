"""Aumentos en GPU, iguales para todos los cortes de un mismo estudio y plano.

Sin flips: en sagital el flip horizontal cambia anterior por posterior y en coronal
cambia medial por lateral, lo que alteraría Medial y Lateral Meniscus/OA.
Se usan rotación leve, escala, traslación, contraste, brillo y gamma.
"""
import math

import torch
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def to_float(x):
    """uint8 (B, D, 3, H, W) → float en [0, 1]."""
    return x.float().div_(255)


def normalize(x):
    """Estandarización con la media y desviación de ImageNet (para ambos modelos)."""
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def random_affine(x, max_deg=10, scale=(0.9, 1.1), max_shift=0.06):
    b, d, c, h, w = x.shape
    dev = x.device
    ang = (torch.rand(b, device=dev) * 2 - 1) * math.radians(max_deg)
    s = torch.empty(b, device=dev).uniform_(*scale)
    t = (torch.rand(b, 2, device=dev) * 2 - 1) * max_shift * 2
    cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.stack([torch.stack([cos, -sin, t[:, 0]], 1), torch.stack([sin, cos, t[:, 1]], 1)], 1)
    theta = theta.repeat_interleave(d, 0)
    flat = x.reshape(b * d, c, h, w)
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    return F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False).view(b, d, c, h, w)


def random_intensity(x, contrast=(0.8, 1.2), brightness=0.1, gamma=(0.8, 1.25)):
    b = x.shape[0]
    shape = (b, 1, 1, 1, 1)
    dev = x.device
    g = torch.exp(torch.empty(shape, device=dev).uniform_(math.log(gamma[0]), math.log(gamma[1])))
    x = x.clamp(0, 1).pow(g)
    k = torch.empty(shape, device=dev).uniform_(*contrast)
    m = x.mean(dim=(1, 2, 3, 4), keepdim=True)
    x = (x - m) * k + m + torch.empty(shape, device=dev).uniform_(-brightness, brightness)
    return x.clamp(0, 1)


def prepare(x, train):
    """uint8 en el dispositivo → tensor normalizado, con aumentos si train."""
    x = to_float(x)
    if train:
        x = random_intensity(random_affine(x))
    return normalize(x)
