"""Pérdida multietiqueta con pesos por elemento (máscara o confianza).

- bce: entropía cruzada binaria, con pos_weight opcional para los hallazgos raros.
- focal: la misma BCE multiplicada por (1 - p_t)^gamma (Lin et al., 2017).
"""
import numpy as np
import torch
import torch.nn.functional as F


def auto_pos_weight(y, w, max_pw=10.0):
    """neg / pos por etiqueta contando solo los elementos con peso > 0, acotado a [1, max_pw]."""
    known = w > 0
    pos = ((y == 1) & known).sum(0).astype(np.float32)
    neg = ((y == 0) & known).sum(0).astype(np.float32)
    return np.clip(neg / np.maximum(pos, 1), 1.0, max_pw).astype(np.float32)


def multilabel_loss(logits, y, w, kind="bce", pos_weight=None, gamma=2.0):
    logits = logits.float()
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
    if kind == "focal":
        p = torch.sigmoid(logits)
        pt = p * y + (1 - p) * (1 - y)
        loss = loss * (1 - pt).pow(gamma)
    elif kind != "bce":
        raise ValueError(f"loss desconocida {kind}")
    return (loss * w).sum() / w.sum().clamp_min(1.0)
