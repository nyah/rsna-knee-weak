"""Modelo a nivel de estudio: codificador 2D compartido por cortes, agregación por plano y cabeza.

Entrada por plano: (B, D, 3, H, W), es decir D ventanas 2.5D de 3 cortes contiguos.
Cada ventana pasa por el mismo codificador, los D vectores se agregan (max, media o
atención, como en MRNet y en el MIL con atención de Ilse et al.), los planos se
concatenan y una capa lineal da 12 logits (una sigmoide independiente por hallazgo).
"""
import torch
from torch import nn

from src.models.cnn_scratch import ScratchCNN


class SlicePool(nn.Module):
    def __init__(self, dim, mode="attn", hidden=128):
        super().__init__()
        self.mode = mode
        if mode == "attn":
            self.att = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        elif mode not in ("max", "mean"):
            raise ValueError(f"pool desconocido {mode}")

    def forward(self, h):
        """h: (B, D, dim) → (B, dim), pesos por corte (B, D) o None."""
        if self.mode == "max":
            return h.max(1).values, None
        if self.mode == "mean":
            return h.mean(1), None
        a = torch.softmax(self.att(h).squeeze(-1).float(), dim=1).to(h.dtype)
        return (a.unsqueeze(-1) * h).sum(1), a


def make_encoder(cfg):
    if cfg["model"] == "scratch":
        return ScratchCNN(widths=tuple(cfg.get("widths", (32, 64, 128, 256))))
    from src.models.backbone_tl import TVBackbone  # torchvision solo hace falta aquí
    return TVBackbone(cfg["model"], pretrained=cfg.get("pretrained", True))


class StudyNet(nn.Module):
    def __init__(self, cfg, n_out=12):
        super().__init__()
        self.planes = list(cfg["planes"])
        self.encoders = nn.ModuleDict({p: make_encoder(cfg) for p in self.planes})
        dim = next(iter(self.encoders.values())).out_dim
        self.pools = nn.ModuleDict({p: SlicePool(dim, cfg.get("pool", "attn")) for p in self.planes})
        self.head = nn.Sequential(nn.Dropout(cfg.get("dropout", 0.3)), nn.Linear(dim * len(self.planes), n_out))

    def forward(self, x, return_attn=False):
        feats, attn = [], {}
        for p in self.planes:
            v = x[p]
            b, d = v.shape[:2]
            h = self.encoders[p](v.flatten(0, 1)).view(b, d, -1)
            z, a = self.pools[p](h)
            feats.append(z)
            attn[p] = a
        logits = self.head(torch.cat(feats, 1))
        return (logits, attn) if return_attn else logits

    def backbone_parameters(self):
        return [q for e in self.encoders.values() for q in e.parameters()]

    def set_backbone_trainable(self, flag):
        """Congela o libera los codificadores. Congelados, su BN queda en modo eval."""
        for e in self.encoders.values():
            for q in e.parameters():
                q.requires_grad_(flag)
            e.train(flag and self.training)
        self._bb_frozen = not flag

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "_bb_frozen", False):
            for e in self.encoders.values():
                e.eval()
        return self
