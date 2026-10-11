"""Fase 5 (entregable 5): Grad-CAM para StudyNet, por plano y por ventana 2.5D.

Para la etiqueta k se toma el gradiente del logit k respecto de la salida de `cam_layer`
(última etapa del codificador de cada plano). El peso de cada canal es la media espacial
del gradiente y el mapa es ReLU(suma de canales ponderados), uno por ventana. Los mapas de
un plano se normalizan con el máximo del estudio, así que las ventanas que casi no aportan
(poca atención) quedan oscuras. Implementación propia (Selvaraju et al., 2017).
"""
import numpy as np
import torch
import torch.nn.functional as F

from src.models.mil import StudyNet
from src.train.augment import prepare
from src.train.data import window_indices


def load_model(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ck["cfg"], pretrained=False)  # los pesos vienen del checkpoint
    model = StudyNet(cfg).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    return model, cfg


def study_input(vols, cfg):
    """{plano: volumen uint8 (n, H, W)} → ({plano: uint8 (1, D, 3, H, W)}, {plano: índices (D, 3)})."""
    x, idx = {}, {}
    for p in cfg["planes"]:
        v = vols[p]
        idx[p] = window_indices(len(v), cfg["n_windows"], cfg["window"])
        x[p] = torch.from_numpy(np.ascontiguousarray(v[idx[p]]))[None]
    return x, idx


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.acts, self.grads, self.handles = {}, {}, []
        for p, enc in model.encoders.items():
            self.handles.append(enc.cam_layer.register_forward_hook(self._hook(p)))

    def _hook(self, plane):
        def fn(module, inp, out):
            self.acts[plane] = out
            out.register_hook(lambda g: self.grads.__setitem__(plane, g))
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()

    def __call__(self, x, label_idx):
        """x: {plano: uint8 (1, D, 3, H, W)} en el dispositivo del modelo.

        Devuelve prob, {plano: cams (D, H, W) en [0, 1]}, {plano: energía por ventana (D,)},
        {plano: pesos de atención (D,) o None}.
        """
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            xin = {p: prepare(v.to(next(self.model.parameters()).device), train=False) for p, v in x.items()}
            logits, attn = self.model(xin, return_attn=True)
            logits[0, label_idx].backward()
        cams, energy = {}, {}
        for p in x:
            a, g = self.acts[p].detach().float(), self.grads[p].float()
            w = g.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((w * a).sum(1, keepdim=True))
            cam = F.interpolate(cam, size=x[p].shape[-2:], mode="bilinear", align_corners=False)[:, 0]
            energy[p] = cam.sum(dim=(1, 2)).cpu().numpy()
            cams[p] = (cam / cam.max().clamp_min(1e-8)).cpu().numpy()
        prob = torch.sigmoid(logits[0, label_idx]).item()
        att = {p: (a[0].detach().float().cpu().numpy() if a is not None else None) for p, a in attn.items()}
        return prob, cams, energy, att
