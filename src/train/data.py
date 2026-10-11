"""Fase 4: lectura de rsna-knee-subset-v1 y armado de ventanas 2.5D por estudio.

Cada volumen (n, 256, 256) uint8 se reduce a D ventanas de `window` cortes contiguos
(los cortes van como canales). Los centros se reparten de forma uniforme en la serie;
en entrenamiento se desplazan todos juntos al azar como aumento.

Objetivos por split:
- train y val: y_ (política POS = 1) con pesos según la estrategia de ruido.
- test: los 58 dorados (gold_*), todos con peso 1.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.select_subset import LABELS


def load_studies(data_dir):
    df = pd.read_csv(Path(data_dir) / "subset_studies.csv")
    assert df.StudyInstanceUID.is_unique
    return df


def label_weights(df, noise="mask", highconf_thr=0.9):
    """Peso por estudio y etiqueta (N, 12) según la estrategia de manejo de ruido.

    - mask: 1 donde la etiqueta es conocida (UNC y WEAK quedan en 0).
    - conf: la confianza del extractor (POS 0.9 a 0.95, NEG 0.9, MIN 0.7, NONE y MILD 0.6).
    - highconf: 1 solo si conf >= highconf_thr; con 0.9 quedan POS y NEG explícitos y se
      descartan los "no mencionados".
    """
    mask = df[[f"mask_{l}" for l in LABELS]].to_numpy(np.float32)
    conf = df[[f"conf_{l}" for l in LABELS]].to_numpy(np.float32)
    if noise == "mask":
        return mask
    if noise == "conf":
        return mask * conf
    if noise == "highconf":
        return mask * (conf >= highconf_thr - 1e-6)
    raise ValueError(f"noise desconocido {noise}")


def targets(df, split, noise="mask", highconf_thr=0.9):
    """(y, w) como arreglos (N, 12) float32 para las filas de `df` (todas del mismo split)."""
    if split == "test":
        y = df[[f"gold_{l}" for l in LABELS]].to_numpy(np.float32)
        assert not np.isnan(y).any(), "test sin etiqueta dorada"
        return y, np.ones_like(y)
    y = df[[f"y_{l}" for l in LABELS]].to_numpy(np.float32)
    return y, label_weights(df, noise, highconf_thr)


def window_indices(n, n_windows=16, window=3, jitter=0.0, rng=None):
    """Índices (n_windows, window) de cortes. jitter en fracción del paso entre centros."""
    half = window // 2
    lo, hi = half, max(half, n - 1 - half)
    centers = np.linspace(lo, hi, n_windows)
    if jitter and rng is not None:
        step = (hi - lo) / max(n_windows - 1, 1)
        centers = centers + rng.uniform(-jitter, jitter) * step
    centers = np.clip(np.round(centers), lo, hi).astype(int)
    idx = centers[:, None] + np.arange(-half, half + 1)[None, :]
    return np.clip(idx, 0, n - 1)


class KneeStudies(Dataset):
    """Devuelve {"x": {plano: uint8 (D, window, H, W)}, "y": (12,), "w": (12,), "i": índice}."""

    PLANE_FILE = {"sag": "sag", "cor": "cor"}

    def __init__(self, df, data_dir, split, planes=("sag",), n_windows=16, window=3,
                 noise="mask", highconf_thr=0.9, train=False, cache=None, seed=0):
        self.df = df.reset_index(drop=True)
        self.dir = Path(data_dir)
        self.planes, self.n_windows, self.window, self.train = list(planes), n_windows, window, train
        self.y, self.w = targets(self.df, split, noise, highconf_thr)
        self.cache = {} if cache is None else cache
        self.rng = np.random.default_rng(seed)
        for uid in self.df.StudyInstanceUID:
            for p in self.planes:
                key = f"{uid}_{self.PLANE_FILE[p]}"
                if key not in self.cache:
                    self.cache[key] = np.load(self.dir / "vol" / f"{key}.npy")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        uid = self.df.StudyInstanceUID.iat[i]
        x = {}
        for p in self.planes:
            v = self.cache[f"{uid}_{self.PLANE_FILE[p]}"]
            idx = window_indices(len(v), self.n_windows, self.window,
                                 jitter=0.5 if self.train else 0.0, rng=self.rng)
            x[p] = torch.from_numpy(np.ascontiguousarray(v[idx]))
        return {"x": x, "y": torch.from_numpy(self.y[i]), "w": torch.from_numpy(self.w[i]), "i": i}


def worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)
