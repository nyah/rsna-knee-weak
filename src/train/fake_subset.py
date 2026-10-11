"""Genera un rsna-knee-subset falso (mismo formato) para probar el entrenamiento en el Mac.

Los estudios positivos en ACL llevan un cuadrado brillante en algunos cortes, así que un
modelo que funcione debe superar 0.5 de AUC en esa etiqueta.

Uso: python -m src.train.fake_subset --out /tmp/fake_subset --n 40 --size 64
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.select_subset import LABELS


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--size", type=int, default=64)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    out = Path(a.out)
    (out / "vol").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(a.n):
        uid = f"1.2.3.{i}"
        split = "train" if i < a.n * 0.6 else ("val" if i < a.n * 0.8 else "test")
        y = (rng.random(len(LABELS)) < 0.4).astype(int)
        y[0] = i % 2
        r = {"StudyInstanceUID": uid, "split": split, "is_gold": int(split == "test"), "lang": "en",
             "Manufacturer": "SIEMENS", "gold_split": "test" if split == "test" else np.nan}
        for j, lab in enumerate(LABELS):
            r[f"y_{lab}"] = y[j]
            r[f"mask_{lab}"] = int(rng.random() > 0.05)
            r[f"conf_{lab}"] = rng.choice([0.6, 0.9, 0.95])
            r[f"score_{lab}"] = float(y[j])
            r[f"gold_{lab}"] = y[j] if split == "test" else np.nan
        rows.append(r)
        for role in ("sag", "cor"):
            n = int(rng.integers(15, 40))
            v = rng.integers(0, 80, (n, a.size, a.size), dtype=np.uint8)
            if y[0]:
                s = a.size // 4
                v[n // 3: 2 * n // 3, s:2 * s, s:2 * s] = 230
            np.save(out / "vol" / f"{uid}_{role}.npy", v)
    pd.DataFrame(rows).to_csv(out / "subset_studies.csv", index=False)
    print(f"{a.n} estudios falsos en {out}")


if __name__ == "__main__":
    main()
