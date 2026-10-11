"""Separa los 58 estudios dorados en ajuste (tune) y prueba (test), estratificado por idioma.

Las reglas del extractor solo se afinan mirando `tune`; `test` se usa una vez para medir.

Uso: python -m src.labels.split_gold
"""
import numpy as np
import pandas as pd

from src.labels.common import DATA, LABELS, load_train

SEED = 42
N_TUNE = 20


def main():
    df = load_train()
    gold = df[df[LABELS].notna().all(axis=1)].copy()
    rng = np.random.default_rng(SEED)

    tune_ids = []
    for lang, g in gold.groupby("lang"):
        k = max(1, round(len(g) * N_TUNE / len(gold))) if len(g) >= 2 else 0
        tune_ids += list(rng.choice(g["StudyInstanceUID"].to_numpy(), size=k, replace=False))

    gold["split"] = np.where(gold["StudyInstanceUID"].isin(tune_ids), "tune", "test")
    gold[["StudyInstanceUID", "lang", "split"]].to_csv(DATA / "gold_split.csv", index=False)

    print(pd.crosstab(gold["lang"], gold["split"], margins=True))
    print("\nPositivos por etiqueta:")
    print(gold.groupby("split")[LABELS].sum().T.astype(int))


if __name__ == "__main__":
    main()
