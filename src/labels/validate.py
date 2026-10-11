"""Acuerdo de las etiquetas débiles contra los dorados (entregables 1 y 7).

Uso:
    python -m src.labels.validate --split tune          # para ajustar reglas
    python -m src.labels.validate --split test          # medición final (una vez)
    python -m src.labels.validate --split tune --unc-as 1

`--unc-as` decide cómo se binariza -1 (incierto o degeneración meniscal): 0 por defecto.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support, roc_auc_score

from src.labels.common import DATA, LABELS, OUT, load_train


def metrics(y, yhat, score):
    p, r, f, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    return {
        "n": len(y), "pos_gold": int(y.sum()), "pos_weak": int(yhat.sum()),
        "acc": (y == yhat).mean(), "prec": p, "rec": r, "f1": f,
        "kappa": cohen_kappa_score(y, yhat) if len(set(y)) > 1 or len(set(yhat)) > 1 else np.nan,
        "auc": roc_auc_score(y, score) if 0 < y.sum() < len(y) else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["tune", "test", "all"], default="tune")
    ap.add_argument("--unc-as", type=int, choices=[0, 1], default=0)
    ap.add_argument("--mild-as", type=int, choices=[0, 1], default=0,
                    help="cómo binarizar un hallazgo descrito como leve o pequeño (estado MILD)")
    ap.add_argument("--errors", action="store_true", help="lista los desacuerdos por estudio")
    args = ap.parse_args()

    gold = load_train()
    gold = gold[gold[LABELS].notna().all(axis=1)]
    split = pd.read_csv(DATA / "gold_split.csv")[["StudyInstanceUID", "split"]]
    weak = pd.read_csv(OUT / "weak_labels.csv")
    df = gold.merge(split, on="StudyInstanceUID").merge(weak, on="StudyInstanceUID", suffixes=("", "_w"))
    if args.split != "all":
        df = df[df["split"] == args.split]
    for label in LABELS:
        df.loc[df[f"{label}__state"] == "MILD", f"{label}_w"] = args.mild_as

    rows = []
    for label in LABELS:
        y = df[label].astype(int).to_numpy()
        w = df[f"{label}_w"].replace(-1, args.unc_as).to_numpy()
        rows.append({"label": label, **metrics(y, w, df[f"{label}__score"].to_numpy())})
    tab = pd.DataFrame(rows).set_index("label")
    tab.loc["MACRO"] = tab.mean(numeric_only=True)
    tab["noise"] = 1 - tab["acc"]

    pd.set_option("display.width", 140)
    print(f"Split={args.split}  n={len(df)}  -1 como {args.unc_as}\n")
    print(tab[["n", "pos_gold", "pos_weak", "acc", "prec", "rec", "f1", "kappa", "auc", "noise"]].round(3))

    # acuerdo por idioma: todas las celdas estudio x etiqueta del idioma
    lang_rows = []
    for lang, g in df.groupby("lang"):
        y = g[LABELS].astype(int).to_numpy().ravel()
        w = g[[f"{l}_w" for l in LABELS]].replace(-1, args.unc_as).to_numpy().ravel()
        p, r, f, _ = precision_recall_fscore_support(y, w, average="binary", zero_division=0)
        lang_rows.append({"lang": lang, "studies": len(g), "acc": (y == w).mean(), "prec": p, "rec": r, "f1": f})
    print("\nPor idioma (celdas estudio x etiqueta):")
    print(pd.DataFrame(lang_rows).set_index("lang").round(3))

    OUT.mkdir(exist_ok=True)
    tab.to_csv(OUT / f"validation_{args.split}.csv")

    if args.errors:
        print("\nDesacuerdos (FN = dorado 1 y débil 0; FP = al revés):")
        for _, r in df.iterrows():
            errs = []
            for label in LABELS:
                g, w = int(r[label]), r[f"{label}_w"]
                wb = args.unc_as if w == -1 else w
                if g != wb:
                    errs.append(f"{'FN' if g else 'FP'}:{label}({r[f'{label}__state']})")
            if errs:
                print(f"  ...{r['StudyInstanceUID'][-6:]} [{r['lang']}] " + ", ".join(errs))


if __name__ == "__main__":
    main()
