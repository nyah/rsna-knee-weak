"""Fase 5 (entregables 3 y 7): evaluación de corridas a partir de preds_val.csv y preds_test.csv.

- AUC por etiqueta y macro en los 58 dorados, con IC95 por bootstrap de estudios.
- Bootstrap pareado de la diferencia de AUC macro entre corridas (mismos remuestreos).
- Umbral por etiqueta elegido en val (máximo F1 con las etiquetas débiles) y aplicado a test:
  precisión, recall y F1. No se usa 0.5 porque pos_weight desplaza las probabilidades.
- AUC en el subconjunto de 38 dorados de test del extractor, para compararlo con las reglas (0.839).
- Desglose por idioma (AUC micro sobre las celdas estudio × etiqueta) para el entregable 7.

Uso (desde rsna-knee-weak/):
    .venv/bin/python -m src.eval.evaluate --runs outputs/runs_v1/resnet18_base outputs/runs_v1/scratch_base \
        --out outputs/eval_v1
    .venv/bin/python -m src.eval.evaluate --runs outputs/runs_v1/* --ref resnet18_base --out outputs/eval_ablation_v1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.data.select_subset import LABELS

N_BOOT, SEED = 2000, 42


def load_run(d):
    d = Path(d)
    return {"name": d.name, "val": pd.read_csv(d / "preds_val.csv"), "test": pd.read_csv(d / "preds_test.csv")}


def arrays(df):
    y = df[[f"y_{l}" for l in LABELS]].to_numpy(float)
    p = df[[f"p_{l}" for l in LABELS]].to_numpy(float)
    w = df[[f"w_{l}" for l in LABELS]].to_numpy(float)
    return y, p, w


def auc_per_label(y, p, w=None):
    w = np.ones_like(y) if w is None else w
    out = np.full(y.shape[1], np.nan)
    for j in range(y.shape[1]):
        m = w[:, j] > 0
        if 0 < y[m, j].sum() < m.sum():
            out[j] = roc_auc_score(y[m, j], p[m, j])
    return out


def boot_indices(n, n_boot=N_BOOT, seed=SEED):
    return np.random.default_rng(seed).integers(0, n, (n_boot, n))


def bootstrap_aucs(y, p, idx):
    """(n_boot, 12) AUC por remuestreo; NaN si en ese remuestreo la etiqueta tiene una sola clase."""
    return np.stack([auc_per_label(y[i], p[i]) for i in idx])


def ci(a):
    a = a[~np.isnan(a)]
    return (np.percentile(a, 2.5), np.percentile(a, 97.5)) if len(a) else (np.nan, np.nan)


def best_f1_thresholds(y, p, w):
    """Umbral por etiqueta que maximiza F1 en val (solo elementos con peso > 0)."""
    th = np.full(y.shape[1], 0.5)
    for j in range(y.shape[1]):
        m = w[:, j] > 0
        yj, pj = y[m, j], p[m, j]
        if yj.sum() == 0:
            continue
        cands = np.unique(pj)
        f1s = [f1_score(yj, pj >= t, zero_division=0) for t in cands]
        th[j] = cands[int(np.argmax(f1s))]
    return th


def prf_at(y, p, th):
    rows = {}
    for j, lab in enumerate(LABELS):
        pred = p[:, j] >= th[j]
        rows[lab] = {"thr": th[j], "precision": precision_score(y[:, j], pred, zero_division=0),
                     "recall": recall_score(y[:, j], pred, zero_division=0),
                     "f1": f1_score(y[:, j], pred, zero_division=0), "n_pred_pos": int(pred.sum())}
    return pd.DataFrame(rows).T


def by_language(df, th):
    y, p, _ = arrays(df)
    rows = []
    for lang, g in df.groupby("lang"):
        i = g.index.values
        yy, pp = y[i].ravel(), p[i].ravel()
        pred = (p[i] >= th).ravel()
        rows.append({"lang": lang, "n_studies": len(i), "pos_cells": int(yy.sum()),
                     "auc_micro": roc_auc_score(yy, pp) if 0 < yy.sum() < len(yy) else np.nan,
                     "acc_at_thr": float((pred == yy).mean()), "f1_at_thr": f1_score(yy, pred, zero_division=0)})
    return pd.DataFrame(rows).sort_values("n_studies", ascending=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--ref", help="corrida de referencia: compara cada corrida solo contra ella (y por etiqueta)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [load_run(d) for d in a.runs]

    test0 = runs[0]["test"]
    for r in runs[1:]:
        assert (r["test"].StudyInstanceUID.values == test0.StudyInstanceUID.values).all(), "test en distinto orden"
    idx = boot_indices(len(test0))
    sub38 = (test0["gold_split"] == "test").values

    per_label, summary, boots, boots_lab, prf_all, lang_all = {}, [], {}, {}, {}, []
    for r in runs:
        y, p, _ = arrays(r["test"])
        yv, pv, wv = arrays(r["val"])
        auc = auc_per_label(y, p)
        b = bootstrap_aucs(y, p, idx)
        boots_lab[r["name"]] = b
        boots[r["name"]] = np.nanmean(b, 1)
        lo, hi = np.array([ci(b[:, j]) for j in range(len(LABELS))]).T
        per_label[r["name"]] = pd.DataFrame({"auc": auc, "ci_lo": lo, "ci_hi": hi,
                                             "pos": y.sum(0).astype(int)}, index=LABELS)
        th = best_f1_thresholds(yv, pv, wv)
        prf = prf_at(y, p, th)
        prf_all[r["name"]] = prf
        mlo, mhi = ci(boots[r["name"]])
        auc38 = np.nanmean(auc_per_label(y[sub38], p[sub38]))
        summary.append({"run": r["name"], "test_auc_macro": np.nanmean(auc), "ci_lo": mlo, "ci_hi": mhi,
                        "test38_auc_macro": auc38, "val_auc_macro": np.nanmean(auc_per_label(yv, pv, wv)),
                        "precision_macro": prf.precision.mean(), "recall_macro": prf.recall.mean(),
                        "f1_macro": prf.f1.mean()})
        lang_all.append(by_language(r["test"].reset_index(drop=True), th).assign(run=r["name"]))

    summary = pd.DataFrame(summary).set_index("run")
    pl = pd.concat(per_label, axis=1)
    prf = pd.concat(prf_all, axis=1)
    lang = pd.concat(lang_all)
    pairs = []
    names = [r["name"] for r in runs]
    if a.ref:
        combos = [(n, a.ref) for n in names if n != a.ref]
    else:
        combos = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    lab_rows = []
    for na, nb in combos:
        d = boots[na] - boots[nb]
        lo, hi = ci(d)
        pairs.append({"a": na, "b": nb, "diff_auc_macro": summary.test_auc_macro[na] - summary.test_auc_macro[nb],
                      "ci_lo": lo, "ci_hi": hi, "p_a_le_b": float((d <= 0).mean())})
        if a.ref:
            dl = boots_lab[na] - boots_lab[nb]
            for j, lab in enumerate(LABELS):
                llo, lhi = ci(dl[:, j])
                lab_rows.append({"a": na, "b": nb, "label": lab,
                                 "diff_auc": per_label[na].auc[lab] - per_label[nb].auc[lab], "ci_lo": llo, "ci_hi": lhi})
    pairs = pd.DataFrame(pairs)
    if lab_rows:
        pd.DataFrame(lab_rows).to_csv(out / "paired_per_label_vs_ref.csv", index=False)

    summary.to_csv(out / "summary.csv")
    pl.to_csv(out / "auc_per_label.csv")
    prf.to_csv(out / "prf_per_label.csv")
    pairs.to_csv(out / "paired_bootstrap.csv", index=False)
    lang.to_csv(out / "by_language.csv", index=False)
    (out / "config.json").write_text(json.dumps({"runs": a.runs, "ref": a.ref, "n_boot": N_BOOT, "seed": SEED}, indent=1))

    pd.set_option("display.width", 200)
    print("== Resumen (58 dorados; IC95 bootstrap) ==\n", summary.round(3))
    print("\n== AUC por etiqueta ==\n", pl.round(3))
    print("\n== Precisión, recall y F1 con umbral de val ==\n", prf.round(3))
    print("\n== Bootstrap pareado ==\n", pairs.round(3))
    print("\n== Por idioma ==\n", lang.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
