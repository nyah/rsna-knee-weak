"""Compara el extractor de reglas con variantes de embeddings, sobre los 38 dorados de test.

Variantes:
  R   reglas (score para AUC; binario con la política fijada: -1 -> 0, MILD -> 0)
  E1  embeddings + regresión logística entrenada con las etiquetas de R en los estudios no dorados
      (destilación; peso de muestra = confianza de la regla; se excluyen los -1 de esa etiqueta)
  E2  embeddings + regresión logística entrenada solo con los 20 dorados de tune
  E3  promedio del score de R y la probabilidad de E1

Intervalos al 95 % por bootstrap pareado sobre estudios (mismo remuestreo para todas las variantes).

Uso: python -m src.labels.compare [--emb outputs/emb_paraphrase-multilingual-MiniLM-L12-v2.npz]
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.labels.common import DATA, LABELS, OUT, load_train

SEED = 42
N_BOOT = 2000


def binarize_rules(weak, label):
    state = weak[f"{label}__state"]
    return (state == "POS").astype(int).to_numpy()


def logreg(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def per_label(y, prob, hard=None):
    """Métricas por etiqueta para matrices (n, 12). `hard`: binario propio; si falta, prob >= 0.5."""
    hard = (prob >= 0.5).astype(int) if hard is None else hard
    rows = []
    for k, label in enumerate(LABELS):
        yk, pk, hk = y[:, k], prob[:, k], hard[:, k]
        rows.append({
            "label": label,
            "acc": (yk == hk).mean(),
            "f1": f1_score(yk, hk, zero_division=0),
            "kappa": cohen_kappa_score(yk, hk) if len(set(yk)) > 1 or len(set(hk)) > 1 else np.nan,
            "auc": roc_auc_score(yk, pk) if 0 < yk.sum() < len(yk) else np.nan,
        })
    return pd.DataFrame(rows).set_index("label")


def macro(y, prob, hard=None):
    t = per_label(y, prob, hard)
    return t[["acc", "f1", "kappa", "auc"]].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=str(OUT / "emb_paraphrase-multilingual-MiniLM-L12-v2.npz"))
    ap.add_argument("--C", type=float, default=0.05, help="regularización de la regresión logística")
    args = ap.parse_args()

    df = load_train()
    weak = pd.read_csv(OUT / "weak_labels.csv")
    split = pd.read_csv(DATA / "gold_split.csv")[["StudyInstanceUID", "split"]]
    emb = np.load(args.emb, allow_pickle=True)
    X_all = pd.DataFrame(emb["X"], index=emb["uids"])

    df = df.merge(weak, on="StudyInstanceUID", suffixes=("", "_w")).merge(split, on="StudyInstanceUID", how="left")
    df["split"] = df["split"].fillna("weak")
    X = X_all.loc[df["StudyInstanceUID"]].to_numpy()
    is_weak, is_tune, is_test = (df["split"] == s for s in ("weak", "tune", "test"))
    y_test = df.loc[is_test, LABELS].astype(int).to_numpy()

    probs = {
        "R": df.loc[is_test, [f"{l}__score" for l in LABELS]].to_numpy(),
        "E1": np.zeros_like(y_test, dtype=float),
        "E2": np.zeros_like(y_test, dtype=float),
    }
    hard = {"R": np.column_stack([binarize_rules(df[is_test], l) for l in LABELS])}  # política: solo POS es 1

    for k, label in enumerate(LABELS):
        yw = binarize_rules(df[is_weak], label)
        keep = (df.loc[is_weak, f"{label}__state"].isin(["UNC", "WEAK"]) == False).to_numpy()  # noqa: E712
        w = df.loc[is_weak, f"{label}__conf"].to_numpy()
        m1 = logreg(args.C).fit(X[is_weak][keep], yw[keep], logisticregression__sample_weight=w[keep])
        probs["E1"][:, k] = m1.predict_proba(X[is_test])[:, 1]

        yt = df.loc[is_tune, label].astype(int).to_numpy()
        m2 = make_pipeline(StandardScaler(), LogisticRegression(C=args.C, max_iter=3000, class_weight="balanced"))
        m2.fit(X[is_tune], yt)
        probs["E2"][:, k] = m2.predict_proba(X[is_test])[:, 1]
    probs["E3"] = (probs["R"] + probs["E1"]) / 2

    print(f"Test n={len(y_test)}  | E1 entrenado con {is_weak.sum()} estudios no dorados | E2 con {is_tune.sum()} dorados\n")
    tabs = {v: per_label(y_test, p, hard.get(v)) for v, p in probs.items()}
    summary = pd.concat({v: t.mean() for v, t in tabs.items()}, axis=1).T
    print("Macro (R binario según la política; E* con umbral 0.5):")

    rng = np.random.default_rng(SEED)
    n = len(y_test)
    boots = {v: [] for v in probs}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        for v, p in probs.items():
            h = hard[v][idx] if v in hard else None
            boots[v].append(macro(y_test[idx], p[idx], h).to_numpy())
    boots = {v: np.array(b) for v, b in boots.items()}
    for v in probs:
        lo, hi = np.nanpercentile(boots[v], [2.5, 97.5], axis=0)
        summary.loc[v, "auc_95ci"] = f"[{lo[3]:.3f}, {hi[3]:.3f}]"
        summary.loc[v, "acc_95ci"] = f"[{lo[0]:.3f}, {hi[0]:.3f}]"
    print(summary.round(3).to_string())

    print("\nDiferencia de AUC macro contra R (bootstrap pareado):")
    for v in ("E1", "E2", "E3"):
        d = boots[v][:, 3] - boots["R"][:, 3]
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        print(f"  {v} - R = {np.nanmean(d):+.3f}  IC95 [{lo:+.3f}, {hi:+.3f}]  P(>0) = {(d > 0).mean():.2f}")

    print("\nAUC por etiqueta:")
    auc = pd.concat({v: t["auc"] for v, t in tabs.items()}, axis=1)
    print(auc.round(3).to_string())

    out = pd.concat(tabs, axis=1)
    out.to_csv(OUT / "compare_test_per_label.csv")
    summary.to_csv(OUT / "compare_test_macro.csv")


if __name__ == "__main__":
    main()
