"""AUC-ROC por etiqueta y macro (métrica oficial), ignorando elementos con peso 0."""
import numpy as np
from sklearn.metrics import roc_auc_score

from src.data.select_subset import LABELS


def auc_table(y, p, w=None):
    """Devuelve ({etiqueta: AUC o NaN}, AUC macro). NaN si la etiqueta tiene una sola clase."""
    w = np.ones_like(y) if w is None else w
    out = {}
    for j, lab in enumerate(LABELS):
        m = w[:, j] > 0
        yj = y[m, j]
        out[lab] = float(roc_auc_score(yj, p[m, j])) if 0 < yj.sum() < len(yj) else float("nan")
    vals = [v for v in out.values() if not np.isnan(v)]
    return out, float(np.mean(vals)) if vals else float("nan")
