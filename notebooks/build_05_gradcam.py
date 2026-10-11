"""Genera notebooks/05_gradcam_rsna_knee.ipynb (entregable 5).

Usa los best.pt del output del notebook 03 (resnet18_base y scratch_base) y los volúmenes de
rsna-knee-subset-v1. No entrena nada: basta con CPU. Para cada hallazgo elegido toma los 3
verdaderos positivos dorados con mayor probabilidad y el falso positivo más fuerte.

Uso (desde rsna-knee-weak/): .venv/bin/python notebooks/build_05_gradcam.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "05_gradcam_rsna_knee.ipynb"
FILES = [
    "src/__init__.py", "src/data/__init__.py", "src/data/select_subset.py",
    "src/models/__init__.py", "src/models/cnn_scratch.py", "src/models/backbone_tl.py", "src/models/mil.py",
    "src/train/__init__.py", "src/train/data.py", "src/train/augment.py",
    "src/interpret/__init__.py", "src/interpret/gradcam.py",
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n")}


files = {f: (ROOT / f).read_text() for f in FILES}

cells = [
    md("""
# 05 · Grad-CAM RSNA Knee (Fase 5, entregable 5)

Settings: **Accelerator None** (CPU basta; no entrena) e Internet Off.
Add Input: el Dataset `rsna-knee-subset-v1` y el output del notebook `03_train_rsna_knee` (pestaña Your Work → Notebooks).

Método: Grad-CAM propio sobre la última etapa convolucional del codificador de cada plano. Hay un mapa por ventana
2.5D; se muestra la ventana con más energía del mapa. Los mapas de un plano se normalizan con el máximo del estudio.

Casos: para cada hallazgo, los 3 verdaderos positivos dorados con mayor probabilidad y el falso positivo con mayor
probabilidad (dorado = 0). Hallazgos: ACL, Medial Meniscus, Medial OA, Effusion y Baker's.

Salida en `/kaggle/working/gradcam/`: una figura por modelo y hallazgo, una figura resumen y `cam_cases.csv`.
Las figuras muestran imágenes de pacientes: van solo al informe y nunca a GitHub ni a un Dataset público.
"""),
    code("""
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

SKIP = {"train_series", "test_series", "vol"}  # carpetas con miles de archivos: no se recorren

def find(rel, root="/kaggle/input", max_depth=7):
    # Busca la ruta relativa `rel` (p. ej. 'resnet18_base/best.pt') sin entrar en SKIP.
    root = Path(root)
    for dirpath, dirnames, _ in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        dirnames[:] = [] if depth >= max_depth else sorted(d for d in dirnames if d not in SKIP)
        if (Path(dirpath) / rel).exists():
            return Path(dirpath) / rel
    raise FileNotFoundError(f"no encuentro {rel} en {root} (¿falta Add Input?)")

print("inputs montados:", sorted(str(p.relative_to("/kaggle/input")) for p in Path("/kaggle/input").glob("*/*/*")))

DATA = find("subset_studies.csv").parent
RUNS = {m: find(f"{m}/best.pt").parent for m in ["resnet18_base", "scratch_base"]}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("/kaggle/working/gradcam"); OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(os.cpu_count())
print(DATA); print(RUNS); print("dispositivo", DEV, "· hilos", torch.get_num_threads())
"""),
    md("## 1. Código del repo (copiado tal cual por `notebooks/build_05_gradcam.py`)"),
    code("CODE = Path('/tmp/rsnak')\nFILES = " + json.dumps(files, ensure_ascii=False, indent=0) + """
for rel, src in FILES.items():
    p = CODE / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(src)
sys.path.insert(0, str(CODE))
from src.data.select_subset import LABELS
from src.interpret.gradcam import GradCAM, load_model, study_input
print(len(FILES), "archivos en", CODE)
"""),
    md("## 2. Selección de casos (desde `preds_test.csv` del notebook 03)"),
    code("""
TARGETS = ["ACL", "Medial Meniscus", "Medial OA", "Effusion", "Baker's"]
N_TP = 3
studies = pd.read_csv(DATA / "subset_studies.csv").set_index("StudyInstanceUID")

def pick_cases(preds, lab):
    tp = preds[preds[f"y_{lab}"] == 1].nlargest(N_TP, f"p_{lab}").assign(kind="VP")
    fp = preds[preds[f"y_{lab}"] == 0].nlargest(1, f"p_{lab}").assign(kind="FP")
    return pd.concat([tp, fp])

preds = {m: pd.read_csv(d / "preds_test.csv") for m, d in RUNS.items()}
# Rango percentil medio de los positivos dorados entre los 58 (0.5 = azar, 1 = todos arriba)
for m, p in preds.items():
    print(m, {lab: round(float(p[f"p_{lab}"].rank(pct=True)[p[f"y_{lab}"] == 1].mean()), 2) for lab in TARGETS})
"""),
    md("## 3. Grad-CAM"),
    code("""
def overlay(ax, img, cam=None, title=""):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    if cam is not None:
        ax.imshow(cam, cmap="jet", alpha=0.35, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8); ax.axis("off")

rows = []
summary = {}
for m, run in RUNS.items():
    model, cfg = load_model(run / "best.pt", DEV)
    gc = GradCAM(model)
    gold = preds[m].set_index("StudyInstanceUID")
    for lab in TARGETS:
        j = LABELS.index(lab)
        cases = pick_cases(preds[m], lab)
        planes = cfg["planes"]
        fig, axes = plt.subplots(len(cases), 2 * len(planes) + 1, figsize=(3 * (2 * len(planes) + 1), 3 * len(cases)))
        for r, case in enumerate(cases.itertuples()):
            uid = case.StudyInstanceUID
            vols = {p: np.load(DATA / "vol" / f"{uid}_{p}.npy") for p in planes}
            x, idx = study_input(vols, cfg)
            t0 = time.time()
            prob, cams, energy, att = gc(x, j)
            print(f"{m} · {lab} · {case.kind} · …{uid[-6:]} · p={prob:.2f} · {time.time()-t0:.1f} s", flush=True)
            ax = axes[r]
            for c, p in enumerate(planes):
                k = int(np.argmax(energy[p]))
                s = int(idx[p][k][len(idx[p][k]) // 2])
                share = float(energy[p][k] / max(energy[p].sum(), 1e-8))
                overlay(ax[2 * c], vols[p][s], None, f"{p} corte {s}/{len(vols[p])}")
                overlay(ax[2 * c + 1], vols[p][s], cams[p][k], f"Grad-CAM {lab} (ventana {k})")
                rows.append({"model": m, "label": lab, "StudyInstanceUID": uid, "kind": case.kind,
                             "prob": prob, "gold": int(gold.at[uid, f"y_{lab}"]),
                             "lang": studies.at[uid, "lang"], "plane": p, "window": k, "slice": s, "n_slices": len(vols[p]),
                             "energy_share": share,
                             "attn_argmax": None if att[p] is None else int(np.argmax(att[p])),
                             "attn_max": None if att[p] is None else float(np.max(att[p]))})
                if m == "resnet18_base" and case.kind == "VP" and lab not in summary:
                    summary[lab] = (uid, prob)
            a = ax[-1]
            for p in planes:
                e = energy[p] / max(energy[p].sum(), 1e-8)
                a.plot(e, marker="o", label=f"CAM {p}")
                if att[p] is not None:
                    a.plot(att[p], ls="--", label=f"atención {p}")
            a.set_title(f"{case.kind} · p={prob:.2f} · {studies.at[uid, 'lang']} · …{uid[-6:]}", fontsize=8)
            a.set_xlabel("ventana", fontsize=8); a.legend(fontsize=6)
        fig.suptitle(f"{m} · {lab}", fontsize=11)
        plt.tight_layout()
        fname = OUT / f"{m}_{lab.replace(' ', '_').replace(chr(39), '')}.png"
        fig.savefig(fname, dpi=110); plt.show(); plt.close(fig)
    gc.remove()

cases_df = pd.DataFrame(rows)
cases_df.to_csv(OUT / "cam_cases.csv", index=False)
print(cases_df.groupby(["model", "label", "kind"]).prob.mean().round(3).unstack())
"""),
    md("## 4. Figura resumen para el informe (ResNet-18, mejor verdadero positivo por hallazgo, ambos planos)"),
    code("""
model, cfg = load_model(RUNS["resnet18_base"] / "best.pt", DEV)
gc = GradCAM(model)
fig, axes = plt.subplots(len(TARGETS), 2 * len(cfg["planes"]), figsize=(12, 3 * len(TARGETS)))
for r, lab in enumerate(TARGETS):
    uid, _ = summary[lab]
    vols = {p: np.load(DATA / "vol" / f"{uid}_{p}.npy") for p in cfg["planes"]}
    x, idx = study_input(vols, cfg)
    prob, cams, energy, att = gc(x, LABELS.index(lab))
    for c, p in enumerate(cfg["planes"]):
        k = int(np.argmax(energy[p])); s = int(idx[p][k][1])
        overlay(axes[r, 2 * c], vols[p][s], None, f"{lab} · {p} · p={prob:.2f}")
        overlay(axes[r, 2 * c + 1], vols[p][s], cams[p][k], "Grad-CAM")
gc.remove()
plt.tight_layout(); fig.savefig(OUT / "gradcam_summary_resnet18.png", dpi=130); plt.show()
"""),
    md("""
## 5. Guardar

**Save Version → Save & Run All** (CPU, sin cola de GPU). Luego bajar `gradcam/` completo a
`rsna-knee-weak/outputs/gradcam_v1/` para revisar las figuras y escribir la discusión de plausibilidad anatómica.
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "kaggle": {"accelerator": "none", "isInternetEnabled": False}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"escrito {OUT} ({len(cells)} celdas)")
