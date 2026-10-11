"""Genera notebooks/03_train_rsna_knee.ipynb con el paquete de entrenamiento incrustado.

El notebook escribe los módulos de src/ y configs/ en /tmp/rsnak (fuera de /kaggle/working,
así no ensucian el output) y corre `python -m src.train.train` por cada configuración.
Si se cambia un módulo o una config, hay que volver a correr este build.

Uso (desde rsna-knee-weak/): .venv/bin/python notebooks/build_03_train.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "03_train_rsna_knee.ipynb"
FILES = [
    "src/__init__.py", "src/data/__init__.py", "src/data/select_subset.py",
    "src/models/__init__.py", "src/models/cnn_scratch.py", "src/models/backbone_tl.py", "src/models/mil.py",
    "src/train/__init__.py", "src/train/data.py", "src/train/augment.py", "src/train/losses.py",
    "src/train/metrics.py", "src/train/train.py",
    "configs/scratch_base.json", "configs/resnet18_base.json",
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n")}


files = {f: (ROOT / f).read_text() for f in FILES}

cells = [
    md("""
# 03 · Entrenamiento RSNA Knee (Fase 4)

Settings del notebook:
- **Accelerator**: GPU T4 x2 (se usa una sola) o P100.
- **Internet**: On, para bajar los pesos ImageNet de torchvision (ResNet-18).
- **Add Input**: el Dataset privado `rsna-knee-subset-v1`.

Modelos (entregable 2):
1. `scratch_base`: CNN 2.5D propia (bloques residuales conv-BN-ReLU), sin pesos previos.
2. `resnet18_base`: ResNet-18 ImageNet; 2 épocas con el backbone congelado y luego fine-tuning con LR × 0.1.

Ambos: 16 ventanas de 3 cortes por serie, sagital y coronal, agregación por atención entre cortes,
12 sigmoides, BCE con `pos_weight`, máscara para UNC y WEAK, sin flips.
El checkpoint se elige por AUC macro en val (etiquetas débiles); los 58 dorados solo se predicen al final.

Salida en `/kaggle/working/runs/<nombre>/`: `best.pt`, `history.csv`, `preds_val.csv`, `preds_test.csv` y `metrics.json`.
"""),
    code("""
import glob, json, os, subprocess, sys
from pathlib import Path
import pandas as pd

def find(pattern, max_depth=4):
    for d in range(1, max_depth + 1):
        hits = sorted(glob.glob("/kaggle/input/" + "*/" * (d - 1) + pattern))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(f"no encuentro {pattern} en /kaggle/input (¿falta Add Input?)")

DATA = find("subset_studies.csv").parent
studies = pd.read_csv(DATA / "subset_studies.csv")
print(DATA, studies.split.value_counts().to_dict(), "· volúmenes:", len(list((DATA / "vol").glob("*.npy"))))
assert studies.split.value_counts().to_dict() == {"train": 400, "val": 100, "test": 58}

import torch, torchvision
print("torch", torch.__version__, "· torchvision", torchvision.__version__, "· cuda", torch.cuda.is_available())
assert torch.cuda.is_available(), "activar GPU en Settings → Accelerator"
print(torch.cuda.get_device_name(0))
RUNS = Path("/kaggle/working/runs"); RUNS.mkdir(parents=True, exist_ok=True)
"""),
    md("## 1. Código del repo (copiado tal cual por `notebooks/build_03_train.py`)"),
    code("CODE = Path('/tmp/rsnak')\nFILES = " + json.dumps(files, ensure_ascii=False, indent=0) + """
for rel, src in FILES.items():
    p = CODE / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(src)
print(len(FILES), "archivos en", CODE)

def train(config, *overrides, out=RUNS):
    cmd = [sys.executable, "-m", "src.train.train", "--config", f"configs/{config}.json",
           "--data", str(DATA), "--out", str(out)]
    if overrides:
        cmd += ["--set", *overrides]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=CODE, check=True)
"""),
    md("## 2. Prueba rápida (1 época con 16 estudios; unos 2 min). Si falla, no seguir."),
    code("""
SMOKE = Path("/tmp/smoke")
train("scratch_base", "epochs=1", "limit_train=16", "limit_eval=16", out=SMOKE)
train("resnet18_base", "epochs=2", "freeze_epochs=1", "limit_train=16", "limit_eval=16", out=SMOKE)
"""),
    md("## 3. Modelo 1: CNN desde cero"),
    code('train("scratch_base")'),
    md("## 4. Modelo 2: ResNet-18 preentrenado"),
    code('train("resnet18_base")'),
    md("## 5. Resumen"),
    code("""
import matplotlib.pyplot as plt
rows, fig, ax = [], *plt.subplots(1, 2, figsize=(12, 4))
for m in sorted(RUNS.glob("*/metrics.json")):
    r = json.loads(m.read_text())
    rows.append({"run": r["name"], "best_epoch": r["best_epoch"], "val_auc": r["val_auc_macro"],
                 "test_auc": r["test_auc_macro"], "min": r["minutes"], "M_params": r["n_params"] / 1e6})
    h = pd.read_csv(m.parent / "history.csv")
    ax[0].plot(h.epoch, h.loss, label=r["name"]); ax[1].plot(h.epoch, h.val_auc, label=r["name"])
ax[0].set_title("pérdida de entrenamiento"); ax[1].set_title("AUC macro en val (débil)")
for a in ax: a.set_xlabel("época"); a.legend()
plt.show()
summary = pd.DataFrame(rows).round(4); summary.to_csv(RUNS / "summary.csv", index=False); print(summary)
per = pd.DataFrame({r: json.loads((RUNS / r / "metrics.json").read_text())["test_auc"] for r in summary.run}).round(3)
print("AUC por etiqueta en los 58 dorados"); print(per)
"""),
    md("""
## 6. Guardar

**Save Version → Save & Run All (Commit)**. El entrenamiento corre en el commit, así que la sesión interactiva se puede cerrar.
La versión guardada deja `runs/` como output del notebook; la Fase 5 (evaluación, ablación y Grad-CAM) lo agrega como input,
igual que el 02 usó el output del 01. Es un derivado de los datos: no se publica ni se sube a GitHub.
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"escrito {OUT} ({len(cells)} celdas)")
