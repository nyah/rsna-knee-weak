"""Genera notebooks/04_ablation_rsna_knee.ipynb: corre las corridas de configs/ablation_v1.json.

Mismo esquema que el 03: los módulos de src/ y configs/ van incrustados y se escriben en
/tmp/rsnak. Cada corrida se salta si ya tiene metrics.json y un fallo no detiene las demás.
Si se cambia un módulo, una config o la lista de ablación, hay que volver a correr este build.

Uso (desde rsna-knee-weak/): .venv/bin/python notebooks/build_04_ablation.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "04_ablation_rsna_knee.ipynb"
FILES = [
    "src/__init__.py", "src/data/__init__.py", "src/data/select_subset.py",
    "src/models/__init__.py", "src/models/cnn_scratch.py", "src/models/backbone_tl.py", "src/models/mil.py",
    "src/train/__init__.py", "src/train/data.py", "src/train/augment.py", "src/train/losses.py",
    "src/train/metrics.py", "src/train/train.py",
    "configs/scratch_base.json", "configs/resnet18_base.json", "configs/ablation_v1.json",
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n")}


files = {f: (ROOT / f).read_text() for f in FILES}
abl = json.loads(files["configs/ablation_v1.json"])
table = "\n".join(f"| `{r['name']}` | {r['factor']} | `{r['config']}` + {json.dumps(r['set'], ensure_ascii=False)} |"
                  for r in abl["runs"])

cells = [
    md(f"""
# 04 · Ablación RSNA Knee (Fase 5)

Settings: **GPU T4 x2**, **Internet On** (pesos ImageNet) y **Add Input** `rsna-knee-subset-v1`.
Antes del commit, detener cualquier sesión interactiva con GPU, porque compite con el commit en la cola.

Cada corrida cambia un solo factor respecto de `resnet18_base` (notebook 03, AUC 0.738 en los 58 dorados).
El checkpoint se elige por AUC en val (débil); los dorados solo se predicen al final.

| Corrida | Factor | Configuración |
|---|---|---|
{table}

Salida: `/kaggle/working/runs/<nombre>/` con `metrics.json`, `history.csv`, `preds_val.csv`, `preds_test.csv` y `best.pt`.
La evaluación con bootstrap se hace en local con `src.eval.evaluate`.
"""),
    code("""
import glob, json, os, subprocess, sys, time
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
print(DATA, studies.split.value_counts().to_dict())
assert studies.split.value_counts().to_dict() == {"train": 400, "val": 100, "test": 58}

import torch, torchvision
print("torch", torch.__version__, "· torchvision", torchvision.__version__)
assert torch.cuda.is_available(), "activar GPU en Settings → Accelerator"
print(torch.cuda.get_device_name(0))
RUNS = Path("/kaggle/working/runs"); RUNS.mkdir(parents=True, exist_ok=True)
"""),
    md("## 1. Código del repo (copiado tal cual por `notebooks/build_04_ablation.py`)"),
    code("CODE = Path('/tmp/rsnak')\nFILES = " + json.dumps(files, ensure_ascii=False, indent=0) + """
for rel, src in FILES.items():
    p = CODE / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(src)
ABL = json.loads(FILES["configs/ablation_v1.json"])["runs"]
print(len(FILES), "archivos en", CODE, "·", len(ABL), "corridas")

def train(config, overrides, out=RUNS):
    cmd = [sys.executable, "-m", "src.train.train", "--config", f"configs/{config}.json",
           "--data", str(DATA), "--out", str(out), "--set", *overrides]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=CODE, check=True)

def overrides(r):
    return [f"name={r['name']}"] + [f"{k}={json.dumps(v)}" for k, v in r["set"].items()]
"""),
    md("## 2. Prueba rápida de las variantes nuevas (focal, highconf, un plano y ResNet-50; unos 3 min)"),
    code("""
SMOKE = Path("/tmp/smoke")
quick = ["epochs=2", "freeze_epochs=1", "limit_train=16", "limit_eval=16"]
for name in ["r18_sag", "r18_focal", "r18_highconf", "r50"]:
    r = next(x for x in ABL if x["name"] == name)
    train(r["config"], overrides(r) + quick, out=SMOKE)
"""),
    md("## 3. Corridas completas (unos 6 min cada una; ResNet-50 y scratch_v2 algo más)"),
    code("""
status = {}
for r in ABL:
    if (RUNS / r["name"] / "metrics.json").exists():
        status[r["name"]] = "ya estaba"; continue
    t0 = time.time()
    try:
        train(r["config"], overrides(r))
        status[r["name"]] = f"ok ({(time.time()-t0)/60:.1f} min)"
    except subprocess.CalledProcessError as e:
        status[r["name"]] = f"FALLÓ (código {e.returncode})"
    print(r["name"], "→", status[r["name"]], flush=True)
print(json.dumps(status, indent=1, ensure_ascii=False))
"""),
    md("## 4. Resumen"),
    code("""
rows = []
for r in ABL:
    m = RUNS / r["name"] / "metrics.json"
    if not m.exists():
        rows.append({"run": r["name"], "factor": r["factor"], "estado": status.get(r["name"], "sin correr")}); continue
    j = json.loads(m.read_text())
    rows.append({"run": r["name"], "factor": r["factor"], "best_epoch": j["best_epoch"], "epochs": j["epochs_run"],
                 "val_auc": j["val_auc_macro"], "test_auc": j["test_auc_macro"], "min": j["minutes"],
                 "gpu_gb": j.get("gpu_peak_gb")})
summary = pd.DataFrame(rows).round(4)
summary.to_csv(RUNS / "summary_ablation.csv", index=False)
print("referencia resnet18_base (notebook 03): val 0.7051 · test 0.7378")
print(summary.to_string(index=False))
"""),
    md("""
## 5. Guardar

**Save Version → Save & Run All (Commit)** con la sesión interactiva detenida.
Después, de cada `runs/<nombre>/` bajar `metrics.json`, `history.csv`, `preds_val.csv` y `preds_test.csv`
a `rsna-knee-weak/outputs/runs_v1/<nombre>/` (los `best.pt` no hacen falta en local).
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"escrito {OUT} ({len(cells)} celdas, {len(abl['runs'])} corridas)")
