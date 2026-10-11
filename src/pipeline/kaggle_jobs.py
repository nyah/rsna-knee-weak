"""Corre los notebooks en Kaggle con la API y baja sus resultados (lo usa run_all.sh).

Cada notebook se sube como versión nueva de su slug privado, con sus entradas encadenadas por
outputs de otros notebooks (configs/kaggle_pipeline.json). Se espera a que termine y se bajan
solo los archivos livianos (CSV, JSON y PNG; los best.pt quedan en Kaggle).

Uso (desde rsna-knee-weak/, con la CLI autenticada: `kaggle auth login`):
    python -m src.pipeline.kaggle_jobs fetch-data        # train.csv de la competencia -> data/
    python -m src.pipeline.kaggle_jobs pull 01           # código del notebook 01 -> notebooks/ (sin salidas)
    python -m src.pipeline.kaggle_jobs upload-labels     # versión nueva del Dataset de etiquetas si cambiaron
    python -m src.pipeline.kaggle_jobs run 03            # sube, espera y baja el output
    python -m src.pipeline.kaggle_jobs download 03       # solo baja el output de la última versión
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads((ROOT / "configs" / "kaggle_pipeline.json").read_text())
OWNER = CFG["owner"]
NB = {n["key"]: n for n in CFG["notebooks"]}
PUSH_DIR = ROOT / "outputs" / "kaggle_push"
DOWN_DIR = ROOT / "outputs" / "run_all" / "kaggle"
KAGGLE = shutil.which("kaggle", path=str(Path(sys.executable).parent)) or "kaggle"


def kaggle(*args, capture=False):
    cmd = [KAGGLE, *map(str, args)]
    print("$", " ".join(cmd), flush=True)
    res = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=capture)
    if res.returncode != 0:
        if capture:
            print(res.stdout, res.stderr, sep="\n")
        raise SystemExit(f"falló: {' '.join(cmd)}")
    return res.stdout if capture else None


def ref(key):
    return f"{OWNER}/{NB[key]['slug']}"


def strip_outputs(path):
    nb = json.loads(Path(path).read_text())
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    Path(path).write_text(json.dumps(nb, ensure_ascii=False, indent=1))


def fetch_data():
    dest = ROOT / "data"
    if (dest / "train.csv").exists():
        print("data/train.csv ya existe"); return
    dest.mkdir(exist_ok=True)
    kaggle("competitions", "download", CFG["competition"], "-f", "train.csv", "-p", dest)
    for z in dest.glob("train.csv*.zip"):
        zipfile.ZipFile(z).extractall(dest); z.unlink()


def pull(key):
    tmp = Path(tempfile.mkdtemp())
    kaggle("kernels", "pull", ref(key), "-p", tmp)
    src = next(tmp.glob("*.ipynb"))
    dst = ROOT / NB[key]["file"]
    shutil.copy(src, dst); strip_outputs(dst)
    print("escrito", dst)


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def upload_labels():
    files = [ROOT / "outputs" / "weak_labels.csv", ROOT / "data" / "gold_split.csv"]
    slug = f"{OWNER}/{CFG['weak_labels_dataset']}"
    cur = Path(tempfile.mkdtemp())
    kaggle("datasets", "download", slug, "-p", cur, "--unzip")
    if all((cur / f.name).exists() and md5(cur / f.name) == md5(f) for f in files):
        print("las etiquetas no cambiaron: se usa la versión actual del Dataset"); return
    up = PUSH_DIR / "weak_labels"
    shutil.rmtree(up, ignore_errors=True); up.mkdir(parents=True)
    for f in files:
        shutil.copy(f, up / f.name)
    (up / "dataset-metadata.json").write_text(json.dumps({"id": slug, "title": CFG["weak_labels_dataset"]}))
    kaggle("datasets", "version", "-p", up, "-m", "run_all.sh: etiquetas regeneradas")
    print("esperando a que Kaggle procese el Dataset..."); time.sleep(90)


def push(key):
    n = NB[key]
    d = PUSH_DIR / key
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    shutil.copy(ROOT / n["file"], d / Path(n["file"]).name)
    datasets = {"weak_labels": f"{OWNER}/{CFG['weak_labels_dataset']}"}
    meta = {
        "id": ref(key), "title": n["title"], "code_file": Path(n["file"]).name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": n["gpu"], "enable_tpu": False, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4" if n["gpu"] else "",
        "dataset_sources": [datasets[k] for k in n["datasets"]],
        "competition_sources": [CFG["competition"]] if n.get("competition") else [],
        "kernel_sources": [ref(k) for k in n["kernels"]],
        "model_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    kaggle("kernels", "push", "-p", d)


def wait(key, poll=60, max_hours=6):
    time.sleep(poll)
    t0 = time.time()
    while True:
        out = kaggle("kernels", "status", ref(key), capture=True).strip()
        status = re.sub(r".*status\s+", "", out).strip('"').lower()
        print(f"[{(time.time() - t0) / 60:5.1f} min] {ref(key)}: {status}", flush=True)
        if "complete" in status:
            return
        if any(s in status for s in ("error", "cancel")):
            raise SystemExit(f"{ref(key)} terminó con estado {status}; revisar el log en Kaggle")
        if time.time() - t0 > max_hours * 3600:
            raise SystemExit(f"{ref(key)} sigue sin terminar tras {max_hours} h")
        time.sleep(poll)


def download(key):
    dest = DOWN_DIR / key
    shutil.rmtree(dest, ignore_errors=True); dest.mkdir(parents=True)
    kaggle("kernels", "output", ref(key), "-p", dest, "--file-pattern", NB[key]["download"], "-o", "-q")
    files = [p for p in dest.rglob("*") if p.is_file()]
    print(f"{len(files)} archivos en {dest}")
    if not files:
        raise SystemExit(f"el output de {ref(key)} no tiene archivos que coincidan con el patrón")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch-data", "pull", "upload-labels", "push", "run", "download"])
    ap.add_argument("key", nargs="?", choices=sorted(NB))
    a = ap.parse_args()
    if a.cmd in ("pull", "push", "run", "download") and not a.key:
        ap.error("falta el notebook (01 a 05)")
    if a.cmd == "fetch-data":
        fetch_data()
    elif a.cmd == "pull":
        pull(a.key)
    elif a.cmd == "upload-labels":
        upload_labels()
    elif a.cmd == "push":
        push(a.key)
    elif a.cmd == "run":
        push(a.key); wait(a.key); download(a.key)
    else:
        download(a.key)


if __name__ == "__main__":
    main()
