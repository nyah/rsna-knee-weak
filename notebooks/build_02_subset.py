"""Genera notebooks/02_subset_rsna_knee.ipynb copiando los módulos de src/data/ en celdas.

Uso (desde rsna-knee-weak/): .venv/bin/python notebooks/build_02_subset.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "02_subset_rsna_knee.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n")}


def module(rel):
    src = (ROOT / rel).read_text()
    return code(f"# ===== {rel} (copiado tal cual del repo) =====\n{src}")


cells = [
    md("""
# 02 · Subconjunto RSNA Knee (Fase 3)

Entradas (Add Input):
1. La competencia `rsna-knee-abnormality-detection`.
2. El Dataset privado `rsna-knee-weak-labels-v1` (`weak_labels.csv`, `gold_split.csv`).
3. El output del notebook `01_eda_rsna_knee` (`series_catalog.csv`, `study_lang.csv`).

Salida en `/kaggle/working`: `vol/*.npy` (uint8, cortes × 256 × 256), `subset_studies.csv`,
`subset_manifest.csv`, `prevalence_*.csv` y `conversion_errors.csv`. Se publica como Dataset privado `rsna-knee-subset-v1`.

Política de etiquetas: solo el estado POS cuenta como 1. MILD, MIN, NEG y NONE valen 0. UNC y WEAK quedan en 0 con máscara 0.
**No usar la columna `<label>` de weak_labels.csv**, porque ahí MILD figura como 1.
"""),
    code("""
import glob, os, time
from pathlib import Path
import numpy as np
import pandas as pd

def find(pattern, max_depth=4):
    # Sin glob recursivo: bajar por train_series/ recorrería los 819,640 DICOM.
    for d in range(1, max_depth + 1):
        hits = sorted(glob.glob("/kaggle/input/" + "*/" * (d - 1) + pattern))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(f"no encuentro {pattern} en /kaggle/input (¿falta Add Input?)")

COMP = find("*knee*abnormal*")
assert (COMP / "train.csv").exists(), COMP
WEAK = find("weak_labels.csv")
GOLD_SPLIT = find("gold_split.csv")
CATALOG = find("series_catalog.csv")
STUDY_LANG = find("study_lang.csv")
WORK = Path("/kaggle/working"); VOL = WORK / "vol"; VOL.mkdir(parents=True, exist_ok=True)
for p in [COMP, WEAK, GOLD_SPLIT, CATALOG, STUDY_LANG]:
    print(p)

N_TRAIN_POOL, MIN_POS, VAL_FRAC, SEED = 500, 60, 0.2, 42
"""),
    module("src/data/select_subset.py"),
    module("src/data/dicom_to_npy.py"),
    md("## 1. Etiquetas con la política (desde `__state`)"),
    code("""
weak = pd.read_csv(WEAK)
gold_split = pd.read_csv(GOLD_SPLIT)
train = pd.read_csv(COMP / "train.csv")
lang = pd.read_csv(STUDY_LANG, usecols=["StudyInstanceUID", "lang"])
labels = apply_policy(weak)
GOLD = set(gold_split["StudyInstanceUID"])
assert len(weak) == 4407 and len(GOLD) == 58

# Comprobación de la advertencia: la columna cruda cuenta MILD como 1 y la política no.
chk = pd.DataFrame({lab: [int((weak[lab] == 1).sum()), int(labels[f"y_{lab}"].sum()),
                          int((labels[f"mask_{lab}"] == 0).sum())] for lab in LABELS},
                   index=["col_cruda==1", "y_politica", "enmascarados"]).T
print(chk)
assert labels["y_Effusion"].sum() == (weak["Effusion__state"] == "POS").sum()
"""),
    md("## 2. Selección de series por estudio"),
    code("""
catalog = pd.read_csv(CATALOG)
t0 = time.time()
series = choose_series(catalog)
print(f"{len(series)} estudios en {time.time()-t0:.0f} s")
for role in ROLES:
    s = series[f"{role}_series"].notna()
    print(f"{role}: presente {s.sum()} · respaldo 0/0 {int((series[f'{role}_fallback'] == 1).sum())}"
          f" · 3D {int((series[f'{role}_is3d'] == 1).sum())}")
g = series[series.StudyInstanceUID.isin(GOLD)]
print("dorados sin sag:", g.sag_series.isna().sum(), "· sin cor:", g.cor_series.isna().sum(),
      "· con respaldo sag/cor:", int(g.sag_fallback.sum()), int(g.cor_fallback.sum()),
      "· 3D sag/cor:", int(g.sag_is3d.sum()), int(g.cor_is3d.sum()))
eligible = eligible_for_training(series) - GOLD
print("no dorados elegibles (sag y cor 2D):", len(eligible), "de", len(series) - len(GOLD))
"""),
    md("## 3. Muestreo estratificado, split y prevalencia (entregable 6)"),
    code("""
pool_uids = sample_studies(labels, eligible, n=N_TRAIN_POOL, min_pos=MIN_POS, seed=SEED)
split = split_train_val(labels, pool_uids, val_frac=VAL_FRAC, seed=SEED)
split.update({u: "test" for u in GOLD})
uids = list(pool_uids) + sorted(GOLD)

studies = (pd.DataFrame({"StudyInstanceUID": uids})
           .assign(split=lambda d: d.StudyInstanceUID.map(split), is_gold=lambda d: d.StudyInstanceUID.isin(GOLD).astype(int))
           .merge(gold_split.rename(columns={"split": "gold_split"})[["StudyInstanceUID", "gold_split"]], how="left")
           .merge(lang, how="left")
           .merge(series, how="left")
           .merge(labels, how="left")
           .merge(train[["StudyInstanceUID"] + LABELS].rename(columns={l: f"gold_{l}" for l in LABELS}), how="left"))
studies.loc[studies.is_gold == 0, [f"gold_{l}" for l in LABELS]] = np.nan
assert studies.StudyInstanceUID.is_unique and studies[[f"gold_{l}" for l in LABELS]].iloc[-58:].notna().all().all()
print(studies.split.value_counts())

all_nongold = set(labels.StudyInstanceUID) - GOLD
prev = pd.concat({
    "pool_4349": prevalence(labels, all_nongold)["pct"],
    "subset_500": prevalence(labels, pool_uids)["pct"],
    "train_pos": prevalence(labels, studies.query("split=='train'").StudyInstanceUID)["pos"],
    "val_pos": prevalence(labels, studies.query("split=='val'").StudyInstanceUID)["pos"],
    "gold58_pos": pd.Series({**{l: int(train.loc[train.StudyInstanceUID.isin(GOLD), l].sum()) for l in LABELS}, "n": 58}),
}, axis=1)
prev.to_csv(WORK / "prevalence_subset.csv")
print(prev)
by_lang = pd.crosstab(studies.lang, studies.split)
by_lang.to_csv(WORK / "subset_by_lang.csv"); print(by_lang)
by_man = pd.crosstab(studies.Manufacturer, studies.split)
by_man.to_csv(WORK / "subset_by_manufacturer.csv"); print(by_man)
"""),
    md("## 4. Prueba de conversión y códecs (antes del bucle largo)"),
    code("""
import pydicom
jobs = []
for r in studies.itertuples():
    for role in ROLES:
        sid = getattr(r, f"{role}_series")
        if isinstance(sid, str):
            jobs.append((r.StudyInstanceUID, role, sid))
print(len(jobs), "series a convertir")

# Sintaxis de transferencia (solo cabecera del primer archivo de cada serie)
ts = {}
for st, role, sid in jobs:
    f = next((COMP / "train_series" / st / sid).glob("*.dcm"))
    t = str(pydicom.dcmread(f, stop_before_pixels=True).file_meta.TransferSyntaxUID.name)
    ts[t] = ts.get(t, 0) + 1
print(ts)
if any(("JPEG" in k) or ("RLE" in k) for k in ts):
    print("Hay series comprimidas: si la prueba de abajo falla, correr  !pip install -q pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg  (requiere Internet en Settings)")

t0 = time.time()
for st, role, sid in jobs[:3]:
    print(convert_series(COMP / "train_series" / st / sid, VOL / f"{st}_{role}.npy"))
print(f"{(time.time()-t0)/3:.1f} s por serie → estimado {len(jobs)*(time.time()-t0)/3/60/max(os.cpu_count(),1):.0f} min con {os.cpu_count()} procesos")
"""),
    code("""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, (st, role, sid) in zip(ax, jobs[:3]):
    v = np.load(VOL / f"{st}_{role}.npy"); a.imshow(v[len(v)//2], cmap="gray"); a.set_title(f"{role} {v.shape}"); a.axis("off")
plt.show()
"""),
    md("## 5. Conversión completa (reanudable: salta los .npy ya guardados)"),
    code("""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

def run(job):
    st, role, sid = job
    out = VOL / f"{st}_{role}.npy"
    try:
        if out.exists():
            v = np.load(out, mmap_mode="r")
            return {**dict(zip(["StudyInstanceUID", "role", "SeriesInstanceUID"], job)), "n_slices_out": v.shape[0], "skipped": 1}
        meta = convert_series(COMP / "train_series" / st / sid, out)
        return {"StudyInstanceUID": st, "role": role, "SeriesInstanceUID": sid, **meta}
    except Exception as e:
        return {"StudyInstanceUID": st, "role": role, "SeriesInstanceUID": sid, "error": repr(e)[:300]}

t0, results = time.time(), []
with ProcessPoolExecutor(max_workers=os.cpu_count(), mp_context=mp.get_context("fork")) as ex:
    futs = [ex.submit(run, j) for j in jobs]
    for i, f in enumerate(as_completed(futs), 1):
        results.append(f.result())
        if i % 100 == 0 or i == len(jobs):
            print(f"{i}/{len(jobs)} · {(time.time()-t0)/60:.1f} min", flush=True)
res = pd.DataFrame(results)
errs = res[res.get("error").notna()] if "error" in res else res.iloc[:0]
errs.to_csv(WORK / "conversion_errors.csv", index=False)
print("errores:", len(errs)); print(errs.head())
"""),
    md("## 6. Manifiesto y resumen"),
    code("""
plane = {"sag": "Sagittal", "cor": "Coronal"}
ok = res[res.get("error").isna()] if "error" in res else res
man = ok.assign(plane=ok.role.map(plane), path=lambda d: "vol/" + d.StudyInstanceUID + "_" + d.role + ".npy")
man["n_slices"] = [np.load(WORK / p, mmap_mode="r").shape[0] for p in man.path]
man["fallback"] = [studies.set_index("StudyInstanceUID").at[s, f"{r}_fallback"] for s, r in zip(man.StudyInstanceUID, man.role)]
keep = ["StudyInstanceUID", "split", "is_gold", "gold_split", "lang", "Manufacturer", "FieldStrength"] \\
       + [f"{p}_{l}" for l in LABELS for p in ("y", "mask", "conf", "score")] + [f"gold_{l}" for l in LABELS]
man = man[["StudyInstanceUID", "role", "plane", "SeriesInstanceUID", "path", "n_slices", "fallback"]].merge(studies[keep], how="left")
man = man.sort_values(["split", "StudyInstanceUID", "role"])
man.to_csv(WORK / "subset_manifest.csv", index=False)

# Estudios con su estado de conversión; un estudio sin sagital válida no sirve para entrenar
have = man.pivot_table(index="StudyInstanceUID", columns="role", values="n_slices", aggfunc="first")
studies["has_sag"] = studies.StudyInstanceUID.map(have.get("sag")).notna().astype(int)
studies["has_cor"] = studies.StudyInstanceUID.map(have.get("cor")).notna().astype(int)
studies.to_csv(WORK / "subset_studies.csv", index=False)
print(studies.groupby("split")[["has_sag", "has_cor"]].sum())
size_gb = sum(f.stat().st_size for f in VOL.glob("*.npy")) / 1e9
print(f"{len(man)} series · {size_gb:.2f} GB · cortes por serie: {man.n_slices.describe()[['min','50%','max']].to_dict()}")
"""),
    code("""
# Control visual: 4 estudios al azar de cada split, corte central sagital y coronal
fig, ax = plt.subplots(3, 8, figsize=(20, 8))
rng = np.random.default_rng(0)
for i, sp in enumerate(["train", "val", "test"]):
    u = rng.choice(studies.query("split==@sp").StudyInstanceUID.values, 4, replace=False)
    for j, st in enumerate(u):
        for k, role in enumerate(ROLES):
            a = ax[i, 2*j + k]; p = VOL / f"{st}_{role}.npy"
            if p.exists():
                v = np.load(p); a.imshow(v[len(v)//2], cmap="gray")
            a.set_title(f"{sp} {role}", fontsize=8); a.axis("off")
plt.tight_layout(); plt.show()
"""),
    md("""
## 7. Publicar

1. **Save Version → Save & Run All (Commit)**. La conversión corre en el commit, así que la sesión interactiva se puede cerrar.
2. En la versión guardada: **Output → New Dataset** → nombre `rsna-knee-subset-v1` → **Private**.
3. Cuando exista el Team, compartir el Dataset con los integrantes (Settings → Sharing). Nunca hacerlo público ni subirlo a GitHub.
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"escrito {OUT} ({len(cells)} celdas)")
