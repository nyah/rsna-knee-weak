"""Fase 4: entrena un modelo a nivel de estudio y guarda checkpoint, predicciones y metrics.json.

Uso (desde rsna-knee-weak/):
    python -m src.train.train --config configs/scratch_base.json --data <dir del subset> --out runs/
    python -m src.train.train --config configs/resnet18_base.json --data ... --set epochs=1 limit_train=16

Selección del checkpoint: AUC macro en val (etiquetas débiles). Los 58 dorados (test) solo se
predicen al final con el mejor checkpoint y nunca intervienen en el entrenamiento ni en la selección.
"""
import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.select_subset import LABELS
from src.models.mil import StudyNet
from src.train.augment import prepare
from src.train.data import KneeStudies, load_studies, worker_init
from src.train.losses import auto_pos_weight, multilabel_loss
from src.train.metrics import auc_table

DEFAULTS = {
    "name": "run",
    "model": "scratch",          # scratch | resnet18 | resnet34 | resnet50 | efficientnet_b0
    "pretrained": True,
    "planes": ["sag", "cor"],
    "n_windows": 16, "window": 3,
    "pool": "attn",              # attn | max | mean
    "loss": "bce", "focal_gamma": 2.0,
    "pos_weight": "auto", "max_pos_weight": 10.0,   # auto | none
    "noise": "mask", "highconf_thr": 0.9,           # mask | conf | highconf
    "epochs": 30, "batch_size": 4, "lr": 1e-3, "bb_lr_mult": 1.0, "freeze_epochs": 0,
    "weight_decay": 1e-4, "warmup_epochs": 1, "dropout": 0.3, "patience": 8,
    "aug": True, "amp": True, "num_workers": 2, "seed": 42,
    "limit_train": 0, "limit_eval": 0,              # >0 solo para pruebas rápidas
}


def parse_value(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def load_config(path, overrides):
    cfg = dict(DEFAULTS)
    if path:
        cfg.update(json.loads(Path(path).read_text()))
    for kv in overrides or []:
        k, v = kv.split("=", 1)
        if k not in DEFAULTS:
            raise KeyError(f"clave desconocida {k}")
        cfg[k] = parse_value(v)
    return cfg


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(batch, dev):
    x = {p: v.to(dev, non_blocking=True) for p, v in batch["x"].items()}
    return x, batch["y"].to(dev), batch["w"].to(dev)


@torch.no_grad()
def predict(model, loader, dev, amp):
    model.eval()
    probs, idx = [], []
    for batch in loader:
        x, _, _ = to_device(batch, dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            logits = model({p: prepare(v, train=False) for p, v in x.items()})
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        idx.append(batch["i"].numpy())
    order = np.argsort(np.concatenate(idx))
    return np.concatenate(probs)[order]


def save_preds(path, ds, p):
    df = pd.DataFrame(p, columns=[f"p_{l}" for l in LABELS])
    df.insert(0, "StudyInstanceUID", ds.df.StudyInstanceUID.values)
    for j, l in enumerate(LABELS):
        df[f"y_{l}"] = ds.y[:, j]
        df[f"w_{l}"] = ds.w[:, j]
    for c in ("lang", "Manufacturer", "gold_split"):
        if c in ds.df:
            df[c] = ds.df[c].values
    df.to_csv(path, index=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--data", required=True, help="carpeta con subset_studies.csv y vol/")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--set", nargs="*", default=[], help="sobrescribe claves: epochs=1 lr=3e-4")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    dev = pick_device()
    amp = bool(cfg["amp"]) and dev.type == "cuda"
    out = Path(args.out) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    print(f"dispositivo {dev} · amp {amp} · salida {out}")
    print(json.dumps(cfg))

    studies = load_studies(args.data)
    parts = {s: studies[studies.split == s] for s in ("train", "val", "test")}
    if cfg["limit_train"]:
        parts["train"] = parts["train"].sample(min(cfg["limit_train"], len(parts["train"])), random_state=0)
    if cfg["limit_eval"]:
        for s in ("val", "test"):
            parts[s] = parts[s].sample(min(cfg["limit_eval"], len(parts[s])), random_state=0)
    common = dict(data_dir=args.data, planes=cfg["planes"], n_windows=cfg["n_windows"], window=cfg["window"],
                  noise=cfg["noise"], highconf_thr=cfg["highconf_thr"])
    cache = {}
    t0 = time.time()
    ds = {s: KneeStudies(parts[s], split=s, train=(s == "train" and cfg["aug"]), cache=cache,
                         seed=cfg["seed"], **common) for s in parts}
    print({s: len(d) for s, d in ds.items()}, f"· volúmenes en memoria: {len(cache)} en {time.time()-t0:.0f} s")

    lk = dict(num_workers=cfg["num_workers"], pin_memory=dev.type == "cuda",
              persistent_workers=cfg["num_workers"] > 0, worker_init_fn=worker_init)
    dl_train = DataLoader(ds["train"], batch_size=cfg["batch_size"], shuffle=True, drop_last=True, **lk)
    dl_eval = {s: DataLoader(ds[s], batch_size=cfg["batch_size"], shuffle=False, **lk) for s in ("val", "test")}

    pw = None
    if cfg["pos_weight"] == "auto":
        pw_np = auto_pos_weight(ds["train"].y, ds["train"].w, cfg["max_pos_weight"])
        pw = torch.tensor(pw_np, device=dev)
        print("pos_weight:", dict(zip(LABELS, pw_np.round(2).tolist())))

    model = StudyNet(cfg, n_out=len(LABELS)).to(dev)
    if dev.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    bb = set(map(id, model.backbone_parameters()))
    head = [q for q in model.parameters() if id(q) not in bb]
    opt = torch.optim.AdamW([
        {"params": model.backbone_parameters(), "lr": cfg["lr"] * cfg["bb_lr_mult"]},
        {"params": head, "lr": cfg["lr"]},
    ], weight_decay=cfg["weight_decay"])
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_epoch = len(dl_train)
    total, warm = cfg["epochs"] * steps_per_epoch, cfg["warmup_epochs"] * steps_per_epoch

    def lr_factor(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))

    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"modelo {cfg['model']} · {n_params/1e6:.2f} M parámetros · {steps_per_epoch} pasos por época")

    best, best_ep, bad, hist, step = -1.0, -1, 0, [], 0
    t_start = time.time()
    for ep in range(cfg["epochs"]):
        frozen = ep < cfg["freeze_epochs"]
        model.set_backbone_trainable(not frozen)
        model.train()
        te, run_loss = time.time(), 0.0
        for batch in dl_train:
            x, y, w = to_device(batch, dev)
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * lr_factor(step)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                logits = model({p: prepare(v, train=cfg["aug"]) for p, v in x.items()})
            loss = multilabel_loss(logits, y, w, cfg["loss"], pw, cfg["focal_gamma"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item()
            step += 1
        p_val = predict(model, dl_eval["val"], dev, amp)
        per, macro = auc_table(ds["val"].y, p_val, ds["val"].w)
        hist.append({"epoch": ep, "loss": run_loss / max(steps_per_epoch, 1), "val_auc": macro,
                     "frozen": int(frozen), "sec": time.time() - te, **{f"val_{k}": v for k, v in per.items()}})
        flag = ""
        if macro > best:
            best, best_ep, bad, flag = macro, ep, 0, " *"
            torch.save({"cfg": cfg, "state_dict": model.state_dict(), "epoch": ep}, out / "best.pt")
        else:
            bad += 1
        print(f"ep {ep:02d} · loss {hist[-1]['loss']:.4f} · val AUC {macro:.4f}{flag} · {hist[-1]['sec']:.0f} s"
              + (" · backbone congelado" if frozen else ""), flush=True)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        if bad >= cfg["patience"] and not frozen:
            print(f"parada temprana en la época {ep}")
            break

    ck = torch.load(out / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    res = {}
    for s in ("val", "test"):
        p = predict(model, dl_eval[s], dev, amp)
        save_preds(out / f"preds_{s}.csv", ds[s], p)
        res[s] = auc_table(ds[s].y, p, ds[s].w)
    metrics = {
        "name": cfg["name"], "cfg": cfg, "best_epoch": best_ep, "epochs_run": len(hist),
        "val_auc_macro": res["val"][1], "test_auc_macro": res["test"][1],
        "val_auc": res["val"][0], "test_auc": res["test"][0],
        "n_params": n_params, "minutes": (time.time() - t_start) / 60,
        "gpu_peak_gb": torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None,
        "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else str(dev),
        "torch": torch.__version__, "python": platform.python_version(),
        "n": {s: len(d) for s, d in ds.items()},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    if dev.type == "cuda":
        print(f"memoria GPU máxima {metrics['gpu_peak_gb']:.1f} GB")
    print(f"mejor época {best_ep} · val AUC {res['val'][1]:.4f} · test (58 dorados) AUC {res['test'][1]:.4f}")
    print(pd.DataFrame({"val": res["val"][0], "test": res["test"][0]}).round(3).T.to_string())
    return metrics


if __name__ == "__main__":
    main()
