"""Embeddings multilingües por oración, agregados por estudio (promedio y máximo).

MiniLM corta en 128 tokens y un reporte típico tiene ~1,000 caracteres, por eso se codifica cada
oración por separado. Salida: outputs/emb_<modelo>.npz con `uids` y `X` (n_estudios, 2 x dim).

Uso: python -m src.labels.embed [--model paraphrase-multilingual-MiniLM-L12-v2]
"""
import argparse

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.labels.common import OUT, load_train
from src.labels.rules import embed_sentences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    df = load_train()
    sents, owner = [], []
    for i, rep in enumerate(df["Report"]):
        ss = embed_sentences(rep) or [rep]
        sents += ss
        owner += [i] * len(ss)
    owner = np.array(owner)
    print(f"{len(df)} estudios, {len(sents)} oraciones")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(f"sentence-transformers/{args.model}", device=device)
    E = model.encode(sents, batch_size=args.batch, normalize_embeddings=True, show_progress_bar=True,
                     convert_to_numpy=True)

    dim = E.shape[1]
    X = np.zeros((len(df), 2 * dim), dtype=np.float32)
    for i in range(len(df)):
        e = E[owner == i]
        X[i, :dim] = e.mean(0)
        X[i, dim:] = e.max(0)

    OUT.mkdir(exist_ok=True)
    path = OUT / f"emb_{args.model.split('/')[-1]}.npz"
    np.savez_compressed(path, uids=df["StudyInstanceUID"].to_numpy(), X=X)
    print(f"-> {path}  X={X.shape}")


if __name__ == "__main__":
    main()
