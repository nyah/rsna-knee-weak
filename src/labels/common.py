"""Rutas y carga de datos compartidas por el pipeline de etiquetas."""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("RSNA_KNEE_DATA", ROOT / "data"))
OUT = ROOT / "outputs"

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]


def load_train():
    """train.csv con la columna `lang` de study_lang.csv (calculada en el EDA)."""
    df = pd.read_csv(DATA / "train.csv")
    lang = pd.read_csv(DATA / "study_lang.csv", usecols=["StudyInstanceUID", "lang"])
    return df.merge(lang, on="StudyInstanceUID", how="left")
