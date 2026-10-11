"""Extractor de etiquetas débiles por reglas multilingües (entregable 1).

Flujo por reporte:
1. Partir en oraciones sobre el texto crudo (un punto seguido de minúscula no corta: "II. stupnja").
2. Normalizar cada oración (minúsculas, sin diacríticos) y cortar antes de encabezados de sección;
   se descartan los de indicación clínica ("Antecedentes clínicos:", "Indication:", ...).
3. Partir cada oración en cláusulas (coma, punto y coma, "but", "pero", "aber", ...).
4. Buscar conceptos del léxico y combinarlos por etiqueta (anatomía + lado + lesión), con negación,
   incertidumbre e intensidad por ventana de tokens.
5. Agregar las menciones del estudio en un estado final por etiqueta.

Salida por etiqueta: `label` en {1, 0, -1}, `conf` en [0, 1] y `score` (probabilidad aproximada,
sirve para calcular AUC contra los dorados).

Uso:
    python -m src.labels.rules                 # etiqueta los 4,407 estudios -> outputs/weak_labels.csv
    python -m src.labels.rules --explain <sufijo>  # muestra la evidencia de un estudio (sufijo del UID)
"""
import argparse
import json
import re
import unicodedata
from collections import defaultdict

import pandas as pd

from src.labels import lexicon as LX
from src.labels.common import LABELS, OUT, load_train

# ---------- normalización ----------
_CHARMAP = str.maketrans({
    "∆": "Δ", "µ": "μ", "ı": "i", "ß": "ss", "ς": "σ",
    "’": "'", "‘": "'", "`": "'", "´": "'", " ": " ",
})


def norm(s):
    s = unicodedata.normalize("NFKC", s.translate(_CHARMAP)).lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return s.translate(_CHARMAP)


def _rx(entries):
    parts = []
    for e in entries:
        if e.startswith("*"):
            parts.append(f"(?:{norm(e[1:])})")
        else:
            parts.append(rf"(?<!\w)(?:{norm(e)})")
    return re.compile("|".join(parts))


R = {name: _rx(getattr(LX, name)) for name in [
    "ACL", "CRUCIATE_GENERIC", "MCL", "COLLATERAL_GENERIC", "MENISCUS", "MENISCUS_PLURAL", "BOTH",
    "SIDE_MED", "SIDE_LAT", "PATELLOFEMORAL", "COMPARTMENT", "TEAR", "SPRAIN", "MENISCAL_DEGEN",
    "OA_STRONG", "CARTILAGE", "CARTILAGE_DAMAGE", "TRICOMPARTMENTAL", "EFFUSION", "SYNOVITIS", "BAKER",
    "CONTUSION", "FRACTURE", "NEG_PRE", "NEG_POST", "DISPLACEMENT", "UNCERTAIN", "NORMAL",
    "INT_MIN", "INT_MILD", "INT_MOD",
]}
R["LIG_LESION"] = re.compile(R["TEAR"].pattern + "|" + R["SPRAIN"].pattern)
_HDR_ALL = re.compile(rf"(?<!\w)(?:{'|'.join(norm(h) for h in LX.HEADERS_DROP + LX.HEADERS_KEEP)})\s*:")
_HDR_DROP = re.compile(rf"^\s*(?:{'|'.join(norm(h) for h in LX.HEADERS_DROP)})\s*:")
_CLAUSE = re.compile(
    r"[,;]|\s(?:but|however|whereas|while|pero|aunque|aber|jedoch|sondern|ali|αλλα|ενω|но|ama|ancak|"
    r"fakat|maar|terwijl|mais)\s")
_W = re.compile(r"\w+")
_JOINT = re.compile(r"joint|gelenk|gleitlager|articul|zglob|eklem|gewricht|αρθρωσ|став|anterieur")
_ABBREV = re.compile(r"(?:(?<![^\W\d_])[^\W\d_]\.[^\W\d_]|(?<![^\W\d_])(?:lig|bzw|ca|vs|dr|m|mm|bek|kl|dif|st|z|evtl|ggf|sog|li|re|nr))$",
                     re.IGNORECASE)
_NOTCH = re.compile(r"intercondyl|notch|cruciat|kreuzb|cruzad|kri[zž]n|χιαστ|кръстн|capraz|kruisb|croise")

POS, MILD, UNC, WEAK, MIN, NEG = "POS", "MILD", "UNC", "WEAK", "MIN", "NEG"


# ---------- segmentación ----------
def raw_sentences(raw):
    raw = re.sub(r"\[BILATERAL NOTE[^\]]*\]", "\n", raw.translate(_CHARMAP))
    raw = unicodedata.normalize("NFKC", raw)
    raw = re.sub(r"[ \t]*\r?\n[ \t]*(?=[^\W\d_])(?=[^A-ZÀ-ÞΑ-ΩА-ЯİŞĞÇÖÜ])", " ", raw)  # línea cortada a ancho fijo
    chunks = re.split(r"[\r\n]+|\s[•>]+\s*|\s+[-–]\s+(?=\S)", raw)
    out = []
    for ch in chunks:
        start = 0
        for m in re.finditer(r"[.!?]+\s*", ch):
            nxt = ch[m.end():m.end() + 1]
            spaced = m.group().strip() != m.group()
            if _ABBREV.search(ch, 0, m.start()) and m.group().startswith(".") and len(m.group().strip()) == 1:
                continue
            if not nxt or nxt.isupper() or (spaced and not nxt.isalpha()):
                out.append(ch[start:m.end()])
                start = m.end()
        out.append(ch[start:])
    return [s.strip() for s in out if s.strip()]


def embed_sentences(raw):
    """Oraciones con mayúsculas y acentos originales (para el modelo de embeddings), sin la indicación clínica."""
    return [s for s in raw_sentences(raw) if not _HDR_DROP.match(norm(s))]


def split_sentences(raw):
    """Oraciones normalizadas, cortadas en encabezados y sin la indicación clínica (para las reglas)."""
    sents = []
    for s in raw_sentences(raw):
        s = norm(s)
        if not s:
            continue
        cuts = [m.start() for m in _HDR_ALL.finditer(s)] + [len(s)]
        pieces = [s[:cuts[0]]] + [s[a:b] for a, b in zip(cuts, cuts[1:])]
        sents += [p.strip() for p in pieces if p.strip() and not _HDR_DROP.match(p)]
    return sents


def tok(s, pos):
    return len(_W.findall(s, 0, pos))


def _toks(rx, s):
    return [(tok(s, m.start()), tok(s, m.end()), m) for m in rx.finditer(s)]


# ---------- modificadores ----------
def assess(sc, start, end, intensity=False):
    """Estado de una lesión encontrada en la cláusula `sc` entre `start` y `end` (posiciones de carácter)."""
    t0, t1 = tok(sc, start), tok(sc, end)
    for tn, te, m in _toks(R["NEG_PRE"], sc):
        if t0 - 7 <= tn < t0:
            after = sc[m.end():m.end() + 30].lstrip(" -")
            if not R["DISPLACEMENT"].match(after):
                return NEG
    for tn, _, _ in _toks(R["NEG_POST"], sc):
        if t1 <= tn <= t1 + 4:
            return NEG
    if any(t0 - 7 <= tn <= t1 + 3 for tn, _, _ in _toks(R["UNCERTAIN"], sc)):
        return UNC
    if intensity:
        # el modificador más cercano al término; los de después pesan un poco menos; empate -> el más fuerte
        best = None
        for rank, (key, st) in enumerate((("INT_MOD", POS), ("INT_MILD", MILD), ("INT_MIN", MIN))):
            for tn, _, _ in _toks(R[key], sc):
                if t0 - 5 <= tn < t0:
                    d = t0 - tn
                elif t1 <= tn <= t1 + 2:
                    d = tn - t1 + 1.5
                else:
                    continue
                if best is None or (d, rank) < best[:2]:
                    best = (d, rank, st)
        if best:
            return best[2]
    return POS


def side_words(sc):
    """Lados (M/L) en la cláusula, ignorando los pegados a una palabra patelofemoral ("medial facet")."""
    pf = [t for t, _, _ in _toks(R["PATELLOFEMORAL"], sc)]
    out = []
    for key, side in (("SIDE_MED", "M"), ("SIDE_LAT", "L")):
        for t, _, _ in _toks(R[key], sc):
            if not any(abs(t - p) <= 3 for p in pf):
                out.append((t, side))
    return out


# ---------- reglas por grupo de etiquetas ----------
def _lesion_status(clauses, i, lesion_rx, intensity=False):
    """Estado de la lesión en la cláusula i; si no hay, la primera lesión de la oración; si tampoco, NORMAL."""
    sc = clauses[i]
    ms = list(lesion_rx.finditer(sc))
    if ms:
        return [assess(sc, m.start(), m.end(), intensity) for m in ms]
    if R["NORMAL"].search(sc):
        return [NEG]
    for j, other in enumerate(clauses):
        m = lesion_rx.search(other) if j != i else None
        if m:
            return [assess(other, m.start(), m.end(), intensity)]
    if any(R["NORMAL"].search(c) for c in clauses):
        return [NEG]
    return []


def ligaments(clauses, add):
    for i, sc in enumerate(clauses):
        for label, spec, generic in (("ACL", "ACL", "CRUCIATE_GENERIC"), ("MCL", "MCL", "COLLATERAL_GENERIC")):
            if R[spec].search(sc):
                for st in _lesion_status(clauses, i, R["LIG_LESION"], intensity=True):
                    add(label, st, sc)
            elif R[generic].search(sc):
                sts = _lesion_status(clauses, i, R["LIG_LESION"])
                if sts:
                    add(label, NEG if NEG in sts else UNC, sc)


def menisci(clauses, add, ctx):
    for i, sc in enumerate(clauses):
        for m in R["MENISCUS"].finditer(sc):
            ta = tok(sc, m.start())
            plural = bool(R["MENISCUS_PLURAL"].match(sc, m.start())) or bool(R["BOTH"].search(sc))
            sides = side_words(sc)
            near = {s for t, s in sides if 0 <= ta - t <= 4 or 0 <= t - ta <= 1}
            if near:
                targets = near
            elif plural:
                targets = {"M", "L"}
            elif sides:
                targets = {min(sides, key=lambda x: abs(x[0] - ta))[1]}
            else:
                other = {s for j, c in enumerate(clauses) if j != i for _, s in side_words(c)}
                targets = other if len(other) == 1 else ({ctx["side"]} if ctx["side"] in ("M", "L") else set())
            if not targets:
                continue
            if R["TEAR"].search(sc):
                sts = [assess(sc, x.start(), x.end()) for x in R["TEAR"].finditer(sc)]
            elif R["MENISCAL_DEGEN"].search(sc):
                x = R["MENISCAL_DEGEN"].search(sc)
                st = assess(sc, x.start(), x.end())
                sts = [WEAK if st in (POS, UNC) else st]
            else:
                sts = _lesion_status(clauses, i, R["TEAR"])
            for side in targets:
                for st in sts:
                    add("Medial Meniscus" if side == "M" else "Lateral Meniscus", st, sc)


def osteoarthritis(clauses, add, ctx):
    names = {"M": "Medial OA", "L": "Lateral OA", "PF": "PF OA"}
    for i, sc in enumerate(clauses):
        hits = list(R["OA_STRONG"].finditer(sc))
        if R["CARTILAGE"].search(sc):
            hits += list(R["CARTILAGE_DAMAGE"].finditer(sc))
        if not hits:
            continue
        comps = {s for _, s in side_words(sc)}
        if R["PATELLOFEMORAL"].search(sc):
            comps.add("PF")
        if R["TRICOMPARTMENTAL"].search(sc):
            comps = {"M", "L", "PF"}
        generic = not comps
        if generic and ctx["side"]:
            comps, generic = {ctx["side"]}, False
        sts = [assess(sc, h.start(), h.end(), intensity=True) for h in hits]
        st = NEG if NEG in sts and POS not in sts else max(sts, key=[NEG, MIN, WEAK, UNC, MILD, POS].index)
        if generic:
            if st in (POS, MILD) and R["OA_STRONG"].search(sc):
                for c in names.values():
                    add(c, UNC, sc)
            continue
        for c in comps:
            add(names[c], st, sc)


def simple_findings(clauses, add):
    spec = (("Effusion", "EFFUSION", True), ("Synovitis", "SYNOVITIS", True), ("Baker's", "BAKER", False),
            ("Contusion", "CONTUSION", False), ("Fracture", "FRACTURE", False))
    for sc in clauses:
        for label, key, intensity in spec:
            for m in R[key].finditer(sc):
                add(label, assess(sc, m.start(), m.end(), intensity), sc)


def update_ctx(sent, ctx):
    """Lado del compartimento vigente ("In the medial compartment, ...", "MEDIAL COMPARTMENT:")."""
    head = " ".join(_W.findall(sent)[:5])
    if R["COMPARTMENT"].search(sent):
        tc = [t for t, _, _ in _toks(R["COMPARTMENT"], sent)]
        near = [s for t, s in side_words(sent) if any(abs(t - c) <= 3 for c in tc)]
        if len(set(near)) == 1:
            ctx["side"] = near[0]
            return
    if R["PATELLOFEMORAL"].search(head) and (_JOINT.search(head) or R["COMPARTMENT"].search(head)):
        ctx["side"] = "PF"
    elif _NOTCH.search(head):
        ctx["side"] = None


# ---------- agregación ----------
STATE = {  # estado -> (label, conf, score)
    POS: (1, 0.9, 0.95), MILD: (1, 0.6, 0.7), UNC: (-1, 0.5, 0.5), WEAK: (-1, 0.4, 0.4),
    MIN: (0, 0.7, 0.2), "NONE": (0, 0.6, 0.15), NEG: (0, 0.9, 0.05),
}
PRIORITY = [POS, MILD, UNC, WEAK, MIN, NEG]


def extract(report):
    """Devuelve ({label: (valor, conf, score, estado)}, evidencia)."""
    found = defaultdict(list)
    evidence = []
    ctx = {"side": None}

    def add(label, status, text):
        found[label].append(status)
        evidence.append((label, status, text))

    for sent in split_sentences(report):
        update_ctx(sent, ctx)
        clauses = [c.strip() for c in _CLAUSE.split(sent) if c and c.strip()]
        ligaments(clauses, add)
        menisci(clauses, add, ctx)
        osteoarthritis(clauses, add, ctx)
        simple_findings(clauses, add)

    out = {}
    for label in LABELS:
        sts = found.get(label, [])
        state = next((s for s in PRIORITY if s in sts), "NONE")
        val, conf, score = STATE[state]
        if state == POS and sts.count(POS) > 1:
            conf, score = 0.95, 0.97
        out[label] = (val, conf, score, state)
    return out, evidence


def run(df):
    rows, ev_rows = [], []
    for uid, rep in zip(df["StudyInstanceUID"], df["Report"]):
        res, ev = extract(rep)
        row = {"StudyInstanceUID": uid}
        for label, (val, conf, score, state) in res.items():
            row[label] = val
            row[f"{label}__conf"] = conf
            row[f"{label}__score"] = score
            row[f"{label}__state"] = state
        rows.append(row)
        ev_rows.append({"StudyInstanceUID": uid, "evidence": ev})
    return pd.DataFrame(rows), ev_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explain", help="sufijo del StudyInstanceUID")
    args = ap.parse_args()
    df = load_train()

    if args.explain:
        r = df[df["StudyInstanceUID"].str.endswith(args.explain)].iloc[0]
        res, ev = extract(r["Report"])
        print(f"{r['StudyInstanceUID']} [{r['lang']}]")
        for label in LABELS:
            gold = "" if pd.isna(r[label]) else f"  gold={int(r[label])}"
            print(f"  {label:17s} {res[label][3]:5s} -> {res[label][0]:2d}{gold}")
        print()
        for label, st, txt in ev:
            print(f"  [{label} {st}] {txt[:160]}")
        return

    out, ev = run(df)
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / "weak_labels.csv", index=False)
    with open(OUT / "weak_evidence.jsonl", "w") as f:
        for e in ev:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"{len(out)} estudios -> {OUT / 'weak_labels.csv'}")
    print((out[LABELS] == 1).mean().round(3).to_string())


if __name__ == "__main__":
    main()
