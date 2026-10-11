#!/usr/bin/env bash
# Pipeline completo: etiquetas débiles (local) -> notebooks 01 a 05 (Kaggle, por API) -> evaluación (local).
#
#   bash run_all.sh                    # todo, desde cero
#   bash run_all.sh --from train       # retoma desde un paso: data eda labels subset train ablation gradcam eval
#   bash run_all.sh --no-kaggle-run    # no corre los notebooks: baja el output de su última versión en Kaggle
#
# Requisitos: entorno de requirements.txt activo y la CLI de Kaggle autenticada (kaggle auth login).
# Resultados en outputs/run_all/. Tiempo aproximado con T4: 2 h (la ablación es la parte larga).
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python}
FROM=data
RUN_KAGGLE=1
while [[ $# -gt 0 ]]; do
  case $1 in
    --from) FROM=$2; shift 2 ;;
    --no-kaggle-run) RUN_KAGGLE=0; shift ;;
    *) echo "opción desconocida: $1"; exit 1 ;;
  esac
done

STEPS=(data eda labels subset train ablation gradcam eval)
want() {  # true si el paso $1 está en o después de --from
  local started=0 s
  for s in "${STEPS[@]}"; do
    [[ $s == "$FROM" ]] && started=1
    [[ $s == "$1" ]] && { [[ $started == 1 ]]; return; }
  done
  return 1
}
[[ " ${STEPS[*]} " == *" $FROM "* ]] || { echo "paso desconocido: $FROM"; exit 1; }

kaggle_nb() {  # corre (o solo baja) el notebook $1
  if [[ $RUN_KAGGLE == 1 ]]; then $PY -m src.pipeline.kaggle_jobs run "$1"
  else $PY -m src.pipeline.kaggle_jobs download "$1"; fi
}

OUT=outputs/run_all
K=$OUT/kaggle
mkdir -p "$OUT"

if want data; then
  echo "== datos: train.csv de la competencia"
  $PY -m src.pipeline.kaggle_jobs fetch-data
  for b in notebooks/build_0*.py; do $PY "$b"; done
fi

if want eda; then
  echo "== 01 EDA (Kaggle, CPU)"
  kaggle_nb 01
  cp "$(find $K/01 -name study_lang.csv | head -1)" data/study_lang.csv
fi

if want labels; then
  echo "== etiquetas débiles (local)"
  $PY -m src.labels.split_gold
  $PY -m src.labels.rules
  $PY -m src.labels.validate --split test | tee $OUT/validate_test.txt
  $PY -m src.labels.embed
  $PY -m src.labels.compare | tee $OUT/compare_embeddings.txt
  if [[ $RUN_KAGGLE == 1 ]]; then $PY -m src.pipeline.kaggle_jobs upload-labels; fi
fi

if want subset;   then echo "== 02 subconjunto (Kaggle, CPU)"; kaggle_nb 02; fi
if want train;    then echo "== 03 entrenamiento (Kaggle, T4)"; kaggle_nb 03; fi
if want ablation; then echo "== 04 ablación (Kaggle, T4)";      kaggle_nb 04; fi
if want gradcam;  then echo "== 05 Grad-CAM (Kaggle, T4)";       kaggle_nb 05; fi

if want eval; then
  echo "== evaluación (local)"
  RUNS=$OUT/runs
  rm -rf "$RUNS"; mkdir -p "$RUNS"
  for m in $(find $K/03 $K/04 -name metrics.json); do
    d=$(dirname "$m"); cp -R "$d" "$RUNS/$(basename "$d")"
  done
  $PY -m src.eval.evaluate --runs $RUNS/resnet18_base $RUNS/scratch_base --out $OUT/eval
  $PY -m src.eval.evaluate --runs $RUNS/*/ --ref resnet18_base --out $OUT/eval_ablation
  echo "listo: resultados en $OUT/"
fi
