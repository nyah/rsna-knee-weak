# rsna-knee-weak

Detección de 12 hallazgos en resonancia magnética de rodilla (competencia RSNA Knee Abnormality Detection) entrenando con etiquetas débiles extraídas de los reportes radiológicos.

Los datos de la competencia y sus derivados no se incluyen en este repositorio.

## Estructura

```
src/labels/     extractor de etiquetas débiles por reglas (9 idiomas) y validación
src/data/       selección del subconjunto y conversión DICOM -> .npy
src/models/     CNN propia, backbones con transfer learning y agregación MIL por estudio
src/train/      datos 2.5D, pérdidas, aumentos y entrenamiento
src/eval/       métricas con IC95 por bootstrap
src/interpret/  Grad-CAM
src/pipeline/   orquestación de los notebooks en Kaggle por API
configs/        hiperparámetros de los modelos y de la ablación
notebooks/      notebook 01 (EDA) y build_*.py, que generan los notebooks 02 a 05
run_all.sh      pipeline completo con un solo comando
```

## Entorno

- Local: Python 3.11 y `pip install -r requirements.txt`. Corre las etiquetas débiles, la evaluación y la orquestación de Kaggle.
- Kaggle: los notebooks corren en la imagen estándar con GPU T4. `requirements-kaggle.txt` registra las versiones de torch y torchvision usadas.
- CLI de Kaggle autenticada con una cuenta del equipo que haya aceptado las reglas de la competencia: `kaggle auth login`.

## Ejecución completa

```
bash run_all.sh
```

Pasos, en orden:

1. `data`: baja `train.csv` de la competencia y genera los notebooks 02 a 05 desde `src/`.
2. `eda`: corre el notebook 01 en Kaggle (catálogo de series e idioma de cada reporte).
3. `labels`: etiquetas débiles en local, validación contra los dorados y comparación con embeddings; sube una versión nueva del Dataset privado de etiquetas solo si cambiaron.
4. `subset`, `train`, `ablation`, `gradcam`: corren los notebooks 02 a 05 en Kaggle, encadenados por sus outputs, y bajan los resultados livianos.
5. `eval`: AUC macro y por etiqueta con IC95, precisión, recall y F1, bootstrap pareado y desglose por idioma.

Los resultados quedan en `outputs/run_all/`. Con GPU T4 el total toma unas 2 horas, la mayor parte en la ablación.

Opciones: `--from <paso>` retoma desde un paso y `--no-kaggle-run` no vuelve a correr los notebooks, solo baja el output de su última versión. Los recursos de Kaggle (cuenta, slugs y entradas de cada notebook) están en `configs/kaggle_pipeline.json`.

## Uso por partes

Local (Python 3.11):

```
pip install -r requirements.txt
python -m src.labels.split_gold
python -m src.labels.rules
python -m src.labels.validate --split test
```

`RSNA_KNEE_DATA` indica la carpeta con los CSV de la competencia (por defecto `data/`).

Kaggle: `python notebooks/build_0X_*.py` genera cada notebook con el código de `src/` incrustado. Se corren en orden 02 (subconjunto), 03 (entrenamiento), 04 (ablación) y 05 (Grad-CAM), los tres últimos con GPU.

Evaluación, con las corridas descargadas en `outputs/runs_v1/`:

```
python -m src.eval.evaluate --runs outputs/runs_v1/*/ --ref resnet18_base --out outputs/eval_ablation_v1
```
