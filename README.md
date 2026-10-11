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
configs/        hiperparámetros de los modelos y de la ablación
notebooks/      build_*.py generan los notebooks de Kaggle 02 a 05
```

## Uso

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
