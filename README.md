# Trabajo Final PDI — Flujo de trabajo completo en dermatoscopia

Pipeline end-to-end para localizar lesiones en imágenes dermoscópicas (dataset IMD,
200 imágenes, 3 clases: Common Nevus / Atypical Nevus / Melanoma), extraer rasgos,
clasificar y **explicar** la decisión con mapas de saliencia.

## Estructura

```
images/images/          200 imágenes dermoscópicas IMD (572x765)
images/masks/           máscaras GT de segmentación
data/raw/labels/        etiquetas (3 clases)
data/processed/preprocessed/   imágenes tras vello+inpainting+MSR
src/
  data_utils.py           carga, split estratificado 60/20/20, constantes
  preprocessing.py        vello (DullRazor multispectral) + Retinex (MSR)
  segmentation_classical.py  línea base clásica (Otsu en LAB + morfología + CC)
  segmentation_deep.py    U-Net y DeepLabv3+ (entrenamiento + eval Dice/IoU)
  eval_segmentation.py    evaluación de segmentación clásica
  features.py             color (HSV/Lab), textura (LBP, Haralick), forma
  features_deep.py        embeddings ResNet-18 (512-d) del recorte de la lesión
  classifiers.py          SVM, Random Forest, embeddings + ensembles
  classifier_deep.py      transferencia + fine-tuning ResNet18 / EfficientNet-B0
  explain.py              Grad-CAM, Integrated Gradients y LIME-style + métricas de fidelidad
  eval_explain.py         comparación cualitativa/cuantitativa de técnicas explicativas
  evaluation.py           ROC/AUC por clase, matrices de confusión y CV estratificada
  app.py                  interfaz web local (Flask): pipeline, comparación y resultados
  run_pipeline.py         demo end-to-end de una imagen + reporte visual
models/                  pesos (.pt) y clasificadores (.joblib)
results/                 métricas y reportes
```

## Cómo ejecutar (venv-dl, Python 3.12)

```bash
source venv-dl/bin/activate    # o usar venv-dl/bin/python directamente

# 1. Preprocesamiento de las 200 imágenes (vello + iluminación)
python src/preprocessing.py --illumination msr

# 2. Segmentación
python src/eval_segmentation.py          # línea base clásica
python src/segmentation_deep.py --arch unet --epochs 20

# 3. Clasificación
python src/classifiers.py                # SVM / RF / embeddings / ensembles
python src/classifier_deep.py --arch resnet18 --epochs 20
python src/classifier_deep.py --arch efficientnet_b0 --epochs 15

# 4. Evaluación (requisito 6)
python src/evaluation.py                 # ROC por clase + confusión (test) y CV clásica 5 folds
python src/evaluation.py --deep          # añade CV estratificada de la CNN (lento en CPU)

# 5. Explicabilidad (requisito 5)
python src/eval_explain.py --n 12        # Grad-CAM vs Integrated Gradients vs LIME + fidelidad
python src/eval_explain.py --ids IMD039 IMD058   # imágenes concretas

# 6. Demo completa de una imagen (reporte con Grad-CAM, IG y LIME)
python src/run_pipeline.py --id IMD039   # IMD039 = melanoma; IMD003 = nevo común
```

## Interfaz web (Flask)

Para no ejecutar scripts uno a uno:

```bash
venv-dl/bin/python src/app.py            # → http://127.0.0.1:5000
```

- **Dataset**: galería por clase con miniaturas; cada imagen tiene botones
  *Pipeline* y *Comparar*.
- **Pipeline**: ejecuta preprocesamiento → segmentación → clasificación →
  explicabilidad y muestra resultado, probabilidades, Dice y todas las etapas.
  Los resultados quedan cacheados en `results/pipeline/<id>/`; `?rerun=1`
  fuerza la re-ejecución.
- **Comparar**: segmentación clásica vs U-Net (Dice/IoU contra GT) y
  Grad-CAM vs Integrated Gradients vs LIME con su fidelidad (pointing game,
  IoU@top20%, AUC de saliencia).
- **Resultados**: dashboard con métricas de test, AUC por clase, curvas ROC,
  matrices de confusión, CV estratificada y comparación de explicabilidad.

Requiere `flask` en el entorno: `venv-dl/bin/python -m pip install flask`.

## Resultados principales

### Segmentación (Dice/IoU contra máscaras GT)

| Método | Dice | IoU | Nota |
|---|---|---|---|
| Clásica (Otsu en LAB + morfología + CC) | 0.470 ± 0.247 | 0.346 | 200 imágenes |
| **U-Net** (20 épocas, CPU) | **0.852** (test) | **0.750** | test n=40, sin posprocesado fino |

### Clasificación (test n = 40: 16 Common, 16 Atypical, 8 Melanoma)

| Modelo | acc | balanced acc | macro-F1 | AUC OVR |
|---|---|---|---|---|
| SVM RBF (rasgos clásicos) | 0.450 | 0.479 | 0.481 | 0.777 |
| Random Forest (rasgos clásicos) | 0.550 | 0.583 | 0.583 | 0.773 |
| SVM (embeddings ResNet-18) | 0.525 | 0.542 | 0.541 | 0.686 |
| Ensemble votación suave (3 vistas) | 0.475 | 0.500 | 0.499 | 0.772 |
| Ensemble stacking (Reg. Logística) | 0.500 | 0.542 | 0.531 | 0.757 |
| ResNet-18 transferencia+FT | 0.450 | 0.479 | 0.475 | — |
| **EfficientNet-B0 transferencia+FT** | **0.550** | **0.604** | **0.568** | — |

Recall de melanoma: 6/8 ensemble stacking, 7/8 EfficientNet-B0 (los modelos dudosos
suelen confundirse con Atypical Nevus, no con Common Nevus — buena señal clínica).

Rasgos más importantes (RF): histograma L de Lab, contraste/homogeneidad/entropía
de Haralick, medias de color — consistente con la regla ABCD (color y textura).

### Evaluación avanzada (`results/classification/`)

**Curvas ROC y AUC por clase** (one-vs-rest, split de test, `roc_curves.png`):

| Modelo | Common Nevus | Atypical Nevus | Melanoma | macro |
|---|---|---|---|---|
| SVM RBF (rasgos) | 0.771 | 0.612 | **0.949** | 0.777 |
| Random Forest (rasgos) | 0.755 | 0.617 | 0.945 | 0.773 |
| SVM (embeddings) | 0.721 | 0.576 | 0.762 | 0.686 |
| CNN ResNet-18 | 0.729 | 0.482 | 0.863 | 0.691 |

En las 3 clases el melanoma es la mejor discriminada; Atypical Nevus es la más
difícil (frontera intermedia por diseño del problema).

**Validación cruzada estratificada (5 folds, n=200, out-of-fold):**

| Modelo | balanced acc | AUC OVR macro | Recall melanoma (OOF) |
|---|---|---|---|
| SVM RBF (rasgos) | 0.600 ± 0.074 | 0.790 ± 0.038 | 32/40 (80%) |
| Random Forest (rasgos) | 0.679 ± 0.060 | 0.822 ± 0.031 | 32/40 (80%) |
| CNN ResNet-18 (6 épocas/fold) | 0.663 ± 0.058 | 0.843 ± 0.059 | 32/40 (80%) |

Fuentes: `cv_classical.json`, `cv_deep.json`, `cv_roc_curves.png`,
`cv_confusion.png`, `cv_deep_roc.png`, `cv_deep_confusion.png`.

### Explicabilidad y fidelidad (`results/explainability/`)

Técnicas: **Grad-CAM**, **Integrated Gradients** (línea base blur, Blur-IG) para
la CNN y **LIME-style** (superpíxeles SLIC) para el clasificador clásico.
Fidelidad medida contra la máscara GT de la lesión (media sobre 12 imágenes de
test estratificadas): `pointing_game` = el píxel de máxima saliencia cae en la
lesión; `IoU@top20%` = IoU(top-20% saliencia, máscara GT); `AUC-sal` = AUC de
la saliencia por píxel frente a la máscara GT.

| Técnica | pointing | IoU@top20% | AUC-sal |
|---|---|---|---|
| Grad-CAM | 0.83 | 0.217 ± 0.067 | 0.594 ± 0.157 |
| Integrated Gradients | 0.92 | 0.245 ± 0.028 | **0.677 ± 0.062** |
| LIME (clásico) | 0.83 | **0.297 ± 0.188** | 0.622 ± 0.149 |

Comparación cualitativa por imagen en `comparison.png` (contorno verde = zona
relevante GT). Fuentes: `explain_metrics.csv`, `explain_summary.json`.

### Demo end-to-end (`results/pipeline/<id>/report.png`)

- `IMD058` (melanoma): segmentación Dice 0.837, ensemble → **Melanoma p=0.89**,
  Grad-CAM sobre la región asimétrica y LIME confirmando el área central.
- `IMD003` (nevo común): ensemble → Common Nevus p=0.42.

Los mapas de saliencia se generan con **Grad-CAM**, **Integrated Gradients**
(CNN) y **LIME-style con superpíxeles SLIC** (pipeline clásico), respaldando
ambas decisiones.
