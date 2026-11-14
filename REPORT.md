# Informe final — Trabajo Final PDI
## Flujo de trabajo completo para análisis de imágenes dermatoscópicas (dataset IMD, 200 imágenes)

**Clases:** Common Nevus (80) · Atypical Nevus (80) · Melanoma (40)
**Split estratificado (semilla 42):** train 120 / val 40 / test 40 (test: 16/16/8)

---

## 1. Preprocesamiento
- **Eliminación de artefactos** (vello, burbujas): DullRazor multispectral — black-hat
  elíptico (kernels 9/17/25) + lineal (0/45/90/135°), histéresis de umbrales
  (12/25, adaptados en franja periférica), componentes con área mínima, **inpainting
  Telea** con radio 3 y mezcla difuminada (feathering) para ocultar bordes duros.
  Se omite el inpainting si la máscara de vello supera el 40 % de la imagen.
- **Corrección de iluminación:** Multi-Scale Retinex (σ = 15/80/250) sobre la
  luminancia Y, recompuesta con los cromas originales; CLAHE disponible como
  alternativa conservadora.
- Salida: `data/processed/preprocessed/` (200 imágenes + máscaras de vello).

## 2. Segmentación
| Método | Dice (test) | IoU (test) |
|---|---|---|
| Línea base clásica: Otsu en LAB (votación 2-de-3 de canal a-CLAHE, L invertida, a+b) + morfología + componente conexa sesgada al centro | 0.470 ± 0.247 (todo el dataset) | 0.346 |
| **U-Net** (32 filtros base, BCE + Dice loss, aumentación flips/rotaciones, 20 épocas CPU, early-best en val) | **0.852** | **0.750** |
| DeepLabv3+ (ResNet-50) | implementado en `segmentation_deep.py --arch deeplab` (no entrenado por tiempo CPU) | |

**Posprocesado** (`postprocess_mask`): apertura+cierre morfológicos para suavizar el
contorno, eliminación de componentes < 200 px (falsos positivos) y relleno de huecos.

Resultados por imagen: `results/segmentation/unet.csv` y `classical.csv`.

## 3. Extracción de características (área de la lesión)
- **Color:** media/desviación + histograma 16 bins de H, S, V (HSV) y L, a, b (CIE-Lab) → 96 rasgos.
- **Textura:** LBP uniforme (P=8, R=1, 10 bins) + Haralick/GLCM (contraste, energía,
  homogeneidad, correlación a distancias 1 y 3; entropía) → 19 rasgos.
- **Forma:** área relativa, compacidad 4πA/P², excentricidad, ratio de ejes,
  solidez y defecto de convexidad, asimetría axial, rugosidad del borde → 8 rasgos.
- **Aprendidas:** embeddings de 512-d de ResNet-18 (ImageNet) sobre el recorte de la lesión.

## 4. Clasificación (test n = 40)
| Modelo | acc | balanced acc | macro-F1 | AUC OVR |
|---|---|---|---|---|
| SVM RBF (C=10, γ=0.001, CV 5-fold) | 0.450 | 0.479 | 0.481 | **0.777** |
| Random Forest (300 árboles) | 0.550 | 0.583 | 0.583 | 0.773 |
| SVM sobre embeddings CNN | 0.525 | 0.542 | 0.541 | 0.686 |
| Ensemble votación suave | 0.475 | 0.500 | 0.499 | 0.772 |
| Ensemble stacking (Reg. Log.) | 0.500 | 0.542 | 0.531 | 0.757 |
| ResNet-18 (transferencia + fine-tuning) | 0.450 | 0.479 | 0.475 | — |
| **EfficientNet-B0 (transferencia + fine-tuning)** | **0.550** | **0.604** | **0.568** | — |

- **Recall de melanoma:** 6/8 (ensemble stacking), **7/8** (EfficientNet-B0). Los
  errores van hacia Atypical Nevus, nunca directo a Common Nevus.
- **Rasgos dominantes (RF):** histograma de L (Lab), Haralick (contraste,
  homogeneidad, entropía), medias de color — coherente con la regla ABCD.
- Con solo 120 imágenes de train, los ensembles no superan claramente al mejor
  modelo individual; el AUC alto (0.77–0.78) indica que los rasgos clásicos
  ordenan bien los casos y el test pequeño (40) domina la varianza del accuracy.

## 5. Explicabilidad (`src/explain.py`)
- **Grad-CAM** sobre la última etapa conv del clasificador CNN: localiza las
  regiones que activaron la clase ganadora.
- **LIME-style** para el clasificador clásico: superpíxeles SLIC dentro de la
  lesión, perturbaciones con fondo de piel media y ajuste de Ridge ponderado →
  saliencia por superpíxel que respalda la decisión del SVM/RF.
- Informes visuales por imagen en `results/pipeline/<id>/report.png`
  (original, preprocesada, máscara de vello, GT, U-Net, clásica, Grad-CAM, LIME).
- Demo destacada: **IMD058** (melanoma): Dice 0.837 y **p(Melanoma)=0.89**.

## 6. Reproducción (venv-dl, Python 3.12, torch 2.15 CPU)
```bash
venv-dl/bin/python src/preprocessing.py
venv-dl/bin/python src/eval_segmentation.py
venv-dl/bin/python src/segmentation_deep.py --arch unet --epochs 20
venv-dl/bin/python src/classifiers.py
venv-dl/bin/python src/classifier_deep.py --arch resnet18 --epochs 20
venv-dl/bin/python src/classifier_deep.py --arch efficientnet_b0 --epochs 15
venv-dl/bin/python src/run_pipeline.py --id IMD058
```

## 7. Limitaciones y trabajo futuro
- 200 imágenes es un dataset pequeño: más aumentación y k-fold repetido darían
  estimaciones más estables que un único split 60/20/20.
- La U-Net se entrena a 256 px; entrenar a resolución completa (572×765) con
  recortes por tiles debería mejorar los contornos finos.
- Posprocesado activo de contornos (snakes/active contours) y ensembles de
  segmentación (U-Net + DeepLab) son extensiones naturales.
