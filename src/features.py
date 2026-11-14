"""
Extracción de rasgos de la lesión a partir de su máscara (binaria 0/1) y su
imagen preprocesada BGR.

Grupos de descriptores (todos concatenados en extract_all()):
  - COLOR   : estadísticos (media/desv) por canal en HSV y CIE-Lab + histogramas
              normalizados (16 bins) de H, S, L, a, b dentro de la lesión.
  - TEXTURA : LBP uniforme (P=8, R=1) en la lesión + Haralick (GLCM) con
              contrast/energy/homogeneity/correlation/entropy en 2 escalas.
  - FORMA   : área relativa, perímetro, compacidad (4*pi*A/P^2), excentricidad,
              solidez (A/A_convexa) -> defectos de convexidad, asimetría axial,
              ratio de ejes (minor/major), número de defectos cóncavos.
  - DEEP    : embedding de 512-d de una ResNet-18 ImageNet recortada a la lesión
              (se calcula aparte en features_deep.py para no exigir torch aquí).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import label as sk_label, regionprops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ------------------------------- COLOR -------------------------------
def color_features(img_bgr, mask, bins=16):
    """Estadísticos + histogramas de color dentro de la lesión."""
    m = mask > 0
    feats, names = [], []
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    for i, ch in enumerate([hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2], lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]]):
        vals = ch[m].astype(np.float32)
        feats += [vals.mean(), vals.std()]
        names += [f"color_mean_{i}", f"color_std_{i}"]
        hist, _ = np.histogram(vals, bins=bins, range=(0, 256))
        feats += (hist / (hist.sum() + 1e-9)).tolist()
        names += [f"color_hist_{i}_{j}" for j in range(bins)]
    return np.array(feats, dtype=np.float32), names


# ------------------------------- TEXTURA -------------------------------
def texture_features(img_bgr, mask):
    """LBP uniforme + Haralick dentro de la lesión."""
    m = mask > 0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_m = np.where(m, gray, 0).astype(np.uint8)

    feats, names = [], []

    # LBP: histograma uniforme (bins 0..9 -> 10 valores) sólo en la lesión
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_les = lbp[m]
    hist, _ = np.histogram(lbp_les, bins=10, range=(0, 10))
    feats += (hist / (hist.sum() + 1e-9)).tolist()
    names += [f"lbp_{j}" for j in range(10)]

    # Haralick (GLCM) a dos escalas de distancia, máscara aplicada por apagado
    # del fondo (pixeles fuera de la lesión = nivel medio para no contaminar).
    bg = int(np.median(gray[m])) if m.sum() else 0
    for dist in (1, 3):
        g2 = np.where(m, gray, bg).astype(np.uint8)
        glcm = graycomatrix(g2, distances=[dist], angles=[0, np.pi / 2], levels=256, symmetric=True, normed=True)
        for prop in ("contrast", "energy", "homogeneity", "correlation"):
            feats.append(float(graycoprops(glcm, prop).mean()))
            names.append(f"haralick_{prop}_d{dist}")
    # entropía aproximada desde el GLCM d=1
    glcm = graycomatrix(g2, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    p = glcm.ravel()
    p = p / (p.sum() + 1e-9)
    feats.append(float(-(p[p > 0] * np.log2(p[p > 0])).sum()))
    names.append("haralick_entropy_d1")
    return np.array(feats, dtype=np.float32), names


# ------------------------------- FORMA -------------------------------
def shape_features(mask):
    """Geometría del contorno de la lesión."""
    m = (mask > 0).astype(np.uint8)
    area = float(m.sum())
    conts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not conts:
        return np.zeros(8, np.float32), ["shape_dummy"] * 8
    cnt = max(conts, key=cv2.contourArea)
    perim = float(cv2.arcLength(cnt, True))
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull)) + 1e-9
    hull_perim = float(cv2.arcLength(hull, True)) + 1e-9

    feats, names = [], []

    # 1) área relativa a la imagen
    feats.append(area / m.size)
    names.append("shape_area_frac")

    # 2) compacidad ( circularidad )
    feats.append(4 * np.pi * area / (perim ** 2 + 1e-9))
    names.append("shape_compacity")

    # 3-4) excentricidad y ejes de la región
    lbl = sk_label(m)
    region = max(regionprops(lbl), key=lambda r: r.area) if len(regionprops(lbl)) else None
    ecc = float(region.eccentricity) if region else 0.0
    axis_ratio = float(region.minor_axis_length / (region.major_axis_length + 1e-9)) if region else 0.0
    feats += [ecc, axis_ratio]
    names += ["shape_eccentricity", "shape_axis_ratio"]

    # 5-6) defectos de convexidad: solidez y su complemento
    solidity = area / hull_area
    feats += [solidity, 1.0 - solidity]
    names += ["shape_solidity", "shape_convexity_defect"]

    # 7) asimetría axial: comparar mitades respecto a los 2 ejes principales
    flip_scores = []
    for flip in (0, 1):
        mm = np.fliplr(m) if flip == 1 else m
        flip_scores.append(float(np.logical_and(mm != m, m > 0).sum()) / (area + 1e-9))
    feats.append(np.mean(flip_scores))
    names.append("shape_asymmetry")

    # 8) rugosidad del borde: perímetro real / perímetro del convex hull
    feats.append(perim / hull_perim)
    names.append("shape_border_roughness")
    return np.array(feats, dtype=np.float32), names


# ------------------------------- COMPLETO -------------------------------
def extract_all(img_bgr, mask):
    """Concatena color + textura + forma. Devuelve (vector, nombres)."""
    v1, n1 = color_features(img_bgr, mask)
    v2, n2 = texture_features(img_bgr, mask)
    v3, n3 = shape_features(mask)
    return np.concatenate([v1, v2, v3]), n1 + n2 + n3


def extract_dataset_features(ids, preprocessed_dir, feature_names_out=None):
    """Extrae rasgos de muchas imágenes preprocesadas.

    La máscara usada para delimitar la lesión puede ser la GT (modo desarrollo)
    o una máscara predicha (modo inferencia). Aquí usamos la GT para evaluar
    el clasificador de forma aislada; en run_pipeline.py se usan máscaras del
    propio pipeline.
    """
    X, y, used = [], [], []
    from data_utils import list_dataset, load_image_and_mask

    _, labels = list_dataset()
    for img_id in ids:
        img, mask = load_image_and_mask(img_id, preprocessed_dir=preprocessed_dir)
        vec, _ = extract_all(img, mask)
        X.append(vec)
        y.append(labels[img_id])
        used.append(img_id)
    return np.vstack(X), np.array(y), used


if __name__ == "__main__":
    from data_utils import load_image_and_mask

    img, mask = load_image_and_mask("IMD002")
    vec, names = extract_all(img, mask)
    print(f"{len(vec)} rasgos extraídos. Primeros 12:")
    for n, v in list(zip(names, vec))[:12]:
        print(f"  {n:28s} = {v:.4f}")
