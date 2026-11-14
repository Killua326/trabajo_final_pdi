"""
Segmentación clásica de la lesión (línea base antes de redes).

Método: en dermatoscopia la lesión se distingue del piel sana sobre todo en
el plano de color a*b* de CIE-Lab y por absorción del canal azul. Estrategia:
  1) Imagen preprocesada -> LAB; CLAHE suave sobre el canal a.
  2) Otsu sobre (a + b) combinados con umbral sobre el canal Y del canal azul.
  3) Fusión de umbrales (intersección) + morfología + mayor componente conexa.
Se evalúa contra la máscara GT con Dice/IoU.
"""

import cv2
import numpy as np


def otsu_map(channel):
    ch = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    th, m = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m > 0


def _fix_polarity(m):
    """Corrige la polaridad del umbral: el borde de la imagen es piel sana,
    por lo que si la máscara cubre los bordes está invertida."""
    border = m[0, 0] or m[0, -1] or m[-1, 0] or m[-1, -1]
    return ~m if border else m


def _largest_cc_center(mask, shape):
    """Componente conexa mayor con sesgo hacia el centro de la imagen
    (la lesión suele estar centrada en dermatoscopia)."""
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros(shape, np.uint8)
    h, w = shape
    best, best_score = 0, -1.0
    for i in range(1, n):
        cx, cy = cent[i]
        score = stats[i, cv2.CC_STAT_AREA] * np.exp(
            -((cx - w / 2) ** 2 + (cy - h / 2) ** 2) / (2 * (0.35 * w) ** 2)
        )
        if score > best_score:
            best_score, best = score, i
    return (lab == best).astype(np.uint8)


def classical_segment(img_bgr):
    """Máscara binaria (uint8 0/1) de la lesión a partir de imagen BGR preprocesada.

    Fusión por votación de 3 candidatos Otsu (canal a con CLAHE, luminancia
    invertida y a+b combinados), morfología, mayor componente conexa centrada
    y relleno de huecos.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = [x.astype(np.float32) for x in cv2.split(lab)]
    h, w = L.shape
    a_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(a.astype(np.uint8)).astype(np.float32)

    m_a = _fix_polarity(otsu_map(a_eq))            # plano a* (rojo-marrón)
    m_L = _fix_polarity(otsu_map(255 - L))         # luminancia invertida
    m_ab = _fix_polarity(otsu_map(a_eq + b))       # plano a*+b*

    fused = m_a.astype(np.uint8) + m_L.astype(np.uint8) + m_ab.astype(np.uint8)
    cand = (fused >= 2).astype(np.uint8) * 255     # votación 2-de-3

    cand = cv2.morphologyEx(
        cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    )
    cand = cv2.morphologyEx(
        cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )

    mask = _largest_cc_center(cand, (h, w))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(mask, cnts, -1, 1, thickness=cv2.FILLED)
    return mask


def dice_iou(pred, gt):
    """Dice e IoU entre dos máscaras binarias 0/1."""
    inter = np.logical_and(pred > 0, gt > 0).sum()
    p, g = (pred > 0).sum(), (gt > 0).sum()
    dice = 2.0 * inter / (p + g + 1e-9)
    iou = inter / (p + g - inter + 1e-9)
    return float(dice), float(iou)


def postprocess_mask(mask, min_area=200, smooth=9):
    """Posprocesado de una máscara de red: quita FP pequeños, suaviza contorno.

    Args:
        mask: uint8 0/1 o 0/255.
        min_area: elimina componentes con área menor (falsos positivos).
        smooth: tamaño de kernel para suavizado del contorno (impar, >=3).
    """
    m = (mask > 127 if mask.max() > 1 else mask > 0).astype(np.uint8)
    if smooth >= 3:
        k = smooth | 1
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        keep = stats[:, cv2.CC_STAT_AREA] >= min_area
        keep[0] = False
        m = keep[labels].astype(np.uint8)
    # relleno de huecos internos
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(m, cnts, -1, 1, thickness=cv2.FILLED)
    return m
