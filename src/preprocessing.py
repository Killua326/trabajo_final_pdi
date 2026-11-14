"""
Pipeline de preprocesamiento para las 200 imágenes dermoscópicas:
  1) Corrección de iluminación: Multi-Scale Retinex (MSR) sobre luminancia,
     con CLAHE como alternativa conservadora.
  2) Eliminación de artefactos (vello/burbujas): reutiliza el algoritmo
     DullRazor multispectral de src/hair_removal_dullrazor.py (black-hat +
     umbrales de histeresis + inpainting Telea + feathering).

Orden elegido: vello -> Retinex. El inpainting se aplica sobre la imagen
original para no propagar el realce del MSR, y el MSR se aplica después
para normalizar la iluminación del resultado ya "limpio".
"""

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hair_removal_dullrazor import remove_hair_multiscale  # noqa: E402


def multi_scale_retinex(img_bgr, sigmas=(15, 80, 250)):
    """MSR sobre el canal de luminancia (Y), recompuesto con los cromas originales."""
    img = img_bgr.astype(np.float32) / 255.0 + 1e-6
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_log = np.log(y)
    retinex = np.zeros_like(y_log)
    for sigma in sigmas:
        illum = cv2.GaussianBlur(y_log, (0, 0), sigma)
        retinex += (y_log - illum)
    retinex /= len(sigmas)
    retinex_norm = cv2.normalize(retinex, None, 0, 1, cv2.NORM_MINMAX)
    out = cv2.cvtColor(cv2.merge([retinex_norm, cr, cb]), cv2.COLOR_YCrCb2BGR)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def clahe_luminance(img_bgr, clip_limit=2.0, tile_grid=(8, 8)):
    """Alternativa conservadora: CLAHE sobre el canal L (CIE-Lab)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def preprocess_image(img_bgr, illumination="msr"):
    """Pipeline completo sobre una imagen BGR.

    Returns:
        (out, hair_mask, status): imagen preprocesada, máscara de vello
        detectada y estado del inpainting.
    """
    cleaned, hair_mask, debug = remove_hair_multiscale(
        img_bgr,
        kernel_sizes=(9, 17, 25),
        low_threshold=12,
        high_threshold=25,
        min_area=15,
        edge_min_area=8,
        dilation_iterations=0,
        inpaint_radius=3.0,
        method="telea",
        max_coverage=0.40,
    )
    out = multi_scale_retinex(cleaned) if illumination == "msr" else clahe_luminance(cleaned)
    return out, hair_mask, debug["status"]


def run_batch(ids=None, out_name="preprocessed", illumination="msr"):
    """Aplica el pipeline a todas (o unas) imágenes y guarda el resultado."""
    from data_utils import RAW_IMAGES_DIR, PROCESSED_DIR, list_dataset

    out_dir = PROCESSED_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if ids is None:
        ids, _ = list_dataset()

    rows, failures = [], []
    t0 = time.perf_counter()
    for k, img_id in enumerate(ids, 1):
        try:
            img = cv2.imread(str(RAW_IMAGES_DIR / f"{img_id}.png"))
            out, hair_mask, status = preprocess_image(img, illumination=illumination)
            cv2.imwrite(str(out_dir / f"{img_id}.png"), out)
            cv2.imwrite(str(out_dir / f"{img_id}_hairmask.png"), hair_mask)
            rows.append({"image_id": img_id, "hair_coverage_pct": 100.0 * np.count_nonzero(hair_mask) / hair_mask.size,
                         "status": status})
            if k % 20 == 0 or k == len(ids):
                print(f"  {k}/{len(ids)} imágenes procesadas ({time.perf_counter() - t0:.0f}s)")
        except Exception as e:
            failures.append(f"{img_id}: {e}")

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "hair_coverage_pct", "status"])
        w.writeheader()
        w.writerows(rows)
    print(f"Listo: {len(rows)} OK, {len(failures)} errores -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Preprocesamiento (vello + iluminación) del dataset IMD")
    ap.add_argument("--ids", nargs="+", default=None, help="IDs concretos; por defecto todas")
    ap.add_argument("--illumination", choices=["msr", "clahe"], default="msr")
    args = ap.parse_args()
    run_batch(ids=args.ids, illumination=args.illumination)
