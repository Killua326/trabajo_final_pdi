"""
Evalúa la segmentación clásica contra las máscaras GT en todo el dataset.
Salida: results/segmentation/classical.csv + resumen por consola.
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_utils import PROCESSED_DIR, RESULTS_DIR, list_dataset, load_image_and_mask  # noqa: E402
from segmentation_classical import classical_segment, dice_iou  # noqa: E402

if __name__ == "__main__":
    ids, _ = list_dataset()
    rows = []
    for k, img_id in enumerate(ids, 1):
        img, gt = load_image_and_mask(img_id, preprocessed_dir="preprocessed")
        pred = classical_segment(img)
        d, iou = dice_iou(pred, gt)
        rows.append({"image_id": img_id, "dice": round(d, 4), "iou": round(iou, 4)})
        if k % 40 == 0:
            print(f"  {k}/{len(ids)}")

    out = RESULTS_DIR / "segmentation" / "classical.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "dice", "iou"])
        w.writeheader()
        w.writerows(rows)
    dices = np.array([r["dice"] for r in rows])
    ious = np.array([r["iou"] for r in rows])
    print(f"Segmentación clásica (200 imgs): Dice {dices.mean():.4f} ± {dices.std():.4f} | "
          f"IoU {ious.mean():.4f} ± {ious.std():.4f} | mediana Dice {np.median(dices):.4f}")
