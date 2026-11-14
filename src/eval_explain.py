"""
Evaluación de explicabilidad (requisito 5): compara TÉCNICAS EXPLICATIVAS de
forma cualitativa y cuantitativa frente a la zona relevante (máscara GT).

Técnicas (todas sobre el mismo modelo CNN ResNet-18 fine-tuned):
  - Grad-CAM             : activaciones de la última etapa conv.
  - Integrated Gradients : integración de gradientes desde imagen negra.
Y, opcionalmente, LIME-style para el clasificador clásico (RF sobre rasgos).

Fidelidad cuantitativa por imagen y técnica (explain.fidelity_metrics):
  - pointing_game : el píxel de máxima saliencia cae dentro de la lesión GT.
  - iou_top       : IoU(top-20% de saliencia, máscara GT de la lesión).
  - sal_auc       : ROC AUC de la saliencia por píxel contra la máscara GT.

Salidas en results/explainability/:
  - explain_metrics.csv     (una fila por imagen x técnica)
  - explain_summary.json    (media ± desviación por técnica)
  - comparison.png          (paneles cualitativos: GT vs Grad-CAM vs IG vs LIME)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_utils import IDX_TO_CLASS, RESULTS_DIR, load_image_and_mask  # noqa: E402
from explain import GradCAM, fidelity_metrics, integrated_gradients, lime_saliency_classic  # noqa: E402
from features_deep import _to_tensor, crop_lesion, lesion_bbox, load_backbone  # noqa: E402

DEVICE = "cpu"
MARGIN = 12      # el mismo margen usado al entrenar el clasificador
SIZE = 224


def load_cnn(arch="resnet18"):
    """CNN de clasificación con pesos entrenados."""
    import torch
    from classifier_deep import build_model

    ckpt = torch.load(ROOT / "models" / f"classifier_{arch}.pt", map_location=DEVICE, weights_only=False)
    net = build_model(arch).to(DEVICE).eval()
    net.load_state_dict(ckpt["state"])
    return net


def load_classic_rf():
    """Random Forest clásico + escalador (predict_fn para LIME)."""
    import joblib

    mp = ROOT / "models"
    scaler, rf = joblib.load(mp / "classic_scaler.joblib"), joblib.load(mp / "rf_features.joblib")
    return lambda v: rf.predict_proba(scaler.transform(v))


def evaluate_image(img_id, cnn, cam_engine, predict_fn=None, use_lime=True):
    """Devuelve filas de métricas + paneles de visualización para UNA imagen."""
    img, gt = load_image_and_mask(img_id, preprocessed_dir="preprocessed")

    # Recorte idéntico al del entrenamiento, aplicado también a la GT
    y0, y1, x0, x1 = lesion_bbox(gt, MARGIN)
    crop, gt_crop = img[y0:y1, x0:x1], gt[y0:y1, x0:x1]
    x = _to_tensor(crop, SIZE)[None].to(DEVICE)

    # --- Grad-CAM ---
    cam, probs, cls = cam_engine(x)
    cam_full = cv2.resize(cam, (crop.shape[1], crop.shape[0]))
    m_cam = fidelity_metrics(cam_full, gt_crop)

    # --- Integrated Gradients ---
    ig, _, _ = integrated_gradients(cnn, x, class_idx=cls)
    ig_full = cv2.resize(ig, (crop.shape[1], crop.shape[0]))
    m_ig = fidelity_metrics(ig_full, gt_crop)

    rows = [
        {"img_id": img_id, "technique": "Grad-CAM", "pred_class": IDX_TO_CLASS[cls], **m_cam},
        {"img_id": img_id, "technique": "IntegratedGradients", "pred_class": IDX_TO_CLASS[cls], **m_ig},
    ]

    # --- LIME-style (clasificador clásico, resolución completa) ---
    if use_lime and predict_fn is not None:
        sal, _, _ = lime_saliency_classic(img, gt, predict_fn)
        m_lime = fidelity_metrics(sal, gt)
        rows.append({"img_id": img_id, "technique": "LIME_classic", "pred_class": IDX_TO_CLASS[cls], **m_lime})

    # Paneles RGB para la figura
    def overlay(sal):
        h = (np.clip(sal, 0, 1) * 255).astype(np.uint8)
        h = cv2.applyColorMap(h, cv2.COLORMAP_JET)
        return cv2.addWeighted(crop, 0.55, h, 0.45, 0)[:, :, ::-1]

    edge = gt_crop.copy()
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    base = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).copy()
    cv2.drawContours(base, contours, -1, (0, 255, 0), 2)

    panels = {
        "GT": base,
        "Grad-CAM": overlay(cam_full),
        "IntegratedGradients": overlay(ig_full),
    }
    if use_lime and predict_fn is not None:
        sal_l, _, _ = lime_saliency_classic(img, gt, predict_fn)
        sal_l_c = cv2.resize(sal_l, (crop.shape[1], crop.shape[0]))
        panels["LIME_classic"] = overlay(sal_l_c)
    panels["pred_class"] = IDX_TO_CLASS[cls]

    return rows, panels


def main(n_images=6, ids=None, use_lime=True, seed=42):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from data_utils import list_dataset, stratified_split

    ids_all, labels = list_dataset()
    if ids is None:
        # muestra estratificada del split de test
        splits = stratified_split()
        rng = np.random.RandomState(seed)
        ids = []
        by_class = {}
        for i in splits["test"]:
            by_class.setdefault(labels[i], []).append(i)
        per_class = max(1, n_images // len(by_class))
        for c in sorted(by_class):
            ids += list(rng.choice(sorted(by_class[c]), size=min(per_class, len(by_class[c])), replace=False))

    cnn = load_cnn("resnet18")
    cam_engine = GradCAM(cnn, "resnet18")
    predict_fn = load_classic_rf() if use_lime else None

    out_dir = RESULTS_DIR / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows, panels_per_img = [], []
    for k, img_id in enumerate(ids, 1):
        rows, panels = evaluate_image(img_id, cnn, cam_engine, predict_fn, use_lime)
        for r in rows:
            r["true_class"] = IDX_TO_CLASS[labels[img_id]]
        all_rows += rows
        panels_per_img.append((img_id, panels))
        print(f"[{k}/{len(ids)}] {img_id}: " +
              " | ".join(f"{r['technique']}: IoU={r['iou_top']:.3f}, AUCsal={r['sal_auc']:.3f}, "
                         f"point={int(r['pointing_game'])}" for r in rows))

    # ------------------------------ CSV ------------------------------
    with (out_dir / "explain_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["img_id", "technique", "true_class", "pred_class",
                                          "pointing_game", "iou_top", "sal_auc"])
        w.writeheader()
        w.writerows(all_rows)

    # --------------------- resumen (media ± std) ---------------------
    summary = {}
    for tech in sorted({r["technique"] for r in all_rows}):
        rs = [r for r in all_rows if r["technique"] == tech]
        summary[tech] = {
            "n": len(rs),
            "pointing_game": float(np.mean([r["pointing_game"] for r in rs])),
            "iou_top": {"mean": float(np.mean([r["iou_top"] for r in rs])),
                        "std": float(np.std([r["iou_top"] for r in rs]))},
            "sal_auc": {"mean": float(np.mean([r["sal_auc"] for r in rs])),
                        "std": float(np.std([r["sal_auc"] for r in rs]))},
        }
    (out_dir / "explain_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n===== FIDELIDAD DE SALIENCIA (media sobre la muestra) =====")
    for tech, s in summary.items():
        print(f"{tech:22s} pointing={s['pointing_game']:.2f} | IoU@top20%={s['iou_top']['mean']:.3f}"
              f" ± {s['iou_top']['std']:.3f} | AUCsal={s['sal_auc']['mean']:.3f} ± {s['sal_auc']['std']:.3f}")

    # ------------------- figura comparativa cualitativa -------------------
    tech_order = ["GT", "Grad-CAM", "IntegratedGradients"]
    if use_lime:
        tech_order.append("LIME_classic")
    fig, axes = plt.subplots(len(panels_per_img), len(tech_order), figsize=(3.2 * len(tech_order), 3.2 * len(panels_per_img)))
    if len(panels_per_img) == 1:
        axes = axes[None, :]
    for i, (img_id, panels) in enumerate(panels_per_img):
        for j, tech in enumerate(tech_order):
            ax = axes[i, j]
            ax.imshow(panels[tech])
            title = tech if i == 0 else ""
            if j == 0:
                title = f"{img_id}\n{title or 'GT'}"
            ax.set_title(title, fontsize=10)
            ax.axis("off")
    fig.suptitle("Comparación de técnicas explicativas (contorno verde = zona relevante GT)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "comparison.png", dpi=130, bbox_inches="tight")
    print(f"\nSalidas -> {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="nº de imágenes si no se pasan --ids")
    ap.add_argument("--ids", nargs="*", default=None, help="ids concretos, ej: IMD039 IMD058")
    ap.add_argument("--skip-lime", action="store_true", help="no evaluar LIME del clasificador clásico")
    args = ap.parse_args()
    main(n_images=args.n, ids=args.ids, use_lime=not args.skip_lime)
