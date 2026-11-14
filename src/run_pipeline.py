"""
Pipeline completo para UNA imagen dermoscópica:
  preprocesamiento (vello+inpainting, MSR) -> segmentación (clásica o U-Net)
  -> posprocesado -> extracción de rasgos -> clasificación (clásico + CNN)
  -> explicabilidad (Grad-CAM para CNN, LIME-style para el clásico)
Genera results/pipeline/<id>/report.png con todas las etapas.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_utils import IDX_TO_CLASS, RESULTS_DIR, load_image_and_mask  # noqa: E402
from explain import GradCAM, integrated_gradients, lime_saliency_classic  # noqa: E402
from features import extract_all  # noqa: E402
from features_deep import extract_embeddings, load_backbone  # noqa: E402
from preprocessing import multi_scale_retinex  # noqa: E402
from segmentation_classical import classical_segment, postprocess_mask  # noqa: E402

DEVICE = "cpu"


def load_classic_models():
    """Carga SVM/RF/ensemble entrenados y el escalador (clasifiers.py debe ejecutarse antes)."""
    import joblib  # sklearn lo trae

    p = Path(ROOT / "models")
    return {
        "scaler": joblib.load(p / "classic_scaler.joblib"),
        "svm": joblib.load(p / "svm_features.joblib"),
        "rf": joblib.load(p / "rf_features.joblib"),
        "e_scaler": joblib.load(p / "embed_scaler.joblib"),
        "esvm": joblib.load(p / "svm_embeddings.joblib"),
        "resnet": load_backbone(),  # backbone para calcular embeddings en inferencia
    }


def load_segmentation_model(arch="unet"):
    import torch
    from segmentation_deep import UNet

    ckpt = torch.load(ROOT / "models" / f"segment_{arch}.pt", map_location=DEVICE, weights_only=False)
    model = UNet().to(DEVICE)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model


def predict_segmentation(model, img_bgr, size=256):
    import torch

    img_r = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 255.0
    x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    with torch.no_grad():
        prob = torch.sigmoid(model(x.to(DEVICE)))[0, 0].numpy()
    prob = cv2.resize(prob, (img_bgr.shape[1], img_bgr.shape[0]))
    return postprocess_mask((prob > 0.5).astype(np.uint8))


def classify_ensemble(models, img_bgr, mask):
    from sklearn.preprocessing import StandardScaler

    vec, _ = extract_all(img_bgr, mask)
    vec2 = vec[None]
    vecs = models["scaler"].transform(vec2)
    ps = models["svm"].predict_proba(vecs)
    pr = models["rf"].predict_proba(vec2)
    emb = extract_embeddings(models["resnet"], img_bgr, mask)[None]
    pe = models["esvm"].predict_proba(models["e_scaler"].transform(emb))
    return (ps + pr + pe) / 3.0


def overlay_heat(img_bgr, heat, alpha=0.45):
    heat_u = (heat * 255).astype(np.uint8)
    heat_c = cv2.applyColorMap(heat_u, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, heat_c, alpha, 0)


def main(img_id, seg_arch="unet"):
    from classifier_deep import build_model
    import torch

    out_dir = RESULTS_DIR / "pipeline" / img_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) preprocesado
    img, gt = load_image_and_mask(img_id)
    cleaned, hair_mask, status = None, None, ""
    from preprocessing import preprocess_image

    img_prep, hair_mask, status = preprocess_image(img)
    cv2.imwrite(str(out_dir / "0_preprocessed.png"), img_prep)

    # 2) segmentación
    seg_model = load_segmentation_model(seg_arch)
    mask = predict_segmentation(seg_model, img_prep, size=256)
    classical = classical_segment(img_prep)
    cv2.imwrite(str(out_dir / "1_mask.png"), mask * 255)
    from segmentation_classical import dice_iou

    d_unet = dice_iou(mask, gt)
    d_class = dice_iou(classical, gt)
    print(f"Segmentación: U-Net Dice={d_unet[0]:.3f} | clásica Dice={d_class[0]:.3f}")

    # 3) clasificación
    models = load_classic_models()
    proba = classify_ensemble(models, img_prep, mask)
    pred_class = int(proba.argmax())
    print(f"Clasificación (ensemble): {IDX_TO_CLASS[pred_class]} "
          + ", ".join(f"{IDX_TO_CLASS[i]}={p:.2f}" for i, p in enumerate(proba[0])))

    # 4) explicabilidad
    # CNN Grad-CAM
    ckpt = torch.load(ROOT / "models" / "classifier_resnet18.pt", map_location=DEVICE, weights_only=False)
    cnn = build_model("resnet18").to(DEVICE).eval()
    cnn.load_state_dict(ckpt["state"])
    from features_deep import crop_lesion, _to_tensor

    cam_engine = GradCAM(cnn, "resnet18")
    crop = crop_lesion(img_prep, mask, margin=12)
    x = _to_tensor(crop, 224).to(DEVICE)
    cam, probs, cls = cam_engine(x)
    cam_full = cv2.resize(cam, (crop.shape[1], crop.shape[0]))
    cv2.imwrite(str(out_dir / "2_gradcam.png"), overlay_heat(crop, cam_full))

    # Integrated Gradients (misma CNN, misma clase)
    ig, _, _ = integrated_gradients(cnn, x, class_idx=cls)
    ig_full = cv2.resize(ig, (crop.shape[1], crop.shape[0]))
    cv2.imwrite(str(out_dir / "2b_integrated_gradients.png"), overlay_heat(crop, ig_full))

    # LIME-style para el pipeline clásico
    def predict_fn(v):
        return models["rf"].predict_proba(models["scaler"].transform(v))

    sal, lcls, sp_labels = lime_saliency_classic(img_prep, mask, predict_fn)
    cv2.imwrite(str(out_dir / "3_lime_classic.png"), overlay_heat(img_prep, sal))

    # 5) reporte visual
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(27, 11))
    panels = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "Original"),
        (cv2.cvtColor(img_prep, cv2.COLOR_BGR2RGB), "Preprocesada (vello+MSR)"),
        (hair_mask, "Máscara de vello detectada"),
        (gt, "Máscara GT"),
        (mask, f"Segmentación U-Net (Dice={d_unet[0]:.2f})"),
        (classical, f"Segmentación clásica (Dice={d_class[0]:.2f})"),
        (cv2.cvtColor(cv2.imread(str(out_dir / "2_gradcam.png")), cv2.COLOR_BGR2RGB), "Grad-CAM (CNN)"),
        (cv2.cvtColor(cv2.imread(str(out_dir / "2b_integrated_gradients.png")), cv2.COLOR_BGR2RGB), "Integrated Gradients (CNN)"),
        (cv2.cvtColor(cv2.imread(str(out_dir / "3_lime_classic.png")), cv2.COLOR_BGR2RGB), "Saliencia LIME (clásico)"),
    ]
    for ax, (im, t) in zip(axes.flat, panels):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"{img_id} | Predicción ensemble: {IDX_TO_CLASS[pred_class]} "
                 f"(p={proba[0][pred_class]:.2f}) | estado vello: {status}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "report.png", dpi=140, bbox_inches="tight")
    print(f"Reporte -> {out_dir / 'report.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="IMD039", help="id de imagen, ej IMD039 (melanoma) o IMD003 (nevus)")
    ap.add_argument("--seg-arch", default="unet", choices=["unet"])
    args = ap.parse_args()
    main(args.id, args.seg_arch)
