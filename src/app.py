"""
Interfaz web local (Flask) para el proyecto de dermoscopia.

Páginas:
  /                  : galería del dataset, agrupada por clase.
  /pipeline/<id>     : ejecuta (o muestra, si ya está cacheada) la demo
                       end-to-end de una imagen: preprocesamiento →
                       segmentación → clasificación → explicabilidad.
  /compare/<id>      : comparación de métodos para una imagen — segmentación
                       clásica vs U-Net (Dice vs GT) y Grad-CAM vs
                       Integrated Gradients vs LIME (fidelidad vs GT).
  /results           : dashboard de métricas: ROC/AUC por clase, matrices de
                       confusión, CV estratificada y fidelidad de saliencia.
  /media/<ruta>      : sirve PNG/CSV de results/ y data/processed/.
  /thumb/<id>        : miniatura de la imagen preprocesada.

Ejecutar:  venv-dl/bin/python src/app.py   →  http://127.0.0.1:5000
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, abort, render_template, request, send_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_utils import IDX_TO_CLASS, ROOT as _ROOT  # noqa: E402  (ROOT ya es el mismo)
from data_utils import list_dataset, load_image_and_mask  # noqa: E402

app = Flask(__name__)
RESULTS = ROOT / "results"
PREP = ROOT / "data" / "processed" / "preprocessed"
THUMBS = ROOT / "data" / "processed" / "thumbs"

# carga perezosa de modelos (se comparten entre peticiones)
_STATE = {}


def get_models():
    """Carga UNA vez los modelos pesados (UNet + CNN + clásicos)."""
    if _STATE:
        return _STATE
    from run_pipeline import load_classic_models, load_segmentation_model

    _STATE["seg_model"] = load_segmentation_model("unet")
    _STATE["classic"] = load_classic_models()

    from eval_explain import load_cnn

    cnn = load_cnn("resnet18")
    _STATE["cnn"] = cnn
    from explain import GradCAM

    _STATE["cam"] = GradCAM(cnn, "resnet18")
    return _STATE


def get_classic_predict_fn():
    """predict_fn(vector) -> probs del RF clásico (para LIME)."""
    import joblib

    mp = ROOT / "models"
    scaler, rf = joblib.load(mp / "classic_scaler.joblib"), joblib.load(mp / "rf_features.joblib")
    return lambda v: rf.predict_proba(scaler.transform(v))


# --------------------------------------------------------------------- /
@app.route("/")
def index():
    ids, labels = list_dataset()
    by_class = {}
    for i in ids:
        by_class.setdefault(IDX_TO_CLASS[labels[i]], []).append(i)
    return render_template("index.html", by_class=by_class)


# ------------------------------------------------------------ /pipeline ---
def _pipeline_fresh(img_id):
    """¿Están todos los artefactos de la demo en results/pipeline/<id>/?"""
    d = RESULTS / "pipeline" / img_id
    need = ["0_preprocessed.png", "1_mask.png", "2_gradcam.png",
            "2b_integrated_gradients.png", "3_lime_classic.png", "report.png", "meta.json"]
    return d.exists() and all((d / f).exists() for f in need)


@app.route("/pipeline/<img_id>")
@app.route("/pipeline/<img_id>/")
def pipeline(img_id):
    force = request.args.get("rerun") == "1"
    out_dir = RESULTS / "pipeline" / img_id
    if force or not _pipeline_fresh(img_id):
        from run_pipeline import main as run_demo

        summary = run_demo(img_id)
        (out_dir / "meta.json").write_text(json.dumps(summary, indent=2))
        rerun = True
    else:
        rerun = False
    meta = json.loads((out_dir / "meta.json").read_text())
    stages = [
        ("0_preprocessed.png", "Preprocesada (vello + MSR)"),
        ("1_mask.png", "Segmentación U-Net"),
        ("2_gradcam.png", "Grad-CAM (CNN)"),
        ("2b_integrated_gradients.png", "Integrated Gradients (CNN)"),
        ("3_lime_classic.png", "LIME (clásico)"),
    ]
    return render_template("pipeline.html", m=meta, img_id=img_id, stages=stages, rerun=rerun)


# -------------------------------------------------------------- /compare ---
def to_b64(img, quality=88):
    """Codifica una imagen (BGR o gris) como data-URI JPEG para HTML."""
    import base64

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


@app.route("/compare/<img_id>")
def compare(img_id):
    img, gt = load_image_and_mask(img_id, preprocessed_dir="preprocessed")

    # ---- segmentación: clásica vs U-Net ----
    from run_pipeline import predict_segmentation
    from segmentation_classical import classical_segment, dice_iou

    st = get_models()
    unet_mask = predict_segmentation(st["seg_model"], img, size=256)
    classic_mask = classical_segment(img)
    d_unet, iou_unet = dice_iou(unet_mask, gt)
    d_clas, iou_clas = dice_iou(classic_mask, gt)

    # ---- explicabilidad: Grad-CAM vs IG vs LIME ----
    from explain import fidelity_metrics, integrated_gradients, lime_saliency_classic
    from features_deep import _to_tensor, lesion_bbox

    cnn, cam = st["cnn"], st["cam"]
    y0, y1, x0, x1 = lesion_bbox(gt, 12)
    crop, gt_crop = img[y0:y1, x0:x1], gt[y0:y1, x0:x1]
    x = _to_tensor(crop, 224)[None]
    cam_sal, probs, cls = cam(x)
    ig_sal, _, _ = integrated_gradients(cnn, x, class_idx=cls)
    sal_l, _, _ = lime_saliency_classic(img, gt, get_classic_predict_fn())

    def overlay_gt(sal, base_bgr, gt_mask):
        """Mapa de saliencia en JET sobre la imagen + contorno GT en verde."""
        h = (np.clip(sal, 0, 1) * 255).astype(np.uint8)
        if h.shape[:2] != base_bgr.shape[:2]:
            h = cv2.resize(h, (base_bgr.shape[1], base_bgr.shape[0]))
        o = cv2.addWeighted(base_bgr, 0.55, cv2.applyColorMap(h, cv2.COLORMAP_JET), 0.45, 0)
        cnts, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(o, cnts, -1, (0, 255, 0), 2)
        return o

    cam_full = cv2.resize(cam_sal, (crop.shape[1], crop.shape[0]))
    ig_full = cv2.resize(ig_sal, (crop.shape[1], crop.shape[0]))

    expl = {
        "Grad-CAM": {"metrics": fidelity_metrics(cam_full, gt_crop),
                     "img": to_b64(overlay_gt(cam_full, crop, gt_crop))},
        "Integrated Gradients": {"metrics": fidelity_metrics(ig_full, gt_crop),
                                 "img": to_b64(overlay_gt(ig_full, crop, gt_crop))},
        "LIME (clásico)": {"metrics": fidelity_metrics(sal_l, gt),
                           "img": to_b64(overlay_gt(sal_l, img, gt))},
    }

    def mask_on_image(mask):
        o = cv2.addWeighted(img, 0.7, cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR), 0.3, 0)
        cnts, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(o, cnts, -1, (0, 255, 0), 2)
        return cv2.cvtColor(o, cv2.COLOR_BGR2RGB)

    imgs = {
        "unet": {"img": to_b64(mask_on_image(unet_mask))},
        "classic": {"img": to_b64(mask_on_image(classic_mask))},
        "gt": {"img": to_b64(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))},
    }

    return render_template(
        "compare.html", img_id=img_id, true_class=IDX_TO_CLASS[cls],
        d_unet=d_unet, iou_unet=iou_unet, d_clas=d_clas, iou_clas=iou_clas,
        expl=expl, imgs=imgs,
    )


# -------------------------------------------------------------- /results ---
@app.route("/results")
def results():
    def load_json(p, default=None):
        try:
            return json.loads((RESULTS / p).read_text())
        except Exception:
            return default

    classic = load_json("classification/classic_metrics.json", {})
    deep = {name: load_json(f"classification/deep_{n}.json", {})
            for name, n in [("ResNet-18", "resnet18"), ("EfficientNet-B0", "efficientnet_b0")]}
    auc = load_json("classification/roc_auc_per_class.json", {})
    cv_classical = load_json("classification/cv_classical.json", {})
    cv_deep = load_json("classification/cv_deep.json", {})
    explain = load_json("explainability/explain_summary.json", {})

    figures = [
        ("Curvas ROC por clase (test)", "classification/roc_curves.png"),
        ("Matrices de confusión por clase (test)", "classification/confusion_matrices.png"),
        ("ROC out-of-fold, CV clásica", "classification/cv_roc_curves.png"),
        ("Confusión out-of-fold, CV clásica", "classification/cv_confusion.png"),
        ("ROC out-of-fold, CV de la CNN", "classification/cv_deep_roc.png"),
        ("Confusión out-of-fold, CV de la CNN", "classification/cv_deep_confusion.png"),
        ("Comparación de técnicas explicativas", "explainability/comparison.png"),
    ]
    figures = [(t, p) for t, p in figures if (RESULTS / p).exists()]
    return render_template("results.html", classic=classic, deep=deep, auc=auc,
                           cv_classical=cv_classical, cv_deep=cv_deep, explain=explain,
                           figures=figures)


# ---------------------------------------------------------------- media ---
@app.route("/media/<path:rel>")
def media(rel):
    p = (ROOT / rel).resolve()
    if not str(p).startswith(str(ROOT.resolve())) or not p.is_file():
        abort(404)
    if not (rel.startswith("results/") or rel.startswith("data/processed/")):
        abort(404)
    return send_file(p, mimetype="image/png" if p.suffix == ".png" else "application/octet-stream")


@app.route("/thumb/<img_id>")
def thumb(img_id):
    THUMBS.mkdir(parents=True, exist_ok=True)
    tp = THUMBS / f"{img_id}.jpg"
    if not tp.exists():
        src = PREP / f"{img_id}.png"
        img = cv2.imread(str(src)) if src.exists() else None
        if img is None:
            abort(404)
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * 180 / h), 180), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(tp), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return send_file(tp, mimetype="image/jpeg")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    print(f"Interfaz -> http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
