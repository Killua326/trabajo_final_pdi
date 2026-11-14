"""
Explicabilidad del clasificador: mapas de saliencia que respaldan la decisión.

1) Grad-CAM            : para los clasificadores CNN (ResNet18/EfficientNet).
   Localiza qué regiones del recorte activaron la clase ganadora.
2) Integrated Gradients: atribución por integración de gradientes desde una
   línea base (imagen negra) hasta la entrada; satisface los axiomas de
   sensibilidad e invariancia por implementación (Sundararajan et al., 2017).
3) LIME-style          : para el clasificador clásico (SVM/RF sobre rasgos).
   Segmente la imagen en superpíxeles (SLIC), "apague" subconjuntos
   sustituyéndolos por la piel de fondo y ajuste un modelo lineal ponderado
   sobre las probabilidades: el coeficiente por superpíxel es su
   contribución a la clase predicha.

Además, métricas de FIDELIDAD de cada mapa frente a la zona relevante
(máscara GT de la lesión): juego de señalar (pointing game), IoU del top-p%
de saliencia y AUC de saliencia a nivel de píxel.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ----------------------------- Grad-CAM -----------------------------
class GradCAM:
    """Grad-CAM estándar para torchvision (última etapa conv)."""

    def __init__(self, model, arch="resnet18"):
        self.model = model.eval()
        # los parámetros deben requerir gradiente para que las activaciones
        # intermedias se guarden en el grafo y se puedan obtener los gradientes
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.acts, self.grads = None, None
        target = model.layer4[-1] if arch == "resnet18" else model.features[-1]
        # hook a nivel de tensor: más fiable entre versiones de torch
        def fwd_hook(m, i, o):
            self.acts = o.detach()
            # sólo registrar el hook de gradiente si el tensor participa en el
            # grafo (p.ej. los forwards en no_grad de otras técnicas no lo tienen)
            if o.requires_grad:
                o.register_hook(lambda g: setattr(self, "grads", g.detach()))
        target.register_forward_hook(fwd_hook)

    def __call__(self, x, class_idx=None):
        """Devuelve (mapa [0,1] a 224x224, probs, clase predicha)."""
        import torch

        x = x if x.dim() == 4 else x[None]
        logits = self.model(x)
        probs = torch.softmax(logits, 1)[0].detach().cpu().numpy()
        cls = int(logits.argmax(1)) if class_idx is None else int(class_idx)
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, cls] = 1.0
        logits.backward(gradient=one_hot)
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
        return cam, probs, cls


# --------------------- LIME-style (pipeline clásico) ---------------------
def _predict_classic(predict_fn, img_bgr, mask):
    """Probabilidades del clasificador clásico para una imagen+máscara."""
    from features import extract_all

    vec, _ = extract_all(img_bgr, mask)
    return predict_fn(vec[None])[0]


def lime_saliency_classic(img_bgr, lesion_mask, predict_fn, n_segments=14, n_samples=48, seed=42):
    """Mapa de saliencia por superpíxeles para el clasificador clásico.

    Args:
        img_bgr: imagen preprocesada BGR.
        lesion_mask: máscara binaria de la lesión (0/1).
        predict_fn: función vector->probs (modelo entrenado con StandardScaler).
        n_segments: superpíxeles aproximados a muestrear dentro de la lesión.
        n_samples: número de perturbaciones.

    Returns:
        (saliency HxW en [0,1], pred_class_idx, superpixel_labels)
    """
    from sklearn.linear_model import Ridge
    from skimage.segmentation import slic

    # Superpíxeles sólo dentro de la lesión (el resto es contexto estable)
    full_labels = np.zeros(img_bgr.shape[:2], np.int32)
    lesion_slic = slic(img_bgr, n_segments=n_segments, compactness=12, start_label=1)
    lesion_slic[lesion_mask == 0] = 0
    sp_ids = sorted(set(lesion_slic.ravel()) - {0})
    if not sp_ids:
        return np.zeros(img_bgr.shape[:2]), 0, lesion_slic
    for i, sp in enumerate(sp_ids, 1):
        full_labels[lesion_slic == sp] = i

    base_mask = lesion_mask.astype(bool)
    pred0 = _predict_classic(predict_fn, img_bgr, lesion_mask)
    cls = int(np.argmax(pred0))

    # Piel de fondo (para apagar superpíxeles): media del anillo alrededor
    ring = cv2.dilate(base_mask.astype(np.uint8), np.ones((41, 41), np.uint8))
    ring = ring & ~base_mask
    skin = img_bgr[ring > 0].mean(axis=0) if ring.sum() else img_bgr[~base_mask].mean(axis=0)

    rng = np.random.RandomState(seed)
    X, ys, weights = [], [], []
    for _ in range(n_samples):
        kept = rng.rand(len(sp_ids)) < 0.5
        pert = img_bgr.copy()
        pm = lesion_mask.copy()
        for i, sp in enumerate(sp_ids):
            if not kept[i]:  # apagar el superpíxel
                sel = full_labels == sp
                pert[sel] = skin
                pm[sel] = 0
        proba = _predict_classic(predict_fn, pert, pm)
        X.append(kept.astype(float))
        ys.append(proba[cls])
        # peso: cercanía de la perturbación al original
        weights.append(np.exp(-np.linalg.norm(kept - np.ones(len(sp_ids))) ** 2 / 2.0))

    ridge = Ridge(alpha=1.0).fit(np.array(X), np.array(ys), sample_weight=np.array(weights))
    sal = np.zeros(img_bgr.shape[:2], np.float32)
    for i, sp in enumerate(sp_ids, 1):
        sal[full_labels == i] = max(ridge.coef_[i - 1], 0.0)
    sal = sal / (sal.max() + 1e-9)
    return sal, cls, full_labels


# ----------------------- Integrated Gradients -----------------------
def integrated_gradients(model, x, class_idx=None, n_steps=32, baseline="blur"):
    """Integrated Gradients (Sundararajan et al., 2017), sin dependencias extra.

    Atribución = (x - línea_base) * media de gradientes sobre la recta entre
    ambas. Satisface los axiomas de sensibilidad e invarancia por
    implementación. Devuelve un mapa de magnitudes en [0,1] a la resolución
    de entrada.

    Línea base:
      - "blur" (por defecto): versión suavizada (Blur-IG). Al partir de una
        imagen cercana a la de entrada y no de un color plano, evita
        atribuciones espúreas en el borde lesión/piel (Qi et al., 2019)
        y localiza mejor la lesión (comparado con la línea base negra).
      - "zero": imagen negra (línea base clásica, en espacio normalizado).

    Args:
        model: red de clasificación (eval mode).
        x: tensor de entrada [C,H,W] o [1,C,H,W] (ya normalizado).
        class_idx: clase a explicar; None = clase predicha.
        n_steps: número de pasos de la integración (regla de Riemann).
        baseline: "blur", "zero" o tensor explícito.

    Returns:
        (saliency [H,W] en [0,1], probs numpy, clase explicada)
    """
    import torch

    m = model.eval()
    x = x if x.dim() == 4 else x[None]
    x = x.detach()
    if isinstance(baseline, str):
        if baseline == "zero":
            baseline = torch.zeros_like(x)
        elif baseline == "blur":
            # volver al espacio RGB [0,1], suavizar y re-normalizar
            mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            rgb01 = (x[0] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            blur = cv2.blur((rgb01 * 255).astype(np.float32), (31, 31)) / 255.0
            baseline = ((torch.from_numpy(blur).permute(2, 0, 1) - mean) / std)[None]
        else:
            raise ValueError(f"baseline desconocida: {baseline}")

    with torch.no_grad():
        logits = m(x)
    probs = torch.softmax(logits, 1)[0].cpu().numpy()
    cls = int(logits.argmax(1)) if class_idx is None else int(class_idx)

    grads = []
    for alpha in torch.linspace(0.0, 1.0, n_steps + 1)[1:]:  # sin el paso alpha=0 (grad ~ 0)
        xi = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        out = m(xi)
        g = torch.autograd.grad(out[0, cls], xi)[0]
        grads.append(g)
    avg_grad = torch.stack(grads).mean(0)
    attr = (x - baseline) * avg_grad
    sal = attr.abs().sum(dim=1)[0]            # agregación sobre canales
    sal = sal / (sal.max() + 1e-9)
    return sal.numpy(), probs, cls


# ------------------- Fidelidad vs zona relevante (GT) -------------------
def binarize_top(sal, frac=0.20):
    """Binariza la saliencia conservando el top-frac de los píxeles."""
    if frac <= 0 or frac >= 1:
        raise ValueError("frac debe estar en (0,1)")
    thr = np.quantile(sal, 1.0 - frac)
    return sal >= thr


def fidelity_metrics(sal, gt_mask, coverage=0.20):
    """Fidelidad del mapa de saliencia frente a la máscara GT de la lesión.

    Args:
        sal: mapa HxW en [0,1] (misma resolución que gt_mask).
        gt_mask: máscara binaria 0/1 de la zona relevante (lesión GT).
        coverage: fracción de píxeles top que define la región atribuida.

    Returns:
        dict con:
          pointing_game : el píxel de saliencia máxima cae dentro de la lesión
          iou_top       : IoU(top-coverage% de saliencia, máscara GT)
          sal_auc       : ROC AUC tratando la saliencia como score por píxel
    """
    from sklearn.metrics import roc_auc_score

    gt = gt_mask > 0
    sal = np.asarray(sal, np.float64)

    y, x = np.unravel_index(np.argmax(sal), sal.shape)
    pointing = bool(gt[y, x])

    top = binarize_top(sal, coverage)
    inter = np.logical_and(top, gt).sum()
    union = np.logical_or(top, gt).sum()
    iou = float(inter) / float(union) if union else 0.0

    if gt.sum() > 0 and (1 - gt).sum() > 0:
        auc = float(roc_auc_score(gt.ravel().astype(np.int8), sal.ravel()))
    else:
        auc = float("nan")

    return {"pointing_game": pointing, "iou_top": iou, "sal_auc": auc}
