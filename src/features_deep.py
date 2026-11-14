"""
Embeddings CNN para clasificación: ResNet-18 ImageNet con cabeza global-avg
removida. Devuelve un vector de 512-d por imagen (recorte de la lesión).
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def lesion_bbox(mask, margin=10):
    """Bounding box de la lesión con margen, dentro de los límites de la imagen."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        h, w = mask.shape[:2]
        return 0, h, 0, w
    h, w = mask.shape[:2]
    y0, y1 = max(int(ys.min()) - margin, 0), min(int(ys.max()) + margin, h)
    x0, x1 = max(int(xs.min()) - margin, 0), min(int(xs.max()) + margin, w)
    return y0, y1, x0, x1


def crop_lesion(img_bgr, mask, margin=10):
    """Recorta el bounding box de la lesión con un margen."""
    y0, y1, x0, x1 = lesion_bbox(mask, margin)
    return img_bgr[y0:y1, x0:x1]


def load_backbone():
    import torch
    import torchvision

    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net.eval()
    return net


def _to_tensor(img_bgr, size=224):
    import torch

    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return x


def extract_embeddings(net, img_bgr, mask, size=224):
    """Embedding 512-d de la ResNet-18 sobre el recorte de la lesión."""
    import torch

    crop = crop_lesion(img_bgr, mask)
    x = _to_tensor(crop, size)[None]
    with torch.no_grad():
        emb = net(x).squeeze(0)
    return emb.numpy()


if __name__ == "__main__":
    from data_utils import load_image_and_mask

    img, mask = load_image_and_mask("IMD002")
    net = load_backbone()
    e = extract_embeddings(net, img, mask)
    print("Embedding:", e.shape, "norm:", np.linalg.norm(e))
