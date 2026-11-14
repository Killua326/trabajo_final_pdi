"""
Clasificación por transferencia + fine-tuning con redes residuales:
  - ResNet-18 (torchvision, pesos ImageNet).
  - EfficientNet-B0 (torchvision, pesos ImageNet).

Protocolo: imágenes preprocesadas, recorte al bounding box de la lesión
(máscara GT para train/val/test del benchmark), redimensionado a 224 px,
aumentación suave. Congelamiento inicial + descongelamiento parcial del
backbone (fine-tuning) y early-stopping sobre validación.
Salida: models/classifier_<arch>.pt + métricas en results/classification/.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from data_utils import CLS_SIZE, IDX_TO_CLASS, MODELS_DIR, RESULTS_DIR, stratified_split  # noqa: E402
from features_deep import crop_lesion  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREP = Path(ROOT / "data" / "processed" / "preprocessed")
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


class LesionDataset(Dataset):
    def __init__(self, ids, augment=False):
        self.ids, self.augment = ids, augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        from data_utils import list_dataset, load_image_and_mask

        img_id = self.ids[i]
        img, mask = load_image_and_mask(img_id, preprocessed_dir="preprocessed")
        _, labels = list_dataset()
        y = labels[img_id]
        crop = crop_lesion(img, mask, margin=12)
        img = cv2.resize(crop, (CLS_SIZE, CLS_SIZE), interpolation=cv2.INTER_AREA)
        if self.augment:
            k = np.random.randint(0, 4)
            img = np.rot90(img, k).copy()
            if np.random.rand() < 0.5:
                img = img[:, ::-1].copy()
            if np.random.rand() < 0.5:
                img = np.flipud(img).copy()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        x = (x - MEAN) / STD
        return x, torch.tensor(y, dtype=torch.long)


def build_model(arch):
    import torchvision

    if arch == "resnet18":
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 3)
        head_params = ["fc."]
    elif arch == "efficientnet_b0":
        m = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 3)
        head_params = ["classifier."]
    else:
        raise ValueError(arch)
    for p in m.parameters():          # fase 1: congelar el backbone
        p.requires_grad = False
    for n, p in m.named_parameters(): # solo entrenar la cabeza
        if any(n.startswith(h) for h in head_params):
            p.requires_grad = True
    return m


def unfreeze_last_stage(model, arch, pct=0.5):
    """Fine-tuning: descongela el último ~pct de bloques del backbone."""
    named = [(n, p) for n, p in model.named_parameters()]
    split = int(len(named) * (1 - pct))
    for n, p in named[split:]:
        p.requires_grad = True


def evaluate(model, loader):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            preds += model(x.to(DEVICE)).argmax(1).cpu().tolist()
            ys += y.tolist()
    return np.array(preds), np.array(ys)


def train(arch="resnet18", epochs=20, head_epochs=3, batch_size=16, lr_head=1e-3, lr_ft=1e-4):
    torch.manual_seed(42)
    np.random.seed(42)
    splits = stratified_split()
    tr = DataLoader(LesionDataset(splits["train"], augment=True), batch_size, shuffle=True)
    va = DataLoader(LesionDataset(splits["val"]), batch_size)
    te = DataLoader(LesionDataset(splits["test"]), batch_size)

    model = build_model(arch).to(DEVICE)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val, best_state, patience = -1.0, None, 0
    for ep in range(1, epochs + 1):
        if ep == head_epochs + 1:
            unfreeze_last_stage(model, arch, pct=0.5)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                    lr=lr_ft, weight_decay=1e-4)
        elif ep == 1:
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                    lr=lr_head, weight_decay=1e-4)
        model.train()
        tot, nb = 0.0, 0
        for x, y in tr:
            opt.zero_grad()
            loss = crit(model(x.to(DEVICE)), y.to(DEVICE))
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        pv, yv = evaluate(model, va)
        val_bacc = balanced_accuracy_score(yv, pv)
        print(f"[{arch}] época {ep:02d}/{epochs} | loss {tot / max(nb, 1):.4f} | val bAcc {val_bacc:.3f}")
        if val_bacc > best_val:
            best_val, best_state, patience = val_bacc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 5:
                print("  early-stopping")
                break

    model.load_state_dict(best_state)
    pt, yt = evaluate(model, te)
    metrics = {
        "model": f"{arch} (transferencia+fine-tuning)",
        "accuracy": accuracy_score(yt, pt),
        "balanced_accuracy": balanced_accuracy_score(yt, pt),
        "macro_f1": f1_score(yt, pt, average="macro"),
        "confusion": confusion_matrix(yt, pt).tolist(),
        "val_balanced_accuracy": best_val,
    }
    print(f"[{arch}] TEST acc={metrics['accuracy']:.3f} | bAcc={metrics['balanced_accuracy']:.3f} | "
          f"macroF1={metrics['macro_f1']:.3f}")
    print("Matriz de confusión (filas: Common, Atypical, Melanoma):")
    print(np.array(metrics["confusion"]))

    MODELS_DIR.mkdir(exist_ok=True)
    torch.save({"arch": arch, "state": best_state}, MODELS_DIR / f"classifier_{arch}.pt")
    out = RESULTS_DIR / "classification" / f"deep_{arch}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: v for k, v in metrics.items() if k != "confusion"}, indent=2))
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["resnet18", "efficientnet_b0"], default="resnet18")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    train(args.arch, epochs=args.epochs)
