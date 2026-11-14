"""
Segmentación con redes: U-Net (implementación propia) y DeepLabv3+ (torchvision).

Uso (CPU):
  python src/segmentation_deep.py --arch unet     --epochs 20
  python src/segmentation_deep.py --arch deeplab  --epochs 15

Guarda el modelo en models/segment_<arch>.pt y una evaluación Dice/IoU en
results/segmentation/<arch>.csv (conjunto de validación + test).
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from data_utils import (IMG_SIZE, MODELS_DIR, PROCESSED_DIR, RESULTS_DIR,  # noqa: E402
                        list_dataset, stratified_split)
from segmentation_classical import dice_iou, postprocess_mask  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREP_DIR = PROCESSED_DIR / "preprocessed"


# ----------------------------- U-Net -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """U-Net compacta (2 niveles de profundidad por lado) para 1 salida binaria."""

    def __init__(self, cin=3, base=32):
        super().__init__()
        self.enc1 = DoubleConv(cin, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 2, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def build_deeplab():
    """DeepLabv3+ (ResNet-50 pretrained, 1 clase de salida)."""
    from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(256, 1, 1)
    return model


def build_model(arch):
    if arch == "unet":
        return UNet()
    if arch == "deeplab":
        return build_deeplab()
    raise ValueError(arch)


# ----------------------------- Dataset -----------------------------
class SegDataset(Dataset):
    def __init__(self, ids, augment=False):
        self.ids, self.augment = ids, augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_id = self.ids[i]
        img = cv2.imread(str(PREP_DIR / f"{img_id}.png"))
        mask = cv2.imread(str(ROOT / "images" / "masks" / f"{img_id}_segmentation.png"), 0)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        if self.augment:  # aumentación: flips + rotaciones de 90°
            k = np.random.randint(0, 4)
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()
            if np.random.rand() < 0.5:
                img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()

        x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        y = torch.from_numpy((mask > 127).astype(np.float32))[None]
        return x, y


# ----------------------------- Entrenamiento -----------------------------
def dice_loss(prob, target, eps=1e-7):
    inter = (prob * target).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


@torch.no_grad()
def eval_split(model, loader):
    model.eval()
    dices, ious = [], []
    for x, y in loader:
        prob = torch.sigmoid(model(x)[0] if isinstance(model(x), dict) else model(x))
        pred = postprocess_mask((prob > 0.5)[0, 0].numpy())
        d, i = dice_iou(pred, y[0, 0].numpy())
        dices.append(d)
        ious.append(i)
    return float(np.mean(dices)), float(np.mean(ious))


def train(arch="unet", epochs=20, batch_size=8, lr=1e-3):
    torch.manual_seed(42)
    np.random.seed(42)
    splits = stratified_split()
    tr_ds = SegDataset(splits["train"], augment=True)
    va_ds = SegDataset(splits["val"])
    te_ds = SegDataset(splits["test"])
    tr = DataLoader(tr_ds, batch_size, shuffle=True, num_workers=0, drop_last=False)
    va = DataLoader(va_ds, 4, shuffle=False)
    te = DataLoader(te_ds, 4, shuffle=False)

    model = build_model(arch).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss()

    best_val, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for x, y in tr:
            opt.zero_grad()
            out = model(x.to(DEVICE))
            logits = out["out"] if isinstance(out, dict) else out
            loss = bce(logits, y.to(DEVICE)) + dice_loss(torch.sigmoid(logits), y.to(DEVICE))
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        val_dice, val_iou = eval_split(model, va)
        print(f"[{arch}] época {ep:02d}/{epochs} | loss {tot / max(nb, 1):.4f} | val Dice {val_dice:.4f} | val IoU {val_iou:.4f}")
        if val_dice > best_val:
            best_val, best_state = val_dice, {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_dice, test_iou = eval_split(model, te)
    print(f"[{arch}] TEST Dice {test_dice:.4f} | IoU {test_iou:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    torch.save({"arch": arch, "state": best_state, "val_dice": best_val, "test_dice": test_dice},
               MODELS_DIR / f"segment_{arch}.pt")

    # Guardar predicciones de test para inspección/CSV
    out_csv = RESULTS_DIR / "segmentation" / f"{arch}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        model.to(DEVICE).eval()
        for i, img_id in enumerate(splits["test"]):
            x, y = te_ds[i]
            prob = torch.sigmoid(model(x[None].to(DEVICE)))
            prob = prob[0] if not isinstance(prob, dict) else prob["out"][0]
            prob = F.interpolate(prob[None], size=(IMG_SIZE, IMG_SIZE), mode="bilinear")[0, 0]
            pred = postprocess_mask((prob > 0.5).cpu().numpy())
            d, iou = dice_iou(pred, y[0].numpy())
            rows.append({"image_id": img_id, "dice": round(d, 4), "iou": round(iou, 4)})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "dice", "iou"])
        w.writeheader()
        w.writerows(rows)
    print(f"Predicciones por imagen -> {out_csv}")
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["unet", "deeplab"], default="unet")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    train(args.arch, args.epochs, args.batch_size, args.lr)
