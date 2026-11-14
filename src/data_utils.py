"""
Utilidades de datos: carga del dataset IMD, split estratificado y constantes.
Dataset: 200 imágenes dermoscópicas (572x765) con máscara GT y 3 clases.
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_IMAGES_DIR = ROOT / "images" / "images"
RAW_MASKS_DIR = ROOT / "images" / "masks"
LABELS_CSV = ROOT / "data" / "raw" / "labels" / "PH2_simple_dataset.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

CLASSES = ["Common Nevus", "Atypical Nevus", "Melanoma"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

IMG_SIZE = 256          # tamaño para segmentación
CLS_SIZE = 224          # tamaño para clasificación (transferencia)
SEED = 42


def load_labels():
    """DataFrame indexado por image_name con columna class_idx."""
    df = pd.read_csv(LABELS_CSV)
    df["class_idx"] = df["diagnosis"].map(CLASS_TO_IDX)
    return df


def list_dataset():
    """Devuelve (ids, labels dict id->class_idx) de las imágenes con etiqueta."""
    df = load_labels()
    ids, labels = [], {}
    for _, row in df.iterrows():
        img_path = RAW_IMAGES_DIR / f"{row.image_name}.png"
        if img_path.exists():
            ids.append(row.image_name)
            labels[row.image_name] = row.class_idx
    return sorted(ids), labels


def stratified_split(seed=SEED, train_frac=0.6, val_frac=0.2):
    """Split 60/20/20 estratificado por clase. Devuelve dict con listas de ids."""
    _, labels = list_dataset()
    rng = random.Random(seed)
    by_class = {}
    for img_id, c in labels.items():
        by_class.setdefault(c, []).append(img_id)

    splits = {"train": [], "val": [], "test": []}
    for c in sorted(by_class):
        ids = sorted(by_class[c])
        rng.shuffle(ids)
        n_train = round(len(ids) * train_frac)
        n_val = round(len(ids) * val_frac)
        splits["train"] += ids[:n_train]
        splits["val"] += ids[n_train:n_train + n_val]
        splits["test"] += ids[n_train + n_val:]
    for s in splits.values():
        s.sort()
    return splits


def load_image_and_mask(img_id, preprocessed_dir=None):
    """Carga imagen BGR y máscara GT binaria (uint8 0/1)."""
    if preprocessed_dir is not None:
        img = cv2.imread(str(PROCESSED_DIR / preprocessed_dir / f"{img_id}.png"))
    else:
        img = cv2.imread(str(RAW_IMAGES_DIR / f"{img_id}.png"))
    mask = cv2.imread(str(RAW_MASKS_DIR / f"{img_id}_segmentation.png"), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        raise FileNotFoundError(f"No se pudo cargar {img_id}")
    return img, (mask > 127).astype(np.uint8)


def resize_pair(img, mask, size):
    """Redimensiona imagen y máscara (nearest) al mismo cuadrado."""
    img_r = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    mask_r = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return img_r, mask_r


if __name__ == "__main__":
    ids, labels = list_dataset()
    counts = np.bincount(list(labels.values()), minlength=3)
    print(f"{len(ids)} imágenes: " + ", ".join(f"{CLASSES[i]}={n}" for i, n in enumerate(counts)))
    sp = stratified_split()
    for k, v in sp.items():
        cc = np.bincount([labels[i] for i in v], minlength=3)
        print(f"{k}: {len(v)} -> {cc.tolist()}")
