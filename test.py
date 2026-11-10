import pandas as pd
import os
import cv2
import numpy as np

base = "data/raw"

# 1. Cargar etiquetas
df = pd.read_csv(os.path.join(base, "labels", "PH2_simple_dataset.csv"))

# 2. Listar imágenes y máscaras
imgs = sorted([f.replace(".png", "") for f in os.listdir(os.path.join(base, "images")) if f.endswith(".png")])
masks = sorted([f.replace("_segmentation.png", "") for f in os.listdir(os.path.join(base, "masks")) if f.endswith(".png")])

print("Imágenes:", len(imgs))
print("Máscaras:", len(masks))
print("Etiquetas:", len(df))
print("Clases:", df["diagnosis"].value_counts().to_dict())
print("Imágenes sin etiqueta:", set(imgs) - set(df["image_name"]))
print("Etiquetas sin imagen:", set(df["image_name"]) - set(imgs))

# 3. Verificar máscara de ejemplo
ejemplo_id = imgs[0]
mask_path = os.path.join(base, "masks", ejemplo_id + "_segmentation.png")  # ← sufijo + extensión
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if mask is None:
    print(f"\nERROR: no se pudo leer la máscara en {mask_path}")
else:
    img_path = os.path.join(base, "images", ejemplo_id + ".png")
    img = cv2.imread(img_path)
    print(f"\nEjemplo: {ejemplo_id}")
    print("Dimensiones imagen:", img.shape)
    print("Dimensiones máscara:", mask.shape)
    print("Valores únicos máscara:", np.unique(mask))