import cv2
import numpy as np
import os

def clahe_luminance(img_bgr, clip_limit=2.0, tile_grid=(8, 8)):
    """Aplica CLAHE al canal L de LAB."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    lab_out = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

# Rutas
input_dir = "data/raw/images"
output_dir = "data/processed/images"
os.makedirs(output_dir, exist_ok=True)

# Procesar todas las imágenes
imgs = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
print(f"Procesando {len(imgs)} imágenes...")

for i, img_name in enumerate(imgs):
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error leyendo: {img_name}")
        continue
        
    img_clahe = clahe_luminance(img)
    output_path = os.path.join(output_dir, img_name)
    cv2.imwrite(output_path, img_clahe)
    
    if (i + 1) % 50 == 0:
        print(f"  Procesadas {i + 1}/{len(imgs)}")

print("¡Preprocesamiento completado!")