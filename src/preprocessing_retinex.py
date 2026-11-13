import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def multi_scale_retinex(img_bgr, sigmas=(15, 80, 250)):
    """
    Multi-Scale Retinex (MSR) aplicado sobre el canal de luminancia (Y).
    Más robusto que SSR y preserva mejor el color de la lesión.
    """
    # 1. Normalizar a [0, 1] y evitar log(0)
    img = img_bgr.astype(np.float32) / 255.0 + 1e-6

    # 2. Pasar a YCrCb
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # 3. Logaritmo de la luminancia
    y_log = np.log(y)

    # 4. Suma de Retinex a varias escalas
    retinex = np.zeros_like(y_log)
    for sigma in sigmas:
        illum = cv2.GaussianBlur(y_log, (0, 0), sigma)
        retinex += (y_log - illum)
    retinex /= len(sigmas)

    # 5. Normalizar el resultado a [0, 1]
    retinex_norm = cv2.normalize(retinex, None, 0, 1, cv2.NORM_MINMAX)

    # 6. Recomponer con Cr y Cb originales (ya están en [0,1])
    ycrcb_out = cv2.merge([retinex_norm, cr, cb])
    out = cv2.cvtColor(ycrcb_out, cv2.COLOR_YCrCb2BGR)

    # 7. Volver a [0, 255]
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def clahe_luminance(img_bgr, clip_limit=2.0, tile_grid=(8, 8)):
    """
    Alternativa suave a Retinex: CLAHE sobre el canal L (LAB).
    Suele dar resultados más conservadores en dermatoscopia.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    lab_out = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)


# ------------------- PRUEBA SOBRE UNA IMAGEN -------------------
base = "data/raw"
img_id = "IMD002"
img = cv2.imread(os.path.join(base, "images", img_id + ".png"))

if img is None:
    raise FileNotFoundError(f"No se encontró la imagen: {os.path.join(base, 'images', img_id + '.png')}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Aplicar MSR
msr = multi_scale_retinex(img)
msr_rgb = cv2.cvtColor(msr, cv2.COLOR_BGR2RGB)

# Aplicar CLAHE (para comparar)
clahe_out = clahe_luminance(img)
clahe_rgb = cv2.cvtColor(clahe_out, cv2.COLOR_BGR2RGB)

# ------------------- VISUALIZACIÓN -------------------
fig, axes = plt.subplots(3, 2, figsize=(14, 14))

# Fila 0: imágenes
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

axes[0, 1].imshow(msr_rgb)
axes[0, 1].set_title("Multi-Scale Retinex")
axes[0, 1].axis("off")

# Fila 1: imagen CLAHE y su histograma
axes[1, 0].imshow(clahe_rgb)
axes[1, 0].set_title("CLAHE (alternativa)")
axes[1, 0].axis("off")

# Fila 2: histogramas de luminancia
y_orig = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
y_msr = cv2.cvtColor(msr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
y_clahe = cv2.cvtColor(clahe_out, cv2.COLOR_BGR2YCrCb)[:, :, 0]

axes[1, 1].hist(y_orig.ravel(), bins=256, range=[0, 256], color="gray", alpha=0.7)
axes[1, 1].hist(y_msr.ravel(), bins=256, range=[0, 256], color="steelblue", alpha=0.5)
axes[1, 1].set_title("Histograma Y: Original (gris) vs MSR (azul)")
axes[1, 1].set_xlim([0, 256])

axes[2, 0].hist(y_orig.ravel(), bins=256, range=[0, 256], color="gray", alpha=0.7)
axes[2, 0].hist(y_clahe.ravel(), bins=256, range=[0, 256], color="orange", alpha=0.5)
axes[2, 0].set_title("Histograma Y: Original (gris) vs CLAHE (naranja)")
axes[2, 0].set_xlim([0, 256])

# Comparativa visual: MSR vs CLAHE
axes[2, 1].imshow(msr_rgb)
axes[2, 1].set_title("MSR de nuevo (para comparar con CLAHE)")
axes[2, 1].axis("off")

plt.tight_layout()
plt.show()

# Guardar resultados de ejemplo (opcional)
os.makedirs("results/preprocessing", exist_ok=True)
cv2.imwrite(f"results/preprocessing/{img_id}_msr.png", msr)
cv2.imwrite(f"results/preprocessing/{img_id}_clahe.png", clahe_out)
print(f"Guardadas versiones MSR y CLAHE en results/preprocessing/")