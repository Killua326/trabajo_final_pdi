import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def remove_reflections_strict(img_bgr):
    # 1. Convertir a HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 2. Umbrales ESTRICTOS
    # V > 235: Solo píxeles extremadamente brillantes (casi blancos)
    # S < 20: Solo píxeles con casi cero color (saturación muy baja)
    lower_white = np.array([0, 0, 235])
    upper_white = np.array([180, 20, 255])
    
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 3. Limpieza morfológica inicial (quitar ruido de 1 píxel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    
    # 4. FILTRO DE FORMA Y ÁREA (El toque final)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered_mask = np.zeros_like(mask)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Las burbujas tienen un área pequeña (entre 5 y 300 píxeles)
        if 5 <= area <= 300:
            filtered_mask[labels == i] = 255
            
    # 5. Dilatación para asegurar que el inpainting cubra el borde del reflejo
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_dilated = cv2.dilate(filtered_mask, kernel_dilate, iterations=1)
    
    # 6. Inpainting
    result = cv2.inpaint(img_bgr, mask_dilated, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    # 7. Mezcla difuminada para integrar el parche
    mask_float = mask_dilated.astype(np.float32) / 255.0
    mask_float = cv2.GaussianBlur(mask_float, (5, 5), 0)
    mask_3ch = cv2.cvtColor(mask_float, cv2.COLOR_GRAY2BGR)
    
    original_float = img_bgr.astype(np.float32)
    restored_float = result.astype(np.float32)
    final_result = (restored_float * mask_3ch) + (original_float * (1.0 - mask_3ch))
    final_result = np.clip(final_result, 0, 255).astype(np.uint8)
    
    return final_result, mask_dilated, hsv

# --- Prueba con las dos imágenes ---
base = "data/raw/images" 
img_ids = ["IMD002", "IMD003"] 

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for i, img_id in enumerate(img_ids):
    img = cv2.imread(os.path.join(base, img_id + ".png"))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    result, mask, hsv = remove_reflections_strict(img)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    h, s, v = cv2.split(hsv)
    
    axes[i, 0].imshow(img_rgb); axes[i, 0].set_title(f"Original - {img_id}")
    axes[i, 1].imshow(v, cmap="gray"); axes[i, 1].set_title("Canal V (Brillo)")
    axes[i, 2].imshow(mask, cmap="gray"); axes[i, 2].set_title("Máscara (V>235, S<20)")
    axes[i, 3].imshow(result_rgb); axes[i, 3].set_title("Resultado Final")

    for ax in axes[i]: ax.axis("off")

plt.tight_layout()
plt.show()