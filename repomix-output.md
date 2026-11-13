This file is a merged representation of the entire codebase, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
data/
  processed/
    .gitkeep
  raw/
    labels/
      PH2_simple_dataset.csv
    .gitkeep
models/
  .gitkeep
notebooks/
  .gitkeep
src/
  .gitkeep
  apply_preprocessing.py
  hair_removal_dullrazor.py
  preprocessing_retinex.py
  reflection_removal_test.py
.gitignore
requirements.txt
test.py
```

# Files

## File: src/apply_preprocessing.py
```python
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
```

## File: src/hair_removal_dullrazor.py
```python
"""
Eliminación conservadora de vello oscuro con tratamiento de bordes, 
suavizado adaptativo y mezcla difuminada (feathering).
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np


def reflect_pad(image, padding):
    return cv2.copyMakeBorder(
        image, padding, padding, padding, padding, cv2.BORDER_REFLECT_101
    )


def crop_padding(image, padding, height, width):
    return image[padding:padding + height, padding:padding + width].copy()


def create_edge_band(shape, width):
    height, image_width = shape
    band = np.zeros(shape, dtype=bool)
    if width <= 0:
        return band
    width = min(width, max(1, min(height, image_width) // 2))
    band[:width, :] = True
    band[-width:, :] = True
    band[:, :width] = True
    band[:, -width:] = True
    return band


def remove_hair_multiscale(
    img_bgr,
    kernel_sizes=(9, 17, 25),
    low_threshold=12, 
    high_threshold=25, 
    min_area=15, 
    edge_min_area=8,
    dilation_iterations=0,
    inpaint_radius=3.0,
    method="telea",
    max_coverage=0.40,
    padding=40,
    edge_width=30,
    edge_factor=0.85,
):
    if (img_bgr is None or img_bgr.dtype != np.uint8 or img_bgr.ndim != 3 or img_bgr.shape[2] != 3):
        raise ValueError("Se requiere una imagen BGR uint8 de tres canales.")
    if not kernel_sizes or any(size < 3 or size % 2 == 0 for size in kernel_sizes):
        raise ValueError("Los kernels deben ser impares y >= 3.")
    if not 0 < low_threshold < high_threshold <= 255:
        raise ValueError("Debe cumplirse 0 < low < high <= 255.")
    if min_area < 1 or edge_min_area < 1:
        raise ValueError("Las áreas mínimas deben ser >= 1.")
    if dilation_iterations < 0 or inpaint_radius <= 0:
        raise ValueError("Dilatación >= 0 y radio > 0.")
    if method not in ("telea", "ns"):
        raise ValueError("Método no válido.")
    if not 0 < max_coverage <= 1:
        raise ValueError("max_coverage debe estar en (0, 1].")
    if padding < 0 or edge_width < 0:
        raise ValueError("padding y edge_width deben ser >= 0.")
    if not 0 < edge_factor <= 1:
        raise ValueError("edge_factor debe estar en (0, 1].")

    height, width = img_bgr.shape[:2]

    actual_padding = max(
        padding, max(kernel_sizes) + 3, int(np.ceil(inpaint_radius)) + 3,
    )

    padded_img = reflect_pad(img_bgr, actual_padding)
    padded_gray = cv2.cvtColor(padded_img, cv2.COLOR_BGR2GRAY)

    detection_gray = cv2.medianBlur(padded_gray, 3)

    padded_response = np.zeros_like(padded_gray)

    # 1. Detección elíptica
    for size in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        current = cv2.morphologyEx(
            detection_gray, cv2.MORPH_BLACKHAT, kernel, borderType=cv2.BORDER_REFLECT_101,
        )
        padded_response = cv2.max(padded_response, current)

    # 2. Detección lineal (para pelos finos)
    line_kernel_size = 15
    for angle in [0, 45, 90, 135]:
        kernel = np.zeros((line_kernel_size, line_kernel_size), dtype=np.uint8)
        if angle == 0: kernel[line_kernel_size//2, :] = 1
        elif angle == 45: np.fill_diagonal(kernel, 1)
        elif angle == 90: kernel[:, line_kernel_size//2] = 1
        elif angle == 135: np.fill_diagonal(np.fliplr(kernel), 1)
            
        current = cv2.morphologyEx(
            detection_gray, cv2.MORPH_BLACKHAT, kernel, borderType=cv2.BORDER_REFLECT_101,
        )
        padded_response = cv2.max(padded_response, current)

    response = crop_padding(padded_response, actual_padding, height, width)

    edge_band = create_edge_band((height, width), edge_width)

    low_map = np.full((height, width), low_threshold, dtype=np.float32)
    high_map = np.full((height, width), high_threshold, dtype=np.float32)

    low_map[edge_band] *= edge_factor
    high_map[edge_band] *= edge_factor

    weak = (response >= low_map).astype(np.uint8) * 255
    strong = (response >= high_map).astype(np.uint8) * 255

    local_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    expanded = cv2.dilate(strong, local_kernel, iterations=1, borderType=cv2.BORDER_REFLECT_101)
    candidates = cv2.bitwise_and(expanded, weak)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)

    border_labels = np.unique(
        np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    )

    touches_border = np.zeros(n_labels, dtype=bool)
    touches_border[border_labels] = True
    touches_border[0] = False

    minimum_areas = np.full(n_labels, min_area, dtype=np.int32)
    minimum_areas[touches_border] = edge_min_area

    keep = stats[:, cv2.CC_STAT_AREA] >= minimum_areas
    keep[0] = False

    mask_before_dilation = (keep[labels].astype(np.uint8) * 255)
    hair_mask = mask_before_dilation.copy()

    if dilation_iterations > 0:
        hair_mask = cv2.dilate(
            hair_mask, local_kernel, iterations=dilation_iterations, borderType=cv2.BORDER_REFLECT_101,
        )

    selected_pixels = int(np.count_nonzero(hair_mask))
    coverage = selected_pixels / hair_mask.size

    edge_pixels = int(np.count_nonzero(edge_band))
    edge_coverage = (
        float(np.count_nonzero(hair_mask[edge_band])) / edge_pixels if edge_pixels else 0.0
    )

    skipped = coverage > max_coverage

    if skipped:
        restored = img_bgr.copy()
        status = "skipped_excessive_mask"
    elif selected_pixels == 0:
        restored = img_bgr.copy()
        status = "no_candidates"
    else:
        flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS

        padded_mask = reflect_pad(hair_mask, actual_padding)
        padded_restored = cv2.inpaint(padded_img, padded_mask, inpaint_radius, flag)
        restored = crop_padding(padded_restored, actual_padding, height, width)

        # =========================================================
        # MEJORA FINAL: SUAVIZADO Y MEZCLA ALFA (FEATHERING)
        # =========================================================
        
        # 1. Encontrar zonas densas y aplicarles un suavizado orgánico (Mediana grande)
        kernel_denso = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dense_cores = cv2.erode(hair_mask, kernel_denso)
        dense_mask = cv2.dilate(dense_cores, kernel_denso)
        
        # Usamos mediana en lugar de Gaussiano para no perder textura natural
        heavy_blur = cv2.medianBlur(restored, 15)
        dense_mask_3ch = cv2.cvtColor(dense_mask, cv2.COLOR_GRAY2BGR)
        restored = np.where(dense_mask_3ch > 0, heavy_blur, restored)

        # 2. Mezcla difuminada (Feathering) para borrar los bordes duros
        # Convertimos la máscara a float [0, 1]
        mask_float = hair_mask.astype(np.float32) / 255.0
        
        # Difuminamos la máscara para que los bordes sean suaves
        # El (7, 7) se puede ajustar según el grosor del vello
        mask_float = cv2.GaussianBlur(mask_float, (7, 7), 0)
        
        # Aseguramos que sea de 3 canales para la mezcla
        mask_3ch = cv2.cvtColor(mask_float, cv2.COLOR_GRAY2BGR)
        
        # Mezcla alfa: (Parche * Máscara) + (Original * (1 - Máscara))
        original_float = img_bgr.astype(np.float32)
        restored_float = restored.astype(np.float32)
        
        final_restored = (restored_float * mask_3ch) + (original_float * (1.0 - mask_3ch))
        restored = np.clip(final_restored, 0, 255).astype(np.uint8)
        
        # =========================================================
        
        status = "inpainted"

    debug = {
        "blackhat": response,
        "weak": weak,
        "strong": strong,
        "candidates": candidates,
        "edge_band": edge_band.astype(np.uint8) * 255,
        "mask_before_dilation": mask_before_dilation,
        "difference": cv2.absdiff(img_bgr, restored),
        "coverage": float(coverage),
        "edge_coverage": float(edge_coverage),
        "components_kept": int(np.count_nonzero(keep)),
        "border_components_kept": int(np.count_nonzero(keep & touches_border)),
        "selected_pixels": selected_pixels,
        "actual_padding": actual_padding,
        "inpainting_skipped": skipped,
        "status": status,
    }

    return restored, hair_mask, debug

# (El resto de funciones: create_overlay, save_image, save_diagnostic_figure, find_image, process_image, parse_args y main se mantienen IGUAL)

def create_overlay(img_bgr, mask, alpha=0.45, color=(0, 0, 255)):
    overlay = img_bgr.copy()
    selected = mask > 0
    overlay[selected] = np.clip(
        (1 - alpha) * img_bgr[selected].astype(np.float32) + alpha * np.array(color, dtype=np.float32),
        0, 255,
    ).astype(np.uint8)
    return overlay

def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"No se pudo guardar: {path}")

def save_diagnostic_figure(img_bgr, restored, hair_mask, debug, image_id, output_path, show):
    import matplotlib.pyplot as plt
    result_title = {"inpainted": "Resultado del inpainting", "no_candidates": "Sin candidatos: original", "skipped_excessive_mask": "OMITIDO: máscara excesiva"}[debug["status"]]
    panels = [("Original", img_bgr), ("Black-hat", debug["blackhat"]), ("Detecciones fuertes", debug["strong"]), ("Candidatos débiles", debug["weak"]), ("Franja periférica", create_overlay(img_bgr, debug["edge_band"], alpha=0.25, color=(255, 255, 0))), ("Máscara final", hair_mask), ("Máscara superpuesta", create_overlay(img_bgr, hair_mask)), (result_title, restored), ("Diferencia absoluta", debug["difference"])]
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    for ax, (title, image) in zip(axes.flat, panels):
        if image.ndim == 2: ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        else: ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{image_id} | Selección: {100 * debug['coverage']:.2f}% | Selección en franja: {100 * debug['edge_coverage']:.2f}%\nEstado: {debug['status']}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(str(output_path), dpi=160, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)

def find_image(input_dir, image_id):
    direct = input_dir / image_id
    if direct.is_file(): return direct
    extensions = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}
    matches = [path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions and path.stem.lower() == Path(image_id).stem.lower()]
    if len(matches) == 1: return matches[0]
    if len(matches) > 1: raise ValueError(f"Varios archivos para {image_id}. Indica la extensión.")
    raise FileNotFoundError(f"No se encontró: {image_id}")

def process_image(image_path, args):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None: raise ValueError(f"No se pudo leer: {image_path}")
    image_id = image_path.stem
    extension = image_path.suffix.lower().lstrip(".")
    output_dir = args.output_dir / f"{image_id}_{extension}"
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    restored, mask, debug = remove_hair_multiscale(
        img_bgr=img, kernel_sizes=tuple(args.kernels), low_threshold=args.low, high_threshold=args.high,
        min_area=args.min_area, edge_min_area=args.edge_min_area, dilation_iterations=args.dilate,
        inpaint_radius=args.radius, method=args.method, max_coverage=args.max_coverage,
        padding=args.padding, edge_width=args.edge_width, edge_factor=args.edge_factor,
    )
    elapsed_ms = 1000 * (time.perf_counter() - start)

    outputs = {"original.png": img, "blackhat.png": debug["blackhat"], "strong_candidates.png": debug["strong"], "weak_candidates.png": debug["weak"], "local_candidates.png": debug["candidates"], "edge_band.png": debug["edge_band"], "mask_before_dilation.png": debug["mask_before_dilation"], "hair_mask.png": mask, "mask_overlay.png": create_overlay(img, mask), "hair_removed.png": restored, "difference.png": debug["difference"]}
    for name, image in outputs.items(): save_image(output_dir / name, image)

    save_diagnostic_figure(img, restored, mask, debug, image_id, output_dir / "diagnostic.png", show=not args.no_show)
    
    message = {"inpainted": "Reconstrucción aplicada.", "no_candidates": "Sin candidatos; se conserva el original.", "skipped_excessive_mask": "Reconstrucción omitida por máscara excesiva."}[debug["status"]]
    (output_dir / "status.txt").write_text(f"Estado: {debug['status']}\n{message}\nCobertura total: {100 * debug['coverage']:.2f}%\nCobertura de la franja: {100 * debug['edge_coverage']:.2f}%\nPadding utilizado: {debug['actual_padding']} px\n", encoding="utf-8")
    
    print(f"{image_id}: máscara={100 * debug['coverage']:.2f}% | franja={100 * debug['edge_coverage']:.2f}% | tiempo={elapsed_ms:.1f} ms")
    print(f"  {message}\n  Resultados: {output_dir}")

    return {"image_id": image_id, "input_file": str(image_path), "height": img.shape[0], "width": img.shape[1], "mask_percentage": 100 * debug["coverage"], "edge_mask_percentage": 100 * debug["edge_coverage"], "selected_pixels": debug["selected_pixels"], "components_kept": debug["components_kept"], "border_components_kept": debug["border_components_kept"], "processing_ms": elapsed_ms, "status": debug["status"], "inpainting_skipped": debug["inpainting_skipped"], "method": args.method, "low_threshold": args.low, "high_threshold": args.high, "kernel_sizes": ";".join(map(str, args.kernels)), "min_area": args.min_area, "edge_min_area": args.edge_min_area, "dilation_iterations": args.dilate, "inpaint_radius": args.radius, "padding_used": debug["actual_padding"], "edge_width": args.edge_width, "edge_factor": args.edge_factor, "max_coverage": args.max_coverage}

def parse_args():
    parser = argparse.ArgumentParser(description="Eliminación de vello con mezcla difuminada.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/preprocessing/hair_edges"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ids", nargs="+", default=None)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--kernels", nargs="+", type=int, default=[9, 17, 25])
    parser.add_argument("--low", type=int, default=12) 
    parser.add_argument("--high", type=int, default=25)
    parser.add_argument("--min-area", type=int, default=15)
    parser.add_argument("--edge-min-area", type=int, default=8)
    parser.add_argument("--dilate", type=int, default=0)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--method", choices=["telea", "ns"], default="telea")
    parser.add_argument("--max-coverage", type=float, default=0.40)
    parser.add_argument("--padding", type=int, default=40)
    parser.add_argument("--edge-width", type=int, default=30)
    parser.add_argument("--edge-factor", type=float, default=0.85)
    parser.add_argument("--no-show", action="store_true")

    args = parser.parse_args()
    if not 0 < args.low < args.high <= 255: parser.error("Debe cumplirse 0 < --low < --high <= 255.")
    if any(size < 3 or size % 2 == 0 for size in args.kernels): parser.error("--kernels requiere tamaños impares y >= 3.")
    if args.min_area < 1 or args.edge_min_area < 1: parser.error("Las áreas mínimas deben ser >= 1.")
    if args.dilate < 0 or args.radius <= 0: parser.error("--dilate >= 0 y --radius > 0.")
    if args.padding < 0 or args.edge_width < 0: parser.error("--padding y --edge-width deben ser >= 0.")
    if not 0 < args.max_coverage <= 1: parser.error("--max-coverage debe estar en (0, 1].")
    if not 0 < args.edge_factor <= 1: parser.error("--edge-factor debe estar en (0, 1].")
    return args

def main():
    args = parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    if not args.input_dir.is_dir(): raise FileNotFoundError(f"No existe {args.input_dir}.")
    extensions = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}
    
    if args.all:
        image_paths = sorted(path for path in args.input_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions)
    else:
        image_paths = [find_image(args.input_dir, image_id) for image_id in (args.ids or ["IMD002", "IMD003"])]
    if not image_paths: raise FileNotFoundError("No se encontraron imágenes.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Imágenes: {len(image_paths)}\nMétodo: {args.method}\nUmbrales: {args.low}/{args.high}\nKernels: {args.kernels}\nFranja: {args.edge_width} px | factor: {args.edge_factor}\n")

    rows, failures = [], []
    for image_path in image_paths:
        try: rows.append(process_image(image_path, args))
        except Exception as error:
            failures.append((str(image_path), str(error)))
            print(f"ERROR en {image_path.name}: {error}")

    if rows:
        summary_path = args.output_dir / "processing_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResumen de esta ejecución: {summary_path}")

    if failures:
        (args.output_dir / "errors.txt").write_text("\n".join(f"{path}: {message}" for path, message in failures), encoding="utf-8")

    print(f"\nFinalizado:\n  Procesadas: {len(rows)}\n  Reconstrucciones aplicadas: {sum(row['status'] == 'inpainted' for row in rows)}\n  Omitidas: {sum(row['inpainting_skipped'] for row in rows)}\n  Errores: {len(failures)}")
    if not rows: raise SystemExit(1)

if __name__ == "__main__":
    main()
```

## File: src/preprocessing_retinex.py
```python
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
```

## File: src/reflection_removal_test.py
```python
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
```

## File: data/processed/.gitkeep
```

```

## File: data/raw/labels/PH2_simple_dataset.csv
```
image_name,diagnosis
IMD003,Common Nevus
IMD009,Common Nevus
IMD016,Common Nevus
IMD022,Common Nevus
IMD024,Common Nevus
IMD025,Common Nevus
IMD035,Common Nevus
IMD038,Common Nevus
IMD042,Common Nevus
IMD044,Common Nevus
IMD045,Common Nevus
IMD050,Common Nevus
IMD092,Common Nevus
IMD101,Common Nevus
IMD103,Common Nevus
IMD112,Common Nevus
IMD118,Common Nevus
IMD125,Common Nevus
IMD132,Common Nevus
IMD134,Common Nevus
IMD135,Common Nevus
IMD144,Common Nevus
IMD146,Common Nevus
IMD147,Common Nevus
IMD150,Common Nevus
IMD152,Common Nevus
IMD156,Common Nevus
IMD159,Common Nevus
IMD161,Common Nevus
IMD162,Common Nevus
IMD175,Common Nevus
IMD177,Common Nevus
IMD182,Common Nevus
IMD198,Common Nevus
IMD200,Common Nevus
IMD010,Common Nevus
IMD017,Common Nevus
IMD020,Common Nevus
IMD039,Common Nevus
IMD041,Common Nevus
IMD105,Common Nevus
IMD107,Common Nevus
IMD108,Common Nevus
IMD133,Common Nevus
IMD142,Common Nevus
IMD143,Common Nevus
IMD160,Common Nevus
IMD173,Common Nevus
IMD176,Common Nevus
IMD196,Common Nevus
IMD197,Common Nevus
IMD199,Common Nevus
IMD203,Common Nevus
IMD204,Common Nevus
IMD206,Common Nevus
IMD207,Common Nevus
IMD208,Common Nevus
IMD364,Common Nevus
IMD365,Common Nevus
IMD367,Common Nevus
IMD371,Common Nevus
IMD372,Common Nevus
IMD374,Common Nevus
IMD375,Common Nevus
IMD378,Common Nevus
IMD379,Common Nevus
IMD380,Common Nevus
IMD381,Common Nevus
IMD383,Common Nevus
IMD384,Common Nevus
IMD385,Common Nevus
IMD389,Common Nevus
IMD390,Common Nevus
IMD392,Common Nevus
IMD394,Common Nevus
IMD395,Common Nevus
IMD397,Common Nevus
IMD399,Common Nevus
IMD400,Common Nevus
IMD402,Common Nevus
IMD002,Atypical Nevus
IMD004,Atypical Nevus
IMD013,Atypical Nevus
IMD015,Atypical Nevus
IMD019,Atypical Nevus
IMD021,Atypical Nevus
IMD027,Atypical Nevus
IMD030,Atypical Nevus
IMD032,Atypical Nevus
IMD033,Atypical Nevus
IMD037,Atypical Nevus
IMD040,Atypical Nevus
IMD043,Atypical Nevus
IMD047,Atypical Nevus
IMD048,Atypical Nevus
IMD049,Atypical Nevus
IMD057,Atypical Nevus
IMD075,Atypical Nevus
IMD076,Atypical Nevus
IMD078,Atypical Nevus
IMD120,Atypical Nevus
IMD126,Atypical Nevus
IMD137,Atypical Nevus
IMD138,Atypical Nevus
IMD139,Atypical Nevus
IMD140,Atypical Nevus
IMD149,Atypical Nevus
IMD153,Atypical Nevus
IMD157,Atypical Nevus
IMD164,Atypical Nevus
IMD166,Atypical Nevus
IMD169,Atypical Nevus
IMD171,Atypical Nevus
IMD210,Atypical Nevus
IMD347,Atypical Nevus
IMD155,Atypical Nevus
IMD376,Atypical Nevus
IMD006,Atypical Nevus
IMD008,Atypical Nevus
IMD014,Atypical Nevus
IMD018,Atypical Nevus
IMD023,Atypical Nevus
IMD031,Atypical Nevus
IMD036,Atypical Nevus
IMD154,Atypical Nevus
IMD170,Atypical Nevus
IMD226,Atypical Nevus
IMD243,Atypical Nevus
IMD251,Atypical Nevus
IMD254,Atypical Nevus
IMD256,Atypical Nevus
IMD278,Atypical Nevus
IMD279,Atypical Nevus
IMD280,Atypical Nevus
IMD304,Atypical Nevus
IMD305,Atypical Nevus
IMD306,Atypical Nevus
IMD312,Atypical Nevus
IMD328,Atypical Nevus
IMD331,Atypical Nevus
IMD339,Atypical Nevus
IMD356,Atypical Nevus
IMD360,Atypical Nevus
IMD368,Atypical Nevus
IMD369,Atypical Nevus
IMD370,Atypical Nevus
IMD382,Atypical Nevus
IMD386,Atypical Nevus
IMD388,Atypical Nevus
IMD393,Atypical Nevus
IMD396,Atypical Nevus
IMD398,Atypical Nevus
IMD427,Atypical Nevus
IMD430,Atypical Nevus
IMD431,Atypical Nevus
IMD432,Atypical Nevus
IMD433,Atypical Nevus
IMD434,Atypical Nevus
IMD436,Atypical Nevus
IMD437,Atypical Nevus
IMD058,Melanoma
IMD061,Melanoma
IMD063,Melanoma
IMD064,Melanoma
IMD065,Melanoma
IMD080,Melanoma
IMD085,Melanoma
IMD088,Melanoma
IMD090,Melanoma
IMD091,Melanoma
IMD168,Melanoma
IMD211,Melanoma
IMD219,Melanoma
IMD240,Melanoma
IMD242,Melanoma
IMD284,Melanoma
IMD285,Melanoma
IMD348,Melanoma
IMD349,Melanoma
IMD403,Melanoma
IMD404,Melanoma
IMD405,Melanoma
IMD407,Melanoma
IMD408,Melanoma
IMD409,Melanoma
IMD410,Melanoma
IMD413,Melanoma
IMD417,Melanoma
IMD418,Melanoma
IMD419,Melanoma
IMD406,Melanoma
IMD411,Melanoma
IMD420,Melanoma
IMD421,Melanoma
IMD423,Melanoma
IMD424,Melanoma
IMD425,Melanoma
IMD426,Melanoma
IMD429,Melanoma
IMD435,Melanoma
```

## File: data/raw/.gitkeep
```

```

## File: models/.gitkeep
```

```

## File: notebooks/.gitkeep
```

```

## File: src/.gitkeep
```

```

## File: .gitignore
```
# Datos (pesados, no deben ir a Git)
data/raw/*
data/processed/*

!data/raw/.gitkeep
!data/processed/.gitkeep
!data/raw/labels/
!data/raw/labels/PH2_simple_dataset.csv

# Modelos entrenados
models/*.pth
models/*.h5
models/*.pt

# Entornos virtuales
venv/
env/
.venv/

# Python
__pycache__/
*.py[cod]
*.egg-info/

# Jupyter
.ipynb_checkpoints/

# Resultados generados (opcional, según prefieras)
# results/

# Sistema
.DS_Store
Thumbs.db
```

## File: test.py
```python
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
```

## File: requirements.txt
```
contourpy==1.4.0
cycler==0.12.1
dlltracer==1.0.3
fonttools==4.68.0
ImageIO==2.38.1
kiwisolver==1.5.1
lazy-loader==0.6
matplotlib==3.11.2
networkx==3.7
numpy==2.5.4
opencv-python==4.11.0.86
packaging==26.3
pandas==3.1.1
pillow==13.0.0
pyparsing==3.3.3
python-dateutil==4.0.0
scikit-image==0.26.0
scipy==1.18.1
tifffile==2026.11.8
tzdata==2026.6
```
