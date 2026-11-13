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