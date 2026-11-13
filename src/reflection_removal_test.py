"""
Detección conservadora y reconstrucción de reflejos brillantes localizados.

Dependencias:
    numpy
    opencv-python
    matplotlib

Ejecutar desde la raíz del proyecto:

    python src/reflection_removal_test.py --ids IMD002 IMD003

Procesar todas las imágenes de la carpeta de entrada:

    python src/reflection_removal_test.py --all --no-show

Procesar las salidas del módulo de vello:

    python src/reflection_removal_test.py \
        --input-dir results/preprocessing/hair_edges \
        --pattern "**/hair_removed.png" \
        --all --no-show

Notas:
    - No es un detector completo de burbujas.
    - No toda región clara corresponde a un reflejo.
    - Los parámetros deben validarse con anotaciones y revisión visual.
    - El inpainting estima el contenido oculto; no recupera su textura real.
    - Reutilizar una carpeta de salida sobrescribe resultados.
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np


EXTENSIONS = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}


def reflect_pad(image, padding):
    return cv2.copyMakeBorder(
        image,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_REFLECT_101,
    )


def remove_reflections(image, args):
    """Devuelve imagen reconstruida, máscara propuesta y diagnóstico."""
    if (
        image is None
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ValueError("Se requiere una imagen BGR uint8.")

    height, width = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=args.sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )

    # Contraste positivo respecto al entorno.
    local_contrast = np.maximum(gray - background, 0)

    strong_bool = (
        (value >= args.v_high)
        & (saturation <= args.s_strong)
        & (local_contrast >= args.contrast_high)
    )

    weak_bool = (
        (value >= args.v_low)
        & (saturation <= args.s_weak)
        & (local_contrast >= args.contrast_low)
    )

    strong = strong_bool.astype(np.uint8) * 255
    weak = weak_bool.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )

    # Crecimiento acotado; no se recorren componentes completos.
    grown = strong.copy()

    for _ in range(args.growth_steps):
        expanded = cv2.dilate(
            grown,
            kernel,
            borderType=cv2.BORDER_REFLECT_101,
        )
        updated = cv2.bitwise_and(expanded, weak)

        if np.array_equal(updated, grown):
            break

        grown = updated

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        grown,
        connectivity=8,
    )

    maximum_area = max(
        args.min_area,
        int(round(args.max_component_fraction * height * width)),
    )

    accepted_ids = np.zeros(n_labels, dtype=bool)
    rejected_ids = np.zeros(n_labels, dtype=bool)

    for label_id in range(1, n_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])

        if args.min_area <= area <= maximum_area:
            accepted_ids[label_id] = True
        elif area > maximum_area:
            rejected_ids[label_id] = True

    mask_before_dilation = (
        accepted_ids[labels].astype(np.uint8) * 255
    )

    rejected_mask = (
        rejected_ids[labels].astype(np.uint8) * 255
    )

    mask = mask_before_dilation.copy()

    if args.dilate > 0:
        mask = cv2.dilate(
            mask,
            kernel,
            iterations=args.dilate,
            borderType=cv2.BORDER_REFLECT_101,
        )

    # No incorporar regiones grandes que se rechazaron explícitamente.
    mask[rejected_mask > 0] = 0

    coverage = float(np.mean(mask > 0))
    skipped = coverage > args.max_coverage

    if skipped:
        restored = image.copy()
        status = "skipped_excessive_mask"

    elif not np.any(mask):
        restored = image.copy()
        status = "no_candidates"

    else:
        padding = max(16, int(np.ceil(args.radius)) + 4)

        padded_image = reflect_pad(image, padding)
        padded_mask = reflect_pad(mask, padding)

        method = (
            cv2.INPAINT_TELEA
            if args.method == "telea"
            else cv2.INPAINT_NS
        )

        padded_restored = cv2.inpaint(
            padded_image,
            padded_mask,
            args.radius,
            method,
        )

        restored = padded_restored[
            padding:padding + height,
            padding:padding + width,
        ].copy()

        # No modificar ningún píxel ajeno a la máscara.
        restored[mask == 0] = image[mask == 0]
        status = "inpainted"

    debug = {
        "value": value,
        "saturation": saturation,
        "contrast": local_contrast,
        "strong": strong,
        "weak": weak,
        "grown": grown,
        "mask_before_dilation": mask_before_dilation,
        "rejected_mask": rejected_mask,
        "difference": cv2.absdiff(image, restored),
        "coverage": coverage,
        "accepted_components": int(np.count_nonzero(accepted_ids)),
        "rejected_large_components": int(np.count_nonzero(rejected_ids)),
        "maximum_component_area": maximum_area,
        "status": status,
        "skipped": skipped,
    }

    return restored, mask, debug


def create_overlay(image, mask, color=(0, 0, 255), alpha=0.45):
    result = image.copy()
    selected = mask > 0

    result[selected] = np.clip(
        (1 - alpha) * image[selected].astype(np.float32)
        + alpha * np.asarray(color, dtype=np.float32),
        0,
        255,
    ).astype(np.uint8)

    return result


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"No se pudo guardar: {path}")


def save_diagnostic(image, restored, mask, debug, title, path, show):
    import matplotlib.pyplot as plt

    result_title = {
        "inpainted": "Reconstrucción",
        "no_candidates": "Sin candidatos: original",
        "skipped_excessive_mask": "OMITIDO: máscara excesiva",
    }[debug["status"]]

    panels = [
        ("Entrada", image),
        ("Brillo: canal V", debug["value"]),
        ("Saturación: canal S", debug["saturation"]),
        ("Contraste local positivo", debug["contrast"]),
        ("Detecciones fuertes", debug["strong"]),
        ("Candidatos débiles", debug["weak"]),
        ("Crecimiento limitado", debug["grown"]),
        ("Regiones grandes rechazadas", create_overlay(
            image, debug["rejected_mask"], color=(255, 0, 255)
        )),
        ("Máscara final", mask),
        ("Máscara superpuesta", create_overlay(image, mask)),
        (result_title, restored),
        ("Diferencia absoluta", debug["difference"]),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))

    for index, (ax, (panel_title, panel)) in enumerate(
        zip(axes.flat, panels)
    ):
        if panel.ndim == 3:
            ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        elif index == 3:
            # Escala propia para visualizar contrastes pequeños.
            ax.imshow(
                panel,
                cmap="magma",
                vmin=0,
                vmax=max(1.0, float(np.max(panel))),
            )
        else:
            ax.imshow(panel, cmap="gray", vmin=0, vmax=255)

        ax.set_title(panel_title, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"{title} | Máscara: {100 * debug['coverage']:.3f}%"
        f" | Componentes aceptados: {debug['accepted_components']}"
        f"\nEstado: {debug['status']}",
        fontsize=12,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(str(path), dpi=160, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def find_image(input_dir, identifier):
    direct = input_dir / identifier

    if direct.is_file():
        return direct

    matches = [
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in EXTENSIONS
        and path.stem.lower() == Path(identifier).stem.lower()
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            f"Varios archivos para {identifier}. Indica la extensión."
        )

    raise FileNotFoundError(f"No se encontró: {identifier}")


def process_image(path, args):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"No se pudo leer: {path}")

    relative = path.relative_to(args.input_dir)
    extension = path.suffix.lower().lstrip(".")

    # Conserva subcarpetas para distinguir varias hair_removed.png.
    output_dir = (
        args.output_dir
        / relative.parent
        / f"{path.stem}_{extension}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    restored, mask, debug = remove_reflections(image, args)
    elapsed_ms = 1000 * (time.perf_counter() - start)

    outputs = {
        "input.png": image,
        "value.png": debug["value"],
        "saturation.png": debug["saturation"],
        "local_contrast.png": np.clip(
            debug["contrast"], 0, 255
        ).astype(np.uint8),
        "strong_candidates.png": debug["strong"],
        "weak_candidates.png": debug["weak"],
        "grown_mask.png": debug["grown"],
        "rejected_large_regions.png": debug["rejected_mask"],
        "mask_before_dilation.png": debug["mask_before_dilation"],
        "reflection_mask.png": mask,
        "mask_overlay.png": create_overlay(image, mask),
        "reflection_removed.png": restored,
        "difference.png": debug["difference"],
    }

    for filename, panel in outputs.items():
        save_image(output_dir / filename, panel)

    save_diagnostic(
        image,
        restored,
        mask,
        debug,
        str(relative),
        output_dir / "diagnostic.png",
        show=not args.no_show,
    )

    message = {
        "inpainted": "Reconstrucción aplicada.",
        "no_candidates": (
            "Sin candidatos aceptados; se conserva la entrada. "
            "Esto no demuestra ausencia de reflejos."
        ),
        "skipped_excessive_mask": (
            "Reconstrucción omitida por máscara excesiva. "
            "reflection_removed.png contiene la entrada."
        ),
    }[debug["status"]]

    (output_dir / "status.txt").write_text(
        f"Estado: {debug['status']}\n"
        f"{message}\n"
        f"Cobertura: {100 * debug['coverage']:.3f}%\n"
        f"Componentes aceptados: {debug['accepted_components']}\n"
        f"Componentes grandes rechazados: "
        f"{debug['rejected_large_components']}\n"
        f"Área máxima permitida: "
        f"{debug['maximum_component_area']} píxeles\n",
        encoding="utf-8",
    )

    print(
        f"{relative}: máscara={100 * debug['coverage']:.3f}%"
        f" | aceptados={debug['accepted_components']}"
        f" | grandes rechazados={debug['rejected_large_components']}"
        f" | tiempo={elapsed_ms:.1f} ms"
    )
    print(f"  {message}")
    print(f"  Resultados: {output_dir}")

    row = {
        "input_file": str(path),
        "height": image.shape[0],
        "width": image.shape[1],
        "mask_percentage": 100 * debug["coverage"],
        "accepted_components": debug["accepted_components"],
        "rejected_large_components": debug["rejected_large_components"],
        "maximum_component_area": debug["maximum_component_area"],
        "processing_ms": elapsed_ms,
        "status": debug["status"],
        "inpainting_skipped": debug["skipped"],
    }

    for key, value in vars(args).items():
        if isinstance(value, (list, tuple)):
            value = ";".join(map(str, value))
        if isinstance(value, Path):
            value = str(value)
        row[f"param_{key}"] = value

    return row


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detección de reflejos brillantes localizados.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/images"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/preprocessing/reflections"),
    )

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ids", nargs="+", default=None)
    selection.add_argument("--all", action="store_true")

    parser.add_argument(
        "--pattern",
        default="*",
        help="Patrón relativo usado con --all; admite ** para subcarpetas.",
    )

    parser.add_argument("--v-high", type=int, default=225)
    parser.add_argument("--v-low", type=int, default=195)
    parser.add_argument("--s-strong", type=int, default=80)
    parser.add_argument("--s-weak", type=int, default=110)

    parser.add_argument("--contrast-high", type=float, default=18.0)
    parser.add_argument("--contrast-low", type=float, default=7.0)
    parser.add_argument("--sigma", type=float, default=7.0)

    parser.add_argument("--growth-steps", type=int, default=4)
    parser.add_argument("--min-area", type=int, default=3)
    parser.add_argument(
        "--max-component-fraction",
        type=float,
        default=0.002,
        help="Área máxima de un componente como fracción de la imagen.",
    )

    parser.add_argument("--dilate", type=int, default=1)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--max-coverage", type=float, default=0.03)
    parser.add_argument(
        "--method", choices=["telea", "ns"], default="telea"
    )
    parser.add_argument("--no-show", action="store_true")

    args = parser.parse_args()

    if not 0 <= args.v_low < args.v_high <= 255:
        parser.error("Debe cumplirse 0 <= v-low < v-high <= 255.")

    if not 0 <= args.s_strong <= args.s_weak <= 255:
        parser.error("Debe cumplirse 0 <= s-strong <= s-weak <= 255.")

    if not 0 <= args.contrast_low < args.contrast_high <= 255:
        parser.error(
            "Debe cumplirse 0 <= contrast-low < contrast-high <= 255."
        )

    if args.sigma <= 0 or args.radius <= 0:
        parser.error("--sigma y --radius deben ser > 0.")

    if args.growth_steps < 0 or args.dilate < 0:
        parser.error("--growth-steps y --dilate deben ser >= 0.")

    if args.min_area < 1:
        parser.error("--min-area debe ser >= 1.")

    if not 0 < args.max_component_fraction <= 1:
        parser.error("--max-component-fraction debe estar en (0, 1].")

    if not 0 < args.max_coverage <= 1:
        parser.error("--max-coverage debe estar en (0, 1].")

    return args


def main():
    args = parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    if not args.input_dir.is_dir():
        raise FileNotFoundError(
            f"No existe {args.input_dir}. "
            "Ejecuta desde la raíz del proyecto o indica --input-dir."
        )

    # Evita que un procesamiento recursivo vuelva a leer sus salidas.
    input_root = args.input_dir.resolve()
    output_root = args.output_dir.resolve()

    if output_root == input_root or input_root in output_root.parents:
        raise ValueError(
            "--output-dir no puede estar dentro de --input-dir."
        )

    if args.all:
        paths = sorted(
            path
            for path in args.input_dir.glob(args.pattern)
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        )
    else:
        paths = [
            find_image(args.input_dir, identifier)
            for identifier in (args.ids or ["IMD002", "IMD003"])
        ]

    if not paths:
        raise FileNotFoundError("No se encontraron imágenes.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Imágenes: {len(paths)}")
    print(f"Brillo: fuerte={args.v_high}, débil={args.v_low}")
    print(
        f"Saturación máxima: fuerte={args.s_strong}, "
        f"débil={args.s_weak}"
    )
    print(
        f"Contraste: fuerte={args.contrast_high}, "
        f"débil={args.contrast_low}"
    )
    print()

    rows = []
    failures = []

    for path in paths:
        try:
            rows.append(process_image(path, args))
        except Exception as error:
            failures.append(f"{path}: {error}")
            print(f"ERROR en {path.name}: {error}")

    if rows:
        summary = args.output_dir / "processing_summary.csv"

        with summary.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nResumen de esta ejecución: {summary}")

    if failures:
        (args.output_dir / "errors.txt").write_text(
            "\n".join(failures),
            encoding="utf-8",
        )

    print(f"\nProcesadas: {len(rows)}")
    print(
        "Reconstrucciones aplicadas: "
        f"{sum(row['status'] == 'inpainted' for row in rows)}"
    )
    print(
        "Omitidas por máscara excesiva: "
        f"{sum(row['inpainting_skipped'] for row in rows)}"
    )
    print(f"Errores: {len(failures)}")

    if not rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()