"""Normalize MegaFall v2 test labels and remove unmatched images."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image

IMAGE_ROOT = Path("/Users/jihunjang/Downloads/dataset/train/megafallv2/images/test")
LABEL_ROOT = Path("/Users/jihunjang/Downloads/dataset/train/megafallv2/labels/test_unnormalized")
OUTPUT_ROOT = LABEL_ROOT.parent / "normalized_test"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXT = ".txt"


def find_image(label_path: Path) -> Path:
    rel_parent = label_path.relative_to(LABEL_ROOT).parent
    stem = label_path.stem
    for ext in IMAGE_EXTS:
        candidate = IMAGE_ROOT / rel_parent / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found for label: {label_path}")


def normalize_bbox(bbox: Iterable[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    try:
        xc, yc, w, h = map(float, bbox)
    except (ValueError, TypeError):
        return None
    if width <= 0 or height <= 0:
        return None

    normalized = (
        0.0 <= xc <= 1.0
        and 0.0 <= yc <= 1.0
        and 0.0 < w <= 1.0
        and 0.0 < h <= 1.0
    )

    if not normalized:
        # detect absolute top-left format (x, y, width, height)
        if xc >= 0 and yc >= 0 and w > 0 and h > 0 and xc + w / 2 <= width * 1.5 and yc + h / 2 <= height * 1.5:
            x_center = xc + w / 2
            y_center = yc + h / 2
            xc = x_center / width
            yc = y_center / height
            w = w / width
            h = h / height
        else:
            xc /= width
            yc /= height
            w /= width
            h /= height

    return (
        max(0.0, min(1.0, xc)),
        max(0.0, min(1.0, yc)),
        max(0.0, min(1.0, w)),
        max(0.0, min(1.0, h)),
    )


def normalize_label(label_path: Path, image_path: Path, output_path: Path) -> int:
    with Image.open(image_path) as img:
        width, height = img.width, img.height

    lines_out: list[str] = []
    original_lines: list[str] = []
    count = 0

    with label_path.open("r", encoding="utf-8") as src:
        for raw in src:
            raw = raw.strip()
            if not raw:
                continue
            original_lines.append(raw)
            parts = raw.split()
            try:
                class_id = int(float(parts[0]))
            except (ValueError, IndexError):
                continue
            bbox = normalize_bbox(parts[1:5], width, height)
            if bbox is None:
                continue
            lines_out.append(
                f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
            )
            count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as dst:
        if lines_out:
            dst.write("\n".join(lines_out) + "\n")
        else:
            dst.write("\n".join(original_lines) + ("\n" if original_lines else ""))
    return count


def normalize_labels() -> tuple[int, int, list[Path]]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_boxes = 0
    missing_images: list[Path] = []

    for label_path in LABEL_ROOT.rglob(f"*{LABEL_EXT}"):
        if not label_path.is_file():
            continue
        total_files += 1
        rel = label_path.relative_to(LABEL_ROOT)
        output_path = OUTPUT_ROOT / rel
        try:
            image_path = find_image(label_path)
        except FileNotFoundError:
            missing_images.append(label_path)
            continue

        count = normalize_label(label_path, image_path, output_path)
        total_boxes += count

    return total_files, total_boxes, missing_images


def log_missing(log_missing: list[Path]) -> None:
    if log_missing:
        print("Missing image for labels:")
        for label in log_missing:
            print("  ", label)


def main() -> None:
    files, boxes, missing = normalize_labels()
    log_missing(missing)
    print(f"Processed labels: {files}")
    print(f"Normalized boxes: {boxes}")
    print(f"Missing image labels: {len(missing)}")
    print(f"Output labels: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
