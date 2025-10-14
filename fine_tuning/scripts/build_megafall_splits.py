"""Generate YOLO train/val image lists excluding problematic MegaFall labels."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

TRAIN_IMG_ROOT = Path("/Users/jihunjang/Downloads/dataset/train/megafallv2/images/train")
TRAIN_LABEL_ROOT = Path("/Users/jihunjang/Downloads/dataset/train/megafallv2/labels/train")
VAL_IMG_ROOT = Path("/Users/jihunjang/Downloads/dataset/train/megafallv2/images/test")
VAL_LABEL_ROOT = Path("/Users/jihunjang/Downloads/dataset/train/megafallv2/labels/test")

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "v4"
TRAIN_LIST_PATH = OUTPUT_DIR / "train.txt"
VAL_LIST_PATH = OUTPUT_DIR / "val.txt"

LABEL_EXT = ".txt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_SIZE = 5e-4  # threshold for width/height after normalization


def has_bad_box(label_path: Path) -> bool:
    """Return True if any bbox in label file has width/height <= MIN_SIZE."""
    try:
        with label_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    _, xc, yc, w, h = parts[0:5]
                    w = float(w)
                    h = float(h)
                except (ValueError, IndexError):
                    return True
                if w <= MIN_SIZE or h <= MIN_SIZE:
                    return True
    except OSError:
        return True
    return False


def collect_images(img_root: Path, label_root: Path) -> list[str]:
    """Collect image paths whose labels exist and have valid boxes."""
    items: list[str] = []
    for label_path in sorted(label_root.rglob(f"*{LABEL_EXT}")):
        if not label_path.is_file():
            continue
        if has_bad_box(label_path):
            continue
        rel = label_path.relative_to(label_root).with_suffix("")
        image_path = None
        for ext in IMAGE_EXTS:
            candidate = img_root / f"{rel}{ext}"
            if candidate.exists():
                image_path = candidate.resolve()
                break
        if image_path is None:
            continue
        items.append(str(image_path))
    return items


def write_list(paths: Iterable[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(paths) + ("\n" if paths else ""))


def main() -> None:
    train_images = collect_images(TRAIN_IMG_ROOT, TRAIN_LABEL_ROOT)
    val_images = collect_images(VAL_IMG_ROOT, VAL_LABEL_ROOT)

    write_list(train_images, TRAIN_LIST_PATH)
    write_list(val_images, VAL_LIST_PATH)

    print(f"train images kept: {len(train_images)}")
    print(f"val images kept: {len(val_images)}")
    print(f"train list: {TRAIN_LIST_PATH}")
    print(f"val list: {VAL_LIST_PATH}")


if __name__ == "__main__":
    main()
