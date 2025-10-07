"""Count image files in MegaFall v2 train/val directories."""
from __future__ import annotations

from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TRAIN_DIR = Path("/Users/jihunjang/Downloads/dataset/train/megafallv2/images/train")
VAL_DIR = Path("/Users/jihunjang/Downloads/dataset/train/megafallv2/images/test")


def count_images(root: Path) -> int:
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def main() -> None:
    train_count = count_images(TRAIN_DIR)
    val_count = count_images(VAL_DIR)

    print(f"train images: {train_count}")
    print(f"val images:   {val_count}")


if __name__ == "__main__":
    main()
