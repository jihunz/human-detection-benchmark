"""Split a YOLO-formatted dataset into train/val subsets for fine-tuning.

Expected input directory structure:

    ROOT_DIR/
        images/
            sample1.jpg
            ...
        labels/
            sample1.txt
            ...

The script shuffles matched image/label pairs, splits them by VAL_RATIO, and
copies them into train/ and val/ sub-folders located under OUTPUT_DIR (defaults
to ROOT_DIR). File transfers use hardlinks when possible to avoid duplication,
falling back to copies.
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path


ROOT_DIR = Path(
    "/Users/jihunjang/Downloads/dataset/train/yolo 학습용 통합"
)
OUTPUT_DIR = ROOT_DIR  # change if you prefer a different destination
VAL_RATIO = 0.2
RANDOM_SEED = 42
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = labels_dir / (image_path.stem + ".txt")
        if not label_path.is_file():
            continue
        pairs.append((image_path, label_path))
    return pairs


def transfer(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    try:
        os.symlink(src, dst)
        return
    except OSError:
        pass
    shutil.copy2(src, dst)


def split_dataset(pairs: list[tuple[Path, Path]]):
    rand = random.Random(RANDOM_SEED)
    rand.shuffle(pairs)
    val_count = max(1, int(len(pairs) * VAL_RATIO)) if pairs else 0
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]
    return train_pairs, val_pairs


def organise_subset(subdir: str, pairs: list[tuple[Path, Path]], dest_root: Path) -> int:
    images_target = ensure_dir(dest_root / subdir / "images")
    labels_target = ensure_dir(dest_root / subdir / "labels")
    for img_src, lbl_src in pairs:
        transfer(img_src, images_target / img_src.name)
        transfer(lbl_src, labels_target / lbl_src.name)
    return len(pairs)


def main() -> None:
    images_dir = ROOT_DIR / "images"
    labels_dir = ROOT_DIR / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError("Expected 'images' and 'labels' directories under ROOT_DIR")

    pairs = iter_pairs(images_dir, labels_dir)
    if not pairs:
        raise RuntimeError("No matched image/label pairs found")

    train_pairs, val_pairs = split_dataset(pairs)
    train_count = organise_subset("train", train_pairs, OUTPUT_DIR)
    val_count = organise_subset("val", val_pairs, OUTPUT_DIR)

    print(f"[done] train pairs: {train_count}, val pairs: {val_count}")


if __name__ == "__main__":
    main()
