"""Move exported fall service assets into YOLO-style images/labels folders."""
from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path("/Users/jihunjang/Downloads/용역데이터")
IMAGES_DIR = ROOT_DIR / "images"
LABELS_DIR = ROOT_DIR / "labels"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_SUFFIX = ".txt"


def ensure_dirs() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)


def move_with_collision(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / src.name
    if not target.exists():
        src.rename(target)
        return target

    stem = src.stem
    suffix = src.suffix
    counter = 1
    while True:
        candidate = dst_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            src.rename(candidate)
            return candidate
        counter += 1


def organize(root: Path) -> None:
    ensure_dirs()
    for entry in root.iterdir():
        if entry.is_dir():
            # Skip already organized folders
            if entry in (IMAGES_DIR, LABELS_DIR):
                continue
            # Leave other directories untouched
            continue

        suffix = entry.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            move_with_collision(entry, IMAGES_DIR)
        elif suffix == LABEL_SUFFIX:
            move_with_collision(entry, LABELS_DIR)


def main() -> None:
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"Root directory not found: {ROOT_DIR}")
    organize(ROOT_DIR)


if __name__ == "__main__":
    main()

