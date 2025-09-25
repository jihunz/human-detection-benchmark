"""Organize AI Hub exports into YOLO-style images/labels folders.

- Scans the root directory for loose .jpg/.jpeg image files and .txt label files.
- Moves images into `images/` and labels into `labels/`, creating folders as needed.
- Avoids overwriting by appending numeric suffixes when name collisions occur.

Update `ROOT_DIR` if your dataset lives elsewhere.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(
    "/Users/jihunjang/Downloads/dataset/train/for-yolo/용역데이터"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg"}
LABEL_SUFFIXES = {".txt"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_collision(dst: Path) -> Path:
    if not dst.exists():
        return dst

    stem = dst.stem
    suffix = dst.suffix
    parent = dst.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def move_file(src: Path, dst_dir: Path) -> Path:
    dst_dir = ensure_dir(dst_dir)
    destination = resolve_collision(dst_dir / src.name)
    src.rename(destination)
    return destination


def organize(root: Path) -> None:
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    images_dir = ensure_dir(root / "images")
    labels_dir = ensure_dir(root / "labels")

    moved_images = 0
    moved_labels = 0
    skipped = 0

    for item in root.iterdir():
        if item.is_dir():
            if item in (images_dir, labels_dir):
                continue
            # Skip subdirectories; handle them manually if needed.
            skipped += 1
            continue

        suffix = item.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            move_file(item, images_dir)
            moved_images += 1
        elif suffix in LABEL_SUFFIXES:
            move_file(item, labels_dir)
            moved_labels += 1
        else:
            skipped += 1

    print(
        f"[done] images moved: {moved_images}, labels moved: {moved_labels}, skipped: {skipped}"
    )


def main() -> None:
    try:
        organize(ROOT_DIR)
    except Exception as exc:  # pragma: no cover - convenience logging
        print(f"[error] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

