"""Copy first and last image/txt pairs from CAUCA Fall directories."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, List

SOURCE_ROOT = Path("/Users/jihunjang/Downloads/dataset/train/caucaFall/CAUCAFall")
TARGET_ROOT = Path("/Users/jihunjang/Downloads/cauca-person-fall")
TARGET_PERSON_IMAGES = TARGET_ROOT / "person" / "images"
TARGET_PERSON_LABELS = TARGET_ROOT / "person" / "labels"
TARGET_FALL_IMAGES = TARGET_ROOT / "fall" / "images"
TARGET_FALL_LABELS = TARGET_ROOT / "fall" / "labels"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXT = ".txt"


def ensure_dirs() -> None:
    TARGET_PERSON_IMAGES.mkdir(parents=True, exist_ok=True)
    TARGET_PERSON_LABELS.mkdir(parents=True, exist_ok=True)
    TARGET_FALL_IMAGES.mkdir(parents=True, exist_ok=True)
    TARGET_FALL_LABELS.mkdir(parents=True, exist_ok=True)


def list_image_files(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def matching_label(image_path: Path) -> Path | None:
    candidate = image_path.with_suffix(LABEL_EXT)
    return candidate if candidate.exists() else None


def rewrite_label(src: Path, dst: Path, class_id: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    with src.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            parts[0] = str(class_id)
            lines.append(" ".join(parts))
    if not lines:
        lines.append(f"{class_id} 0.5 0.5 1.0 1.0")
    with dst.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def copy_pair(image: Path, label: Path, prefix: str, class_id: int, is_fall: bool) -> None:
    img_dir = TARGET_FALL_IMAGES if is_fall else TARGET_PERSON_IMAGES
    lbl_dir = TARGET_FALL_LABELS if is_fall else TARGET_PERSON_LABELS
    dst_image = img_dir / f"{prefix}_{image.name}"
    dst_label = lbl_dir / f"{prefix}_{label.name}"
    counter = 1
    while dst_image.exists() or dst_label.exists():
        dst_image = img_dir / f"{prefix}_{counter}_{image.name}"
        dst_label = lbl_dir / f"{prefix}_{counter}_{label.name}"
        counter += 1
    shutil.copy2(image, dst_image)
    rewrite_label(label, dst_label, class_id)
    print(f"[copy] {image} -> {dst_image}")


def process_fall_directory(fall_dir: Path) -> None:
    images = list_image_files(fall_dir)
    if not images:
        print(f"[skip] No images in {fall_dir}")
        return
    selections: List[Path]
    if len(images) == 1:
        selections = [(images[0], False)]
    else:
        selections = [(images[0], False), (images[-1], True)]
    for image, is_fall in selections:
        label = matching_label(image)
        if label is None:
            print(f"[warn] Missing label for {image}")
            continue
        prefix = fall_dir.relative_to(SOURCE_ROOT).as_posix().replace("/", "_")
        class_id = 80 if is_fall else 0
        copy_pair(image, label, prefix, class_id, is_fall)


def main() -> None:
    ensure_dirs()
    subject_dirs = sorted([p for p in SOURCE_ROOT.iterdir() if p.is_dir() and "Subject" in p.name])
    if not subject_dirs:
        print(f"No Subject directories found under {SOURCE_ROOT}")
        return
    for subject in subject_dirs:
        fall_dirs = sorted([p for p in subject.iterdir() if p.is_dir() and "Fall" in p.name])
        for fall_dir in fall_dirs:
            process_fall_directory(fall_dir)


if __name__ == "__main__":
    main()
