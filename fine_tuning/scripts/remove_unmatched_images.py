"""Delete MegaFall v2 train images that lack a normalized label."""
from __future__ import annotations

from pathlib import Path

IMAGE_ROOT = Path("/Users/jihunjang/Downloads/dataset/train/megafallv2/images/train")
NORM_LABEL_ROOT = Path("/Users/jihunjang/Downloads/dataset/train/megafallv2/labels/normalized_train")
LABEL_ROOT = Path("/Users/jihunjang/Downloads/dataset/train/megafallv2/labels/train")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXT = ".txt"


def main() -> None:
    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(f"Image root missing: {IMAGE_ROOT}")
    if not NORM_LABEL_ROOT.exists():
        raise FileNotFoundError(f"Normalized labels missing: {NORM_LABEL_ROOT}")

    removed = 0
    for img_path in IMAGE_ROOT.rglob("*"):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = img_path.relative_to(IMAGE_ROOT)
        label_path = NORM_LABEL_ROOT / rel.with_suffix(LABEL_EXT)
        if not label_path.is_file():
            print(f"[remove] {img_path}")
            img_path.unlink(missing_ok=True)
            # remove corresponding original label if it exists without image
            orig_label = LABEL_ROOT / rel.with_suffix(LABEL_EXT)
            if orig_label.is_file():
                orig_label.unlink(missing_ok=True)
            removed += 1

    print(f"Total images removed: {removed}")


if __name__ == "__main__":
    main()
