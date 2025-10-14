"""Create MegaFall train/val lists excluding invalid labels."""
from __future__ import annotations

from pathlib import Path
import math

TRAIN_LIST_IN = Path("/Users/jihunjang/workspace/ust/human-detection/fine_tuning/v4/train.txt")
VAL_LIST_IN = Path("/Users/jihunjang/workspace/ust/human-detection/fine_tuning/v4/val.txt")
TRAIN_LIST_OUT = Path("/Users/jihunjang/workspace/ust/human-detection/fine_tuning/v4/train2.txt")
VAL_LIST_OUT = Path("/Users/jihunjang/workspace/ust/human-detection/fine_tuning/v4/val2.txt")
LABEL_ROOT = Path("/Users/jihunjang/Downloads/dataset/train/megafallv2/labels")

LABEL_EXT = ".txt"
IMAGE_TOKEN = "images"


def label_path_for_image(img_path: Path) -> Path | None:
    parts = list(img_path.parts)
    if IMAGE_TOKEN not in parts:
        return None
    idx = parts.index(IMAGE_TOKEN)
    parts[idx] = "labels"
    rel = Path(*parts[idx:])
    return LABEL_ROOT / rel.relative_to("labels").with_suffix(LABEL_EXT)


def label_is_valid(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    valid = True
    try:
        with label_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    return False
                try:
                    cls = int(float(parts[0]))
                    values = [float(v) for v in parts[1:5]]
                except ValueError:
                    return False
                for v in values:
                    if not math.isfinite(v):
                        return False
                    if v < 0.0 or v > 1.0:
                        return False
                if values[2] <= 0.0 or values[3] <= 0.0:
                    return False
    except OSError:
        valid = False
    return valid


def filter_list(list_in: Path, list_out: Path) -> int:
    kept: list[str] = []
    with list_in.open("r", encoding="utf-8") as fh:
        for line in fh:
            img_path = Path(line.strip())
            if not img_path:
                continue
            label_path = label_path_for_image(img_path)
            if label_path is None:
                continue
            if label_is_valid(label_path):
                kept.append(str(img_path))
    list_out.parent.mkdir(parents=True, exist_ok=True)
    with list_out.open("w", encoding="utf-8") as out:
        out.write("\n".join(kept) + ("\n" if kept else ""))
    return len(kept)


def main() -> None:
    train_kept = filter_list(TRAIN_LIST_IN, TRAIN_LIST_OUT)
    val_kept = filter_list(VAL_LIST_IN, VAL_LIST_OUT)
    print(f"train images kept: {train_kept}")
    print(f"val images kept: {val_kept}")
    print(f"train list → {TRAIN_LIST_OUT}")
    print(f"val list → {VAL_LIST_OUT}")


if __name__ == "__main__":
    main()
