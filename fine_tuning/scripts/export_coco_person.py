"""Extract COCO train2017 person-like annotations into YOLO format."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


ANN_PATH = Path(
    "/Users/jihunjang/Downloads/dataset/train/coco/annotations_trainval2017/instances_train2017.json"
)
IMAGES_SRC = Path("/Users/jihunjang/Downloads/dataset/train/coco/train2017")
OUTPUT_ROOT = Path("/Users/jihunjang/Downloads/dataset/train/coco/person")
IMAGES_DST = OUTPUT_ROOT / "images"
LABELS_DST = OUTPUT_ROOT / "labels"
TARGET_CATEGORY_ID = 0  # fall back to category name when id not present
TARGET_CATEGORY_NAME = "person"
YOLO_TARGET_CLASS = 1


def ensure_dirs() -> None:
    IMAGES_DST.mkdir(parents=True, exist_ok=True)
    LABELS_DST.mkdir(parents=True, exist_ok=True)


def resolve_target_id(categories: list[dict]) -> int:
    for cat in categories:
        if cat.get("id") == TARGET_CATEGORY_ID:
            print(f"Using target category id={TARGET_CATEGORY_ID}")
            return TARGET_CATEGORY_ID
    for cat in categories:
        if cat.get("name") == TARGET_CATEGORY_NAME:
            resolved = int(cat["id"])
            print(
                f"Category id {TARGET_CATEGORY_ID} not found; using '{TARGET_CATEGORY_NAME}' id={resolved}"
            )
            return resolved
    raise ValueError(
        f"Neither category id {TARGET_CATEGORY_ID} nor name '{TARGET_CATEGORY_NAME}' found"
    )


def load_annotations() -> tuple[dict[int, dict], dict[int, list[list[float]]], int]:
    with ANN_PATH.open("r", encoding="utf-8") as fh:
        ann = json.load(fh)

    categories = ann.get("categories", [])
    target_id = resolve_target_id(categories)

    images_info: dict[int, dict] = {img["id"]: img for img in ann.get("images", [])}

    ann_by_image: dict[int, list[list[float]]] = {}
    for det in ann.get("annotations", []):
        if det.get("iscrowd", 0) != 0:
            continue
        if det.get("category_id") != target_id:
            continue
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        ann_by_image.setdefault(det["image_id"], []).append(bbox)
    return images_info, ann_by_image, target_id


def coco_bbox_to_yolo(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    x, y, w, h = bbox
    if width <= 0 or height <= 0 or w <= 0 or h <= 0:
        return None
    cx = (x + w / 2) / width
    cy = (y + h / 2) / height
    nw = w / width
    nh = h / height
    if nw <= 0 or nh <= 0:
        return None
    return cx, cy, nw, nh


def transfer_image(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    shutil.copy2(src, dst)


def resolve_image_path(image_id: int) -> Path | None:
    # Primary: zero-padded 12-digit filename (standard COCO layout)
    candidate = IMAGES_SRC / f"{image_id:012d}.jpg"
    if candidate.is_file():
        return candidate

    # Fallback: scan for file whose stem matches the id without leading zeros
    stripped = str(image_id)
    for ext in (".jpg", ".jpeg", ".png"):
        alt = IMAGES_SRC / f"{stripped}{ext}"
        if alt.is_file():
            return alt

    # Brute-force search limited to matching stems to avoid full directory walk
    for path in IMAGES_SRC.glob("*.jpg"):
        if path.stem.lstrip("0") == stripped:
            return path
    return None


def export_person_dataset() -> None:
    ensure_dirs()
    images_info, ann_by_image, target_id = load_annotations()

    total = len(ann_by_image)
    if total == 0:
        print(
            f"No annotations found for category id {target_id}; nothing to export."
        )
        return

    processed = 0
    for image_id, bboxes in ann_by_image.items():
        info = images_info.get(image_id)
        if info is None:
            continue

        width = info.get("width")
        height = info.get("height")
        src_path = resolve_image_path(image_id)
        if src_path is None:
            continue

        if width in (None, 0) or height in (None, 0):
            width = info.get("width")
            height = info.get("height")
        if width in (None, 0) or height in (None, 0):
            continue

        yolo_lines: list[str] = []
        for bbox in bboxes:
            converted = coco_bbox_to_yolo(bbox, width, height)
            if converted is None:
                continue
            cx, cy, nw, nh = converted
            yolo_lines.append(f"{YOLO_TARGET_CLASS} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if not yolo_lines:
            continue

        img_dst = IMAGES_DST / src_path.name
        transfer_image(src_path, img_dst)

        label_dst = LABELS_DST / f"{Path(src_path.name).stem}.txt"
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        with label_dst.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(yolo_lines))
            fh.write("\n")

        processed += 1
        if processed % 500 == 0:
            print(f"Exported {processed}/{total} person images")

    print(f"[done] Exported {processed} person images with labels")


def main() -> None:
    if not ANN_PATH.is_file():
        raise FileNotFoundError(f"Annotation file not found: {ANN_PATH}")
    if not IMAGES_SRC.is_dir():
        raise FileNotFoundError(f"COCO train2017 directory not found: {IMAGES_SRC}")
    export_person_dataset()


if __name__ == "__main__":
    main()
