"""Select highest-scoring senior-behaviour frames and export YOLO labels."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from PIL import Image


IMAGES_ROOT = Path("/Users/jihunjang/Downloads/시니어 이상행동 영상/images")
LABELS_ROOT = Path("/Users/jihunjang/Downloads/시니어 이상행동 영상/labels")
OUTPUT_ROOT = Path("/Users/jihunjang/Downloads/시니어 이상행동 영상/exports")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
YOLO_CLASS_ID = 1


@dataclass
class SelectedSample:
    image_path: Path
    label_path: Path
    rel_dir: Path


def ensure_output_dirs() -> None:
    (OUTPUT_ROOT / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "labels").mkdir(parents=True, exist_ok=True)


def _flat_stem(sample: SelectedSample) -> str:
    parts = list(sample.rel_dir.parts)
    parts.append(sample.label_path.stem)
    return "__".join(parts)


def _extract_annotation_blocks(data: dict) -> List[dict]:
    content = data.get("content")
    if not isinstance(content, dict):
        return []
    obj = content.get("object")
    if obj is None:
        return []

    annotations: List[dict] = []
    candidates: Sequence
    if isinstance(obj, list):
        candidates = [item.get("annotation") for item in obj if isinstance(item, dict)]
    elif isinstance(obj, dict):
        candidates = [obj.get("annotation")]
    else:
        candidates = []

    for ann in candidates:
        if isinstance(ann, dict):
            annotations.append(ann)
        elif isinstance(ann, list):
            annotations.extend([a for a in ann if isinstance(a, dict)])
    return annotations


def read_score(label_path: Path) -> float:
    try:
        with label_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError:
        return float("-inf")

    best = float("-inf")
    for ann in _extract_annotation_blocks(data):
        scores = ann.get("scores")
        if not isinstance(scores, list):
            continue
        for val in scores:
            try:
                best = max(best, float(val))
            except (TypeError, ValueError):
                continue
    return best


def select_best_label(label_dir: Path) -> Optional[Path]:
    candidates = [p for p in label_dir.iterdir() if p.suffix.lower() == ".json"]
    if not candidates:
        return None
    scored = [(read_score(p), p) for p in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_path = scored[0]
    if best_score == float("-inf"):
        return None
    return best_path


def find_image_for_label(label_path: Path) -> Optional[Path]:
    rel_dir = label_path.parent.relative_to(LABELS_ROOT)
    stem = label_path.stem
    image_dir = IMAGES_ROOT / rel_dir
    if not image_dir.exists():
        return None
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    # fallback: case-insensitive search
    lower_stem = stem.lower()
    for file in image_dir.iterdir():
        if file.suffix.lower() in IMAGE_EXTENSIONS and file.stem.lower() == lower_stem:
            return file
    return None


def convert_json_to_yolo(label_path: Path, image_path: Path) -> List[str]:
    with label_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    try:
        with Image.open(image_path) as im:
            width, height = im.size
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to open image {image_path}: {exc}")

    def normalise_bbox(bbox: Iterable[float], cls: Optional[int]) -> Optional[str]:
        try:
            x1, y1, x2, y2 = map(float, bbox)
        except (TypeError, ValueError):
            return None

        if width <= 0 or height <= 0:
            return None

        # Detect whether bbox values are normalised
        normalised = max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5
        if not normalised:
            # Interpret as absolute pixel coordinates (x1, y1, x2, y2)
            cx = ((x1 + x2) / 2.0) / width
            cy = ((y1 + y2) / 2.0) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
        else:
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                # Assume (cx, cy, w, h) ordering
                cx = x1
                cy = y1
                bw = x2
                bh = y2
            else:
                cx = x1 + bw / 2.0
                cy = y1 + bh / 2.0

        if bw <= 0 or bh <= 0:
            return None

        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        bw = min(max(bw, 0.0), 1.0)
        bh = min(max(bh, 0.0), 1.0)

        class_id = YOLO_CLASS_ID if cls is None else cls
        return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    entries: List[str] = []
    for ann in _extract_annotation_blocks(data):
        raw_boxes = ann.get("bboxes") or ann.get("bbox")
        if not isinstance(raw_boxes, list) or not raw_boxes:
            continue
        if isinstance(raw_boxes[0], (list, tuple)):
            bbox_list = [list(b) for b in raw_boxes]
        elif len(raw_boxes) == 4 and all(isinstance(v, (int, float)) for v in raw_boxes):
            bbox_list = [raw_boxes]
        else:
            continue

        scores = ann.get("scores")
        best_index = 0
        if isinstance(scores, list) and scores:
            try:
                if len(scores) == len(bbox_list):
                    best_index = max(range(len(scores)), key=lambda i: float(scores[i]))
                elif len(scores) == 1 and len(bbox_list[0]) == len(scores):
                    best_index = 0
                else:
                    best_index = max(range(len(scores)), key=lambda i: float(scores[i]))
            except (ValueError, TypeError):
                best_index = 0

        ids = ann.get("ids")
        if isinstance(ids, list) and len(ids) != len(bbox_list):
            ids = None

        if best_index >= len(bbox_list):
            best_index = len(bbox_list) - 1

        cls = None
        if isinstance(ids, list):
            raw_cls = ids[best_index]
            if isinstance(raw_cls, (int, float)):
                cls = int(raw_cls)
            elif isinstance(raw_cls, str) and raw_cls.isdigit():
                cls = int(raw_cls)

        line = normalise_bbox(bbox_list[best_index], cls)
        if line:
            entries.append(line)
    return entries


def process_folder(label_dir: Path) -> Optional[SelectedSample]:
    best_label = select_best_label(label_dir)
    if best_label is None:
        return None
    image_path = find_image_for_label(best_label)
    if image_path is None:
        return None
    rel_dir = best_label.parent.relative_to(LABELS_ROOT)
    return SelectedSample(image_path=image_path, label_path=best_label, rel_dir=rel_dir)


def export_sample(sample: SelectedSample) -> None:
    yolo_lines = convert_json_to_yolo(sample.label_path, sample.image_path)
    if not yolo_lines:
        print(f"[skip] {sample.label_path}: no bbox entries")
        return

    flat_stem = _flat_stem(sample)
    dst_image = OUTPUT_ROOT / "images" / f"{flat_stem}{sample.image_path.suffix}"
    dst_image.write_bytes(sample.image_path.read_bytes())

    dst_label = OUTPUT_ROOT / "labels" / f"{flat_stem}.txt"
    dst_label.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
    print(f"[export] {sample.label_path}" )


def main() -> None:
    ensure_output_dirs()
    processed = 0
    skipped = 0
    for label_dir in sorted(p for p in LABELS_ROOT.rglob("*") if p.is_dir()):
        sample = process_folder(label_dir)
        if sample is None:
            continue
        try:
            export_sample(sample)
            processed += 1
        except Exception as exc:
            print(f"[error] {sample.label_path}: {exc}")
            skipped += 1
    print(f"[done] exported={processed}, errors={skipped}")


if __name__ == "__main__":
    main()
