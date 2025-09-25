"""Extract fall_end frames and YOLO labels from AI Hub dataset using OpenCV sequential reads."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2

VIDEO_ROOT = Path(
    "/Users/jihunjang/Downloads/dataset/train/실내 사람 이상행동 데이터/01-1.정식개방데이터/Training/01.원천데이터"
)
LABEL_ROOT = Path(
    "/Users/jihunjang/Downloads/dataset/train/실내 사람 이상행동 데이터/01-1.정식개방데이터/Training/02.라벨링데이터"
)
OUTPUT_ROOT = Path("/Users/jihunjang/Downloads/ai_hub")
OUTPUT_IMAGES = OUTPUT_ROOT / "images"
OUTPUT_LABELS = OUTPUT_ROOT / "labels"
FALL_CLASS_ID = 80
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


Box = Tuple[float, float, float, float]


def ensure_dirs() -> None:
    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)


def load_fall_end_boxes(xml_path: Path) -> Dict[int, List[Box]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes_by_frame: Dict[int, List[Box]] = defaultdict(list)

    for track in root.findall(".//track[@label='fall_end']"):
        for box in track.findall("box"):
            frame = int(box.attrib.get("frame", 0))
            xtl = float(box.attrib.get("xtl", 0.0))
            ytl = float(box.attrib.get("ytl", 0.0))
            xbr = float(box.attrib.get("xbr", 0.0))
            ybr = float(box.attrib.get("ybr", 0.0))
            boxes_by_frame[frame].append((xtl, ytl, xbr, ybr))
    return boxes_by_frame


def build_video_map(video_root: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for video in video_root.iterdir():
        if video.suffix.lower() in VIDEO_EXTS:
            mapping[video.stem] = video
    return mapping


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def convert_to_yolo(box: Box, width: int, height: int) -> Tuple[float, float, float, float]:
    xtl, ytl, xbr, ybr = box
    xtl = clamp(xtl, 0.0, width)
    xbr = clamp(xbr, 0.0, width)
    ytl = clamp(ytl, 0.0, height)
    ybr = clamp(ybr, 0.0, height)
    bw = max(0.0, xbr - xtl)
    bh = max(0.0, ybr - ytl)
    if bw <= 0 or bh <= 0:
        return (0.5, 0.5, 0.0, 0.0)
    cx = xtl + bw / 2
    cy = ytl + bh / 2
    return (cx / width, cy / height, bw / width, bh / height)


def write_label(label_path: Path, entries: List[Tuple[float, float, float, float]]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as f:
        for cx, cy, bw, bh in entries:
            f.write(f"{FALL_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def grab_frame(video_path: Path, frame_idx: int) -> Tuple[bool, Any, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, None, frame_idx

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_idx = frame_idx
    if total_frames > 0:
        target_idx = max(0, min(frame_idx, total_frames - 1))

    if target_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    success, frame = cap.read()

    if not success or frame is None:
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, None, target_idx
        success = False
        for _ in range(target_idx + 1):
            success, frame = cap.read()
            if not success or frame is None:
                break
        if not success or frame is None:
            cap.release()
            return False, None, target_idx

    cap.release()
    return True, frame, target_idx


def process_video(video_path: Path, frame_idx: int, boxes: List[Box], stem: str) -> Tuple[int, int]:
    success, frame, actual_idx = grab_frame(video_path, frame_idx)
    if not success or frame is None:
        print(f"[warn] Could not read frame {frame_idx} from {video_path.name}")
        return (0, 1)
    if actual_idx != frame_idx:
        print(
            f"[info] Adjusted frame index {frame_idx} -> {actual_idx} for {video_path.name}"
        )

    height, width = frame.shape[:2]
    yolo_boxes = [convert_to_yolo(box, width, height) for box in boxes]
    image_name = f"{stem}_frame_{actual_idx:05d}.jpg"
    label_name = image_name.replace(".jpg", ".txt")
    image_path = OUTPUT_IMAGES / image_name
    label_path = OUTPUT_LABELS / label_name
    cv2.imwrite(str(image_path), frame)
    write_label(label_path, yolo_boxes)
    return (1, 0)


def process_xml(xml_path: Path, video_map: Dict[str, Path]) -> Tuple[int, int]:
    boxes_by_frame = load_fall_end_boxes(xml_path)
    if not boxes_by_frame:
        return (0, 0)

    stem = xml_path.stem
    video_path = video_map.get(stem)
    if video_path is None:
        print(f"[skip] Video not found for XML: {xml_path.name}")
        return (0, len(boxes_by_frame))

    target_frame = max(boxes_by_frame.keys())
    boxes = boxes_by_frame[target_frame]
    return process_video(video_path, target_frame, boxes, stem)


def main() -> None:
    ensure_dirs()
    video_map = build_video_map(VIDEO_ROOT)
    xml_files = sorted(LABEL_ROOT.glob("*.xml"))
    if not xml_files:
        print(f"No XML files found in {LABEL_ROOT}")
        return

    total_saved = 0
    total_skipped = 0
    for idx, xml_path in enumerate(xml_files, 1):
        print(f"[progress] ({idx}/{len(xml_files)}) {xml_path.name}")
        saved, skipped = process_xml(xml_path, video_map)
        total_saved += saved
        total_skipped += skipped

    print(
        f"\n[done] Saved {total_saved} frame(s). Skipped {total_skipped} frame(s) due to read errors."
    )


if __name__ == "__main__":
    main()
