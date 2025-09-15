"""
AI-Hub Indoor Abnormal Behavior dataset helper
-------------------------------------------------
비-CLI 방식: 파일 상단의 [User Config] 값을 수정한 뒤,
그냥 실행하면 됩니다. 영상-XML 1쌍 또는 디렉토리(배치) 모두 지원.

기능
- 각 영상당 fall_end 프레임 1장을 추출하여 이미지(.jpg)와 YOLO 라벨(.txt) 저장
- XML의 <track label="fall_end"> 내 <box frame xtl ytl xbr ybr>를 사용
- keyframe=1을 우선, 없으면 가장 뒤 프레임 선택, outside=1은 제외

출력 구조
- out_root/
  - images/{stem}_frame{N}.jpg
  - labels/{stem}_frame{N}.txt

YOLO 포맷
- class_id cx cy w h  (이미지 크기로 정규화)

의존성: opencv-python
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Optional, Tuple, List, Dict

import cv2


# -------------------------
# User Config (edit below)
# -------------------------
# 모드 선택:
#   - 단일 파일 모드: VIDEO_PATH, XML_PATH 를 채우면 단일 모드로 실행
#   - 배치 모드: 위 2개를 비워두고, VIDEO_ROOT, XML_ROOT 를 채우면 배치 실행

# 단일 파일 모드 (예시 경로를 주석으로 남깁니다)
VIDEO_PATH = ''  # 예: "/path/to/C_3_7_1_BU_DYA_07-31_15-15-25_CA_RGB_DF2_M1.mp4"
XML_PATH   = ''  # 예: "/Users/USER/Downloads/.../C_3_7_1_BU_DYA_07-31_15-15-25_CA_RGB_DF2_M1.xml"
DEFAULT_CLASS_ID = 0
# 배치 모드 (재귀 탐색, 파일명 stem 매칭)
VIDEO_ROOT = '/Users/jihunjang/Downloads/실내 사람 이상행동 데이터/01-1.정식개방데이터/Training/01.원천데이터/TS_03.이상행동_07.전도'  # 예: "/path/to/videos"
XML_ROOT   = '/Users/jihunjang/Downloads/실내 사람 이상행동 데이터/01-1.정식개방데이터/Training/02.라벨링데이터/TL_03.이상행동_07.전도'  # 예: "/path/to/xmls"

# 출력 루트
OUT_DIR    = "fine_tuning_v2/_fall_frame_dataset"

# 라벨 클래스 id (기본 0=person)
CLASS_ID   = 0

# 저장 이미지에 박스 시각화 여부
VISUALIZE  = False

# 지원 비디오 확장자
SUPPORTED_VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
TARGET_SIZE = 640  # resize to fixed 640x640 (no letterbox)


@dataclass
class FallEndBox:
    frame: int
    xtl: float
    ytl: float
    xbr: float
    ybr: float

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        return self.xtl, self.ytl, self.xbr, self.ybr


def parse_fall_end(xml_path: Path) -> FallEndBox:
    """
    Parse XML to find the fall_end frame and bbox.

    Expected (CVAT-like) structure:
      <annotations>
        <track id="1" label="fall_end" source="manual">
           <box frame="123" xtl="..." ytl="..." xbr="..." ybr="..." .../>
           ...
        </track>
      </annotations>

    Selection strategy:
      - Find track[label='fall_end'] (case-insensitive)
      - Collect <box> with attributes xtl,ytl,xbr,ybr
      - Prefer boxes with outside != '1' if present
      - If multiple remain, prefer keyframe='1'
      - Else choose the one with the largest frame index (end-most)
    """
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception as e:
        raise RuntimeError(f"XML parse failed: {xml_path}\n{e}")

    def is_fall_end_track(el: ET.Element) -> bool:
        label = (el.attrib.get("label") or "").strip().lower()
        return label == "fall_end"

    tracks = [t for t in root.iter("track") if is_fall_end_track(t)]
    if not tracks:
        # Some datasets may use <image> annotations (not tracks). Provide a helpful error.
        raise ValueError(f"No <track label='fall_end'> found in: {xml_path}")

    # Heuristics: If there are multiple fall_end tracks, use the first one that has a valid box.
    candidates: List[FallEndBox] = []
    keyframe_boxes: List[FallEndBox] = []

    for tr in tracks:
        for box in tr.iter("box"):
            try:
                xtl = float(box.attrib["xtl"])  # raises if missing
                ytl = float(box.attrib["ytl"])  # raises if missing
                xbr = float(box.attrib["xbr"])  # raises if missing
                ybr = float(box.attrib["ybr"])  # raises if missing
                frame = int(box.attrib.get("frame", "0"))
            except Exception:
                continue

            outside = box.attrib.get("outside")
            if outside == "1":
                # outside=1 means outside of frame (CVAT), skip
                continue

            feb = FallEndBox(frame=frame, xtl=xtl, ytl=ytl, xbr=xbr, ybr=ybr)
            candidates.append(feb)

            if box.attrib.get("keyframe") == "1":
                keyframe_boxes.append(feb)

    if not candidates:
        raise ValueError(f"No usable <box> under <track label='fall_end'> in: {xml_path}")

    # Prefer keyframe if available, else choose the box with the largest frame index (end-most)
    selected = None
    if keyframe_boxes:
        selected = sorted(keyframe_boxes, key=lambda b: b.frame)[-1]
    else:
        selected = sorted(candidates, key=lambda b: b.frame)[-1]

    return selected


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Tuple[float, float, float, float]:
    # Clamp and convert to center-based normalized coords
    x1 = max(0.0, min(x1, w - 1))
    y1 = max(0.0, min(y1, h - 1))
    x2 = max(0.0, min(x2, w - 1))
    y2 = max(0.0, min(y2, h - 1))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size: {w}x{h}")
    return cx / w, cy / h, bw / w, bh / h


def get_video_size(cap: cv2.VideoCapture) -> Tuple[int, int]:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return width, height


def read_frame_at(video_path: Path, frame_idx: int) -> Tuple[Optional[any], Tuple[int, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, (0, 0)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or -1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, frame = cap.read()
    if not ok and 0 <= frame_idx < total and total > 0:
        # fallback: try sequential read
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
        idx = -1
        while True:
            ok, f = cap.read()
            if not ok:
                break
            idx += 1
            if idx == frame_idx:
                frame = f
                break
    w, h = get_video_size(cap)
    cap.release()
    return frame, (w, h)


def resize_to_square(image, size: int = TARGET_SIZE):
    if image is None:
        return None
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def save_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # BGR → fixed 640x640
    img = resize_to_square(image, TARGET_SIZE)
    cv2.imwrite(str(path), img)


def save_yolo_label(path: Path, class_id: int, cx: float, cy: float, bw: float, bh: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def find_matching_video(xml_path: Path, video_root: Path) -> Optional[Path]:
    stem = xml_path.stem
    # Try exact stem match recursively
    for p in video_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_VID_EXTS and p.stem == stem:
            return p
    return None


def process_pair(xml_path: Path, video_path: Path, out_root: Path, class_id: int = DEFAULT_CLASS_ID, visualize: bool = False) -> Optional[Tuple[Path, Path]]:
    feb = parse_fall_end(xml_path)
    frame, (w, h) = read_frame_at(video_path, feb.frame)
    if frame is None:
        print(f"[skip] Cannot read frame {feb.frame} from {video_path.name}")
        return None
    if w <= 0 or h <= 0:
        h, w = frame.shape[:2]

    cx, cy, bw, bh = xyxy_to_yolo(feb.xtl, feb.ytl, feb.xbr, feb.ybr, w, h)

    stem = xml_path.stem  # base for naming
    img_name = f"{stem}_frame{feb.frame}.jpg"
    lbl_name = f"{stem}_frame{feb.frame}.txt"
    img_out = out_root / "images" / img_name
    lbl_out = out_root / "labels" / lbl_name

    if visualize:
        # draw rectangle for sanity
        x1, y1, x2, y2 = int(feb.xtl), int(feb.ytl), int(feb.xbr), int(feb.ybr)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    save_image(img_out, frame)
    save_yolo_label(lbl_out, class_id, cx, cy, bw, bh)

    print(f"[ok] {video_path.name} -> {img_out.name}, {lbl_out.name} (frame={feb.frame})")
    return img_out, lbl_out


def collect_xmls(xml_root: Path) -> List[Path]:
    return [p for p in xml_root.rglob("*.xml") if p.is_file()]


def is_video_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in SUPPORTED_VID_EXTS


def run_with_config() -> int:
    out_root = Path(OUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    # If single pair is provided, run single mode
    if VIDEO_PATH.strip() and XML_PATH.strip():
        single_video = Path(VIDEO_PATH)
        single_xml = Path(XML_PATH)
        if not is_video_file(single_video):
            print(f"[error] Not a supported video file: {single_video}")
            return 2
        if not single_xml.is_file():
            print(f"[error] XML not found: {single_xml}")
            return 2
        process_pair(single_xml, single_video, out_root, class_id=CLASS_ID, visualize=VISUALIZE)
        print("\n[done] processed: 1")
        return 0

    # Else, try batch mode using roots
    if not (VIDEO_ROOT.strip() and XML_ROOT.strip()):
        print("[error] 설정이 비어있습니다. 단일 모드는 VIDEO_PATH/XML_PATH, 배치 모드는 VIDEO_ROOT/XML_ROOT 를 채워주세요.")
        return 2

    video_root = Path(VIDEO_ROOT)
    xml_root = Path(XML_ROOT)
    if not video_root.is_dir() or not xml_root.is_dir():
        print(f"[error] Invalid roots. video_root={video_root}, xml_root={xml_root}")
        return 2

    xmls = collect_xmls(xml_root)
    if not xmls:
        print(f"[error] No XMLs found under: {xml_root}")
        return 2

    processed = 0
    for xml_path in xmls:
        vid_path = find_matching_video(xml_path, video_root)
        if not vid_path:
            print(f"[skip] Matching video not found for: {xml_path.name}")
            continue
        try:
            process_pair(xml_path, vid_path, out_root, class_id=CLASS_ID, visualize=VISUALIZE)
            processed += 1
        except Exception as e:
            print(f"[error] {xml_path.name}: {e}")

    print(f"\n[done] processed: {processed} (out: {out_root.resolve()})")
    return 0


if __name__ == "__main__":
    sys.exit(run_with_config())
