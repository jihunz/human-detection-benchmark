# ==============================================
# Folder → YOLO12n person detection (recursive)
# - No COCO / No XML / No evaluation
# - For each video: save annotated .mp4 + per-frame CSV
# ==============================================

# !pip -q install ultralytics opencv-python pillow

import os, csv, cv2, time
from pathlib import Path
from ultralytics import YOLO

# ---------------------------
# Config
# ---------------------------
CONF_THRES = 0.25
NMS_IOU   = 0.70
IMGSZ     = 640
CLASSES   = [0]    # COCO class 0 == person
VID_EXTS  = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

INPUT_VIDEO_DIR = "/content/falldown_cut"  # 폴더 경로
OUTPUT_DIR      = "/content/outputs"   # 결과 저장 폴더

# ---------------------------
# Utils
# ---------------------------
def ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def list_videos(root: str | Path):
    """폴더를 재귀적으로 검색하여 지원 확장자의 비디오를 전부 반환."""
    p = Path(root)
    files = []
    if p.is_file() and p.suffix.lower() in VID_EXTS:
        return [p]
    for ext in VID_EXTS:
        files.extend(p.rglob(f"*{ext}"))
    return sorted(files)

def make_out_paths(in_path: Path, out_root: Path):
    out_root = ensure_dir(out_root)
    stem = in_path.stem
    return out_root / f"{stem}_yolo12n.mp4", out_root / f"{stem}_yolo12n.csv"

# ---------------------------
# Inference core
# ---------------------------
def run_yolo12n_on_video(
    model: YOLO,
    in_path: str | Path,
    out_video_path: str | Path,
    out_csv_path: str | Path,
    conf: float = CONF_THRES,
    iou: float = NMS_IOU,
    imgsz: int = IMGSZ,
    classes = CLASSES,
    show_progress: bool = True,
):
    in_path = Path(in_path)
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {in_path}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (w, h))

    csv_file = open(out_csv_path, "w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_file)
    csv_w.writerow(["frame_idx", "time_sec", "x1", "y1", "x2", "y2", "conf"])

    frame_idx = -1
    t0 = time.time()
    for res in model.predict(
        source=str(in_path),
        stream=True,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        classes=classes,
        agnostic_nms=False,
        verbose=False,
    ):
        frame_idx += 1
        annotated = res.plot()  # np.ndarray (BGR)

        if getattr(res, "boxes", None):
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            t_sec = frame_idx / fps
            for (x1, y1, x2, y2), sc in zip(xyxy, confs):
                csv_w.writerow([frame_idx, f"{t_sec:.3f}",
                                f"{x1:.2f}", f"{y1:.2f}",
                                f"{x2:.2f}", f"{y2:.2f}",
                                f"{sc:.4f}"])
        writer.write(annotated)

        if show_progress and total > 0 and frame_idx % 50 == 0:
            pct = 100.0 * frame_idx / max(1, total)
            print(f"[{in_path.name}] {frame_idx}/{total} frames ({pct:.1f}%)")

    writer.release()
    csv_file.close()
    cap.release()

    dt = time.time() - t0
    print(f"[Done] {in_path.name}")
    print(f"  Video saved: {out_video_path}")
    print(f"  CSV saved  : {out_csv_path}")
    print(f"  Elapsed    : {dt:.1f}s")

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # YOLO12n 모델 로드 (허브 자동 다운로드)
    model = YOLO("yolo12n.pt")

    # 폴더 내 모든 비디오 수집 (재귀)
    videos = list_videos(INPUT_VIDEO_DIR)
    if not videos:
        raise FileNotFoundError(f"No videos found under: {INPUT_VIDEO_DIR}")

    print(f"Found {len(videos)} video(s) under {INPUT_VIDEO_DIR}")
    for vid in videos:
        try:
            out_mp4, out_csv = make_out_paths(Path(vid), Path(OUTPUT_DIR))
            run_yolo12n_on_video(model, vid, out_mp4, out_csv)
        except Exception as e:
            print(f"[ERROR] {vid}: {e}")