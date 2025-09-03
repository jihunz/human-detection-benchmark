# ================================================================
# Recall-only evaluation for KISA-style clips (Start-5s pre-trimmed)
# - 평가 구간: [clip 5s, clip 5s + duration]
# - 프레임 TP: person max_conf >= 0.90
# - 프레임 FN: 구간 내인데 max_conf < 0.90 (또는 미검출)
# - 이벤트 구간 밖은 평가대상 제외 (precision/FP 미계산)
# - 모델: YOLO12n, YOLO11n (Ultralytics), RT-DETRv2, D-FINE (HF)
# - 결과: metrics.csv / per_event.csv + 간단 시각화
# - 사람 라벨은 각 모델의 라벨 이름에서 동적으로 탐색 (COCO index 가정 X)
# ================================================================

# !pip -q install ultralytics transformers torch pillow opencv-python matplotlib lxml

import os, time, csv, math
from pathlib import Path
from typing import List, Tuple, Optional
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------
# Config
# ---------------------------
VIDEO_ROOT  = "/content/falldown_cut"   # 영상/GT(XML) 폴더 (파일명 통일 가정)
OUTPUT_ROOT = "/content/outputs_eval"   # 결과 저장 폴더
os.makedirs(OUTPUT_ROOT, exist_ok=True)

TARGET_EVENTS     = {"Falldown"}  # 평가 대상 이벤트 (None이면 XML의 모든 이벤트 사용)
CONF_HARD_TH      = 0.90          # TP 프레임으로 인정할 최소 confidence
CLIP_PRE_OFFSET_S = 5.0           # 클립이 Start-5s부터 시작 → 평가 시작은 5.0s
VID_EXTS          = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# 사람 라벨 이름 후보(소문자 비교)
PERSON_ALIASES = {"person", "human", "pedestrian", "people"}

# ---------------------------
# 파일 유틸
# ---------------------------
def list_videos(root: str | Path) -> List[Path]:
    """재귀 검색 + 숨김/리소스포크(. , ._) + 극소 용량 파일 제외"""
    root = Path(root)
    vids = []
    for ext in VID_EXTS:
        vids.extend(root.rglob(f"*{ext}"))
    out = []
    for p in sorted({Path(x) for x in vids}):
        nm = p.name
        if nm.startswith(".") or nm.startswith("._"):
            continue
        try:
            if p.stat().st_size < 100 * 1024:
                continue
        except Exception:
            continue
        out.append(p)
    return out

# ---------------------------
# XML (KISA GT) 파서
# ---------------------------
def parse_time_hms(s: str) -> float:
    hh, mm, ss = s.strip().split(":")
    return int(hh)*3600 + int(mm)*60 + int(ss)

def parse_kisa_xml(xml_path: Path, target_events: Optional[set]) -> List[dict]:
    """
    반환: [{"event": str, "start": float, "duration": float}, ...]
    - KISA 형식: <Alarms><Alarm><StartTime><AlarmDuration><AlarmDescription>
    - 평가는 [5s, 5s+duration] (클립 좌표)로 진행하므로 start는 로깅용
    """
    out = []
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    for a in root.findall(".//Alarms/Alarm"):
        ev  = (a.findtext("AlarmDescription") or "").strip()
        if target_events and ev not in target_events:
            continue
        st  = parse_time_hms(a.findtext("StartTime") or "00:00:00")
        dur = parse_time_hms(a.findtext("AlarmDuration") or "00:00:00")
        out.append({"event": ev, "start": float(st), "duration": float(dur)})
    return out

# ---------------------------
# Detector 공통 (사람 라벨 자동 감지)
# ---------------------------
def _norm(s: str) -> str:
    return (s or "").strip().lower()

class Detector:
    def __init__(self, name: str):
        self.name = name
    def infer_person(self, frame_bgr: np.ndarray) -> Tuple[bool, dict]:
        """반환: (has_person, {'ms': float, 'n': int, 'max_conf': float})"""
        raise NotImplementedError

# --- Ultralytics(YOLO) ---
class UltralyticsYOLO(Detector):
    def __init__(self, name: str, weights: str):
        super().__init__(name)
        from ultralytics import YOLO
        self.model = YOLO(weights)
        # model.names: {idx: class_name}
        self.person_class_ids = [i for i, n in self.model.names.items() if _norm(n) in PERSON_ALIASES]
        self.use_classes = self.person_class_ids if self.person_class_ids else None

    def infer_person(self, frame_bgr: np.ndarray) -> Tuple[bool, dict]:
        t0 = time.time()
        r = self.model.predict(
            source=frame_bgr, conf=0.25, iou=0.70, imgsz=640,
            classes=self.use_classes, verbose=False
        )[0]
        ms = (time.time() - t0) * 1000.0
        n  = int(len(r.boxes) if getattr(r, "boxes", None) is not None else 0)

        # classes를 못썼다면 결과에서 라벨 이름으로 2차 필터
        max_conf = 0.0
        if n > 0:
            if self.use_classes is None:
                kept = []
                cls = r.boxes.cls.cpu().numpy().astype(int)
                conf = r.boxes.conf.cpu().numpy().astype(float)
                for ci, sc in zip(cls, conf):
                    if _norm(self.model.names.get(int(ci), "")) in PERSON_ALIASES:
                        kept.append(sc)
                n = len(kept)
                max_conf = float(max(kept)) if kept else 0.0
            else:
                max_conf = float(r.boxes.conf.max().item())
        return (n > 0), {"ms": ms, "n": n, "max_conf": max_conf}

# --- HF(Transformers) ---
class HFDetector(Detector):
    def __init__(self, name: str, model_id: str):
        super().__init__(name)
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForObjectDetection.from_pretrained(model_id, trust_remote_code=True).to(self.device).eval()
        id2label = getattr(self.model.config, "id2label", {}) or {}
        self.person_label_ids = {int(i) for i, n in id2label.items() if _norm(n) in PERSON_ALIASES}

    def infer_person(self, frame_bgr: np.ndarray) -> Tuple[bool, dict]:
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)

        t0 = time.time()
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        ms = (time.time() - t0) * 1000.0

        target_sizes = self.torch.tensor([[pil.size[1], pil.size[0]]], device=self.device)
        det = self.processor.post_process_object_detection(outputs, threshold=0.25, target_sizes=target_sizes)[0]
        labels = det.get("labels"); scores = det.get("scores")
        if labels is None or scores is None:
            return False, {"ms": ms, "n": 0, "max_conf": 0.0}

        labels = labels.detach().cpu().numpy().astype(int)
        scores = scores.detach().cpu().numpy().astype(float)

        keep = [sc for lb, sc in zip(labels, scores) if int(lb) in self.person_label_ids] if self.person_label_ids else []
        n = len(keep); max_conf = float(max(keep)) if keep else 0.0
        return (n > 0), {"ms": ms, "n": n, "max_conf": max_conf}

# ---------------------------
# 평가 함수 (Recall-only)
# ---------------------------
def evaluate_events_for_video(detector: Detector, video_path: Path, events: List[dict]):
    """
    정책:
      - 평가 구간: [clip 5초, clip 5초 + duration]
      - 프레임 TP : max_conf >= 0.90
      - 프레임 FN : 구간 내인데 max_conf < 0.90
      - 구간 밖은 평가 제외 (precision/FP 계산 X)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # 이벤트 구간 생성
    intervals = []
    for e in events:
        dur = max(0.0, float(e["duration"]))
        s   = CLIP_PRE_OFFSET_S
        t   = CLIP_PRE_OFFSET_S + dur
        s = max(0.0, min(s, N / fps))
        t = max(0.0, min(t, N / fps))
        if t <= s:
            continue
        s_idx = int(math.floor(s * fps))
        t_idx = int(math.ceil (t * fps))
        intervals.append((s_idx, t_idx, s, t, dur))

    # 프레임별 max_conf 수집
    maxconfs = np.zeros(N, dtype=np.float32)
    per_frame_ms = []
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        _, dbg = detector.infer_person(frame)
        maxconfs[idx] = float(dbg.get("max_conf", 0.0))
        per_frame_ms.append(dbg["ms"])
    cap.release()

    # 이벤트별 집계
    per_event = []
    TP_total = FN_total = 0
    latencies = []

    for (s_idx, t_idx, s, t, dur) in intervals:
        seg = maxconfs[s_idx:t_idx] if t_idx > s_idx else np.array([], dtype=np.float32)
        frames_pos = int(len(seg))
        if frames_pos == 0:
            continue

        tp_mask   = (seg >= CONF_HARD_TH)
        tp_frames = int(tp_mask.sum())
        fn_frames = frames_pos - tp_frames
        recall    = tp_frames / frames_pos

        if tp_frames > 0:
            first_tp  = int(np.argmax(tp_mask))
            latency_s = first_tp / fps
            latencies.append(latency_s)
        else:
            latency_s = float('inf')

        TP_total += tp_frames
        FN_total += fn_frames

        per_event.append({
            "video": video_path.name, "event": "Falldown",
            "start_s": round(s, 3), "end_s": round(t, 3), "duration_s": round(dur, 3),
            "frames_pos": frames_pos, "tp_frames": tp_frames, "fn_frames": fn_frames,
            "recall": round(recall, 4),
            "latency_s": None if not np.isfinite(latency_s) else round(latency_s, 3),
        })

    recall_all   = TP_total / max(1, (TP_total + FN_total))
    avg_ms       = float(np.mean(per_frame_ms)) if per_frame_ms else 0.0
    med_ms       = float(np.median(per_frame_ms)) if per_frame_ms else 0.0
    total_time_s = sum(per_frame_ms) / 1000.0
    latency_avg  = float(np.mean(latencies)) if latencies else None
    latency_med  = float(np.median(latencies)) if latencies else None

    summary = {
        "TP_frames": TP_total, "FN_frames": FN_total, "recall": recall_all,
        "latency_avg_s": None if latency_avg is None else round(latency_avg, 3),
        "latency_med_s": None if latency_med is None else round(latency_med, 3),
        "total_time_s": total_time_s, "avg_ms_per_frame": avg_ms, "median_ms_per_frame": med_ms,
        "pixels": W*H, "fps_nominal": fps
    }
    return summary, per_event

# ---------------------------
# 실행 & 집계
# ---------------------------
def run_all():
    videos = list_videos(VIDEO_ROOT)
    if not videos:
        raise FileNotFoundError(f"No videos under: {VIDEO_ROOT}")

    models: List[Detector] = [
        UltralyticsYOLO("YOLO12n", "yolo12n.pt"),
        UltralyticsYOLO("YOLO11n", "yolo11n.pt"),
        HFDetector("RT-DETRv2 (HF)", "PekingU/rtdetr_v2_r18vd"),
        HFDetector("D-FINE (HF)",    "ustc-community/dfine_x_coco"),
    ]

    metrics_rows, per_event_rows = [], []

    for det in models:
        print(f"\n=== [{det.name}] ===")
        for vp in videos:
            # 통일된 파일명 가정: video.ext ↔ video.xml
            xml_path = vp.with_suffix(".xml")
            if not xml_path.exists():
                print(f"  - Skip (no XML): {vp.name}")
                continue
            try:
                events = parse_kisa_xml(xml_path, TARGET_EVENTS)
            except Exception as e:
                print(f"  - Skip (XML parse error): {xml_path.name} ({e})")
                continue
            if not events:
                print(f"  - Skip (no target events): {vp.name}")
                continue

            print(f"  - Eval: {vp.name}  (#events={len(events)})")
            summary, per_events = evaluate_events_for_video(det, vp, events)

            metrics_rows.append([
                det.name, vp.name,
                summary["TP_frames"], summary["FN_frames"],
                f"{summary['recall']:.4f}",
                summary["latency_avg_s"] if summary["latency_avg_s"] is not None else "",
                summary["latency_med_s"] if summary["latency_med_s"] is not None else "",
                f"{summary['total_time_s']:.2f}",
                f"{summary['avg_ms_per_frame']:.2f}",
                f"{summary['median_ms_per_frame']:.2f}",
                int(summary["pixels"]), f"{summary['fps_nominal']:.2f}",
            ])

            for row in per_events:
                per_event_rows.append([
                    det.name, row["video"], row["event"],
                    f"{row['start_s']:.3f}", f"{row['end_s']:.3f}", f"{row['duration_s']:.3f}",
                    row["frames_pos"], row["tp_frames"], row["fn_frames"],
                    f"{row['recall']:.4f}",
                    "" if row["latency_s"] is None else f"{row['latency_s']:.3f}",
                ])

    metrics_csv = os.path.join(OUTPUT_ROOT, "metrics.csv")
    per_event_csv = os.path.join(OUTPUT_ROOT, "per_event.csv")

    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model","video","TP_frames","FN_frames","recall",
                    "latency_avg_s","latency_med_s",
                    "total_time_s","avg_ms_per_frame","median_ms_per_frame",
                    "pixels","fps_nominal"])
        w.writerows(metrics_rows)

    with open(per_event_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model","video","event","start_s","end_s","duration_s",
                    "frames_pos","tp_frames","fn_frames","recall","latency_s"])
        w.writerows(per_event_rows)

    print(f"\nSaved: {metrics_csv}")
    print(f"Saved: {per_event_csv}")
    plot_results(metrics_csv)

def plot_results(metrics_csv: str):
    import pandas as pd
    df = pd.read_csv(metrics_csv)
    g = df.groupby("model").agg({
        "recall":"mean",
        "avg_ms_per_frame":"mean",
        "total_time_s":"sum"
    }).reset_index()

    x = np.arange(len(g))
    # Recall
    plt.figure(figsize=(max(6, len(g)*1.3), 4.0))
    plt.bar(x, g["recall"])
    plt.xticks(x, g["model"], rotation=20, ha="right")
    plt.ylim(0, 1.0); plt.ylabel("Recall"); plt.title("Per-Model Recall (avg)")
    plt.tight_layout(); plt.show()

    # ms/frame
    plt.figure(figsize=(max(6, len(g)*1.3), 4.0))
    plt.bar(x, g["avg_ms_per_frame"])
    plt.xticks(x, g["model"], rotation=20, ha="right")
    plt.ylabel("Avg ms / frame"); plt.title("Per-Model Inference Time")
    plt.tight_layout(); plt.show()

# ---------------------------
# main
# ---------------------------
if __name__ == "__main__":
    run_all()