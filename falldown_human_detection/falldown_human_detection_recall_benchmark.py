# ================================================================
# Recall-only evaluation for KISA-style clips (Start-5s pre-trimmed)
# - 평가 구간: [clip 5s, clip 5s + duration]
# - 프레임 TP: max_conf(사람) >= 0.90
# - 프레임 FN: 구간 내인데 max_conf < 0.90
# - 이벤트 구간 밖은 평가 제외 (Precision/FP 미계산)
# - 모델: YOLO12n, YOLO11n (Ultralytics), RT-DETRv2, D-FINE (HF)
# - 결과:
#     * metrics.csv / per_event.csv + 간단 시각화
#     * 모델별 폴더에 주석 영상 저장
#     * 마지막에 각 모델 폴더 ZIP 압축
# ================================================================

# !pip -q install ultralytics transformers torch pillow opencv-python matplotlib lxml

import os, time, csv, math, shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------
# Config
# ---------------------------
VIDEO_ROOT   = "/kaggle/input/falldown-cut2"    # 영상/GT(XML) 폴더
OUTPUT_ROOT  = "/kaggle/working/outputs_eval"   # 결과 저장 루트
os.makedirs(OUTPUT_ROOT, exist_ok=True)

TARGET_EVENTS       = {"Falldown"}  # None이면 XML 모든 이벤트 사용
CONF_HARD_TH        = 0.90          # TP 판정용 하드 스레시홀드
CLIP_PRE_OFFSET_S   = 5.0           # 클립이 Start-5s부터 시작 → 평가 시작 5.0s 고정
WRITE_ANNOTATED_MP4 = True          # 주석 영상 저장 여부
VID_EXTS            = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# ---------------------------
# 작은 유틸
# ---------------------------
def ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def list_videos(root: str | Path) -> List[Path]:
    root = Path(root)
    if root.is_file() and root.suffix.lower() in VID_EXTS and not root.name.startswith("._"):
        return [root]
    files = []
    for ext in VID_EXTS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted({Path(p) for p in files if not Path(p).name.startswith("._")})

def parse_time_hms(s: str) -> float:
    hh, mm, ss = s.strip().split(":")
    return int(hh)*3600 + int(mm)*60 + int(ss)

def parse_kisa_xml(xml_path: Path,
                   target_events: Optional[set] = TARGET_EVENTS) -> List[dict]:
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
# 공통: 사람 라벨 이름 자동 매핑
# ---------------------------
PERSON_ALIASES = {"person", "human", "pedestrian", "people"}

def find_person_ids_from_names(names: Dict[int, object]) -> List[int]:
    out = []
    for i, n in names.items():
        if isinstance(n, str):
            if n.strip().lower() in PERSON_ALIASES:
                out.append(int(i))
    return out

# ---------------------------
# Detector 베이스
# ---------------------------
class Detector:
    def __init__(self, name: str):
        self.name = name
    def infer(self, frame_bgr: np.ndarray) -> Tuple[float, List[Tuple[int, int, int, int, float]]]:
        """
        반환:
          max_conf (사람만): float
          boxes (사람만): [(x1,y1,x2,y2,conf), ...] - 그리기용
        """
        raise NotImplementedError

# ---------------------------
# Ultralytics YOLO
# ---------------------------
class UltralyticsYOLO(Detector):
    def __init__(self, name: str, weights: str):
        super().__init__(name)
        from ultralytics import YOLO
        self.model = YOLO(weights)
        names = {int(i): n for i, n in enumerate(self.model.names)} if hasattr(self.model, "names") else {}
        self.person_ids = find_person_ids_from_names(names) or [0]  # 안전장치

    def infer(self, frame_bgr: np.ndarray):
        r = self.model.predict(
            source=frame_bgr, conf=0.25, iou=0.70, imgsz=640,
            verbose=False
        )[0]
        boxes = []
        max_conf = 0.0
        if getattr(r, "boxes", None) is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls  = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), sc, c in zip(xyxy, conf, cls):
                if int(c) in self.person_ids:
                    boxes.append((int(x1), int(y1), int(x2), int(y2), float(sc)))
                    if sc > max_conf: max_conf = float(sc)
        return max_conf, boxes

# ---------------------------
# HuggingFace Object Detection (RT-DETRv2 / D-FINE)
# ---------------------------
class HFDetector(Detector):
    def __init__(self, name: str, model_id: str):
        super().__init__(name)
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model     = AutoModelForObjectDetection.from_pretrained(model_id, trust_remote_code=True).to(self.device).eval()
        id2label = getattr(self.model.config, "id2label", {}) or {}
        self.person_ids = []
        for k, v in id2label.items():
            kid = int(k) if not (isinstance(k, str) and k.isdigit()) else int(k)
            if isinstance(v, str) and v.strip().lower() in PERSON_ALIASES:
                self.person_ids.append(kid)

    def infer(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        ts = self.torch.tensor([[pil.size[1], pil.size[0]]], device=self.device)
        det = self.processor.post_process_object_detection(outputs, threshold=0.25, target_sizes=ts)[0]
        boxes = []
        max_conf = 0.0
        if "boxes" in det and "scores" in det and "labels" in det:
            b = det["boxes"].detach().cpu().numpy()
            s = det["scores"].detach().cpu().numpy()
            l = det["labels"].detach().cpu().numpy().astype(int)
            for (x1,y1,x2,y2), sc, lb in zip(b, s, l):
                if self.person_ids and int(lb) not in self.person_ids:
                    continue
                boxes.append((int(x1), int(y1), int(x2), int(y2), float(sc)))
                if sc > max_conf: max_conf = float(sc)
        return max_conf, boxes

# ---------------------------
# 프레임 단위 평가 + (선택) 주석 저장
# ---------------------------
def evaluate_single_video(det, video_path: Path, events: List[dict], out_dir: Path):
    """
    - 평가 구간: [5s, 5s+duration] 프레임만 TP/FN 집계
    - 영상 전체 프레임에 대해 주석(사람 박스) 그려서 mp4 저장(옵션)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    intervals = []
    for e in events:
        dur = max(0.0, float(e["duration"]))
        s = CLIP_PRE_OFFSET_S
        t = CLIP_PRE_OFFSET_S + dur
        s = max(0.0, min(s, N / fps))
        t = max(0.0, min(t, N / fps))
        if t <= s:
            continue
        s_idx = int(math.floor(s * fps))
        t_idx = int(math.ceil (t * fps))
        intervals.append((s_idx, t_idx))

    maxconfs = np.zeros(N, dtype=np.float32)

    writer = None
    if WRITE_ANNOTATED_MP4:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_mp4 = out_dir / f"{video_path.stem}_{det.name}.mp4"
        writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (W, H))

    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        max_conf, boxes = det.infer(frame)

        in_window = any(s<=idx<t for (s,t) in intervals)
        if in_window:
            maxconfs[idx] = max_conf

        if writer is not None:
            for (x1,y1,x2,y2,sc) in boxes:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"{sc:.2f}", (x1, max(0,y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()

    per_event = []
    TP_total = FN_total = 0
    for (s_idx, t_idx) in intervals:
        seg = maxconfs[s_idx:t_idx] if t_idx > s_idx else np.array([], dtype=np.float32)
        frames_pos = int(len(seg))
        if frames_pos == 0:
            continue
        tp_frames = int((seg >= CONF_HARD_TH).sum())
        fn_frames = frames_pos - tp_frames
        recall    = tp_frames / frames_pos
        TP_total += tp_frames
        FN_total += fn_frames
        per_event.append((frames_pos, tp_frames, fn_frames, recall))

    recall_all = TP_total / max(1, (TP_total + FN_total))
    return recall_all, per_event

# ---------------------------
# 실행 & 집계 + 저장/압축
# ---------------------------
def run_all():
    videos = list_videos(VIDEO_ROOT)
    if not videos:
        raise FileNotFoundError(f"No videos under: {VIDEO_ROOT}")

    models: List[Detector] = [
        UltralyticsYOLO("YOLO12n", "yolo12n.pt"),
        UltralyticsYOLO("YOLO11n", "yolo11n.pt"),
        HFDetector("RT-DETRv2 (HF, R18vd)", "PekingU/rtdetr_v2_r18vd"),
        HFDetector("D-FINE-S (HF)", "ustc-community/dfine_s_coco"),
    ]

    metrics_rows = []
    per_event_rows = []

    for det in models:
        print(f"\n=== [{det.name}] ===")
        model_out_dir = ensure_dir(Path(OUTPUT_ROOT) / det.name)

        for vp in videos:
            xml_path = Path(str(vp)[:-len(vp.suffix)]).with_suffix(".xml")
            if not xml_path.exists():
                print(f"  - Skip (no XML): {vp.name}")
                continue

            events = parse_kisa_xml(xml_path, TARGET_EVENTS)
            if not events:
                print(f"  - Skip (no target events): {vp.name}")
                continue

            print(f"  - Eval: {vp.name}  (#events={len(events)})")
            start_time = time.time()
            recall, evs = evaluate_single_video(det, vp, events, model_out_dir)
            total_time_s = time.time() - start_time

            cap = cv2.VideoCapture(str(vp))
            N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            ms_per_frame = (total_time_s * 1000.0 / max(1, N))

            tp_sum = sum(tp for _, tp, _, _ in evs) if evs else 0
            fn_sum = sum(fn for _, _, fn, _ in evs) if evs else 0
            metrics_rows.append([
                det.name, vp.name, tp_sum, fn_sum, f"{recall:.4f}",
                f"{total_time_s:.2f}", f"{ms_per_frame:.2f}", W*H, f"{fps:.2f}"
            ])

            for (frames_pos, tp_frames, fn_frames, rec) in evs:
                per_event_rows.append([
                    det.name, vp.name, "Falldown",
                    frames_pos, tp_frames, fn_frames, f"{rec:.4f}"
                ])

        zip_base = str(Path(OUTPUT_ROOT) / f"{det.name}")
        shutil.make_archive(zip_base, "zip", root_dir=str(Path(OUTPUT_ROOT) / det.name))
        print(f"  - Zipped: {zip_base}.zip")

    metrics_csv = Path(OUTPUT_ROOT) / "metrics.csv"
    per_event_csv = Path(OUTPUT_ROOT) / "per_event.csv"

    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model","video","TP_frames","FN_frames","recall",
                    "total_time_s","avg_ms_per_frame","pixels","fps_nominal"])
        w.writerows(metrics_rows)

    with open(per_event_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model","video","event","frames_pos","tp_frames","fn_frames","recall"])
        w.writerows(per_event_rows)

    print(f"\nSaved: {metrics_csv}")
    print(f"Saved: {per_event_csv}")
    plot_results(str(metrics_csv))

# ---------------------------
# 간단 시각화 (디자인 업그레이드)
# ---------------------------
def _apply_clean_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.5)

def _annotate_bars(ax, rects, horizontal=False, fmt="{:.1f}"):
    for r in rects:
        if horizontal:
            w = r.get_width()
            y = r.get_y() + r.get_height()/2
            ax.text(w + (0.01 if w>=0 else -0.01), y, fmt.format(w),
                    va="center", ha="left" if w>=0 else "right", fontsize=10)
        else:
            h = r.get_height()
            x = r.get_x() + r.get_width()/2
            ax.text(x, h + (max(ax.get_ylim())*0.02), fmt.format(h),
                    va="bottom", ha="center", fontsize=10)

def plot_results(metrics_csv: str):
    import pandas as pd
    df = pd.read_csv(metrics_csv)

    # 모델 평균 집계
    g = df.groupby("model").agg({
        "recall":"mean",
        "avg_ms_per_frame":"mean",
        "total_time_s":"sum"
    }).reset_index()

    # 색상 팔레트 (모델 수만큼 순서대로 매핑)
    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i % 10) for i, m in enumerate(g["model"])}

    # ---------- Recall (Horizontal bar) ----------
    fig, ax = plt.subplots(figsize=(8, max(3.8, 0.55*len(g)+1)))
    y_pos = np.arange(len(g))[::-1]  # 위에서 아래로
    vals = (g["recall"].values*100.0)[::-1]
    labels = g["model"].values[::-1]
    rects = ax.barh(y_pos, vals, color=[colors[m] for m in labels], height=0.6)
    _apply_clean_style(ax)
    ax.set_yticks(y_pos, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Recall (%)")
    ax.set_title("Per-Model Recall (avg)")
    _annotate_bars(ax, rects, horizontal=True, fmt="{:.1f}%")
    plt.tight_layout()
    plt.show()

    # ---------- Inference time (ms/frame) ----------
    fig, ax = plt.subplots(figsize=(8, max(3.6, 0.5*len(g)+1)))
    x = np.arange(len(g))
    rects = ax.bar(x, g["avg_ms_per_frame"], color=[colors[m] for m in g["model"]], width=0.6)
    _apply_clean_style(ax)
    ax.set_xticks(x, g["model"], rotation=15, ha="right")
    ax.set_ylabel("Avg ms / frame")
    ax.set_title("Per-Model Inference Time")
    # 값 라벨 (소수 1자리)
    for r in rects:
        h = r.get_height()
        ax.text(r.get_x()+r.get_width()/2, h + (h*0.03 + 1), f"{h:.1f}",
                ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.show()

# ---------------------------
# main
# ---------------------------
if __name__ == "__main__":
    run_all()