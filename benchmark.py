# =========================
# 0) 설치 (Colab 권장)
# =========================

import os, cv2, json, numpy as np, xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.metrics import f1_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# =========================
# 1) 경로/설정: 폴더에 *.mp4와 동일 스템의 *.xml이 있다고 가정
# =========================
DATA_DIR = "/content/datasets"  # ✅ 데이터셋 경로
SAMPLE_FPS = 5  # 프레임 샘플링 속도
CONF = 0.25  # 검출 신뢰도 임계치
IMGZ = 640  # 추론 입력 크기
IOU = 0.5  # 박스형 평가 IoU 기준


# =========================
# 2) 유틸: 페어 탐색, 프레임 샘플링, XML 파싱
# =========================
def find_pairs(root):
    root = Path(root)
    pairs = []
    for mp4 in sorted(root.rglob("*.mp4")):
        xml = mp4.with_suffix(".xml")
        if xml.exists():
            pairs.append((mp4.stem, str(mp4), str(xml)))
    if not pairs:
        raise FileNotFoundError("*.mp4와 같은 스템의 *.xml 페어가 없습니다.")
    return pairs


def sample_frames(video_path, sample_fps=None):
    cap = cv2.VideoCapture(video_path);
    assert cap.isOpened(), f"open fail: {video_path}"
    nat_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(nat_fps / (sample_fps or nat_fps))))
    frames, fidxs = [], []
    f = 0
    while True:
        ok = cap.grab()
        if not ok: break
        if f % step == 0:
            ok2, img = cap.retrieve()
            if not ok2: break
            frames.append(img[:, :, ::-1])  # BGR->RGB
            fidxs.append(f)
        f += 1
    cap.release()
    return frames, np.array(fidxs), nat_fps, total


def _hhmmss_to_sec(s):
    h, m, ss = s.strip().split(':');
    return int(h) * 3600 + int(m) * 60 + int(ss)


def parse_xml(xml_path, fps, total_frames):
    # VOC(박스) 또는 이벤트형(StartTime/AlarmDuration) 자동 감지
    root = ET.parse(xml_path).getroot()
    tags = {t.tag for t in root.iter()}

    if 'object' in tags and 'bndbox' in tags:
        # 단순화: 예시 XML에 프레임 정보가 없다고 가정 → frame 0에 GT 박스만 배치 (실무는 프레임 매핑 확장)
        gt = {0: []}
        for obj in root.iter('object'):
            name = (obj.findtext('name') or '').lower()
            if name not in ['person', 'human', 'people']: continue
            b = obj.find('bndbox');
            if b is None: continue
            x1 = float(b.findtext('xmin'));
            y1 = float(b.findtext('ymin'))
            x2 = float(b.findtext('xmax'));
            y2 = float(b.findtext('ymax'))
            gt[0].append([x1, y1, x2, y2])
        return {'mode': 'bbox', 'gt_boxes_by_frame': gt}

    st = next((n.text for n in root.iter('StartTime')), None)
    du = next((n.text for n in root.iter('AlarmDuration')), None)
    mask = np.zeros((total_frames,), dtype=np.uint8)
    if st and du:
        s = _hhmmss_to_sec(st);
        e = s + _hhmmss_to_sec(du)
        s_f = max(0, int(s * fps));
        e_f = min(total_frames - 1, int(e * fps))
        mask[s_f:e_f + 1] = 1
    return {'mode': 'event', 'gt_event_mask': mask}


# =========================
# 3) 박스 매칭 → scikit-learn F1
# =========================
def iou_mat(A, B):
    if not A or not B: return np.zeros((len(A), len(B)), dtype=np.float32)
    A = np.array(A);
    B = np.array(B)
    areaA = np.clip(A[:, 2] - A[:, 0], 0, None) * np.clip(A[:, 3] - A[:, 1], 0, None)
    areaB = np.clip(B[:, 2] - B[:, 0], 0, None) * np.clip(B[:, 3] - B[:, 1], 0, None)
    I = np.zeros((len(A), len(B)), np.float32)
    for i, a in enumerate(A):
        xx1 = np.maximum(a[0], B[:, 0]);
        yy1 = np.maximum(a[1], B[:, 1])
        xx2 = np.minimum(a[2], B[:, 2]);
        yy2 = np.minimum(a[3], B[:, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        I[i] = inter / np.clip(areaA[i] + areaB - inter, 1e-6, None)
    return I


def f1_boxes(gt_by_f, pr_by_f, iou_thr=0.5):
    # 프레임별 매칭 → (y_true,y_pred) 구성 → f1_score
    y_true, y_pred = [], []
    frames = sorted(set(gt_by_f.keys()) | set(pr_by_f.keys()))
    for f in frames:
        G = gt_by_f.get(f, []);
        P = pr_by_f.get(f, [])
        if G and P:
            M = iou_mat(G, P);
            cost = 1.0 - M
            gi, pj = linear_sum_assignment(cost)
            M_g, M_p = set(), set()
            for g, p in zip(gi, pj):
                if M[g, p] >= iou_thr:
                    y_true.append(1);
                    y_pred.append(1);
                    M_g.add(g);
                    M_p.add(p)
            for p in range(len(P)):
                if p not in M_p: y_true.append(0); y_pred.append(1)  # FP
            for g in range(len(G)):
                if g not in M_g: y_true.append(1); y_pred.append(0)  # FN
        else:
            y_true += [0] * len(P);
            y_pred += [1] * len(P)  # all FP
            y_true += [1] * len(G);
            y_pred += [0] * len(G)  # all FN
    return f1_score(y_true, y_pred, average='binary', zero_division=0)


# =========================
# 4) 모델 어댑터 (핵심만)
# - YOLOv11: 바로 동작
# - YOLOv12/DEYO/RF-DETR: 아래 1줄만 해당 레포 추론함수로 교체
#   (반드시 사람(person) 박스만 [[x1,y1,x2,y2],...] 형태로 리턴)
# =========================
def infer_yolo11(frames, conf=0.25, imgsz=640):
    model = YOLO('yolo11s.pt')  # 자동 다운로드
    names = model.model.names
    pid = {i for i, n in names.items() if str(n).lower() == 'person'}
    out = {}
    for i, img in enumerate(frames):
        r = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]
        boxes = []
        for xyxy, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
            if c in pid:
                x1, y1, x2, y2 = xyxy;
                boxes.append([float(x1), float(y1), float(x2), float(y2)])
        if boxes: out[i] = boxes
    return out


def infer_yolo12(frames, conf=0.25, imgsz=640):
    # TODO: return {i: predict_person_boxes_yolo12(img, conf, imgsz) for i,img in enumerate(frames)}
    # 데모용(즉시 실행 보장): YOLOv11 대체 → 실사용 시 ↑ 한 줄만 실제 레포 infer로 교체
    return infer_yolo11(frames, conf, imgsz)


def infer_deyo(frames, conf=0.25, imgsz=640):
    # TODO: return {i: predict_person_boxes_deyo(img, conf, imgsz) for i,img in enumerate(frames)}
    return infer_yolo11(frames, conf, imgsz)


def infer_rfdetr(frames, conf=0.25, imgsz=640):
    # TODO: return {i: predict_person_boxes_rfdetr(img, conf, imgsz) for i,img in enumerate(frames)}
    return infer_yolo11(frames, conf, imgsz)


MODELS = {
    "YOLOv11": infer_yolo11,
    "YOLOv12": infer_yolo12,  # ← 레포 infer 1줄 교체
    "DEYO": infer_deyo,  # ← 레포 infer 1줄 교체
    "RF-DETR": infer_rfdetr  # ← 레포 infer 1줄 교체
}


# =========================
# 5) 한 영상 평가 → 모델별 F1 계산
# =========================
def evaluate_one_video(video_path, xml_path, models, sample_fps, conf, imgsz, iou):
    frames, fidxs, fps, total = sample_frames(video_path, sample_fps)
    gt = parse_xml(xml_path, fps, total)
    result = {}
    for name, infer in models.items():
        pred_local = infer(frames, conf, imgsz)  # {sample_i: [[...],...]}
        pred_global = {int(fidxs[i]): b for i, b in pred_local.items()}
        if gt['mode'] == 'bbox':
            f1 = f1_boxes(gt['gt_boxes_by_frame'], pred_global, iou)
        else:
            # 이벤트형: 프레임 단위 이진 분류(사람 탐지 여부)
            y_true = gt['gt_event_mask']
            y_pred = np.zeros_like(y_true)
            for f, boxes in pred_global.items():
                if len(boxes) > 0: y_pred[f] = 1
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        result[name] = f1
    return result


# =========================
# 6) 전체 폴더 일괄 평가 + 평균 도식화
# =========================
pairs = find_pairs(DATA_DIR)
print(f"Found {len(pairs)} pairs")

per_video = []
agg = {m: [] for m in MODELS.keys()}

for stem, vpath, xpath in tqdm(pairs, desc="Videos"):
    scores = evaluate_one_video(vpath, xpath, MODELS, SAMPLE_FPS, CONF, IMGZ, IOU)
    per_video.append((stem, scores))
    for m, v in scores.items():
        agg[m].append(v)

avg_scores = {m: (np.mean(v) if len(v) > 0 else 0.0) for m, v in agg.items()}
print("\n--- Per-video F1 ---")
for stem, sc in per_video:
    print(stem, {k: f"{v:.3f}" for k, v in sc.items()})

print("\n--- Average F1 over videos ---")
for k, v in avg_scores.items():
    print(k, f"{v:.3f}")

plt.figure(figsize=(6, 3.5))
plt.bar(avg_scores.keys(), avg_scores.values());
plt.ylim(0, 1)
plt.ylabel("Avg F1-score")
plt.title("Human detection (Avg F1 over datasets)")
for i, (k, v) in enumerate(avg_scores.items()):
    plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
plt.xticks(rotation=10);
plt.show()






