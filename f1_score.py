# =========================================================
# Real-time detectors F1 benchmark (COCO person, 200 imgs)
# - YOLO12n (Ultralytics hub auto-download)
# - YOLO11n (Ultralytics hub auto-download)
# - RT-DETRv2 (HF Transformers)
# - D-FINE    (HF Transformers)
# Greedy 1:1 matching, Precision/Recall/F1
# + matplotlib bar plot
# + matplotlib table (sorted by F1)
# + console print table + CSV save
# =========================================================

# !pip -q install ultralytics transformers torch pillow matplotlib

import os, json
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------
# Paths / Constants
# -----------------------
ROOT = "/content"
COCO_ANN = os.path.join(ROOT, "instances_val2017.json")  # COCO val2017 annotation(json)
IMG_DIR  = os.path.join(ROOT, "data_200")                # images (12-digit COCO filenames)

# -----------------------
# Helpers
# -----------------------
def xywh_to_xyxy(box_xywh: List[float]) -> List[float]:
    x, y, w, h = box_xywh
    return [x, y, x + w, y + h]

def iou_xyxy(a: List[float], b: List[float]) -> float:
    xx1 = max(a[0], b[0]); yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2]); yy2 = min(a[3], b[3])
    w = max(0.0, xx2 - xx1); h = max(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def get_image_id_from_path(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem.lstrip("0") or "0")

def list_coco_like_images(img_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
             if os.path.splitext(f)[1].lower() in exts]
    files.sort()
    return files

def get_dir_ids(img_dir: str) -> List[int]:
    return [get_image_id_from_path(p) for p in list_coco_like_images(img_dir)]

# -----------------------
# GT loader (person only)
# -----------------------
def get_gt_by_img(ann_path: str,
                  allowed_img_ids: Optional[Iterable[int]]=None,
                  limit_n: Optional[int]=200) -> Dict[int, List[List[float]]]:
    result: Dict[int, List[List[float]]] = {}
    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    for item in ann["annotations"]:
        if item.get("iscrowd", 0) != 0:
            continue
        if item.get("category_id") != 1:  # person
            continue
        img_id = item["image_id"]
        if allowed_img_ids is not None and img_id not in allowed_img_ids:
            continue
        result.setdefault(img_id, []).append(xywh_to_xyxy(item["bbox"]))
    ids = sorted(result.keys())
    if limit_n is not None:
        ids = ids[:limit_n]
        result = {i: result[i] for i in ids}
    return result

def get_eval_gts(ann_path: str, img_dir: str, limit_n: int=200) -> Dict[int, List[List[float]]]:
    dir_ids = set(get_dir_ids(img_dir))
    return get_gt_by_img(ann_path, allowed_img_ids=dir_ids, limit_n=limit_n)

# -----------------------
# Inference (Ultralytics YOLO/RT-DETR)
# -----------------------
def pred_ultralytics_any(weights: str, image_paths: List[str],
                         conf=0.25, nms_iou=0.7, imgsz: Optional[int]=None,
                         arch: Optional[str]=None):
    from ultralytics import YOLO, RTDETR
    model = RTDETR(weights) if arch == "rtdetr" else YOLO(weights)
    kwargs = dict(conf=conf, iou=nms_iou, agnostic_nms=False)
    if imgsz is not None:
        kwargs["imgsz"] = imgsz
    results = model(image_paths, **kwargs)

    out: Dict[int, List[Tuple[List[float], float, int]]] = {}
    for r, p in zip(results, image_paths):
        img_id = get_image_id_from_path(p)
        out.setdefault(img_id, [])
        if getattr(r, "boxes", None) is None:
            continue
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clses = r.boxes.cls.cpu().numpy().astype(int)
        for bb, sc, ci in zip(boxes_xyxy, confs, clses):
            if ci == 0:  # person
                out[img_id].append((bb.tolist(), float(sc), 0))
        out[img_id].sort(key=lambda x: x[1], reverse=True)
    return out

# -----------------------
# HF Transformers inference (RT-DETRv2 / D-FINE)
# -----------------------
def pred_hf_od(model_id: str,
               image_paths: List[str],
               conf: float = 0.25,
               batch_size: int = 8,
               use_fast: bool = True,
               fp16: bool = True,
               trust_remote_code: bool = True):
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(
        model_id, use_fast=use_fast, trust_remote_code=trust_remote_code
    )
    model = AutoModelForObjectDetection.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    ).to(device).eval()

    id2label = getattr(model.config, "id2label", {}) or {}
    out: Dict[int, List[Tuple[List[float], float, int]]] = {}

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    for batch_paths in chunks(image_paths, batch_size):
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            if fp16 and device == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
        target_sizes = torch.tensor(
            [[im.size[1], im.size[0]] for im in imgs], device=device
        )
        dets = processor.post_process_object_detection(
            outputs, threshold=conf, target_sizes=target_sizes
        )
        for p, res in zip(batch_paths, dets):
            img_id = get_image_id_from_path(p)
            boxes = res["boxes"].detach().cpu().numpy()
            scores = res["scores"].detach().cpu().numpy()
            labels = res["labels"].detach().cpu().numpy().astype(int)
            lst = []
            for bb, sc, lb in zip(boxes, scores, labels):
                name = id2label.get(int(lb), "")
                if name == "person" or int(lb) == 1:
                    lst.append((bb.tolist(), float(sc), 0))
            lst.sort(key=lambda x: x[1], reverse=True)
            out[img_id] = lst
    return out

# -----------------------
# Matcher + Evaluation
# -----------------------
def greedy_match_one_image(
    preds: List[Tuple[List[float], float, int]],
    gts: List[List[float]],
    iou_thr: float = 0.5,
) -> Tuple[int, int, int]:
    matched_gt = set()
    TP = FP = 0
    for (p_box, _conf, _ci) in preds:
        best_iou, best_gi = 0.0, -1
        for gi, g_box in enumerate(gts):
            if gi in matched_gt:
                continue
            iou = iou_xyxy(p_box, g_box)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_gi >= 0 and best_iou >= iou_thr:
            TP += 1; matched_gt.add(best_gi)
        else:
            FP += 1
    FN = len(gts) - len(matched_gt)
    return TP, FP, FN

def evaluate(preds_by_img: Dict[int, List[Tuple[List[float], float, int]]],
             gts_by_img: Dict[int, List[List[float]]],
             iou_thr: float = 0.5) -> Tuple[float, float, float]:
    all_ids = set(gts_by_img.keys()) | set(preds_by_img.keys())
    TP_tot = FP_tot = FN_tot = 0
    for img_id in all_ids:
        preds = preds_by_img.get(img_id, [])
        gts   = gts_by_img.get(img_id, [])
        tp, fp, fn = greedy_match_one_image(preds, gts, iou_thr=iou_thr)
        TP_tot += tp; FP_tot += fp; FN_tot += fn
    precision = TP_tot / (TP_tot + FP_tot) if (TP_tot + FP_tot) > 0 else 0.0
    recall    = TP_tot / (TP_tot + FN_tot) if (TP_tot + FN_tot) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

# -----------------------
# Plotting + Tables
# -----------------------
def plot_bars(results: List[Tuple[str, float, float, float]], title="Precision / Recall / F1"):
    names = [r[0] for r in results]; P = [r[1] for r in results]
    R = [r[2] for r in results];     F1 = [r[3] for r in results]
    x = np.arange(len(names)); width = 0.25
    plt.figure(figsize=(max(6, len(names)*1.3), 4.5))
    plt.bar(x - width, P, width, label='Precision')
    plt.bar(x,         R, width, label='Recall')
    plt.bar(x + width, F1, width, label='F1')
    plt.xticks(x, names, rotation=20, ha='right')
    plt.ylabel("Score"); plt.ylim(0, 1.0)
    plt.title(title); plt.legend(); plt.tight_layout(); plt.show()

def make_table_rows(results: List[Tuple[str, float, float, float]]):
    headers = ["Model", "Precision", "Recall", "F1"]
    rows = [[name, f"{P:.4f}", f"{R:.4f}", f"{F1:.4f}"] for name, P, R, F1 in results]
    return headers, rows

def plot_table(results: List[Tuple[str, float, float, float]],
               sort_by: str = "F1",
               title: str = "Per-Model Metrics (sorted)"):
    # sort
    idx = {"Precision": 1, "Recall": 2, "F1": 3}.get(sort_by, 3)
    sorted_res = sorted(results, key=lambda r: r[idx], reverse=True)
    headers, rows = make_table_rows(sorted_res)

    # figure with table
    fig_h = 0.6 * len(rows) + 1.6
    fig_w = max(6, len(rows)*1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.scale(1.0, 1.2)
    ax.set_title(f"{title} by {sort_by}", pad=10)
    plt.tight_layout()
    plt.show()
    return sorted_res

def print_table(results: List[Tuple[str, float, float, float]], sort_by: str = "F1"):
    idx = {"Precision": 1, "Recall": 2, "F1": 3}.get(sort_by, 3)
    sorted_res = sorted(results, key=lambda r: r[idx], reverse=True)
    headers, rows = make_table_rows(sorted_res)
    # console pretty print
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    def fmt_row(r): return " | ".join(s.ljust(w) for s, w in zip(r, widths))
    sep = "-+-".join("-"*w for w in widths)
    print(fmt_row(headers))
    print(sep)
    for r in rows:
        print(fmt_row(r))
    return sorted_res

# -----------------------
# Runner
# -----------------------
def run_benchmark(models_cfg: List[dict],
                  ann_path: str = COCO_ANN,
                  img_dir: str = IMG_DIR,
                  iou_thr: float = 0.5,
                  conf_default: float = 0.25,
                  nms_iou_default: float = 0.7):
    if not os.path.isfile(ann_path):
        raise FileNotFoundError(f"Missing COCO annotation: {ann_path}")
    if not os.path.isdir(img_dir) or not list_coco_like_images(img_dir):
        raise FileNotFoundError(f"Missing images in: {img_dir}")

    gts = get_eval_gts(ann_path, img_dir, limit_n=200)
    img_paths = list_coco_like_images(img_dir)
    valid_ids = set(gts.keys())
    img_paths = [p for p in img_paths if get_image_id_from_path(p) in valid_ids]

    results = []
    for cfg in models_cfg:
        name = cfg["name"]
        print(f"\n[Eval] {name}")
        backend = cfg.get("backend", "ultralytics")

        if backend == "ultralytics":
            preds = pred_ultralytics_any(
                weights=cfg["weights"],
                image_paths=img_paths,
                conf=cfg.get("conf", conf_default),
                nms_iou=cfg.get("nms_iou", nms_iou_default),
                imgsz=cfg.get("imgsz", None),
                arch=cfg.get("arch", None),
            )
        elif backend == "hf":
            preds = pred_hf_od(
                model_id=cfg["hf_id"],
                image_paths=img_paths,
                conf=cfg.get("conf", conf_default),
                batch_size=cfg.get("batch_size", 8),
                use_fast=cfg.get("use_fast", True),
                fp16=cfg.get("fp16", True),
                trust_remote_code=cfg.get("trust_remote_code", True),
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        P, R, F1 = evaluate(preds, gts, iou_thr=iou_thr)
        results.append((name, P, R, F1))
        print(f"{name}: P={P:.4f}, R={R:.4f}, F1={F1:.4f}")
    return results

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    MODELS = [
        # ✅ YOLO12n: 허브에서 자동 다운로드
        {"name": "YOLO12n", "backend": "ultralytics",
         "weights": "yolo12n.pt",
         "conf": 0.25, "nms_iou": 0.7, "imgsz": 640, "arch": "yolo"},

        # YOLO11n: 허브에서 자동 다운로드
        {"name": "YOLO11n", "backend": "ultralytics",
         "weights": "yolo11n.pt",
         "conf": 0.25, "nms_iou": 0.7, "imgsz": 640, "arch": "yolo"},

        # HF Transformers
        {"name": "RT-DETRv2 (HF)", "backend": "hf",
         "hf_id": "PekingU/rtdetr_v2_r18vd", "conf": 0.25, "batch_size": 8, "fp16": True},
        {"name": "D-FINE (HF)", "backend": "hf",
         "hf_id": "ustc-community/dfine_x_coco", "conf": 0.25, "batch_size": 8, "fp16": True},
    ]

    IOU_MATCH = 0.7  # 필요 시 0.5/0.75 등으로 변경

    results = run_benchmark(MODELS, ann_path=COCO_ANN, img_dir=IMG_DIR, iou_thr=IOU_MATCH)

    # 1) 막대그래프
    plot_bars(results, title=f"Precision / Recall / F1 (IoU={IOU_MATCH})")

    # 2) 표(그림) + 3) 콘솔 표 + 4) CSV 저장
    sorted_res = plot_table(results, sort_by="F1", title="Per-Model Metrics")
    print_table(sorted_res, sort_by="F1")
