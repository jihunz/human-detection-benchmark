# !pip install ultralytics

import json
import os
from typing import Dict, List, Tuple
import numpy as np
from ultralytics import YOLO
import shutil

root_dir = "/content"
yolo12n = YOLO("sample/yolo12n.pt")


def xywh_to_xyxy(box_xywh: List[float]) -> List[float]:
    x, y, w, h = box_xywh
    return [x, y, x + w, y + h]


def iou_xyxy(a: List[float], b: List[float]) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def get_image_id_from_path(path: str) -> int:
    # COCO: zero-padded 12-digit filename -> image_id
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem.lstrip("0") or "0")


def get_gt_by_img() -> Dict[int, List[List[float]]]:
    result: Dict[int, List[List[float]]] = {}
    path = os.path.join(root_dir, "instances_val2017.json")
    # path = '/Users/jihunjang/workspace/ust/human-detection/content/instances_val2017.json'
    with open(path, "r", encoding="utf-8") as f:
        ann = json.load(f)
        for item in ann["annotations"]:
            if item.get("iscrowd", 0) != 0:
                continue
            if item.get("category_id") != 1:  # COCO person
                continue
            img_id = item["image_id"]
            if img_id not in result:
                result[img_id] = []
            result[img_id].append(xywh_to_xyxy(item["bbox"]))

    # Sort image IDs and keep only the first 200
    sorted_img_ids = sorted(result.keys())[:200]
    filtered_result = {img_id: result[img_id] for img_id in sorted_img_ids}
    size = len(filtered_result)

    return filtered_result


def get_pred(
        img_ids_to_process: List[int],
        conf: float = 0.25,
        iou: float = 0.7,
        agnostic_nms: bool = False,
) -> Dict[int, List[Tuple[List[float], float, int]]]:
    image_dir = '/content/data_200'
    # image_dir = '/Users/jihunjang/workspace/ust/human-detection/content/data_200'

    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

    if not image_paths:
        return {}

    results = yolo12n(image_paths, conf=conf, iou=iou, agnostic_nms=agnostic_nms)
    out: Dict[int, List[Tuple[List[float], float, int]]] = {}

    for r, p in zip(results, image_paths):
        img_id = get_image_id_from_path(p)
        out[img_id] = []

        if getattr(r, "boxes", None) is None:
            continue

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clses = r.boxes.cls.cpu().numpy().astype(int)

        for bb, sc, ci in zip(boxes_xyxy, confs, clses):
            if int(ci) == 0:  # person only
                out[img_id].append((bb.tolist(), float(sc), int(ci)))

        # Sort by confidence descending
        out[img_id].sort(key=lambda x: x[1], reverse=True)

    return out


def greedy_match_one_image(
        preds: List[Tuple[List[float], float, int]],
        gts: List[List[float]],
        iou_thr: float = 0.5,
) -> Tuple[int, int, int]:
    """
    preds: [(bbox_xyxy, conf, cls=0), ...]  # Assumes already sorted by conf desc
    gts  : [bbox_xyxy, ...]
    return: TP, FP, FN
    """
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
            TP += 1
            matched_gt.add(best_gi)
        else:
            FP += 1

    FN = len(gts) - len(matched_gt)
    return TP, FP, FN


def get_f1_score(
        iou_match_thr: float = 0.5,
        conf: float = 0.25,
        nms_iou: float = 0.7,
) -> Tuple[float, float, float]:
    gts_by_img = get_gt_by_img()
    # Get the list of image IDs that were loaded from the ground truth
    img_ids_to_process = list(gts_by_img.keys())

    preds_by_img = get_pred(img_ids_to_process, conf=conf, iou=nms_iou)

    all_ids = set(gts_by_img.keys()) | set(preds_by_img.keys())
    TP_tot = FP_tot = FN_tot = 0

    for img_id in all_ids:
        preds = preds_by_img.get(img_id, [])
        gts = gts_by_img.get(img_id, [])
        tp, fp, fn = greedy_match_one_image(preds, gts, iou_thr=iou_match_thr)
        TP_tot += tp
        FP_tot += fp
        FN_tot += fn

    precision = TP_tot / (TP_tot + FP_tot) if (TP_tot + FP_tot) > 0 else 0.0
    recall    = TP_tot / (TP_tot + FN_tot) if (TP_tot + FN_tot) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def extract_human_img():
    gt = get_gt_by_img()
    img_dir = '/datasets/coco/val2017/'
    listdir = os.listdir(img_dir)
    for f in listdir:
        img_id = get_image_id_from_path(f)
        if img_id in gt.keys():
            src = img_dir + f
            target = '/Users/jihunjang/workspace/ust/human-detection/content/data_200/' + f
            shutil.copy(src, target)


if __name__ == "__main__":
    P, R, F1 = get_f1_score(
        iou_match_thr=0.5,  # GT-예측 매칭 기준 IoU
        conf=0.25,  # 예측 필터링 기준(conf)
        nms_iou=0.7,  # NMS IoU
    )
    print(f"Precision={P:.4f}, Recall={R:.4f}, F1={F1:.4f}")
