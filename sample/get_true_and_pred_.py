import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# -----------------------
# 설정
# -----------------------
COCO_ANN = "/path/to/annotations/instances_val2017.json"
CONF_THR = 0.25
IOU_THR = 0.5

# 각 모델의 COCO results 포맷 json 경로
MODEL_RESULT_FILES = {
    "YOLOv8": "/path/to/preds/yolov8_person_results.json",
    "EfficientDet": "/path/to/preds/efficientdet_person_results.json",
    "Faster-RCNN": "/path/to/preds/fasterrcnn_person_results.json",
}


# -----------------------
# 유틸
# -----------------------
def xywh_to_xyxy(box):
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=float)


def iou_xyxy(a, b):
    # a,b: [x1,y1,x2,y2]
    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    union = area_a + area_b - inter + 1e-12
    return inter / union


# -----------------------
# GT(person) 로딩
# -----------------------
def load_coco_person_gt(coco_ann_path):
    with open(coco_ann_path, "r") as f:
        data = json.load(f)

    # person category id 찾기
    name_to_id = {c["name"]: c["id"] for c in data["categories"]}
    assert "person" in name_to_id, "COCO categories에 person이 없습니다."
    person_id = name_to_id["person"]

    # image_id -> list of GT boxes(xyxy)
    gt_by_img = defaultdict(list)
    for ann in data["annotations"]:
        if ann["category_id"] != person_id:
            continue
        if ann.get("iscrowd", 0) == 1:
            continue
        box_xyxy = xywh_to_xyxy(ann["bbox"])
        gt_by_img[ann["image_id"]].append(box_xyxy)

    return gt_by_img, person_id


# -----------------------
# 예측(person) 로딩
# -----------------------
def load_person_predictions(results_json_path, person_id, conf_thr=0.25):
    with open(results_json_path, "r") as f:
        preds = json.load(f)

    pred_by_img = defaultdict(list)  # image_id -> list of (xyxy, score)
    for det in preds:
        if det["category_id"] != person_id:
            continue
        score = float(det["score"])
        if score < conf_thr:
            continue
        box_xyxy = xywh_to_xyxy(det["bbox"])
        pred_by_img[det["image_id"]].append((box_xyxy, score))

    # 점수 내림차순 정렬 (그리디 매칭 안정화)
    for img_id in pred_by_img:
        pred_by_img[img_id].sort(key=lambda x: x[1], reverse=True)

    return pred_by_img


# -----------------------
# 매칭 및 배열 생성 (binary)
# -----------------------
def build_arrays_for_model(gt_by_img, pred_by_img, iou_thr=0.5):
    y_true, y_pred = [], []

    # 전체 이미지의 합산 성능을 보려면 "양쪽 키의 합집합"을 도는 게 안전
    all_img_ids = set(gt_by_img.keys()) | set(pred_by_img.keys())

    for img_id in all_img_ids:
        gts = [g.copy() for g in gt_by_img.get(img_id, [])]  # list of np.array([x1,y1,x2,y2])
        preds = pred_by_img.get(img_id, [])  # list of (np.array([x1,y1,x2,y2]), score)

        gt_used = [False] * len(gts)

        # 예측을 점수순으로 순회하며 그리디 매칭
        for p_box, _ in preds:
            best_iou, best_j = 0.0, -1
            for j, g_box in enumerate(gts):
                if gt_used[j]:
                    continue
                iou = iou_xyxy(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_thr and best_j >= 0:
                # 매칭 성공 → TP
                gt_used[best_j] = True
                y_true.append(1)
                y_pred.append(1)
            else:
                # 매칭 실패 예측 → FP
                y_true.append(0)
                y_pred.append(1)

        # 남은 GT → FN
        for used in gt_used:
            if not used:
                y_true.append(1)
                y_pred.append(0)

    return np.array(y_true, dtype=int), np.array(y_pred, dtype=int)


# -----------------------
# 평가 및 시각화
# -----------------------
def evaluate_models(coco_ann, model_files, conf_thr=0.25, iou_thr=0.5):
    gt_by_img, person_id = load_coco_person_gt(coco_ann)

    scores = {}
    for model_name, res_path in model_files.items():
        pred_by_img = load_person_predictions(res_path, person_id, conf_thr)
        y_true, y_pred = build_arrays_for_model(gt_by_img, pred_by_img, iou_thr)
        f1 = f1_score(y_true, y_pred, average="binary", pos_label=1)
        scores[model_name] = f1
        print(f"{model_name}: F1={f1:.4f} (conf>={conf_thr}, IoU>={iou_thr}) | n={len(y_true)})")
    return scores


def plot_scores(f1_scores, title="Model-wise F1 (person, COCO val2017)"):
    plt.figure(figsize=(8, 5))
    plt.bar(list(f1_scores.keys()), list(f1_scores.values()))
    plt.title(title)
    plt.ylabel("F1 Score")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    # PyCharm 백엔드 문제 있으면 show 대신 저장:
    # plt.savefig("f1_scores.png", dpi=150, bbox_inches="tight")
    plt.show()


def get_gt_by_img():
    result = {}
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = root_dir + '/datasets/coco/annotations/instances_val2017.json'

    with open(path, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
        for item in annotation['annotations']:
            image_id = item['image_id']

            if image_id not in result:
                result[image_id] = []

            if item['category_id'] == 1 and item['iscrowd'] == 0:
                result[image_id].append(item['bbox'])

    return sorted(result.items())


if __name__ == "__main__":
    get_gt_by_img()
    # f1_scores = evaluate_models(COCO_ANN, MODEL_RESULT_FILES, CONF_THR, IOU_THR)
    # plot_scores(f1_scores)
