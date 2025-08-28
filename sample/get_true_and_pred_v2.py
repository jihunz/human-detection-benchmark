import json
import os

from ultralytics import YOLO

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

yolo12n = YOLO("yolo12n.pt")


def get_pred():
    result = {}

    pred = yolo12n(root_dir + "/datasets/coco/val2017/000000000785.jpg")
    for p in pred:
        img_id = int(p.path.split("/")[-1].split(".")[0].lstrip("0"))
        result[img_id] = []

        boxes = p.boxes
        bbox_list = boxes.xyxy
        cls_list = boxes.cls
        conf_list = boxes.conf

        for i in range(0, len(cls_list)):
            result[img_id].append({
                'cls': cls_list[i],
                'conf': conf_list[i],
                'bbox': bbox_list[i]
            })

            sorted(result[img_id], key=lambda x: x['conf'], reverse=True)

    return result


def get_gt_by_img():
    result = {}
    path = root_dir + '/datasets/coco/annotations/instances_val2017.json'

    with open(path, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
        for item in annotation['annotations']:
            img_id = item['image_id']

            if img_id not in result:
                result[img_id] = []

            if item['category_id'] == 1 and item['iscrowd'] == 0: # 군중 이미지는 예측 박스와의 매칭 복잡성의 문제로 제외
                # if img_id == 554002: # TODO: 군중이 아니지만 사람에 대한 여러 gt가 있는 경우
                #     print(item)
                result[img_id].append(item['bbox'])

    return dict(sorted(result.items()))


# 예측과 gt의 box iou 계산
def get_iou_between_gt_pred(pred, gt):
    result = []

    for gt_idx in range(0, len(gt)):
        gt_box = gt[gt_idx]
        pred_box = pred[gt_idx]['bbox']

        xx1 = max(pred_box[0], gt_box[0])
        yy1 = max(pred_box[1], gt_box[1])
        xx2 = min(pred_box[2], gt_box[2])
        yy2 = min(pred_box[3], gt_box[3])
        w = max(0.0, xx2 - xx1)
        h = max(0.0, yy2 - yy1)
        inter = w * h
        area_a = max(0.0, (pred_box[2] - pred_box[0])) * max(0.0, (pred_box[3] - pred_box[1]))
        area_b = max(0.0, (gt_box[2] - gt_box[0])) * max(0.0, (gt_box[3] - gt_box[1]))
        union = area_a + area_b - inter

        result.append(inter / union if union > 0 else 0.0)
    return result


if __name__ == "__main__":
    gts = get_gt_by_img()
    preds = get_pred()

    img_ids = set(gts.keys()) | set(preds.keys())
    for img_id in img_ids:
        gt = gts.get(img_id, [])
        pred = preds.get(img_id, [])

        if len(pred) == 0:  # TODO: 테스트 후 모든 db 대상 예측 시 제거
            continue

        iou_list = get_iou_between_gt_pred(pred, gt)
        print(iou_list)

    # f1_scores = evaluate_models(COCO_ANN, MODEL_RESULT_FILES, CONF_THR, IOU_THR)
    # plot_scores(f1_scores)
