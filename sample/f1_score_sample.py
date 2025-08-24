import os
import numpy as np
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# sklearn.metrics의 f1-score 메서드를 사용하여 점수를 구함
# f1-score를 구하기 위해 정답 리스트 각 모델들의 예측 결과 배열들의 딕셔너리가 필요. 아래 가정 참고.
# 가정: 각 모델별 예측 결과와 정답이 준비됨

# y_true: 정답 라벨
# y_preds: { "YOLOv8": pred1, "EfficientDet": pred2, ... }

y_true = [0, 1, 0, 2, 1, 2, 0]  # 예시 ground truth
y_preds = {
    "YOLOv8": [0, 1, 0, 2, 1, 2, 1],
    "EfficientDet": [0, 1, 1, 2, 0, 2, 0],
    "Faster-RCNN": [0, 1, 0, 2, 1, 1, 0]
}

f1_scores = {}
for model, pred in y_preds.items(): # y_preds 를 순회하며 정답과 각 모델들의 예측을 이용하여 점수 계산 -> 딕셔너리에 모델별 점수 저장
    f1 = f1_score(y_true, pred, average="macro")  # macro: 클래스별 평균
    f1_scores[model] = f1

# 결과 시각화
plt.figure(figsize=(8, 5))
plt.bar(f1_scores.keys(), f1_scores.values(), color="skyblue")
plt.title("Model-wise F1 Score Comparison")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.show()