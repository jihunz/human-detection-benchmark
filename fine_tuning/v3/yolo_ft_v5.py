from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = '../yolo12n.pt'


def run_train():
    model = YOLO(DEFAULT_WEIGHTS)

    model.train(
        # 훈련
        data='data.yaml',
        imgsz=640,
        epochs=100,
        patience=20,
        batch=8,
        single_cls=False,
        freeze=10, # 백본만 동결 -> 넥, 헤드 학습

        # 하이퍼파라미터
        lr0=1e-3,
        # close_mosaic=5,

        # 컴퓨팅
        device='mps',
        workers=4,
        cache=True,

        # 저장 및 로깅
        project=str(BASE_DIR / 'result'),
        verbose=True

    )


if __name__ == "__main__":
    run_train()
