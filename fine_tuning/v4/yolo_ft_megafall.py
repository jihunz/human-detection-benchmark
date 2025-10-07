from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = '../yolo12n.pt'
DATA_CFG = BASE_DIR / 'data.yaml'


def run_train() -> None:
    model = YOLO(DEFAULT_WEIGHTS)

    model.train(
        data=str(DATA_CFG),
        imgsz=640,
        epochs=100,
        patience=20,
        batch=8,
        single_cls=False,
        freeze=10,
        lr0=1e-3,
        device='mps',
        workers=4,
        cache=True,
        project=str(BASE_DIR / 'result'),
        verbose=True,
    )


if __name__ == '__main__':
    run_train()
