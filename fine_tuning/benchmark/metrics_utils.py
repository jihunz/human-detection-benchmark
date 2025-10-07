from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


BASE_WEIGHTS = Path(
    "/Users/jihunjang/wozrkspace/ust/human-detection/fine_tuning/v3/result/train2/weights/best.pt"
)
DEFAULT_DATA_YAML = Path("/Users/jihunjang/workspace/ust/human-detection/fine_tuning/v3/data.yaml")
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.6
DEFAULT_IMGSZ = 640
DEFAULT_DEVICE = "mps"
OUTPUT_ROOT = Path("/Users/jihunjang/workspace/ust/human-detection/fine_tuning/benchmark")


def _run_validation(weights: Path, data_yaml: Path) -> Dict[str, float]:
    model = YOLO(str(weights))
    results = model.val(
        data=str(data_yaml),
        conf=DEFAULT_CONF,
        iou=DEFAULT_IOU,
        imgsz=DEFAULT_IMGSZ,
        device=DEFAULT_DEVICE,
        half=False,
        save=False,
    )
    summary = results.results_dict or {}

    precision = float(summary.get("metrics/precision(B)", 0.0))
    recall = float(summary.get("metrics/recall(B)", 0.0))
    map50 = float(summary.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(summary.get("metrics/mAP50-95(B)", 0.0))
    denom = precision + recall
    f1 = (2.0 * precision * recall / denom) if denom else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map50_95,
    }


def benchmark_pair(finetuned_weights: str | Path, data_yaml: str | Path = DEFAULT_DATA_YAML) -> Dict[str, Dict[str, float]]:
    """Validate baseline and fine-tuned checkpoints on the same dataset."""
    finetuned_path = Path(finetuned_weights).resolve()
    data_path = Path(data_yaml).resolve()
    base_metrics = _run_validation(BASE_WEIGHTS.resolve(), data_path)
    finetuned_metrics = _run_validation(finetuned_path, data_path)
    return {"baseline": base_metrics, "finetuned": finetuned_metrics}


def save_benchmark_report(results: Dict[str, Dict[str, float]]) -> Path:
    """Persist metric summary (JSON + plot) under a timestamped directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["precision", "recall", "f1", "map50", "map50_95"]
    labels = list(results.keys())
    values = np.array([[results[label][metric] for label in labels] for metric in metrics])

    x = np.arange(len(metrics))
    width = 0.35 if len(labels) == 2 else 0.6

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for idx, label in enumerate(labels):
        offset = (idx - (len(labels) - 1) / 2) * width
        ax.bar(x + offset, values[:, idx], width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("YOLO Benchmark")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(run_dir / "metrics.png", dpi=200)
    plt.close(fig)

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    return run_dir


__all__ = ["benchmark_pair", "save_benchmark_report"]
