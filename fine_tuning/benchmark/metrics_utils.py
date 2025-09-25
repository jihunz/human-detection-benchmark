from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO


def benchmark_detector(
    weights: str | Path,
    data_yaml: str | Path,
    conf: float = 0.25,
    iou: float = 0.6,
    imgsz: int = 640,
    device: Optional[str] = None,
    half: bool = False,
) -> Dict[str, float]:
    """Run YOLO validation and return core metrics (Precision/Recall/mAP/F1).

    Parameters
    ----------
    weights: Path to the fine-tuned `.pt` file or Ultralytics alias.
    data_yaml: Ultralytics data config describing val split.
    conf, iou: Confidence and IoU thresholds for evaluation.
    imgsz: Input resolution.
    device: Optional device string (e.g. ``"0"``, ``"cuda:0"``, ``"mps"``).
    half: Whether to enable half precision during validation.

    Returns
    -------
    dict
        Metric summary with keys ``precision``, ``recall``, ``f1``, ``map50`` and ``map50_95``.
    """

    model = YOLO(str(weights))
    val_kwargs: Dict[str, Any] = {
        "data": str(data_yaml),
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "half": half,
        "save": False,
    }
    if device is not None:
        val_kwargs["device"] = device

    results = model.val(**val_kwargs)
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


__all__ = ["benchmark_detector"]

