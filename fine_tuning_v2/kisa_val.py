"""Evaluate YOLO12n baseline and fine-tuned weights on KISA val subsets.

Two subsets are evaluated separately:
  - `val/person`: images representing normal human presence
  - `val/fall`: images representing fall events

For each subset the script rebuilds a temporary YOLO-style dataset, runs
`model.val()` for both baseline YOLO12n and the fine-tuned checkpoint, prints
metrics, and produces professional visualizations comparing Precision/Recall/F1
across subsets and models.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ---------------------------
# Configuration
# ---------------------------
BASE_DIR = Path("/Users/jihunjang/workspace/ust/human-detection/fine_tuning_v2")
DATASET_ROOT = BASE_DIR / "val"
TMP_EVAL_ROOT = DATASET_ROOT / "_eval_dataset"
PLOT_PATH = TMP_EVAL_ROOT / "kisa_eval_metrics.png"
BASELINE_WEIGHTS = "yolo12n.pt"
FINETUNED_WEIGHTS = BASE_DIR / "runs/detect/train/weights/best.pt"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
IMG_SIZE = 640
WORKERS = 2
MODEL_NAMES = ["YOLO12n (base)", "YOLO12n (finetuned)"]
SUBSETS = {
    "person": DATASET_ROOT / "person",
    "fall": DATASET_ROOT / "fall",
}

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)


@dataclass
class Sample:
    image: Path
    label: Path


def collect_samples(root: Path) -> List[Sample]:
    """Collect image/label pairs under a subset directory."""
    if not root.is_dir():
        return []
    samples: List[Sample] = []
    for img in sorted(root.rglob("*")):
        if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
            continue
        lbl = img.with_suffix(".txt")
        if not lbl.exists():
            print(f"[warn] Missing label for {img}")
            continue
        samples.append(Sample(image=img, label=lbl))
    return samples


def safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return
    except Exception:
        pass
    try:
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src)
        return
    except Exception:
        pass
    shutil.copy2(src, dst)


def rebuild_eval_dataset(samples: Iterable[Sample], out_root: Path) -> List[Path]:
    if out_root.exists():
        shutil.rmtree(out_root)
    images_dir = out_root / "images"
    labels_dir = out_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    collected: List[Path] = []
    for sample in samples:
        dst_img = images_dir / sample.image.name
        dst_lbl = labels_dir / sample.label.name
        safe_link_or_copy(sample.image, dst_img)
        shutil.copy2(sample.label, dst_lbl)
        collected.append(dst_img)
    return collected


def write_list_file(paths: Iterable[Path], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as f:
        for p in sorted(paths):
            f.write(str(p.resolve()) + "\n")


def make_data_yaml(root: Path, list_file: Path) -> Path:
    data = {
        "path": str(root.resolve()),
        "train": str(list_file.resolve()),
        "val": str(list_file.resolve()),
        "names": {0: "person"},
    }
    out = root / "data.auto.yaml"
    with out.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out


def f1_score(precision: float, recall: float) -> float:
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


def evaluate(weights: Path | str, data_yaml: Path) -> Dict[str, float]:
    print(f"\n[Eval] weights = {weights}")
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        imgsz=IMG_SIZE,
        device=DEVICE,
        workers=WORKERS,
        verbose=False,
        plots=False,
    )
    mp = float(getattr(metrics.box, "mp", 0.0))
    mr = float(getattr(metrics.box, "mr", 0.0))
    map50 = float(getattr(metrics.box, "map50", 0.0))
    map5095 = float(getattr(metrics.box, "map", 0.0))
    f1 = f1_score(mp, mr)
    print(
        f"  Precision: {mp:.4f} | Recall: {mr:.4f} | F1: {f1:.4f} | "
        f"mAP50: {map50:.4f} | mAP50-95: {map5095:.4f}"
    )
    return {
        "precision": mp,
        "recall": mr,
        "f1": f1,
        "map50": map50,
        "map5095": map5095,
    }


def visualize_results(results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    if not results:
        return

    TMP_EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    subsets = list(results.keys())
    metrics_to_plot = ["precision", "recall", "f1", "map50"]
    metric_titles = {
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "map50": "mAP@0.50",
    }
    colors = {
        MODEL_NAMES[0]: "#4c72b0",
        MODEL_NAMES[1]: "#dd8452",
    }

    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "figure.dpi": 160,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes = axes.flatten()

    x = np.arange(len(subsets))
    bar_width = 0.35

    for ax, metric in zip(axes, metrics_to_plot):
        for idx, model_name in enumerate(MODEL_NAMES):
            offsets = x + (idx - (len(MODEL_NAMES) - 1) / 2) * bar_width
            values = [results[subset][model_name][metric] for subset in subsets]
            ax.bar(
                offsets,
                values,
                width=bar_width,
                label=model_name,
                color=colors[model_name],
            )
            for ox, val in zip(offsets, values):
                ax.text(ox, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(subsets, rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(metric_titles.get(metric, metric.capitalize()))
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    fig.suptitle("KISA Validation Metrics", fontsize=16, fontweight="bold")
    fig.legend(MODEL_NAMES, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=len(MODEL_NAMES))
    fig.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved metric comparison to {PLOT_PATH}")


def summarize(results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    if not results:
        print("\n[Summary] No subsets were evaluated.")
        return
    print("\n[Summary]")
    for subset, model_metrics in results.items():
        print(f"Subset: {subset}")
        for model_name in MODEL_NAMES:
            metrics = model_metrics[model_name]
            print(
                f"  {model_name:<18} -> F1: {metrics['f1']:.4f} | "
                f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | "
                f"mAP50: {metrics['map50']:.4f}"
            )
    all_pairs = [
        (subset, model, metrics["f1"])
        for subset, subset_metrics in results.items()
        for model, metrics in subset_metrics.items()
    ]
    if all_pairs:
        best_subset, best_model, best_f1 = max(all_pairs, key=lambda x: x[2])
        print(f"\nBest F1: {best_model} on {best_subset} ({best_f1:.4f})")


def main() -> None:
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for subset_name, subset_path in SUBSETS.items():
        samples = collect_samples(subset_path)
        if not samples:
            print(f"[skip] No labeled samples found for subset '{subset_name}'")
            continue
        print(f"\n[Build] Subset '{subset_name}': {len(samples)} labeled images")

        subset_eval_root = TMP_EVAL_ROOT / subset_name
        img_paths = rebuild_eval_dataset(samples, subset_eval_root)
        lists_dir = subset_eval_root / "_lists"
        val_txt = lists_dir / "val.txt"
        write_list_file(img_paths, val_txt)
        data_yaml = make_data_yaml(subset_eval_root, val_txt)
        print(f"[YAML] {data_yaml}")

        subset_results: Dict[str, Dict[str, float]] = {}
        subset_results[MODEL_NAMES[0]] = evaluate(BASELINE_WEIGHTS, data_yaml)
        assert FINETUNED_WEIGHTS.exists(), f"Fine-tuned weights not found: {FINETUNED_WEIGHTS}"
        subset_results[MODEL_NAMES[1]] = evaluate(FINETUNED_WEIGHTS, data_yaml)
        results[subset_name] = subset_results

    summarize(results)
    visualize_results(results)


if __name__ == "__main__":
    main()
