"""Evaluate 파인튜닝 YOLO12n(class_fall 클래스 포함) 모델을 KISA val 서브셋에서 검증.

서브셋별 평가 대상:
  - `val/person`: 라벨 클래스 0 (person)
  - `val/class_fall`: 라벨 클래스 80 (class_fall)

각 서브셋을 단일 클래스(0)로 정규화한 임시 평가 세트를 생성하고,
YOLO12n 기본 가중치와 class_fall 클래스로 파인튜닝된 가중치를 모두 평가한다.
Precision/Recall/F1/mAP 지표를 비교 그래프로 시각화하고 JSON 리포트로 저장한다.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import datetime
import json
from functools import lru_cache
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
BASELINE_WEIGHTS = "yolo12n.pt"
FINETUNED_WEIGHTS = BASE_DIR / "runs/detect/02/train2/weights/last.pt"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
IMG_SIZE = 640
WORKERS = 2
MODEL_NAMES = ["YOLO12n (base)", "YOLO12n (finetuned)"]
BENCHMARK_ROOT = BASE_DIR / "benchmark"
SUBSETS = {
    "person": {
        "path": DATASET_ROOT / "person",
        "src_class": 0,
        "ft_class": 0,
        "base_class": 0,
        "label_name": "person",
    },
    "class_fall": {
        "path": DATASET_ROOT / "class_fall",
        "src_class": 80,
        "ft_class": 80,
        "base_class": 0,
        "label_name": "class_fall",
    },
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



def rewrite_label_single_class(src: Path, dst: Path, src_class: int, target_class: int) -> None:
    """라벨 파일에서 지정 클래스만 남기고 id를 target_class로 치환."""
    lines_out: List[str] = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cls_id = int(float(parts[0]))
            except ValueError:
                continue
            if cls_id != src_class:
                continue
            parts[0] = str(target_class)
            lines_out.append(" ".join(parts))

    dst.parent.mkdir(parents=True, exist_ok=True)
    if not lines_out:
        # 대상 클래스가 없으면 빈 파일로 저장하여 무라벨 처리
        with dst.open("w", encoding="utf-8") as f:
            f.write("")
        return

    with dst.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + "\n")


def rebuild_eval_dataset(samples: Iterable[Sample], out_root: Path, src_class: int, target_class: int) -> List[Path]:
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
        rewrite_label_single_class(sample.label, dst_lbl, src_class, target_class)
        collected.append(dst_img)
    return collected


def write_list_file(paths: Iterable[Path], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as f:
        for p in sorted(paths):
            f.write(str(p.resolve()) + "\n")




@lru_cache(None)
def load_base_names() -> Dict[int, str]:
    model = YOLO(str(BASELINE_WEIGHTS))
    names = getattr(model.model, "names", None)
    if names is None and hasattr(model, "names"):
        names = model.names
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return {i: n for i, n in enumerate(names)}


def make_data_yaml(root: Path, list_file: Path, target_class: int, label_name: str) -> Path:
    base_names = load_base_names().copy()
    for idx in range(target_class + 1):
        base_names.setdefault(idx, f"class_{idx}")
    base_names[target_class] = label_name
    names = {idx: base_names[idx] for idx in range(target_class + 1)}
    data = {
        "path": str(root.resolve()),
        "train": str(list_file.resolve()),
        "val": str(list_file.resolve()),
        "names": names,
    }
    out = root / "data.auto.yaml"
    with out.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out


def prepare_benchmark_dir() -> Path:
    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    target = BENCHMARK_ROOT / date_str
    idx = 1
    while target.exists():
        idx += 1
        target = BENCHMARK_ROOT / f"{date_str}_{idx:02d}"
    target.mkdir(parents=True, exist_ok=True)
    return target


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


def visualize_results(results: Dict[str, Dict[str, Dict[str, float]]], plot_path: Path) -> None:
    if not results:
        return

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
            "figure.dpi": 170,
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
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved metric comparison to {plot_path}")




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

    best_subset, best_model, best_metrics = max(
        (
            (subset, model_name, metrics)
            for subset, subset_metrics in results.items()
            for model_name, metrics in subset_metrics.items()
        ),
        key=lambda item: item[2]["f1"],
    )
    print(f"\nBest F1: {best_model} on {best_subset} ({best_metrics['f1']:.4f})")



def main() -> None:
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for subset_name, subset_cfg in SUBSETS.items():
        subset_path = subset_cfg["path"]
        samples = collect_samples(subset_path)
        if not samples:
            print(f"[skip] No labeled samples found for subset '{subset_name}'")
            continue
        print(f"\n[Build] Subset '{subset_name}': {len(samples)} labeled images")

        subset_eval_root = TMP_EVAL_ROOT / subset_name
        img_paths = rebuild_eval_dataset(samples, subset_eval_root, subset_cfg["src_class"], subset_cfg["ft_class"])
        lists_dir = subset_eval_root / "_lists"
        val_txt = lists_dir / "val.txt"
        write_list_file(img_paths, val_txt)
        data_yaml_ft = make_data_yaml(subset_eval_root, val_txt, subset_cfg["ft_class"], subset_cfg["label_name"])
        print(f"[YAML] {data_yaml_ft}")

        if subset_name == "class_fall":
            base_eval_root = TMP_EVAL_ROOT / "class_fall-yolo12-base"
            base_img_paths = rebuild_eval_dataset(samples, base_eval_root, subset_cfg["src_class"], subset_cfg["base_class"])
            base_lists_dir = base_eval_root / "_lists"
            base_val_txt = base_lists_dir / "val.txt"
            write_list_file(base_img_paths, base_val_txt)
            data_yaml_base = make_data_yaml(base_eval_root, base_val_txt, subset_cfg["base_class"], "person")
        else:
            data_yaml_base = data_yaml_ft

        subset_results: Dict[str, Dict[str, float]] = {}
        subset_results[MODEL_NAMES[0]] = evaluate(BASELINE_WEIGHTS, data_yaml_base)
        assert FINETUNED_WEIGHTS.exists(), f"Fine-tuned weights not found: {FINETUNED_WEIGHTS}"
        subset_results[MODEL_NAMES[1]] = evaluate(FINETUNED_WEIGHTS, data_yaml_ft)
        results[subset_name] = subset_results

    summarize(results)

    bench_dir = prepare_benchmark_dir()
    plot_path = bench_dir / "kisa_eval_metrics.png"
    visualize_results(results, plot_path)

    metrics_path = bench_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"models": MODEL_NAMES, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[Benchmark] Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
