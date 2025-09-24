"""Benchmark YOLO models on KISA fall/person validation datasets.

Steps
-----
1. Combine fall validation sources into a fresh evaluation dataset under
   `fine_tuning_v2/val/<date>/{fall,fall_base,person}`.
2. Remap fall labels to class 80 for the fine-tuned model and to class 0 for the
   base model.
3. Run Ultralytics `model.val` on both fall/person subsets for
   - fine-tuned weights (expects fall at class id 80)
   - baseline YOLO12n weights (expects standard COCO 80 classes)
4. Store metrics and plots in `fine_tuning_v2/benchmark/<date>`.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
VAL_BASE_DIR = BASE_DIR / "val"
BENCH_BASE_DIR = BASE_DIR / "benchmark"

# source datasets
FALL_SOURCES = [
    {
        "name": "kisa_fall",
        "images": Path("/Users/jihunjang/Downloads/dataset/val/kisa-fall/images"),
        "labels": Path("/Users/jihunjang/Downloads/dataset/val/kisa-fall/labels"),
        "target_class": 80,
    },
    {
        "name": "kisa2_fall",
        "images": Path("/Users/jihunjang/Downloads/dataset/val/kisa-2-fall-only/images"),
        "labels": Path("/Users/jihunjang/Downloads/dataset/val/kisa-2-fall-only/labels"),
        "target_class": 80,
    },
]
PERSON_SOURCE = {
    "name": "kisa_person",
    "images": Path("/Users/jihunjang/Downloads/dataset/val/kisa-person/images"),
    "labels": Path("/Users/jihunjang/Downloads/dataset/val/kisa-person/labels"),
    "target_class": 0,
}

NEW_WEIGHTS = BASE_DIR / "runs/detect/fall_adapter/weights/best.pt"
BASE_WEIGHTS = BASE_DIR / "yolo12n.pt"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LBL_EXT = ".txt"
IMG_SIZE = 640
WORKERS = 2
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DatasetSubset:
    name: str
    root: Path
    images: Path
    labels: Path
    val_txt: Path
    image_count: int


def ensure_unique_dir(base_dir: Path, base_name: str) -> Path:
    target = base_dir / base_name
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)
        return target
    idx = 2
    while True:
        candidate = base_dir / f"{base_name}_{idx:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        idx += 1


def collect_pairs(images_root: Path, labels_root: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    if not images_root.exists() or not labels_root.exists():
        return pairs
    for img in sorted(images_root.rglob("*")):
        if img.is_file() and img.suffix.lower() in IMG_EXTS:
            rel = img.relative_to(images_root)
            lbl = labels_root / rel.with_suffix(LBL_EXT)
            if lbl.exists():
                pairs.append((img, lbl))
    return pairs


def rewrite_label(src: Path, dst: Path, class_id: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    with src.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            parts[0] = str(class_id)
            lines.append(" ".join(parts))
    with dst.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return
    except (FileExistsError, OSError):
        pass
    shutil.copy2(src, dst)


def copy_pairs(
    pairs: Sequence[Tuple[Path, Path]],
    dst_images: Path,
    dst_labels: Path,
    prefix: str,
    class_id: int,
) -> int:
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    count = 0
    for idx, (img, lbl) in enumerate(pairs, 1):
        base_name = f"{prefix}_{idx:04d}_{img.stem}"
        dst_img = dst_images / f"{base_name}{img.suffix.lower()}"
        dst_lbl = dst_labels / f"{base_name}{LBL_EXT}"
        link_or_copy(img, dst_img)
        rewrite_label(lbl, dst_lbl, class_id)
        count += 1
    return count


def write_list_file(images_dir: Path, out_path: Path) -> int:
    image_paths = [p.resolve() for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    image_paths.sort()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for path in image_paths:
            f.write(str(path) + "\n")
    return len(image_paths)


def build_fall_dataset(eval_root: Path) -> Tuple[DatasetSubset, DatasetSubset]:
    fall_root = eval_root / "fall"
    fall_images = fall_root / "images"
    fall_labels = fall_root / "labels"
    fall_root.mkdir(parents=True, exist_ok=True)

    total = 0
    for src in FALL_SOURCES:
        images_root = src["images"]
        labels_root = src["labels"]
        pairs = collect_pairs(images_root, labels_root)
        if not pairs and images_root.joinpath("train").exists():
            pairs = collect_pairs(images_root / "train", labels_root / "train")
        if not pairs:
            print(f"[warn] No pairs found in {images_root}")
            continue
        total += copy_pairs(pairs, fall_images, fall_labels, src["name"], src["target_class"])

    lists_dir = fall_root / "_lists"
    val_txt = lists_dir / "val.txt"
    count = write_list_file(fall_images, val_txt)
    fall_subset = DatasetSubset("fall", fall_root, fall_images, fall_labels, val_txt, count)

    # build base-view with class mapped to 0
    fall_base_root = eval_root / "fall_base"
    base_images = fall_base_root / "images"
    base_labels = fall_base_root / "labels"
    fall_base_root.mkdir(parents=True, exist_ok=True)

    for img in sorted(fall_images.iterdir()):
        if img.suffix.lower() in IMG_EXTS:
            link_or_copy(img, base_images / img.name)
    for lbl in sorted(fall_labels.iterdir()):
        if lbl.suffix.lower() == LBL_EXT:
            rewrite_label(lbl, base_labels / lbl.name, class_id=0)

    base_lists = fall_base_root / "_lists"
    base_val_txt = base_lists / "val.txt"
    write_list_file(base_images, base_val_txt)
    fall_base_subset = DatasetSubset("fall_base", fall_base_root, base_images, base_labels, base_val_txt, count)

    return fall_subset, fall_base_subset


def build_person_dataset(eval_root: Path) -> DatasetSubset:
    person_root = eval_root / "person"
    person_images = person_root / "images"
    person_labels = person_root / "labels"
    person_root.mkdir(parents=True, exist_ok=True)

    src = PERSON_SOURCE
    pairs = collect_pairs(src["images"], src["labels"])
    total = copy_pairs(pairs, person_images, person_labels, src["name"], src["target_class"])

    lists_dir = person_root / "_lists"
    val_txt = lists_dir / "val.txt"
    count = write_list_file(person_images, val_txt)
    return DatasetSubset("person", person_root, person_images, person_labels, val_txt, count)


def prepare_eval_dataset() -> Tuple[Path, Dict[str, DatasetSubset]]:
    date_str = datetime.now().strftime("%Y%m%d")
    eval_root = ensure_unique_dir(VAL_BASE_DIR, date_str)
    fall_subset, fall_base_subset = build_fall_dataset(eval_root)
    person_subset = build_person_dataset(eval_root)
    subsets = {
        "fall": fall_subset,
        "fall_base": fall_base_subset,
        "person": person_subset,
    }
    return eval_root, subsets


def make_data_yaml(subset: DatasetSubset, names_map: Dict[int, str], tag: str) -> Path:
    num_classes = max(names_map.keys()) + 1 if names_map else 0
    data = {
        "path": str(subset.root.resolve()),
        "train": str(subset.val_txt.resolve()),
        "val": str(subset.val_txt.resolve()),
        "names": {i: names_map.get(i, f"class_{i}") for i in range(num_classes)},
    }
    yaml_path = subset.root / f"data.{tag}.yaml"
    import yaml  # local import to avoid dependency at top level if unused

    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=True)
    return yaml_path


def evaluate_model(model_tag: str, weights: Path, subset_keys: Dict[str, str], subsets: Dict[str, DatasetSubset], bench_dir: Path) -> Dict[str, Dict[str, float]]:
    assert weights.exists(), f"Weights not found: {weights}"
    model = YOLO(str(weights))
    names = model.model.names if hasattr(model, "model") else model.names
    if isinstance(names, dict):
        names_map = {i: names[i] for i in range(len(names))}
    else:
        names_map = {i: name for i, name in enumerate(names)}

    metrics_per_subset: Dict[str, Dict[str, float]] = {}
    for logical_name, subset_key in subset_keys.items():
        subset = subsets[subset_key]
        data_yaml = make_data_yaml(subset, names_map, f"{model_tag}_{subset.name}")
        metrics = model.val(
            data=str(data_yaml),
            imgsz=IMG_SIZE,
            device=DEVICE,
            workers=WORKERS,
            verbose=False,
            plots=False,
        )
        box_metrics = getattr(metrics, "box", None)
        mp = float(getattr(box_metrics, "mp", 0.0))
        mr = float(getattr(box_metrics, "mr", 0.0))
        map50 = float(getattr(box_metrics, "map50", 0.0))
        f1 = 0.0 if (mp + mr) == 0 else 2 * mp * mr / (mp + mr)
        metrics_per_subset[logical_name] = {
            "precision": mp,
            "recall": mr,
            "f1": f1,
            "mAP50": map50,
        }
        print(
            f"[{model_tag}] {logical_name}: P={mp:.4f} R={mr:.4f} F1={f1:.4f} mAP50={map50:.4f}"
        )
    return metrics_per_subset


def prepare_benchmark_dir(base_name: str) -> Path:
    bench_dir = ensure_unique_dir(BENCH_BASE_DIR, base_name)
    return bench_dir


def plot_results(results: Dict[str, Dict[str, Dict[str, float]]], bench_dir: Path) -> Path:
    subsets = sorted({subset for model_res in results.values() for subset in model_res.keys()})
    metrics = ["precision", "recall", "f1", "mAP50"]
    model_tags = list(results.keys())

    fig, axes = plt.subplots(len(subsets), 1, figsize=(8.0, 3.2 * len(subsets)), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    x = np.arange(len(metrics))
    width = 0.35

    for ax, subset in zip(axes, subsets):
        for idx, model_tag in enumerate(model_tags):
            values = [results[model_tag].get(subset, {}).get(metric, 0.0) for metric in metrics]
            offset = width * (idx - (len(model_tags) - 1) / 2)
            ax.bar(x + offset, values, width=width, label=model_tag)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(f"Subset: {subset}")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend()

    plot_path = bench_dir / "benchmark_comparison.png"
    fig.suptitle("KISA Validation Benchmark", fontsize=14, fontweight="bold")
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved comparison to {plot_path}")
    return plot_path


def main() -> None:
    eval_root, subsets = prepare_eval_dataset()
    date_tag = eval_root.name
    bench_dir = prepare_benchmark_dir(date_tag)

    print(f"[data] Evaluation dataset prepared at {eval_root}")
    print(f"[bench] Results will be stored in {bench_dir}")

    models = {
        "base_yolo12n": {
            "weights": BASE_WEIGHTS,
            "subsets": {"fall": "fall_base", "person": "person"},
        },
        "fall_adapter": {
            "weights": NEW_WEIGHTS,
            "subsets": {"fall": "fall", "person": "person"},
        },
    }

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for tag, cfg in models.items():
        metrics = evaluate_model(tag, cfg["weights"], cfg["subsets"], subsets, bench_dir)
        all_results[tag] = metrics

    plot_path = plot_results(all_results, bench_dir)

    payload = {
        "eval_root": str(eval_root.resolve()),
        "benchmark_dir": str(bench_dir.resolve()),
        "models": {
            tag: {
                "weights": str(cfg["weights"]),
                "metrics": all_results[tag],
            }
            for tag, cfg in models.items()
        },
        "subset_sizes": {name: subset.image_count for name, subset in subsets.items()},
        "plot": str(plot_path.resolve()),
    }

    metrics_path = bench_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[done] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
