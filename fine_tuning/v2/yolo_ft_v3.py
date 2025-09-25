"""Fine-tune YOLO12n on fall incidents while preserving person detection.

- Consolidates fall datasets under `train/class_fall`.
- Creates train/val splits and Ultralytics-compatible data.yaml with a new `fall` class (id 80).
- Launches YOLOv8 training following the official Ultralytics API.
"""
from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
import yaml
from ultralytics import YOLO

# ================= user configuration =================
VAL_RATIO = 0.2
RANDOM_SEED = 42
MAX_DATASETS: int | None = None  # optionally limit datasets used for quick experiments

# training hyper-parameters (see Ultralytics YOLO docs)
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 8
NUM_WORKERS = 4
LEARNING_RATE = 0.01  # maps to lr0 in YOLO.train
MOMENTUM = 0.937
WEIGHT_DECAY = 5e-4
PATIENCE = 30
FREEZE_LAYERS = 10  # freeze backbone layers to retain person knowledge
PROJECT_NAME = "fall_finetune"
# ======================================================

BASE_DIR = Path(__file__).resolve().parent
TRAIN_ROOT = BASE_DIR / "train"
FALL_DATA_ROOT = TRAIN_ROOT / "class_fall"
MERGED_ROOT = TRAIN_ROOT / "_fall_finetune_dataset"
LISTS_DIR = MERGED_ROOT / "_lists"
DEFAULT_WEIGHTS = BASE_DIR / "yolo12n.pt"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LBL_EXT = ".txt"

FALL_CLASS_NAME = "fall"

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)


@dataclass
class ImageLabelPair:
    image: Path
    label: Path
    dataset: str


def discover_fall_datasets(root: Path) -> List[Path]:
    datasets = []
    if not root.is_dir():
        raise FileNotFoundError(f"Fall dataset root not found: {root}")
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "images").is_dir() and (entry / "labels").is_dir():
            datasets.append(entry)
    if MAX_DATASETS is not None:
        datasets = datasets[:MAX_DATASETS]
    if not datasets:
        raise RuntimeError(f"No fall datasets discovered in {root}")
    return datasets


def gather_pairs(dataset_dir: Path) -> List[ImageLabelPair]:
    pairs: List[ImageLabelPair] = []
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    for img in sorted(images_dir.rglob("*")):
        if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
            continue
        rel = img.relative_to(images_dir)
        lbl = labels_dir / rel.with_suffix(LBL_EXT)
        if not lbl.exists():
            print(f"[warn] Missing label for {rel} in {dataset_dir.name}; skipping")
            continue
        pairs.append(ImageLabelPair(img, lbl, dataset_dir.name))
    return pairs


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


def merge_datasets(pairs: Sequence[ImageLabelPair]) -> Tuple[List[Path], List[Path]]:
    if MERGED_ROOT.exists():
        shutil.rmtree(MERGED_ROOT)
    images_dir = MERGED_ROOT / "images"
    labels_dir = MERGED_ROOT / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    merged_images: List[Path] = []
    merged_labels: List[Path] = []
    for pair in pairs:
        new_name = f"{pair.dataset}_{pair.image.stem}{pair.image.suffix.lower()}"
        dst_img = images_dir / new_name
        dst_lbl = labels_dir / (Path(new_name).stem + LBL_EXT)
        safe_link_or_copy(pair.image, dst_img)
        shutil.copy2(pair.label, dst_lbl)
        merged_images.append(dst_img)
        merged_labels.append(dst_lbl)
    return merged_images, merged_labels


def train_val_split(images: Sequence[Path], val_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    rand = random.Random(seed)
    shuffled = list(images)
    rand.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio)) if shuffled else 0
    val_set = set(shuffled[:val_count])
    train_imgs: List[Path] = []
    val_imgs: List[Path] = []
    for img in shuffled:
        (val_imgs if img in val_set else train_imgs).append(img)
    return train_imgs, val_imgs


def write_list_file(paths: Iterable[Path], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as fh:
        for p in sorted(paths):
            fh.write(str(p.resolve()) + "\n")


def resolve_class_info(weights: Path | str) -> Tuple[int, List[str]]:
    model = YOLO(str(weights))
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        base_names = [names[i] for i in range(len(names))]
    elif isinstance(names, list):
        base_names = list(names)
    else:
        raise ValueError("Unable to resolve class names from YOLO weights")
    fall_class_id = len(base_names)
    extended_names = base_names + [FALL_CLASS_NAME]
    return fall_class_id, extended_names


def build_data_yaml(train_txt: Path, val_txt: Path, class_names: List[str]) -> Path:
    data = {
        "path": str(MERGED_ROOT.resolve()),
        "train": str(train_txt.resolve()),
        "val": str(val_txt.resolve()),
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    yaml_path = MERGED_ROOT / "data.fall.yaml"
    with yaml_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return yaml_path


def pick_weights() -> Path | str:
    if DEFAULT_WEIGHTS.is_file():
        print(f"[weights] Using local checkpoint: {DEFAULT_WEIGHTS}")
        return DEFAULT_WEIGHTS
    alias = "yolo12n.pt"
    print(f"[weights] Local checkpoint missing; falling back to Ultralytics alias '{alias}'")
    return alias


def check_labels(labels: Sequence[Path], expected_class_id: int) -> None:
    for lbl in labels:
        with lbl.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    cls_id = int(float(parts[0]))
                except (ValueError, IndexError):
                    raise ValueError(f"Malformed label line in {lbl}: '{line}'")
                if cls_id != expected_class_id:
                    raise ValueError(
                        f"Unexpected class id {cls_id} in {lbl}; expected only {expected_class_id} (fall)"
                    )


def launch_training(weights: Path | str, data_yaml: Path, class_names: List[str]) -> None:
    model = YOLO(str(weights))
    print(f"[train] Starting fine-tuning on device={DEVICE}")
    freeze_layers = list(range(FREEZE_LAYERS)) if FREEZE_LAYERS else 0

    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=NUM_WORKERS,
        lr0=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        project=str(BASE_DIR / "runs" / "detect"),
        name=PROJECT_NAME,
        exist_ok=False,
        cache=True,
        close_mosaic=10,
        single_cls=False,
        freeze=freeze_layers,
        verbose=True,
    )
    print("[train] Training complete. Review metrics in ./runs/detect")


def main() -> None:
    datasets = discover_fall_datasets(FALL_DATA_ROOT)
    print(f"[data] Found {len(datasets)} fall datasets under {FALL_DATA_ROOT}")

    all_pairs: List[ImageLabelPair] = []
    for ds in datasets:
        ds_pairs = gather_pairs(ds)
        print(f"[data] {ds.name}: {len(ds_pairs)} labeled samples")
        all_pairs.extend(ds_pairs)
    if not all_pairs:
        raise RuntimeError("No labeled fall samples collected")

    merged_images, merged_labels = merge_datasets(all_pairs)
    print(f"[data] Merged dataset created at {MERGED_ROOT} with {len(merged_images)} images")

    weights = pick_weights()
    fall_class_id, class_names = resolve_class_info(weights)
    print(f"[info] Base model has {fall_class_id} classes; new class '{FALL_CLASS_NAME}' -> id {fall_class_id}")

    check_labels(merged_labels, fall_class_id)

    train_imgs, val_imgs = train_val_split(merged_images, VAL_RATIO, RANDOM_SEED)
    print(f"[split] train={len(train_imgs)} | val={len(val_imgs)}")

    train_txt = LISTS_DIR / "train.txt"
    val_txt = LISTS_DIR / "val.txt"
    write_list_file(train_imgs, train_txt)
    write_list_file(val_imgs, val_txt)

    data_yaml = build_data_yaml(train_txt, val_txt, class_names)
    print(f"[yaml] Data config saved to {data_yaml}")

    launch_training(weights, data_yaml, class_names)


if __name__ == "__main__":
    main()
