"""Fine-tune YOLO12n with a new fall class while freezing legacy person weights.

This script consolidates fall-only datasets and trains a new class head without
updating pre-trained parameters, so existing classes (e.g., person) keep their
performance. Only the additional fall channel in the detection head receives
gradients during optimisation.
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
MAX_DATASETS: int | None = None  # optionally limit datasets for quick tests

# training hyper-parameters
EPOCHS = 50
IMG_SIZE = 640 #TODO: rect=True로 종횡비 유지해야함
BATCH_SIZE = 8
NUM_WORKERS = 4
LEARNING_RATE = 1e-3  # smaller LR since only a few params update
PATIENCE = 20
PROJECT_NAME = "fall_adapter"
FALL_CLASS_NAME = "fall"
FREEZE_BACKBONE = True  # keep True to preserve legacy classes
FREEZE_NORM_STATS = True
# ======================================================

BASE_DIR = Path(__file__).resolve().parent
TRAIN_ROOT = BASE_DIR / "train"
FALL_DATA_ROOT = TRAIN_ROOT / "class_fall"
MERGED_ROOT = TRAIN_ROOT / "fall_finetune_dataset2"
LISTS_DIR = MERGED_ROOT / "lists"
DEFAULT_WEIGHTS = BASE_DIR / "yolo12n.pt"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LBL_EXT = ".txt"

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)


@dataclass
class ImageLabelPair:
    image: Path
    label: Path
    dataset: str


# ---------------------------------------------------------------------------
# Dataset preparation helpers (identical to v3 script logic)
# ---------------------------------------------------------------------------

def discover_fall_datasets(root: Path) -> List[Path]:
    datasets: List[Path] = []
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


def build_data_yaml(train_txt: Path, val_txt: Path, class_names: List[str]) -> Path:
    data = {
        "path": str(MERGED_ROOT.resolve()),
        "train": str(train_txt.resolve()),
        "val": str(val_txt.resolve()),
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    yaml_path = MERGED_ROOT / "data.fall_adapter.yaml"
    with yaml_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return yaml_path


# ---------------------------------------------------------------------------
# Model surgery: add fall class while freezing legacy weights
# ---------------------------------------------------------------------------

def load_pretrained() -> YOLO:
    weights = DEFAULT_WEIGHTS if DEFAULT_WEIGHTS.is_file() else "yolo12n.pt"
    print(f"[weights] Loading base checkpoint: {weights}")
    model = YOLO(str(weights))
    return model


def expand_detection_head_for_fall(model: YOLO, fall_name: str) -> Tuple[int, int, List[torch.nn.Module]]:
    """Extend YOLO detect head with one extra class channel for fall."""
    net = model.model  # ultralytics.nn.tasks.DetectionModel
    det = net.model[-1]
    assert hasattr(det, "nc"), "Unexpected detect layer structure"

    if fall_name in net.names:
        print(f"[detect] '{fall_name}' already present; skipping expansion")
        return det.nc - 1, det.nc, []

    old_nc: int = det.nc
    new_nc = old_nc + 1
    net.names = list(net.names) + [fall_name]
    net.nc = new_nc
    det.nc = new_nc
    det.no = getattr(det, "reg_max", 16) * 4 + new_nc

    new_classifiers: List[torch.nn.Module] = []
    for seq in det.cv3:  # classification convs per detection level
        classifier = seq[-1]
        if not isinstance(classifier, torch.nn.Conv2d):
            raise TypeError("Unexpected classifier module structure in detect head")

        with torch.no_grad():
            old_w = classifier.weight.data
            old_b = classifier.bias.data
            in_channels = old_w.shape[1]
            new_classifier = torch.nn.Conv2d(
                in_channels,
                new_nc,
                kernel_size=classifier.kernel_size,
                stride=classifier.stride,
                padding=classifier.padding,
                bias=True,
            ).to(old_w.device, dtype=old_w.dtype)

            new_w = new_classifier.weight.data
            new_b = new_classifier.bias.data
            new_w.zero_()
            new_b.zero_()
            new_w[:old_nc] = old_w
            new_b[:old_nc] = old_b
            # initialise fall channel close to person weights (class 0)
            new_w[old_nc] = old_w[0]
            new_b[old_nc] = old_b[0]

            seq[-1] = new_classifier
            new_classifiers.append(new_classifier)

    print(f"[detect] Expanded head: nc {old_nc} -> {new_nc}")
    return old_nc, new_nc, new_classifiers


def _mask_grad_except(param: torch.Tensor, allowed_idx: int):
    def hook(grad: torch.Tensor | None) -> torch.Tensor | None:
        if grad is None:
            return grad
        masked = torch.zeros_like(grad)
        if grad.ndim == 1:
            masked[allowed_idx] = grad[allowed_idx]
        else:
            masked[allowed_idx : allowed_idx + 1] = grad[allowed_idx : allowed_idx + 1]
        return masked

    return param.register_hook(hook)


def freeze_legacy_weights(model: YOLO, old_nc: int, classifiers: Sequence[torch.nn.Module]) -> None:
    net = model.model

    if FREEZE_BACKBONE:
        for _, param in net.named_parameters():
            param.requires_grad = False

    handles = []
    for clf in classifiers:
        for param in clf.parameters():
            param.requires_grad = True

        handles.append(_mask_grad_except(clf.weight, old_nc))
        handles.append(_mask_grad_except(clf.bias, old_nc))

    if FREEZE_NORM_STATS:
        for module in net.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
                module.track_running_stats = False

    print("[freeze] Enabled gradient masking so only fall channel updates")
    net._fall_hooks = handles  # type: ignore[attr-defined]


def prepare_data_yaml(class_names: List[str]) -> Path:
    train_txt = LISTS_DIR / "train.txt"
    val_txt = LISTS_DIR / "val.txt"
    return build_data_yaml(train_txt, val_txt, class_names)


def launch_training(model: YOLO, data_yaml: Path) -> None:
    print(f"[train] Starting adapter training on device={DEVICE}")
    # Keep ultralytics trainer but with tiny LR and early stopping
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=NUM_WORKERS,
        lr0=LEARNING_RATE,
        patience=PATIENCE,
        project=str(BASE_DIR / "runs" / "detect"),
        name=PROJECT_NAME,
        exist_ok=False,
        cache=True,
        close_mosaic=5,
        single_cls=False,
        verbose=True,
    )
    print("[train] Training complete. Review runs/detect for results")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

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

    merged_images, _ = merge_datasets(all_pairs)
    print(f"[data] Merged dataset created at {MERGED_ROOT} with {len(merged_images)} images")

    train_imgs, val_imgs = train_val_split(merged_images, VAL_RATIO, RANDOM_SEED)
    print(f"[split] train={len(train_imgs)} | val={len(val_imgs)}")

    write_list_file(train_imgs, LISTS_DIR / "train.txt")
    write_list_file(val_imgs, LISTS_DIR / "val.txt")

    model = load_pretrained()
    old_nc, new_nc, classifiers = expand_detection_head_for_fall(model, FALL_CLASS_NAME)
    freeze_legacy_weights(model, old_nc, classifiers)

    data_yaml = prepare_data_yaml(model.model.names)
    print(f"[yaml] Data config saved to {data_yaml}")

    launch_training(model, data_yaml)


if __name__ == "__main__":
    main()
