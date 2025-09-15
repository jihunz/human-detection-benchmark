# train_yolo_mps_localpt.py
# ------------------------------------------------------------
# - 영상별 폴더(C00xxxx_xxx/ ...) 하위의 images/labels 구조를 자동 스캔
# - 라벨(.txt) 기준으로 동명 이미지 매칭 → train/val 리스트 작성
# - data.yaml 자동 생성
# - Mac MPS(device='mps')로, yolo12n 가중치 자동 다운로드 후 파인튜닝
#   * 원본 .pt는 덮어쓰지 않으며, 결과는 runs/detect/train*/weights/ 로 저장
# ------------------------------------------------------------

import os
import sys
from pathlib import Path
import random
import yaml
import torch

from ultralytics import YOLO

# ========== [사용자 설정] ==========
DATASET_ROOT = Path("/Users/jihunjang/Downloads/kisadb-c-fall-label")

# 로컬 .pt를 쓰고 싶으면 경로를 넣고, 비워두거나 존재하지 않으면 yolo12n을 자동 다운로드/사용
CKPT_PATH = Path("")                 # 예: Path("/Users/you/weights/yolo12n.pt")
CKPT_ALIAS = "yolo12n.pt"            # 자동 다운로드 트리거

VAL_RATIO = 0.1
RANDOM_SEED = 42
IMG_SIZE = 640
EPOCHS = 100
BATCH_SIZE = 8
NUM_WORKERS = 2
CLASS_NAMES = ["person"]
# ===================================

# MPS 환경 (없으면 CPU)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LBL_EXT = ".txt"


def find_image_for_label(lbl_path: Path) -> Path | None:
    """라벨(.txt)과 동명 이미지 탐색."""
    stem = lbl_path.stem

    # 1) 같은 폴더
    for ext in IMG_EXTS:
        cand = lbl_path.with_suffix(ext)
        if cand.exists():
            return cand

    # 2) labels → images 치환
    parts = list(lbl_path.parts)
    if "labels" in parts:
        i = parts.index("labels")
        for ext in IMG_EXTS:
            cand = Path(*parts[:i], "images", *parts[i + 1:]).with_suffix(ext)
            if cand.exists():
                return cand

    # 3) 부모/동일 폴더 탐색
    parent = lbl_path.parent
    for ext in IMG_EXTS:
        cand = parent / f"{stem}{ext}"
        if cand.exists():
            return cand

    return None


def collect_pairs(dataset_root: Path) -> list[tuple[Path, Path, str]]:
    """(img_path, lbl_path, top_video_dir) 튜플 수집."""
    label_txts = list(dataset_root.glob("**/labels/train/*.txt"))
    if not label_txts:
        label_txts = list(dataset_root.glob("**/labels/**/*.txt"))

    pairs = []
    root_resolved = dataset_root.resolve()
    for lbl in label_txts:
        if lbl.suffix.lower() != LBL_EXT:
            continue
        img = find_image_for_label(lbl)
        if not img or not img.exists():
            continue
        try:
            rel = img.resolve().relative_to(root_resolved)
            top_dir = rel.parts[0]  # 예: C001100_003
        except Exception:
            top_dir = img.parent.name
        pairs.append((img, lbl, top_dir))
    return pairs


def split_by_topdir(pairs: list[tuple[Path, Path, str]], val_ratio: float, seed: int):
    """비디오(상위 폴더) 단위 분할 → 누수 방지."""
    random.seed(seed)
    from collections import defaultdict

    groups = defaultdict(list)
    for img, lbl, top in pairs:
        groups[top].append((img, lbl))

    tops = list(groups.keys())
    random.shuffle(tops)

    val_count = max(1, int(len(tops) * val_ratio))
    val_tops = set(tops[:val_count])

    train_list, val_list = [], []
    for top, items in groups.items():
        (val_list if top in val_tops else train_list).extend(items)
    return train_list, val_list


def write_list_file(paths: list[Path], out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as f:
        for p in paths:
            f.write(str(p.resolve()) + "\n")  # 절대경로 저장


def make_data_yaml(root: Path, train_txt: Path, val_txt: Path, class_names: list[str]) -> Path:
    """Ultralytics용 data.yaml 생성 (리스트 파일 사용)."""
    data = {
        "path": str(root.resolve()),
        "train": str(train_txt.resolve()),
        "val": str(val_txt.resolve()),
        "names": {i: n for i, n in enumerate(class_names)},
    }
    out = root / "data.auto.yaml"
    with out.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out


def resolve_ckpt() -> str:
    """로컬 CKPT가 있으면 그걸, 없으면 yolo12n(alias)을 사용(자동 다운로드)."""
    if isinstance(CKPT_PATH, Path) and CKPT_PATH.exists() and CKPT_PATH.is_file():
        print(f"[weights] Using local checkpoint: {CKPT_PATH}")
        return str(CKPT_PATH)
    print(f"[weights] Local ckpt not provided/found → using alias '{CKPT_ALIAS}' (auto-download)")
    return CKPT_ALIAS


def main():
    assert DATASET_ROOT.exists(), f"Dataset root not found: {DATASET_ROOT}"

    print(f"[1/5] 스캔 시작: {DATASET_ROOT}")
    pairs = collect_pairs(DATASET_ROOT)
    if not pairs:
        print("라벨(.txt)과 이미지 페어를 찾지 못했습니다. 폴더 구조/확장자를 확인하세요.")
        sys.exit(1)
    print(f"  발견된 페어 수: {len(pairs)}")

    print("[2/5] 비디오 폴더 단위로 train/val 분할...")
    train_pairs, val_pairs = split_by_topdir(pairs, VAL_RATIO, RANDOM_SEED)
    print(f"  train imgs: {len(train_pairs)} | val imgs: {len(val_pairs)}")

    print("[3/5] 이미지 리스트 파일 작성(train/val)...")
    train_imgs = [img for img, _lbl in train_pairs]
    val_imgs = [img for img, _lbl in val_pairs]
    out_dir = DATASET_ROOT / "_lists"
    train_txt = out_dir / "train.txt"
    val_txt = out_dir / "val.txt"
    write_list_file(train_imgs, train_txt)
    write_list_file(val_imgs, val_txt)
    print(f"  - {train_txt}")
    print(f"  - {val_txt}")

    print("[4/5] data.yaml 생성...")
    data_yaml = make_data_yaml(DATASET_ROOT, train_txt, val_txt, CLASS_NAMES)
    print(f"  - {data_yaml}")

    ckpt_to_use = resolve_ckpt()

    print(f"[5/5] 학습 시작 | device={DEVICE} | ckpt={ckpt_to_use}")
    try:
        model = YOLO(ckpt_to_use)  # 없으면 yolo12n.pt 자동 다운로드
    except Exception as e:
        print("[ERROR] YOLO 가중치 로드 실패:", e)
        print("        pip install -U ultralytics 로 업데이트 후 다시 시도하세요.")
        sys.exit(1)

    # MPS 주의: AMP/half 미지원 → amp=False 권장
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,     # 'mps' 또는 'cpu'
        workers=NUM_WORKERS,
        optimizer="SGD",   # 또는 'AdamW'
        lr0=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        amp=False,
        patience=20,
        cos_lr=True,
        cache=False
    )

    metrics = model.val(
        data=str(data_yaml),
        imgsz=IMG_SIZE,
        device=DEVICE,
        workers=NUM_WORKERS
    )
    print("Validation metrics:", metrics)


if __name__ == "__main__":
    main()