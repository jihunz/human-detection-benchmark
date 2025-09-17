# eval_caucafall_fallonly_person_f1.py
# ------------------------------------------------------------
# - CAUCAFall/Subject.1~10/** 중 폴더명에 'Fall' 포함된 폴더만 사용
# - 각 폴더 내부의 *.png ↔ 동명 *.txt 라벨만 수집
# - 라벨 변환: 첫 토큰(class id) 0↔1 스왑, 스왑 이후 class==0(=person)만 남김
# - 원본 파일 보존: 임시 평가 폴더에 이미지 링크(가능하면 하드링크/불가시 복사) + 변환 라벨 생성
# - data.yaml/val.txt 자동 생성 후, yolo12n(베이스)와 파인튜닝 가중치 둘 다 F1 비교/시각화
# ------------------------------------------------------------

import os
import shutil
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ====== 사용자 입력 ======
DATASET_ROOT = Path('/Users/jihunjang/Downloads/Dataset CAUCAFall/CAUCAFall')           # 예: "/Users/me/datasets/CAUCAFall"
BASELINE_W   = "yolo12n.pt"                          # 베이스(없으면 자동 다운로드)
FINETUNED_W  = Path('/fine_tuning_v1/runs/detect/train/weights/best.pt')

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
TMP_EVAL_ROOT = DATASET_ROOT / "_eval_fall_person_swapped"  # 임시 평가셋 생성 위치
IMG_SIZE = 768          # 평가 해상도(학습/배포 해상도와 맞추면 공정)
WORKERS = 2             # macOS/MPS 권장
# ========================

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def iter_fall_images(root: Path):
    """
    CAUCAFall/Subject.1~10/** 중 폴더명에 'Fall'이 들어간 디렉터리만 탐색하여
    (img_path, lbl_path or None) 튜플 스트림을 생성.
    """
    assert root.exists(), f"Dataset root not found: {root}"
    for subj in sorted(root.glob("Subject.*")):
        if not subj.is_dir():
            continue
        for sub in sorted(subj.rglob("*")):
            if not sub.is_dir():
                continue
            if "class_fall" not in sub.name.lower():   # 'Fall' 포함 폴더만
                continue
            for img in sorted(sub.glob("*")):
                if img.suffix.lower() in IMG_EXTS:
                    lbl = img.with_suffix(".txt")
                    yield img, (lbl if lbl.exists() else None)


def safe_link_or_copy(src: Path, dst: Path):
    """가능하면 하드링크 → 안되면 심볼릭링크 → 그것도 안되면 복사."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)  # 동일 파일시스템에서 빠르고 공간 절약
    except Exception:
        try:
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src)
        except Exception:
            shutil.copy2(src, dst)


def swap_and_keep_person_only(src_lbl: Path, dst_lbl: Path):
    """
    YOLO 라벨 파일에서 class id 0<->1 스왑 후, 스왑 결과 class==0(=person) 라인만 남겨 저장.
    다른 클래스(0/1 외)는 그대로 두지 않음(단일 클래스 평가 목적).
    """
    lines_out = []
    with src_lbl.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            cls = parts[0]
            # 0 <-> 1 스왑
            if cls == "0":
                cls_swapped = "1"
            elif cls == "1":
                cls_swapped = "0"
            else:
                cls_swapped = cls  # 0/1 외엔 그대로

            # 스왑 이후 class==0(=person)만 유지
            if cls_swapped == "0":
                parts[0] = "0"
                lines_out.append(" ".join(parts))

    if lines_out:
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        with dst_lbl.open("w") as f:
            f.write("\n".join(lines_out) + "\n")
    else:
        # 사람(=class0) 라벨이 하나도 없으면 라벨 파일 생성 안 함(무라벨 이미지로 처리)
        if dst_lbl.exists():
            dst_lbl.unlink(missing_ok=True)


def build_eval_set(src_root: Path, out_root: Path):
    """
    'Fall' 폴더만 모아 임시 평가셋(out_root)에 이미지 링크 + 변환 라벨 생성.
    out_root 구조는 원본과 동일한 상대경로를 유지.
    반환: 생성된 모든 이미지 경로 리스트
    """
    if out_root.exists():
        shutil.rmtree(out_root)
    created_imgs = []

    for img, lbl in iter_fall_images(src_root):
        rel = img.relative_to(src_root)
        out_img = out_root / rel
        safe_link_or_copy(img, out_img)

        if lbl is not None:
            out_lbl = out_img.with_suffix(".txt")
            swap_and_keep_person_only(lbl, out_lbl)
        # 라벨이 없거나 필터링 후 비었으면 .txt 미생성 → 무라벨 이미지로 처리
        created_imgs.append(out_img)

    return created_imgs


def write_val_list(imgs, out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as f:
        for p in sorted(imgs):
            f.write(str(p.resolve()) + "\n")


def make_data_yaml(root: Path, val_txt: Path) -> Path:
    """
    검증 전용 data.yaml. (train은 더미로 val 재사용)
    단일 클래스 person으로 명시.
    """
    data = {
        "path": str(root.resolve()),
        "train": str(val_txt.resolve()),  # dummy
        "val": str(val_txt.resolve()),
        "names": {0: "person"},
    }
    out = root / "data.auto.yaml"
    with out.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out


def f1_from_pr(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def evaluate(weights, data_yaml: Path):
    print(f"\n[Eval] weights = {weights}")
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        imgsz=IMG_SIZE,
        device=DEVICE,
        workers=WORKERS,
        verbose=False,
        plots=False
    )
    mp = float(getattr(metrics.box, "mp", 0.0))        # mean precision
    mr = float(getattr(metrics.box, "mr", 0.0))        # mean recall
    map50 = float(getattr(metrics.box, "map50", 0.0))  # mAP@0.50
    map5095 = float(getattr(metrics.box, "map", 0.0))  # mAP@0.50-0.95
    f1 = f1_from_pr(mp, mr)

    print(f"  Precision: {mp:.4f} | Recall: {mr:.4f} | F1: {f1:.4f} | mAP50: {map50:.4f} | mAP50-95: {map5095:.4f}")
    return {"precision": mp, "recall": mr, "f1": f1, "map50": map50, "map5095": map5095}


def main():
    # 1) 임시 평가셋 구성(‘Fall’ 폴더만, 라벨 0↔1 스왑 후 class0만 유지)
    imgs = build_eval_set(DATASET_ROOT, TMP_EVAL_ROOT)
    assert len(imgs) > 0, "No images found under 'Fall' folders."
    print(f"[Build] eval images: {len(imgs)}  (root: {TMP_EVAL_ROOT})")

    # 2) val.txt & data.yaml 작성
    lists_dir = TMP_EVAL_ROOT / "_lists"
    val_txt = lists_dir / "val.txt"
    write_val_list(imgs, val_txt)
    data_yaml = make_data_yaml(TMP_EVAL_ROOT, val_txt)
    print(f"[YAML] {data_yaml}")

    # 3) 두 모델 평가
    base_res = evaluate(BASELINE_W, data_yaml)
    ft_res   = evaluate(FINETUNED_W, data_yaml)

    # 4) F1 시각화
    labels = ["YOLO12n (base)", "YOLO12n (finetuned)"]
    f1s = [base_res["f1"], ft_res["f1"]]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, f1s)
    plt.title("F1 on CAUCAFall (Fall folders, person only, swapped labels)")
    plt.ylabel("F1")
    for i, v in enumerate(f1s):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    out_png = TMP_EVAL_ROOT / "f1_compare.png"
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"[Saved] {out_png}")

if __name__ == "__main__":
    main()