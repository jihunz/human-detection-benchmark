# make_shards.py
# - train/test 이미지를 10k개 단위로 이미지 전용 .tar 샤드 생성
# - labels(train+test) 전체를 단일 .tar(내부는 shard-인덱스 경로 포함)로 생성
# 실행: python make_shards.py

import os, math, tarfile, glob
from pathlib import Path

# ====== 사용자 설정 ======
ROOT = Path("/Users/jihunjang/Downloads/dataset/train/megafallv2")
IMG_TRAIN = ROOT / "images/train"
IMG_TEST  = ROOT / "images/test"
LBL_TRAIN = ROOT / "labels/train"
LBL_TEST  = ROOT / "labels/test"

OUT_DIR = ROOT / "shards_out"
SHARD_SIZE = 10_000   # 이미지 1만 장 단위
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ====== 공통 유틸 ======
def list_images(img_dir: Path):
    imgs = [Path(p) for p in glob.glob(str(img_dir / "**/*"), recursive=True)]
    imgs = [p for p in imgs if p.is_file() and p.suffix.lower() in IMG_EXTS]
    # 상대경로(이미지 루트 기준)도 함께 반환
    rels = [p.relative_to(img_dir) for p in imgs]
    return imgs, rels

def label_for_image(rel_path, labels_root: Path):
    # 이미지 상대경로와 동일한 하위경로/베이스명에 .txt
    lp = labels_root / rel_path.with_suffix(".txt")
    return lp if lp.exists() else None

def write_image_shards(img_root: Path, lbl_root: Path, split_name: str):
    imgs, rels = list_images(img_root)

    # 라벨이 있는 이미지만 사용 (짝 없는 것 제거)
    pairs = []
    for img, rel in zip(imgs, rels):
        lbl = label_for_image(rel, lbl_root)
        if lbl is not None:
            pairs.append((img, rel, lbl))
    total = len(pairs)
    n_shards = math.ceil(total / SHARD_SIZE)
    print(f"[{split_name}] usable pairs = {total}, shards = {n_shards}")

    shard_maps = []  # labels_all.tar를 만들 때 쓸 매핑: [(split, shard_idx, rel_label_path), ...]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for si in range(n_shards):
        start = si * SHARD_SIZE
        end = min(total, (si + 1) * SHARD_SIZE)
        shard_pairs = pairs[start:end]
        tar_path = OUT_DIR / f"images_{split_name}-{si:03d}.tar"
        with tarfile.open(tar_path, "w") as tar:
            for img, rel, lbl in shard_pairs:
                # 이미지 tar 내부 경로: images/<split>/shard-XXX/<원래상대경로>
                img_arc = Path("images") / split_name / f"shard-{si:03d}" / rel
                tar.add(str(img), arcname=str(img_arc))
                # 라벨 tar는 여기서 안 넣음(이미지 전용). 대신 나중 labels_all 만들 때 경로 기록
                lbl_rel = lbl.relative_to(lbl_root)
                shard_maps.append((split_name, si, lbl_rel))
        print(f"  wrote {tar_path} ({end-start} images)")
    return shard_maps  # 라벨 단일 샤드에 반영하기 위한 매핑 반환

def write_labels_all(shard_maps_train, shard_maps_test):
    # shard_maps_*: [(split_name, shard_idx, label_relpath), ...]
    tar_path = OUT_DIR / "labels_all.tar"
    with tarfile.open(tar_path, "w") as tar:
        # train 라벨
        for split, si, lbl_rel in shard_maps_train:
            assert split == "train"
            src = LBL_TRAIN / lbl_rel
            arc = Path("labels") / split / f"shard-{si:03d}" / lbl_rel
            tar.add(str(src), arcname=str(arc))
        # test 라벨
        for split, si, lbl_rel in shard_maps_test:
            assert split == "test"
            src = LBL_TEST / lbl_rel
            arc = Path("labels") / split / f"shard-{si:03d}" / lbl_rel
            tar.add(str(src), arcname=str(arc))
    print(f"[labels] wrote single tar: {tar_path}")

if __name__ == "__main__":
    # 1) 이미지 샤드 생성 (train)
    train_maps = write_image_shards(IMG_TRAIN, LBL_TRAIN, "train")
    # 2) 이미지 샤드 생성 (test)
    test_maps  = write_image_shards(IMG_TEST,  LBL_TEST,  "test")
    # 3) 라벨 단일 샤드 생성
    write_labels_all(train_maps, test_maps)
    print("Done.")