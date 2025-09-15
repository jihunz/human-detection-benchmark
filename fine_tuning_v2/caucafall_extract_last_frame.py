"""
CAUCAFall last-frame collector (non-CLI)
---------------------------------------
폴더 구조: <ROOT>/subject1..10/.../<...Fall...>/*.{png,txt}
- 최하위(leaf)이며 디렉토리명에 'Fall'이 포함된 폴더만 대상으로 합니다.
- 각 대상 폴더에서 마지막 프레임의 PNG와 동일 stem의 TXT(YOLO)를 찾아
  하나의 출력 폴더로 복사하며, 라벨의 클래스 번호는 모두 0으로 덮어써 저장합니다.

주의/가정
- 프레임 파일은 이름에 번호가 포함되어 있는 경우가 많음(e.g., img_000123.png).
  번호 기준 내림차순으로 가장 큰 것을 '마지막'으로 간주. 번호가 없으면 사전식 정렬로 선택.
- TXT는 이미지와 동일 stem의 .txt가 동일 폴더에 존재한다고 가정. 없으면 경고 후 건너뜀.
- 파일명 충돌 방지를 위해, 출력 파일명은 입력의 상대 경로를 '_'로 합쳐서 접두사로 사용.

사용 방법
1) 아래 [User Config] 값을 채운 뒤 파일을 실행하세요.
2) 출력은 OUT_DIR 하위 단일 폴더에 PNG/TXT가 함께 모입니다.

"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
import cv2


# -------------------------
# User Config (edit below)
# -------------------------
ROOT = '/Users/jihunjang/Downloads/Dataset CAUCAFall/CAUCAFall'  # 예: "/Users/USER/Downloads/CAUCAFall"
OUT_DIR = "fine_tuning_v2/_caucafall_last_frames"  # 모을 단일 폴더

# 매칭 규칙
FALL_KEYWORD = "fall"      # 경로에 이 문자열이 포함되면 Fall 폴더로 간주(대소문자 무시)
EXCLUDE_KEYWORD = "nonfall" # 제외할 키워드가 있다면 지정(없으면 빈 문자열)

# 이미지 확장자 (PNG만 사용)
IMG_EXTS = {".png"}

# 리사이즈: 640x640 고정(비율 무시, 스퀘어 워프)
TARGET_SIZE = 640


def has_keyword(path: Path, key: str) -> bool:
    if not key:
        return False
    low = key.lower()
    return any(low in part.lower() for part in path.parts)


def is_leaf_directory(dir_path: Path) -> bool:
    return not any(p.is_dir() for p in dir_path.iterdir())


def is_target_fall_leaf(dir_path: Path) -> bool:
    # 디렉토리명에 'fall' 포함(대소문자 무시), 제외 키워드 미포함, 하위 디렉토리 없음, PNG 존재
    if EXCLUDE_KEYWORD and has_keyword(dir_path, EXCLUDE_KEYWORD):
        return False
    if FALL_KEYWORD.lower() not in dir_path.name.lower():
        return False
    # 루트 하위 경로 중 'Subject'가 포함된 경로만 허용
    if not any('subject' in part.lower() for part in dir_path.parts):
        return False
    if not is_leaf_directory(dir_path):
        return False
    return any(p.is_file() and p.suffix.lower() in IMG_EXTS for p in dir_path.iterdir())


def extract_numbers(s: str) -> List[int]:
    return [int(x) for x in re.findall(r"\d+", s)]


def image_sort_key(p: Path) -> Tuple[int, str]:
    # 파일명에서 가장 큰 정수를 우선 키로 사용, 없으면 -1
    nums = extract_numbers(p.stem)
    max_num = max(nums) if nums else -1
    return (max_num, p.name)


def pick_last_image(dir_path: Path) -> Optional[Path]:
    imgs = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not imgs:
        return None
    imgs.sort(key=image_sort_key)
    return imgs[-1]


def find_label_for_image(img_path: Path) -> Optional[Path]:
    cand = img_path.with_suffix(".txt")
    return cand if cand.is_file() else None


def sanitize_relpath(rel: Path) -> str:
    # 상대 경로의 구분자와 비문자열을 '_' 로 치환하여 파일명 접두사로 사용
    s = "_".join(rel.parts)
    s = re.sub(r"[^a-zA-Z0-9._-]", "_", s)
    return s


def gather_fall_folders(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_dir() and is_target_fall_leaf(p):
            out.append(p)
    out.sort()
    return out


def copy_pair(img: Path, lbl: Path, out_dir: Path, root: Path) -> Tuple[Path, Path]:
    rel = img.parent.resolve().relative_to(root.resolve())
    prefix = sanitize_relpath(rel)
    stem = f"{prefix}__{img.stem}"
    out_img = out_dir / f"{stem}{img.suffix.lower()}"
    out_lbl = out_dir / f"{stem}.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_resized_image(img, out_img)
    rewrite_label_to_class0(lbl, out_lbl)
    return out_img, out_lbl


def rewrite_label_to_class0(src: Path, dst: Path) -> None:
    """YOLO 라벨 파일의 클래스 id를 전부 0으로 통일하여 저장."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines_out: list[str] = []
    try:
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                if not parts:
                    continue
                parts[0] = "0"  # 강제 person=0
                lines_out.append(" ".join(parts))
    except Exception:
        # 문제 발생 시 원본을 그대로 복사(최후 수단)
        shutil.copy2(src, dst)
        return
    with dst.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + ("\n" if lines_out else ""))


def save_resized_image(src_img: Path, dst_img: Path) -> None:
    """이미지를 640x640 고정 크기로 리사이즈하여 저장."""
    img = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
    if img is None:
        # 실패 시 원본 복사
        shutil.copy2(src_img, dst_img)
        return
    img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst_img), img)


def run() -> int:
    root = Path(ROOT)
    out_dir = Path(OUT_DIR)
    if not root.is_dir():
        print(f"[error] ROOT not found: {root}")
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    fall_dirs = gather_fall_folders(root)
    if not fall_dirs:
        print(f"[warn] No 'Fall' folders found under: {root}")
    print(f"[info] Fall folders: {len(fall_dirs)}")

    copied = 0
    skipped = 0
    for d in fall_dirs:
        img = pick_last_image(d)
        if img is None:
            print(f"[skip] No images in: {d}")
            skipped += 1
            continue
        lbl = find_label_for_image(img)
        if lbl is None:
            print(f"[skip] Label not found for: {img}")
            skipped += 1
            continue
        out_img, out_lbl = copy_pair(img, lbl, out_dir, root)
        print(f"[ok] {d.name}: {out_img.name}, {out_lbl.name}")
        copied += 1

    print(f"\n[done] copied={copied}, skipped={skipped}, out={out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
