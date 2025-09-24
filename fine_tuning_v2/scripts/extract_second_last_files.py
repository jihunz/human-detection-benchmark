"""Extract second-last image/txt from each zip without full extraction, update labels."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
import re
from typing import Iterable, Optional

SOURCE_DIR = Path("/Volumes/작업데이터/실신 재작업 위치 정보파일")
TARGET_DIR = Path("/Users/jihunjang/Downloads/용역데이터")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TEXT_EXT = ".txt"
FALL_CLASS_ID = 80


def ensure_dirs() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)


def natural_sort_key(name: str):
    nums = [int(match) for match in re.findall(r"\d+", name)]
    return (nums, name)


def iter_second_last_infos(infos: Iterable[zipfile.ZipInfo]) -> Optional[zipfile.ZipInfo]:
    entries = sorted(
        [info for info in infos if not info.is_dir()],
        key=lambda i: natural_sort_key(Path(i.filename).name),
    )
    if len(entries) < 2:
        return None
    return entries[-2]


def unique_prefixed_path(dst_dir: Path, prefix: str, src_name: str) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / f"{prefix}_{src_name}"
    counter = 1
    while dst_path.exists():
        dst_path = dst_dir / f"{prefix}_{counter}_{src_name}"
        counter += 1
    return dst_path


def process_zip(zip_path: Path) -> None:
    prefix = zip_path.stem
    with zipfile.ZipFile(zip_path, "r") as zf:
        image_infos = [info for info in zf.infolist() if Path(info.filename).suffix.lower() in IMAGE_EXTS]
        text_infos = [info for info in zf.infolist() if Path(info.filename).suffix.lower() == TEXT_EXT]

        img_info = iter_second_last_infos(image_infos)
        txt_info = iter_second_last_infos(text_infos)

        if img_info is None:
            print(f"[skip] {zip_path.name}: not enough image files")
        else:
            dst_image = unique_prefixed_path(TARGET_DIR, prefix, Path(img_info.filename).name)
            with zf.open(img_info, "r") as src, dst_image.open("wb") as out:
                shutil.copyfileobj(src, out)
            print(f"[img] {zip_path.name} -> {dst_image.name}")

        if txt_info is None:
            print(f"[skip] {zip_path.name}: not enough text files")
        else:
            dst_text = unique_prefixed_path(TARGET_DIR, prefix, Path(txt_info.filename).name)
            with zf.open(txt_info, "r") as src:
                content = src.read().decode("utf-8", "ignore")
            lines = []
            for raw in content.splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                parts = raw.split()
                parts[0] = str(FALL_CLASS_ID)
                lines.append(" ".join(parts))
            if not lines:
                lines.append(f"{FALL_CLASS_ID} 0.5 0.5 1.0 1.0")
            with dst_text.open("w", encoding="utf-8") as out:
                for line in lines:
                    out.write(line + "\n")
            print(f"[txt] {zip_path.name} -> {dst_text.name}")


def main() -> None:
    ensure_dirs()
    archives = sorted(p for p in SOURCE_DIR.iterdir() if p.suffix.lower() == ".zip")
    if not archives:
        print(f"No zip files found in {SOURCE_DIR}")
        return
    total = len(archives)
    for idx, zip_path in enumerate(archives, 1):
        print(f"\n[progress] ({idx}/{total}) processing {zip_path.name}")
        try:
            process_zip(zip_path)
        except Exception as exc:
            print(f"[error] Failed to process {zip_path.name}: {exc}")


if __name__ == "__main__":
    main()
