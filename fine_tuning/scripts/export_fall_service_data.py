"""Extract AI Hub fall frames with class filtering and remap fall labels."""
from __future__ import annotations

import shutil
import uuid
import zipfile
from io import TextIOWrapper
from pathlib import Path
from typing import List

SOURCE_DIR = Path("/Volumes/작업데이터/실신 재작업 위치 정보파일")
TARGET_DIR = Path("/Users/jihunjang/Downloads/용역데이터")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SUFFIXES = tuple(IMAGE_EXTS)
TEXT_EXT = ".txt"
TARGET_FALL_CLASS = 2
TARGET_PERSON_CLASS = 0
OUTPUT_FALL_CLASS = 1


def ensure_dirs() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)


def unique_prefixed_path(dst_dir: Path, prefix: str, src_name: str) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    return dst_dir / f"{prefix}_{token}_{src_name}"


def load_and_convert_label(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> List[str] | None:
    converted: List[str] = []
    has_fall = False
    with zf.open(info, "r") as raw:
        for line in TextIOWrapper(raw, encoding="utf-8", errors="ignore"):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if not parts:
                continue
            try:
                cls = int(float(parts[0]))
            except ValueError:
                return None

            if cls == TARGET_FALL_CLASS:
                parts[0] = str(OUTPUT_FALL_CLASS)
                has_fall = True
            elif cls == TARGET_PERSON_CLASS:
                parts[0] = str(TARGET_PERSON_CLASS)
            else:
                return None

            converted.append(" ".join(parts))

    if not has_fall or not converted:
        return None
    return converted


def process_zip(zip_path: Path) -> None:
    prefix = zip_path.stem
    with zipfile.ZipFile(zip_path, "r") as zf:
        image_map: dict[str, zipfile.ZipInfo] = {}
        text_infos: List[zipfile.ZipInfo] = []
        for info in zf.infolist():
            if info.is_dir():
                continue
            lower_name = info.filename.lower()
            if lower_name.endswith(TEXT_EXT):
                text_infos.append(info)
            elif lower_name.endswith(IMAGE_SUFFIXES):
                filename = info.filename
                basename = filename.rsplit("/", 1)[-1]
                dot = basename.rfind(".")
                if dot == -1:
                    continue
                stem = basename[:dot]
                image_map[stem] = info

        if not text_infos:
            print(f"[skip] {zip_path.name}: no label files")
            return

        exported = 0
        for txt_info in text_infos:
            label_name = txt_info.filename.rsplit("/", 1)[-1]
            stem = label_name.rsplit(".", 1)[0]

            converted_lines = load_and_convert_label(zf, txt_info)
            if not converted_lines:
                continue

            img_info = image_map.get(stem)
            if img_info is None:
                print(f"[warn] {zip_path.name}: image missing for label {label_name}")
                continue

            img_name = img_info.filename.rsplit("/", 1)[-1]
            dst_image = unique_prefixed_path(TARGET_DIR, prefix, img_name)
            with zf.open(img_info, "r") as src, dst_image.open("wb") as out:
                shutil.copyfileobj(src, out)

            dst_text = unique_prefixed_path(TARGET_DIR, prefix, label_name)
            with dst_text.open("w", encoding="utf-8") as out:
                out.write("\n".join(converted_lines))
                out.write("\n")

            exported += 1
            print(f"[export] {zip_path.name}: {label_name} ({len(converted_lines)} obj)")
            break

        if exported == 0:
            print(f"[skip] {zip_path.name}: no labels matched criteria")


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
