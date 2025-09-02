# ==============================================
# KISA XML → StartTime + AlarmDuration 기반 영상 클립 추출
# - XML과 동일한 스템명의 비디오를 찾아 구간만 잘라 저장
# - 빠른 복사(-c copy) 실패시 재인코딩 폴백
# ==============================================

import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

INPUT_DIR = Path("/Users/jihunjang/Downloads/falldown")  # 현재 폴더 (원하면 "/content/falldown" 등으로 변경)
OUTPUT_DIR = Path("/Users/jihunjang/Downloads/falldown_cut")  # 클립 저장 폴더
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
DESC_FILTER = {"Falldown"}  # 이 집합에 들어있는 AlarmDescription만 자름 (빈 집합이면 모두 자름)
FAST_COPY = True  # True: 빠른 복사(-c copy), 실패시 자동 재인코딩 폴백
FFMPEG_BIN = "ffmpeg"  # ffmpeg 바이너리 경로 (colab/리눅스 기본: "ffmpeg")


def parse_hms(hms: str) -> float:
    """HH:MM:SS → seconds(float)"""
    h, m, s = hms.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def find_video_for_xml(xml_path: Path) -> Optional[Path]:
    """XML과 스템이 같은 비디오 파일을 폴더에서 탐색"""
    stem = xml_path.stem
    for ext in VIDEO_EXTS:
        cand = xml_path.with_suffix(ext)
        if cand.exists():
            return cand
        # 혹시 같은 폴더 내에 다른 확장자로 있을 수 있으므로 스캔
    parent = xml_path.parent
    for p in parent.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS and p.stem == stem:
            return p
    return None


def load_alarms_from_xml(xml_path: Path) -> List[Tuple[float, float, str]]:
    """XML에서 (start_sec, duration_sec, desc) 리스트 추출"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    out = []
    for al in root.findall(".//Alarm"):
        desc = (al.findtext("AlarmDescription") or "").strip()
        if DESC_FILTER and desc not in DESC_FILTER:
            continue
        st = al.findtext("StartTime")
        dur = al.findtext("AlarmDuration")
        if not st or not dur:
            continue
        start = parse_hms(st)
        start = max(0, start - 5)
        duration = parse_hms(dur)
        if duration <= 0:
            continue
        out.append((start, duration, desc))
    return out


def run_ffmpeg_cut(in_vid: Path, out_vid: Path, start: float, duration: float,
                   fast_copy: bool = True) -> None:
    """
    fast_copy=True: -ss before -i + -t + -c copy (빠름, 키프레임 컷 오차 가능)
    실패/거부 시 재인코딩 폴백(-c:v libx264 -c:a aac)
    """
    out_vid.parent.mkdir(parents=True, exist_ok=True)

    def _run(cmd):
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return proc.returncode, proc.stdout.decode("utf-8", "ignore"), proc.stderr.decode("utf-8", "ignore")

    if fast_copy:
        cmd_copy = [
            FFMPEG_BIN, "-y",
            "-ss", f"{start:.3f}",
            "-i", str(in_vid),
            "-t", f"{duration:.3f}",
            "-c", "copy",
            str(out_vid),
        ]
        rc, so, se = _run(cmd_copy)
        if rc == 0 and out_vid.exists() and out_vid.stat().st_size > 0:
            return  # 성공

        print(f"[warn] fast copy failed or produced empty file → re-encode fallback: {out_vid.name}")

    # 안전 모드(재인코딩)
    cmd_enc = [
        FFMPEG_BIN, "-y",
        "-ss", f"{start:.3f}",
        "-i", str(in_vid),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-ac", "2", "-b:a", "128k",
        str(out_vid),
    ]
    rc, so, se = _run(cmd_enc)
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed for {in_vid.name} → {out_vid.name}\n{se}")


def sanitize_desc(desc: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", desc.strip()) or "Alarm"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    xml_files = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() == ".xml"])

    if not xml_files:
        raise FileNotFoundError(f"No XML files found in: {INPUT_DIR}")

    total_jobs = 0
    for xml_path in xml_files:
        alarms = load_alarms_from_xml(xml_path)
        if not alarms:
            print(f"[skip] No matching alarms in XML: {xml_path.name} (filter={DESC_FILTER or 'ALL'})")
            continue

        vid_path = find_video_for_xml(xml_path)
        if not vid_path:
            print(f"[skip] Video not found for XML: {xml_path.name}")
            continue

        stem = xml_path.stem
        for idx, (start, dur, desc) in enumerate(alarms, 1):
            tag = sanitize_desc(desc)
            out_name = f"{stem}_{tag}_{idx:02d}_{int(start)}s_{int(dur)}s.mp4"
            out_path = OUTPUT_DIR / out_name
            print(f"[cut] {vid_path.name} | {desc} | start={start:.2f}s, dur={dur:.2f}s → {out_name}")
            run_ffmpeg_cut(vid_path, out_path, start, dur, fast_copy=FAST_COPY)
            total_jobs += 1

    print(f"\n[done] generated {total_jobs} clip(s) in: {OUTPUT_DIR.resolve()}")

def copy_xml_files(src: Path, dst: Path, exts={".xml"}):
    """
    src 디렉토리를 재귀적으로 탐색하여 xml만 dst로 동일 구조 복사
    """
    src = Path(src); dst = Path(dst)
    for path in src.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            rel = path.relative_to(src)       # 상대 경로 유지
            out_path = dst / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out_path)
            print(f"[copy] {path} → {out_path}")
    print(f"\n[done] XML files copied to: {dst.resolve()}")

if __name__ == "__main__":
    copy_xml_files(INPUT_DIR, OUTPUT_DIR)
    # main()
