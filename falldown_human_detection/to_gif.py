# ================================
# Folder → GIF batch converter
# (non-recursive by default)
# Requirements: ffmpeg
# ================================

import subprocess
from pathlib import Path

# -------- Config --------
SRC_DIR   = Path("/Users/jihunjang/Downloads/kisadb-c/falldown-cut-pred")  # 변환할 동영상들이 있는 폴더
DST_DIR   = Path("//Users/jihunjang/Downloads/falldown-gifs")      # GIF 저장 폴더
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".wmv"}  # 처리할 확장자
RECURSIVE = False          # True면 하위 폴더까지 전부 처리

# GIF 품질/크기 옵션
GIF_FPS   = 10             # 프레임레이트
GIF_WIDTH = 480            # 가로 해상도 (세로는 비율 유지, -1)
START     = 0              # 시작 시각(초) | 0이면 처음부터
DURATION  = None           # 길이(초) | None이면 끝까지
LOOP      = 0              # 0: 무한 반복

FFMPEG    = "ffmpeg"       # ffmpeg 바이너리 경로

def build_filter():
    # 고품질 GIF를 위한 palettegen/paletteuse + lanczos 스케일
    vf = f"fps={GIF_FPS},scale={GIF_WIDTH}:-1:flags=lanczos,split[s0][s1];" \
         f"[s0]palettegen=stats_mode=full[p];[s1][p]paletteuse=new=1:dither=bayer:bayer_scale=5"
    return vf

def video_to_gif(in_path: Path, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vf = build_filter()

    cmd = [FFMPEG, "-y"]
    if START and START > 0:
        cmd += ["-ss", str(float(START))]
    cmd += ["-i", str(in_path)]
    if DURATION and DURATION > 0:
        cmd += ["-t", str(float(DURATION))]
    cmd += [
        "-vf", vf,
        "-loop", str(LOOP),
        str(out_path)
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {in_path.name}\nSTDERR:\n{proc.stderr.decode('utf-8', 'ignore')}"
        )

def list_videos(folder: Path):
    if RECURSIVE:
        return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXT]
    else:
        return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXT]

def main():
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Source folder not found: {SRC_DIR}")
    DST_DIR.mkdir(parents=True, exist_ok=True)

    vids = list_videos(SRC_DIR)
    if not vids:
        print(f"No videos found in: {SRC_DIR}")
        return

    print(f"Found {len(vids)} video(s) in {SRC_DIR} (recursive={RECURSIVE})")
    for v in vids:
        out_gif = DST_DIR / f"{v.stem}.gif"
        try:
            print(f"[GIF] {v.name} → {out_gif.name}")
            video_to_gif(v, out_gif)
        except Exception as e:
            print(f"[ERROR] {v.name}: {e}")

    print(f"\n[done] GIFs saved to: {DST_DIR.resolve()}")

if __name__ == "__main__":
    main()