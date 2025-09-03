import os
import shlex
import subprocess
from pathlib import Path
from typing import List

# ---------------------------
# Config
# ---------------------------
SRC_DIR   = Path("/Users/jihunjang/Downloads/falldown-gifs")  # 입력 GIF 폴더
OUT_PATH  = SRC_DIR / "falldown-grid.gif"                     # 출력 파일
MAX_COLS  = 4                                                 # 가로 최대 개수
TILE_W    = 480                                               # 각 타일 가로 크기(px)
TILE_H    = 480                                               # 각 타일 세로 크기(px)
FPS       = 10                                                # 출력 FPS
FFMPEG    = "ffmpeg"                                          # ffmpeg 경로

def list_gifs(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".gif"])

def build_filter_and_cmd(files: List[Path]) -> List[str]:
    """
    각 입력을 동일 타일 크기로 스케일/패드 → xstack로 격자 → palettegen/paletteuse
    """
    n = len(files)
    if n == 0:
        raise FileNotFoundError(f"No GIFs found in: {SRC_DIR}")

    # 입력들
    cmd = [FFMPEG, "-y"]
    for f in files:
        cmd += ["-ignore_loop", "0", "-i", str(f)]  # GIF 루프 무시(한 번만 읽음)

    # per-input 가공: fps 맞추고, 스케일/패드로 타일 크기 통일
    per_input = []
    for i in range(n):
        per_input.append(
            f"[{i}:v]"
            f"fps={FPS},"
            f"scale={TILE_W}:{TILE_H}:force_original_aspect_ratio=decrease,"
            f"pad={TILE_W}:{TILE_H}:(ow-iw)/2:(oh-ih)/2:color=white"
            f"[v{i}]"
        )

    # xstack layout 문자열 생성 (동일 타일 크기 가정)
    layout_elems = []
    for idx in range(n):
        row = idx // MAX_COLS
        col = idx % MAX_COLS
        x = col * TILE_W
        y = row * TILE_H
        layout_elems.append(f"{x}_{y}")
    layout = "|".join(layout_elems)

    # xstack로 그리드 합성 → 팔레트 생성/적용
    # shortest=1: 가장 짧은 클립 길이에 맞춰 종료 (너무 긴 입력으로 무한 커지는 것 방지)
    filter_graph = (
        ";".join(per_input) + ";" +
        "".join(f"[v{i}]" for i in range(n)) +
        f"xstack=inputs={n}:layout={layout}:shortest=1[vout];"
        f"[vout]split[v0][v1];"
        f"[v0]palettegen=stats_mode=full[p];"
        f"[v1][p]paletteuse=new=1:dither=bayer:bayer_scale=5[vf]"
    )

    # 최종 맵/출력
    cmd += [
        "-filter_complex", filter_graph,
        "-map", "[vf]",
        "-loop", "0",                   # 무한 반복
        str(OUT_PATH)
    ]
    return cmd

def main():
    gifs = list_gifs(SRC_DIR)
    print(f"Found {len(gifs)} GIF(s) in {SRC_DIR}")
    cmd = build_filter_and_cmd(gifs)

    # 디버깅용 명령 표시
    print("\nRunning ffmpeg:\n", " ".join(shlex.quote(c) for c in cmd), "\n")

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "ignore")
        raise RuntimeError(f"ffmpeg failed:\n{err}")

    print(f"[OK] Saved grid GIF → {OUT_PATH}")

if __name__ == "__main__":
    main()