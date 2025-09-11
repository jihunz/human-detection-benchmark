# -*- coding: utf-8 -*-
"""
Frame Extractor (Editor-Friendly)
- 특정 폴더의 .mp4만 처리
- 영상별 폴더 생성(폴더명 = 영상 파일명)
- 프레임 파일명: frame_000000.jpg 형식(6자리 zero-pad)

사용법:
1) 아래 CONFIG의 SRC_DIR(입력 폴더)만 본인 경로로 바꾸세요.
2) OUT_DIR=None이면 SRC_DIR 밑에 결과가 저장됩니다.
3) 에디터에서 바로 실행(Run)하세요.

필요 패키지: pip install opencv-python
"""

from pathlib import Path
import sys
import cv2


# ======== CONFIG ========
SRC_DIR      = r"/Users/jihunjang/Downloads/kisadb-c/falldown_cut"  # 처리할 mp4들이 있는 폴더
OUT_DIR      = None                     # None -> SRC_DIR에 저장. 경로를 주면 그 경로에 저장
START_INDEX  = 0                        # 프레임 시작 번호(0 -> frame_000000.jpg부터)
JPEG_QUALITY = 95                       # 1~100
VIDEO_EXT    = ".mp4"                   # 소문자 확장자 기준
# ========================


def extract_frames_from_video(video_path: Path, out_root: Path,
                              start_index: int = 0,
                              jpg_quality: int = 95) -> int:
    """
    비디오의 모든 프레임을 JPEG로 저장.
    저장 경로: out_root / <video_stem> / frame_XXXXXX.jpg
    반환값: 저장된 프레임 수
    """
    if not video_path.exists():
        print(f"[WARN] 파일이 존재하지 않음: {video_path}", file=sys.stderr)
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] 비디오를 열 수 없음: {video_path}", file=sys.stderr)
        return 0

    out_dir = out_root / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = start_index
    saved = 0
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        fname = f"frame_{frame_idx:06d}.jpg"  # 6자리 zero-pad
        out_path = out_dir / fname

        if not cv2.imwrite(str(out_path), frame, params):
            print(f"[WARN] 저장 실패: {out_path}", file=sys.stderr)
        else:
            saved += 1

        frame_idx += 1

    cap.release()
    return saved


def run():
    src_dir = Path(SRC_DIR).expanduser().resolve()
    out_root = Path(OUT_DIR).expanduser().resolve() if OUT_DIR else src_dir

    if not src_dir.exists() or not src_dir.is_dir():
        print(f"[ERROR] 입력 폴더를 찾을 수 없습니다: {src_dir}", file=sys.stderr)
        return

    # 하위폴더는 탐색하지 않고, 해당 폴더의 .mp4만 처리
    mp4_list = sorted(
        [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() == VIDEO_EXT]
    )

    if not mp4_list:
        print(f"[INFO] .mp4 파일이 없습니다: {src_dir}")
        return

    total_saved = 0
    for v in mp4_list:
        print(f"[INFO] 처리 중: {v.name}")
        saved = extract_frames_from_video(
            v, out_root=out_root, start_index=START_INDEX, jpg_quality=JPEG_QUALITY
        )
        print(f"  -> 저장된 프레임: {saved}")
        total_saved += saved

    print(f"[DONE] 전체 저장 프레임 수: {total_saved}")
    print(f"[DONE] 출력 루트: {out_root}")


if __name__ == "__main__":
    run()