from __future__ import annotations

import json
from pathlib import Path
from typing import List
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
import json as pyjson
import time
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

APP_ROOT = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(APP_ROOT / "templates"))
STATIC_DIR = APP_ROOT / "static"

app = FastAPI(title="MegaFall Dataset Viewer")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXT = ".txt"


def list_images(img_dir: Path, label_dir: Path) -> List[str]:
    images: List[str] = []
    depth_one = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    depth_dirs = sorted(p for p in img_dir.iterdir() if p.is_dir())

    for img_path in depth_one:
        rel = img_path.relative_to(img_dir)
        if (label_dir / rel.with_suffix(LABEL_EXT)).is_file():
            images.append(str(rel))

    for subdir in depth_dirs:
        sub_images = [p for p in sorted(subdir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        for img_path in sub_images:
            rel = img_path.relative_to(img_dir)
            if (label_dir / rel.with_suffix(LABEL_EXT)).is_file():
                images.append(str(rel))

    return images


def collect_images(img_dir: Path, label_dir: Path) -> List[str]:
    return list_images(img_dir, label_dir)


def safe_join(base: Path, rel: str) -> Path:
    candidate = (base / rel).resolve()
    if not candidate.is_file() or base.resolve() not in candidate.parents:
        raise HTTPException(status_code=404, detail="File not found")
    return candidate


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


@app.get("/viewer", response_class=HTMLResponse)
def viewer(request: Request, img_dir: str = Query(...), label_dir: str = Query(...)):
    img_dir_path = resolve_dataset_path(img_dir)
    label_dir_path = resolve_dataset_path(label_dir)

    if not img_dir_path.exists() or not img_dir_path.is_dir():
        return TEMPLATES.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": f"Invalid image directory: {img_dir}",
                "prefill": {"img_dir": img_dir, "label_dir": label_dir},
            },
            status_code=400,
        )
    if not label_dir_path.exists() or not label_dir_path.is_dir():
        return TEMPLATES.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": f"Invalid label directory: {label_dir}",
                "prefill": {"img_dir": img_dir, "label_dir": label_dir},
            },
            status_code=400,
        )

    images = collect_images(img_dir_path, label_dir_path)
    if not images:
        return TEMPLATES.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "No images with matching labels found.",
                "prefill": {"img_dir": img_dir, "label_dir": label_dir},
            },
            status_code=400,
        )

    data = {
        "images": images,
        "img_dir": str(img_dir_path.resolve()),
        "label_dir": str(label_dir_path.resolve()),
    }
    return TEMPLATES.TemplateResponse(
        "viewer.html",
        {"request": request, "data_json": json.dumps(data)},
    )


@app.get("/image")
def get_image(img_dir: Path, rel_path: str):
    img_path = safe_join(img_dir, rel_path)
    return FileResponse(img_path)


@app.get("/api/labels")
def get_labels(img_dir: Path, label_dir: Path, rel_path: str):
    img_path = safe_join(img_dir, rel_path)
    label_rel = Path(rel_path).with_suffix(LABEL_EXT)
    label_path = safe_join(label_dir, str(label_rel))

    labels = []
    with label_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cls = int(float(parts[0]))
                bbox = [float(val) for val in parts[1:5]]
            except (ValueError, IndexError):
                continue
            labels.append({"class": cls, "bbox": bbox})

    return JSONResponse(
        {
            "image": rel_path,
            "label": str(label_rel),
            "labels": labels,
        }
    )


@app.get("/api/progress")
def scan_progress(img_dir: str, label_dir: str):
    img_dir_path = resolve_dataset_path(img_dir)
    label_dir_path = resolve_dataset_path(label_dir)

    def event_stream():
        start = time.time()
        if not img_dir_path.exists() or not img_dir_path.is_dir():
            yield f"data: {pyjson.dumps({'status': 'error', 'message': 'Invalid image directory'})}\n\n"
            return
        if not label_dir_path.exists() or not label_dir_path.is_dir():
            yield f"data: {pyjson.dumps({'status': 'error', 'message': 'Invalid label directory'})}\n\n"
            return

        rel_images = list_images(img_dir_path, label_dir_path)
        total = len(rel_images)
        if not total:
            yield f"data: {pyjson.dumps({'status': 'error', 'message': 'No images with matching labels found.'})}\n\n"
            return

        step = max(1, total // 200)
        for idx, _ in enumerate(rel_images, 1):
            if idx % step == 0 or idx == total:
                progress = idx / total
                elapsed = time.time() - start
                remaining = (elapsed / progress) - elapsed if progress > 0 else None
                payload = {
                    'status': 'progress',
                    'progress': round(progress * 100, 2),
                    'elapsed': round(elapsed, 2),
                    'eta': round(remaining, 2) if remaining is not None else None,
                }
                yield f"data: {pyjson.dumps(payload)}\n\n"

        viewer_url = f"/viewer?img_dir={quote(img_dir, safe='')}&label_dir={quote(label_dir, safe='')}"
        final_payload = {
            'status': 'done',
            'progress': 100.0,
            'elapsed': round(time.time() - start, 2),
            'viewer_url': viewer_url,
        }
        yield f"data: {pyjson.dumps(final_payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
HOST_DATA_ROOT = Path("/Users/jihunjang/Downloads/dataset/train")
CONTAINER_DATA_ROOT = Path("/datasets")


def resolve_dataset_path(raw_path: str) -> Path:
    candidate = Path(raw_path.strip()).expanduser()
    if candidate.exists():
        return candidate
    try:
        relative = Path(raw_path.strip()).expanduser().relative_to(HOST_DATA_ROOT)
        mapped = (CONTAINER_DATA_ROOT / relative).resolve()
        if mapped.exists():
            return mapped
    except Exception:
        pass
    return candidate
