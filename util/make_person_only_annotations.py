#!/usr/bin/env python3
"""
Make a 'person-only' COCO annotation file from instances_val2017.json.

Now supports setting paths **inside this script** via USER CONFIG below,
while still allowing optional CLI overrides.

USER CONFIG (edit these as needed):
  DATA_ROOT = "/data/coco"  # base dir containing 'annotations' and 'images'
  SRC = None                # if None -> uses f"{DATA_ROOT}/annotations/instances_val2017.json"
  DST = None                # if None -> uses f"{DATA_ROOT}/annotations/instances_val2017_person.json"

CLI (overrides USER CONFIG when provided):
  python make_person_only_annotations.py \
      --src /path/to/coco/annotations/instances_val2017.json \
      --dst /path/to/coco/annotations/instances_val2017_person.json
"""
import json, argparse
from pathlib import Path

# ========== USER CONFIG (edit here) ==========
DATA_ROOT = "/data/coco"   # e.g., "/data/coco" or "D:/datasets/coco"
SRC = '/Users/jihunjang/Downloads/ust/db/coco/annotations/instances_val2017.json'                 # e.g., "/data/coco/annotations/instances_val2017.json"
DST = '/Users/jihunjang/Downloads/ust/db/coco/annotations/instances_val2017_person.json'          # e.g., "/data/coco/annotations/instances_val2017_person.json"
# ============================================

def resolve_paths(args_src: str|None, args_dst: str|None):
    # Priority: CLI args > explicit SRC/DST > derived from DATA_ROOT
    base = Path(DATA_ROOT) if DATA_ROOT else None

    src = None
    if args_src:
        src = Path(args_src)
    elif SRC:
        src = Path(SRC)
    elif base:
        src = base / "annotations" / "instances_val2017.json"
    else:
        raise RuntimeError("No source path. Set --src or SRC or DATA_ROOT.")

    dst = None
    if args_dst:
        dst = Path(args_dst)
    elif DST:
        dst = Path(DST)
    elif base:
        dst = base / "annotations" / "instances_val2017_person.json"
    else:
        raise RuntimeError("No destination path. Set --dst or DST or DATA_ROOT.")

    return src, dst

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="instances_val2017.json (CLI override)")
    ap.add_argument("--dst", default=None, help="output json path (CLI override)")
    args = ap.parse_args()

    src_path, dst_path = resolve_paths(args.src, args.dst)

    if not src_path.exists():
        raise FileNotFoundError(f"Source not found: {src_path}")

    with src_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Find 'person' category id without assuming fixed ID
    person_cat = next((c for c in data.get("categories", []) if c.get("name") == "person"), None)
    if person_cat is None:
        raise RuntimeError("Could not find 'person' in categories")
    person_id = person_cat["id"]

    out = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": data.get("images", []),  # keep all images so negatives remain
        "annotations": [ann for ann in data.get("annotations", []) if ann.get("category_id") == person_id],
        "categories": [person_cat],
    }

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"Wrote person-only annotations to: {dst_path}")

if __name__ == "__main__":
    main()
