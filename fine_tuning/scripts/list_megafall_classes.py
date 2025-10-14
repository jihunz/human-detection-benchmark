"""Summarize MegaFall train/val class usage based on YOLO split files (with PNG visualization)."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path("/Users/jihunjang/workspace/ust/human-detection/fine_tuning/v4")
TRAIN_LIST = BASE_DIR / "train2.txt"
VAL_LIST = BASE_DIR / "val2.txt"
if not TRAIN_LIST.exists():
    TRAIN_LIST = BASE_DIR / "train.txt"
if not VAL_LIST.exists():
    VAL_LIST = BASE_DIR / "val.txt"
LABEL_ROOT = Path("/Users/jihunjang/Downloads/dataset/train/megafallv2/labels")
OUTPUT_DIR = Path(__file__).resolve().parent / "mf_cls_result"

LABEL_EXT = ".txt"
TOKEN_IMAGES = "images"
FONT = ImageFont.load_default()


def label_for_image(img_path: Path) -> Path | None:
    parts = list(img_path.parts)
    if TOKEN_IMAGES not in parts:
        return None
    idx = parts.index(TOKEN_IMAGES)
    parts[idx] = "labels"
    rel = Path(*parts[idx:])
    return LABEL_ROOT / rel.relative_to("labels").with_suffix(LABEL_EXT)


def parse_label(label_path: Path) -> tuple[Counter[int], str | None]:
    counter: Counter[int] = Counter()
    classes: set[int] = set()
    try:
        with label_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    cls = int(float(parts[0]))
                except (ValueError, IndexError):
                    continue
                counter[cls] += 1
                classes.add(cls)
    except OSError:
        return Counter(), None
    combo = ",".join(str(c) for c in sorted(classes)) if classes else None
    return counter, combo


def gather(list_path: Path) -> tuple[Counter[int], Counter[str], int]:
    class_counter: Counter[int] = Counter()
    combo_counter: Counter[str] = Counter()
    total = 0
    with list_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            img = Path(line.strip())
            if not img:
                continue
            label = label_for_image(img)
            if label is None:
                continue
            per_file, combo = parse_label(label)
            if not per_file:
                continue
            class_counter.update(per_file)
            total += 1
            if combo:
                combo_counter[combo] += 1
    return class_counter, combo_counter, total


def top_items(counter: Counter[int | str], limit: int) -> list[tuple[int | str, int]]:
    return counter.most_common(limit)


def draw_bar_block(draw: ImageDraw.ImageDraw, origin: tuple[int, int], size: tuple[int, int], title: str, data: list[tuple[str, int]]):
    x0, y0 = origin
    width, height = size
    draw.rectangle([x0, y0, x0 + width, y0 + height], outline="gray")
    draw.text((x0 + 10, y0 + 6), title, font=FONT, fill="black")
    bar_area_top = y0 + 24
    if not data:
        draw.text((x0 + 10, bar_area_top + 10), "No data", font=FONT, fill="black")
        return
    max_value = max(v for _, v in data)
    bar_height = max(10, min(24, (height - 40) // max(len(data), 1)))
    spacing = bar_height + 4
    for idx, (label, count) in enumerate(data):
        bar_y = bar_area_top + idx * spacing
        if bar_y + bar_height > y0 + height - 5:
            break
        frac = count / max_value if max_value else 0
        bar_w = int((width - 120) * frac)
        bar_w = max(bar_w, 1)
        draw.rectangle([x0 + 10, bar_y, x0 + 10 + bar_w, bar_y + bar_height], fill="#4E79A7")
        draw.text((x0 + 14 + bar_w, bar_y), f"{count}", font=FONT, fill="black")
        draw.text((x0 + width - 100, bar_y), str(label), font=FONT, fill="black")


def plot_summary(train_stats, val_stats, out_path: Path) -> None:
    img = Image.new("RGB", (1200, 900), "white")
    draw = ImageDraw.Draw(img)

    train_classes, train_combos, _ = train_stats
    val_classes, val_combos, _ = val_stats

    blocks = [
        ((40, 40), (520, 380), "Train Class Counts", top_items(train_classes, 20)),
        ((640, 40), (520, 380), "Val Class Counts", top_items(val_classes, 20)),
        ((40, 460), (520, 380), "Train Class Combos", top_items(train_combos, 15)),
        ((640, 460), (520, 380), "Val Class Combos", top_items(val_combos, 15)),
    ]

    draw.text((20, 10), "MegaFall Class Distribution", font=FONT, fill="black")
    for origin, size, title, data in blocks:
        draw_bar_block(draw, origin, size, title, data)

    img.save(out_path)


def write_section(out_path: Path, tag: str, stats: tuple[Counter[int], Counter[str], int]) -> None:
    class_counts, combo_counts, total = stats
    lines = [f"[{tag}] total label files: {total}"]
    lines.append("class_id: count")
    for cls, count in class_counts.most_common():
        lines.append(f"{cls}: {count}")
    lines.append("")
    lines.append("combination: frames")
    for combo, count in combo_counts.most_common(50):
        lines.append(f"{combo}: {count}")
    lines.append("")
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_txt = OUTPUT_DIR / f"megafall_splits_{timestamp}.txt"
    out_png = OUTPUT_DIR / f"megafall_splits_{timestamp}.png"

    train_stats = gather(TRAIN_LIST)
    val_stats = gather(VAL_LIST)

    write_section(out_txt, "train", train_stats)
    write_section(out_txt, "val", val_stats)

    plot_summary(train_stats, val_stats, out_png)

    print(f"[done] Summary written to {out_txt}")
    print(f"[plot] Visualization saved to {out_png}")


if __name__ == "__main__":
    main()
