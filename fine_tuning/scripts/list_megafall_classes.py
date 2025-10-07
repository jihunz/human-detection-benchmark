"""Utility to list unique YOLO class IDs under the MegaFall v2 labels directory."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt


LABELS_ROOT = Path("/Users/jihunjang/Downloads/dataset/train/megafallv2/labels/train")
LABEL_EXT = ".txt"
OUTPUT_DIR = Path(__file__).resolve().parent / "mf_cls_result"


def gather_stats(root: Path) -> tuple[Counter[int], Counter[str], int]:
    class_counter: Counter[int] = Counter()
    combo_counter: Counter[str] = Counter()
    label_files = 0
    for label_path in root.rglob(f"*{LABEL_EXT}"):
        if not label_path.is_file():
            continue
        try:
            classes_in_file: list[int] = []
            with label_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    try:
                        class_id = int(float(parts[0]))
                    except (ValueError, IndexError):
                        continue
                    class_counter[class_id] += 1
                    classes_in_file.append(class_id)
            if classes_in_file:
                label_files += 1
                combo = ",".join(str(c) for c in sorted(set(classes_in_file)))
                combo_counter[combo] += 1
        except OSError as exc:
            print(f"[warn] Could not read {label_path}: {exc}")
    return class_counter, combo_counter, label_files


def main() -> None:
    if not LABELS_ROOT.exists():
        raise FileNotFoundError(f"Labels root not found: {LABELS_ROOT}")

    class_counts, combo_counts, label_count = gather_stats(LABELS_ROOT)
    if not class_counts:
        print("[info] No labels found.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_txt = OUTPUT_DIR / f"megafall_classes_{timestamp}.txt"
    out_png = OUTPUT_DIR / f"megafall_classes_{timestamp}.png"

    lines: list[str] = []
    lines.append(f"labels_root: {LABELS_ROOT}")
    lines.append(f"total_label_files: {label_count}")
    lines.append("")

    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    lines.append("[class_counts]")
    for class_id, count in sorted_classes:
        lines.append(f"class {class_id}: {count}")
    lines.append("")

    sorted_combos = sorted(combo_counts.items(), key=lambda x: x[1], reverse=True)
    lines.append("[class_combinations]")
    for combo, count in sorted_combos:
        lines.append(f"{combo}: {count}")

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_stats(sorted_classes, sorted_combos, out_png)
    print(f"[done] Summary written to {out_txt}")
    print(f"[plot] Visualization saved to {out_png}")


def plot_stats(
    class_items: list[tuple[int, int]],
    combo_items: list[tuple[str, int]],
    out_path: Path,
) -> None:
    if not class_items:
        return

    top_combo_items = combo_items[: min(10, len(combo_items))]
    table_items = combo_items[: min(20, len(combo_items))]

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.5, 1.5], height_ratios=[1, 1.1])

    ax_classes = fig.add_subplot(gs[0, 0])
    ax_combos = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[:, 1])

    # Class count bar chart
    class_ids = [str(cls) for cls, _ in class_items]
    class_counts = [count for _, count in class_items]
    ax_classes.barh(class_ids[::-1], class_counts[::-1], color="#4E79A7")
    ax_classes.set_xlabel("Instances", fontweight="bold")
    ax_classes.set_title("Class Distribution", fontweight="bold", fontsize=12)
    ax_classes.grid(axis="x", linestyle="--", alpha=0.4)
    ax_classes.tick_params(axis="y", labelsize=9)
    for y, count in zip(class_ids[::-1], class_counts[::-1]):
        ax_classes.text(count, y, f" {count:,}", va="center", ha="left", fontsize=9, fontweight="bold")

    # Combination bar chart
    if top_combo_items:
        combo_labels = [combo for combo, _ in top_combo_items]
        combo_counts = [count for _, count in top_combo_items]
        ax_combos.barh(combo_labels[::-1], combo_counts[::-1], color="#F28E2B")
        ax_combos.set_xlabel("Frames", fontweight="bold")
        ax_combos.set_title("Top Class Combinations", fontweight="bold", fontsize=12)
        ax_combos.grid(axis="x", linestyle="--", alpha=0.4)
        ax_combos.tick_params(axis="y", labelsize=9)
        for y, count in zip(combo_labels[::-1], combo_counts[::-1]):
            ax_combos.text(count, y, f" {count:,}", va="center", ha="left", fontsize=9, fontweight="bold")
    else:
        ax_combos.axis("off")

    # Combination table
    ax_table.axis("off")
    table_data = [[combo, count] for combo, count in table_items] or [["N/A", 0]]
    table = ax_table.table(
        cellText=table_data,
        colLabels=["Combination", "Frames"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    ax_table.set_title("Combination Summary", fontweight="bold", fontsize=12, pad=10)

    fig.suptitle("MegaFall v2 Label Statistics", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
