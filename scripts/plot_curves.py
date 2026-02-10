#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw


def read_metrics_csv(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "iter": int(r["iter"]),
                    "train_loss": float(r["train_loss"]),
                    "val_loss": float(r["val_loss"]),
                }
            )
    return rows


def draw_curve_plot(
    series: list[tuple[str, list[tuple[int, float]], tuple[int, int, int]]],
    out_path: Path,
    title: str,
) -> None:
    width, height = 1400, 900
    margin_l, margin_r, margin_t, margin_b = 120, 60, 80, 110
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    all_x = [x for _, pts, _ in series for (x, _) in pts]
    all_y = [y for _, pts, _ in series for (_, y) in pts]
    if not all_x or not all_y:
        raise ValueError("No points to plot")

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
    y_min -= y_pad
    y_max += y_pad

    def map_x(x: float) -> float:
        if x_max == x_min:
            return margin_l + plot_w * 0.5
        return margin_l + (x - x_min) / (x_max - x_min) * plot_w

    def map_y(y: float) -> float:
        if y_max == y_min:
            return margin_t + plot_h * 0.5
        return margin_t + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Axes
    draw.line([(margin_l, margin_t), (margin_l, margin_t + plot_h)], fill="black", width=3)
    draw.line([(margin_l, margin_t + plot_h), (margin_l + plot_w, margin_t + plot_h)], fill="black", width=3)
    draw.text((margin_l, 28), title, fill="black")
    draw.text((width // 2 - 40, height - 45), "Iteration", fill="black")
    draw.text((20, margin_t - 10), "Loss", fill="black")

    # Grid
    for i in range(6):
        yv = y_min + (y_max - y_min) * i / 5
        py = map_y(yv)
        draw.line([(margin_l, py), (margin_l + plot_w, py)], fill=(220, 220, 220), width=1)
        draw.text((35, py - 8), f"{yv:.3f}", fill="black")

    for i in range(6):
        xv = x_min + (x_max - x_min) * i / 5
        px = map_x(xv)
        draw.line([(px, margin_t), (px, margin_t + plot_h)], fill=(230, 230, 230), width=1)
        draw.text((px - 18, margin_t + plot_h + 10), f"{int(xv)}", fill="black")

    # Lines
    for name, pts, color in series:
        if len(pts) < 2:
            continue
        mapped = [(map_x(x), map_y(y)) for x, y in pts]
        draw.line(mapped, fill=color, width=4)
        lx, ly = mapped[-1]
        draw.text((lx + 8, ly - 8), name, fill=color)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot experiment loss curves from metrics.csv files")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Run directories containing metrics.csv")
    parser.add_argument("--which", choices=["val", "train"], default="val")
    parser.add_argument("--title", default="Loss Curves")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    colors = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ]

    series = []
    for i, run in enumerate(args.run_dirs):
        run_dir = Path(run)
        rows = read_metrics_csv(run_dir / "metrics.csv")
        if args.which == "val":
            pts = [(r["iter"], r["val_loss"]) for r in rows]
        else:
            pts = [(r["iter"], r["train_loss"]) for r in rows]
        series.append((run_dir.name, pts, colors[i % len(colors)]))

    draw_curve_plot(series, Path(args.output), title=args.title)
    print(f"Saved plot: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
