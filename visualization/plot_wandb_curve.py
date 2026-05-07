#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


RUN_DIR = Path("/mnt/d/cl/mrl/wandb/offline-run-20260506_055038-8b0mj82x")
METRICS = ["train/loss", "train/lr", "train/volume_mean"]
OUT_DIR = Path("/mnt/d/cl/mrl/visualizations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot hard-coded W&B offline metrics.")
    parser.add_argument("--run_dir", type=Path, default=RUN_DIR)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--metrics", nargs="+", default=METRICS)
    parser.add_argument("--smooth", type=float, default=0.0, help="EMA smoothing weight in [0, 1).")
    return parser.parse_args()


def coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_value_json(value_json: str) -> Any:
    if value_json == "":
        return None
    try:
        return json.loads(value_json)
    except json.JSONDecodeError:
        return value_json


def read_history_jsonl(run_dir: Path) -> list[dict[str, Any]]:
    candidates = [
        run_dir / "wandb-history.jsonl",
        run_dir / "history.jsonl",
        run_dir / "files" / "wandb-history.jsonl",
        run_dir / "files" / "history.jsonl",
    ]
    rows: list[dict[str, Any]] = []
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        if rows:
            return rows
    return rows


def wandb_record_to_history(record: Any) -> dict[str, Any] | None:
    history = getattr(record, "history", None)
    items = list(getattr(history, "item", [])) if history is not None else []
    if not items:
        return None

    row: dict[str, Any] = {}
    for item in items:
        nested_key = list(getattr(item, "nested_key", []))
        key = ".".join(nested_key) if nested_key else getattr(item, "key", "")
        if not key:
            continue
        row[key] = parse_value_json(getattr(item, "value_json", ""))
    if "_step" not in row:
        step = getattr(history, "step", None)
        step_value = getattr(step, "num", None) if step is not None else None
        if step_value is not None:
            row["_step"] = step_value
    return row or None


def read_wandb_binary(run_dir: Path) -> list[dict[str, Any]]:
    wandb_files = sorted(run_dir.glob("*.wandb")) + sorted((run_dir / "files").glob("*.wandb"))
    if not wandb_files:
        return []

    try:
        from wandb.proto import wandb_internal_pb2
        from wandb.sdk.internal.datastore import DataStore
    except ImportError as exc:
        raise RuntimeError(
            "No JSONL history file was found and the `wandb` package is unavailable, "
            "so the offline .wandb file cannot be parsed in this environment."
        ) from exc

    rows: list[dict[str, Any]] = []
    for wandb_file in wandb_files:
        store = DataStore()
        store.open_for_scan(str(wandb_file))
        try:
            while True:
                data = store.scan_data()
                if data is None:
                    break
                record = wandb_internal_pb2.Record()
                record.ParseFromString(data)
                row = wandb_record_to_history(record)
                if row is not None:
                    rows.append(row)
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                close()
    return rows


def load_history(run_dir: Path) -> list[dict[str, Any]]:
    rows = read_history_jsonl(run_dir)
    if rows:
        return rows
    rows = read_wandb_binary(run_dir)
    if rows:
        return rows
    raise FileNotFoundError(f"No W&B history data found under {run_dir}.")


def metric_points(rows: list[dict[str, Any]], metric: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    fallback_step = 0
    for row in rows:
        if metric not in row:
            continue
        y = coerce_float(row[metric])
        if y is None:
            continue
        step = coerce_float(row.get("_step"))
        if step is None:
            step = float(fallback_step)
        points.append((step, y))
        fallback_step += 1
    return points


def smooth_points(points: list[tuple[float, float]], weight: float) -> list[tuple[float, float]]:
    if not points or weight <= 0:
        return points
    if weight >= 1:
        raise ValueError("--smooth must be smaller than 1.")
    smoothed: list[tuple[float, float]] = []
    last = points[0][1]
    for step, value in points:
        last = weight * last + (1.0 - weight) * value
        smoothed.append((step, last))
    return smoothed


def collect_metrics(
    rows: list[dict[str, Any]],
    metrics: list[str],
    *,
    smooth: float,
) -> dict[str, list[tuple[float, float]]]:
    curves = {metric: smooth_points(metric_points(rows, metric), smooth) for metric in metrics}
    missing = [metric for metric, points in curves.items() if not points]
    if missing:
        available = sorted({key for row in rows for key in row if not key.startswith("_")})
        raise KeyError(f"Metrics not found: {missing}. Available metrics include: {available[:50]}")
    return curves


def write_csv(curves: dict[str, list[tuple[float, float]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_step: dict[float, dict[str, float]] = defaultdict(dict)
    for metric, points in curves.items():
        for step, value in points:
            by_step[step][metric] = value

    metrics = list(curves.keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", *metrics])
        for step in sorted(by_step):
            writer.writerow([step, *[by_step[step].get(metric, "") for metric in metrics]])


def format_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value != 0 and (abs_value < 0.001 or abs_value >= 10000):
        return f"{value:.3e}"
    return f"{value:.4g}"


def polyline_points(
    points: list[tuple[float, float]],
    *,
    min_step: float,
    max_step: float,
    min_value: float,
    max_value: float,
    left: int,
    top: int,
    width: int,
    height: int,
) -> str:
    x_span = max(max_step - min_step, 1.0)
    y_span = max(max_value - min_value, 1.0e-12)
    mapped = []
    for step, value in points:
        x = left + (step - min_step) / x_span * width
        y = top + height - (value - min_value) / y_span * height
        mapped.append(f"{x:.2f},{y:.2f}")
    return " ".join(mapped)


def write_svg(curves: dict[str, list[tuple[float, float]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel_width = 980
    panel_height = 260
    margin_left = 86
    margin_right = 32
    margin_top = 46
    plot_width = panel_width - margin_left - margin_right
    plot_height = 165
    total_height = panel_height * len(curves)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{panel_width}" height="{total_height}" '
        f'viewBox="0 0 {panel_width} {total_height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}'
        '.axis{stroke:#333;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}'
        '.line{fill:none;stroke-width:2.2}</style>',
    ]

    for idx, (metric, points) in enumerate(curves.items()):
        top = idx * panel_height + margin_top
        left = margin_left
        steps = [step for step, _ in points]
        values = [value for _, value in points]
        min_step, max_step = min(steps), max(steps)
        min_value, max_value = min(values), max(values)
        if min_value == max_value:
            pad = abs(min_value) * 0.05 or 1.0
            min_value -= pad
            max_value += pad
        else:
            pad = (max_value - min_value) * 0.08
            min_value -= pad
            max_value += pad

        parts.append(f'<text x="{left}" y="{top - 20}" font-size="18" font-weight="700">{html.escape(metric)}</text>')
        latest_step, latest_value = points[-1]
        parts.append(
            f'<text x="{panel_width - margin_right}" y="{top - 20}" text-anchor="end" font-size="13">'
            f'latest step={format_number(latest_step)} value={format_number(latest_value)}</text>'
        )

        for grid_idx in range(5):
            y = top + grid_idx / 4 * plot_height
            value = max_value - grid_idx / 4 * (max_value - min_value)
            parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>')
            parts.append(
                f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12">'
                f'{format_number(value)}</text>'
            )
        for grid_idx in range(5):
            x = left + grid_idx / 4 * plot_width
            step = min_step + grid_idx / 4 * (max_step - min_step)
            parts.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/>')
            parts.append(
                f'<text x="{x:.2f}" y="{top + plot_height + 22}" text-anchor="middle" font-size="12">'
                f'{format_number(step)}</text>'
            )

        parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>')
        parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>')
        polyline = polyline_points(
            points,
            min_step=min_step,
            max_step=max_step,
            min_value=min_value,
            max_value=max_value,
            left=left,
            top=top,
            width=plot_width,
            height=plot_height,
        )
        parts.append(f'<polyline class="line" stroke="{colors[idx % len(colors)]}" points="{polyline}"/>')
        parts.append(f'<text x="{left + plot_width / 2}" y="{top + plot_height + 46}" text-anchor="middle" font-size="13">step</text>')

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_png_if_available(curves: dict[str, list[tuple[float, float]]], output_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(curves), 1, figsize=(10, 3 * len(curves)), squeeze=False)
    for axis, (metric, points) in zip(axes[:, 0], curves.items()):
        steps = [step for step, _ in points]
        values = [value for _, value in points]
        axis.plot(steps, values, linewidth=1.8)
        axis.set_title(metric)
        axis.set_xlabel("step")
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    if args.smooth < 0 or args.smooth >= 1:
        raise ValueError("--smooth must be in [0, 1).")

    rows = load_history(args.run_dir)
    curves = collect_metrics(rows, args.metrics, smooth=args.smooth)

    suffix = f"_smooth_{args.smooth:g}" if args.smooth > 0 else ""
    csv_path = args.out_dir / f"wandb_metrics{suffix}.csv"
    svg_path = args.out_dir / f"wandb_metrics{suffix}.svg"
    png_path = args.out_dir / f"wandb_metrics{suffix}.png"
    write_csv(curves, csv_path)
    write_svg(curves, svg_path)
    wrote_png = write_png_if_available(curves, png_path)

    print(f"Loaded {len(rows)} history rows from {args.run_dir}")
    for metric, points in curves.items():
        print(f"{metric}: {len(points)} points, last step={points[-1][0]}, last value={points[-1][1]}")
    print(f"CSV: {csv_path}")
    print(f"SVG: {svg_path}")
    if wrote_png:
        print(f"PNG: {png_path}")
    else:
        print("PNG skipped: matplotlib is not installed; SVG was written with stdlib only.")


if __name__ == "__main__":
    main()
