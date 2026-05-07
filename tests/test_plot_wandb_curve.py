from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "visualization"))

from plot_wandb_curve import collect_metrics, load_history, write_csv, write_svg


def test_plot_wandb_curve_reads_history_and_writes_outputs() -> None:
    with TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "offline-run-test"
        run_dir.mkdir()
        history_path = run_dir / "wandb-history.jsonl"
        rows = [
            {"_step": 1, "train/loss": 3.0, "train/lr": 1.0e-4, "train/volume_mean": 0.2},
            {"_step": 2, "train/loss": 2.5, "train/lr": 9.0e-5, "train/volume_mean": 0.3},
        ]
        with history_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        loaded = load_history(run_dir)
        curves = collect_metrics(
            loaded,
            ["train/loss", "train/lr", "train/volume_mean"],
            smooth=0.0,
        )
        assert curves["train/loss"] == [(1.0, 3.0), (2.0, 2.5)]
        assert curves["train/lr"][-1] == (2.0, 9.0e-5)

        out_dir = Path(tmp) / "out"
        csv_path = out_dir / "metrics.csv"
        svg_path = out_dir / "metrics.svg"
        write_csv(curves, csv_path)
        write_svg(curves, svg_path)

        assert "train/loss" in csv_path.read_text(encoding="utf-8")
        assert "<svg" in svg_path.read_text(encoding="utf-8")
        assert "train/volume_mean" in svg_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    test_plot_wandb_curve_reads_history_and_writes_outputs()
    print("ok")
