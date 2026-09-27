#!/usr/bin/env python3
"""Convert JSON evaluation reports into CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_BASE = PROJECT_ROOT / "experiments" / "reports"
REPORTS_ROOT = REPORTS_BASE / "standard_vs_tail_targeted"
DEFAULT_OUTPUT_DIR = REPORTS_BASE / "csv"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    if isinstance(value, dict):
        flattened: dict[str, float] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_numeric(child, child_prefix))
        return flattened
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {prefix: float(value)}
    return {}


def report_identity(path: Path) -> dict[str, str]:
    relative = path.relative_to(REPORTS_ROOT)
    parts = relative.parts
    dataset = parts[0] if len(parts) > 0 else ""
    model = parts[1] if len(parts) > 1 else ""
    seed = ""
    if len(parts) > 2 and parts[2].startswith("seed_"):
        seed = parts[2].replace("seed_", "", 1)
    return {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "source_file": str(path),
    }


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_wide(path: Path, id_fields: list[str], rows: list[dict[str, Any]]) -> None:
    metric_fields = sorted({key for row in rows for key in row if key not in id_fields})
    write_rows(path, id_fields + metric_fields, rows)


def build_seed_tables(report_files: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    for path in report_files:
        payload = load_json(path)
        identity = report_identity(path)
        metrics = flatten_numeric(payload)
        wide_rows.append({**identity, **metrics})
        for metric, value in sorted(metrics.items()):
            long_rows.append({**identity, "metric": metric, "value": value})
    return long_rows, wide_rows


def build_summary_tables(summary_files: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    for path in summary_files:
        payload = load_json(path)
        identity = report_identity(path)
        identity["dataset"] = str(payload.get("dataset", identity["dataset"]))
        identity["model"] = str(payload.get("model", identity["model"]))
        summary = payload.get("metric_summary", {})
        if not isinstance(summary, dict):
            continue

        wide_mean_row = {**identity}
        for metric, stats in sorted(summary.items()):
            if not isinstance(stats, dict):
                continue
            row = {**identity, "metric": metric}
            for stat_name in ("mean", "std", "min", "max"):
                if stat_name in stats and isinstance(stats[stat_name], (int, float)):
                    row[stat_name] = float(stats[stat_name])
            long_rows.append(row)
            if "mean" in row:
                wide_mean_row[metric] = row["mean"]
        wide_rows.append(wide_mean_row)
    return long_rows, wide_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export JSON report metrics to CSV.")
    parser.add_argument("--reports-root", default=str(REPORTS_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--expected-seeds",
        nargs="*",
        type=int,
        default=None,
        help="Fail unless every dataset/model cell has reports for these seeds.",
    )
    args = parser.parse_args()

    reports_root = Path(args.reports_root)
    output_dir = Path(args.output_dir)

    seed_files = sorted(
        reports_root.glob("*/*/seed_*/generation_evaluation_report.json")
    )
    summary_files = sorted(reports_root.glob("*/*/multi_seed_summary.json"))

    if args.expected_seeds:
        expected = {str(seed) for seed in args.expected_seeds}
        discovered: dict[tuple[str, str], set[str]] = {}
        for path in seed_files:
            identity = report_identity(path)
            key = (identity["dataset"], identity["model"])
            discovered.setdefault(key, set()).add(identity["seed"])

        summary_cells = {
            (report_identity(path)["dataset"], report_identity(path)["model"])
            for path in summary_files
        }
        problems = []
        for dataset, model in sorted(summary_cells):
            missing = expected - discovered.get((dataset, model), set())
            if missing:
                problems.append(
                    f"{dataset}/{model}: missing seeds {', '.join(sorted(missing))}"
                )
        if problems:
            raise SystemExit(
                "Cannot export complete multi-seed results:\n  " + "\n  ".join(problems)
            )

    seed_long, seed_wide = build_seed_tables(seed_files)
    summary_long, summary_wide = build_summary_tables(summary_files)

    write_rows(
        output_dir / "seed_metrics_long.csv",
        ["dataset", "model", "seed", "source_file", "metric", "value"],
        seed_long,
    )
    write_wide(
        output_dir / "seed_metrics_wide.csv",
        ["dataset", "model", "seed", "source_file"],
        seed_wide,
    )
    write_rows(
        output_dir / "summary_metrics_long.csv",
        ["dataset", "model", "seed", "source_file", "metric", "mean", "std", "min", "max"],
        summary_long,
    )
    write_wide(
        output_dir / "summary_metrics_wide.csv",
        ["dataset", "model", "seed", "source_file"],
        summary_wide,
    )

    print(f"Wrote CSV files to {output_dir}")
    print(f"Seed reports: {len(seed_files)}")
    print(f"Summary reports: {len(summary_files)}")


if __name__ == "__main__":
    main()
