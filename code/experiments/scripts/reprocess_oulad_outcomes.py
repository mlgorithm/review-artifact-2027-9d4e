#!/usr/bin/env python3
"""Version and reevaluate existing OULAD trajectories with behavioral dropout.

This script never invokes a synthetic generator. It preserves the original
artifacts, rewrites only the three derived terminal-outcome columns into a
versioned compressed CSV, and writes evaluation reports to a separate tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from data.dataset_scripts.oulad_weekly import dataset as adapter
from evaluation.general_evaluation import run_general_evaluation
from evaluation.publication import SCHEMA_VERSION, build_publication_report
from pipeline import (
    final_report_issues,
    flatten_numeric,
    sha256_file,
    summarize_metrics,
    summarize_publication_statuses,
    write_json,
)


VARIANT = "behavioral_dropout_v2"
BASE_OUTPUT_ROOT = EXPERIMENTS / "outputs" / "oulad_weekly"
OUTPUT_ROOT = EXPERIMENTS / "outputs" / "oulad_weekly_behavioral_dropout_v2"
REPORT_ROOT = EXPERIMENTS / "reports" / "oulad_weekly_engagement_behavioral_dropout_v2"
DEFAULT_MODELS = ("markov_ngram", "sequence_vae_timevae", "timegan")
DEFAULT_SEEDS = (20260703, 20260704, 20260705)
OUTCOME_COLUMNS = tuple(adapter.TERMINAL_OUTCOME_COLUMNS)


def _arm_folder(root: Path, model: str, seed: int, targeted: bool) -> Path:
    folder = root / model / f"seed_{seed}"
    return folder / "tail_targeted" if targeted else folder


def _write_compressed_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.gz")
    frame.to_csv(
        temporary,
        index=False,
        compression={"method": "gzip", "compresslevel": 3, "mtime": 0},
    )
    temporary.replace(path)


def _learner_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("learner_id", sort=False)[list(OUTCOME_COLUMNS)]
        .first()
        .sort_index()
    )


def reprocess_arm(
    split,
    model: str,
    seed: int,
    targeted: bool,
    fitted,
    *,
    base_output_root: Path = BASE_OUTPUT_ROOT,
    output_root: Path = OUTPUT_ROOT,
    base_frame: pd.DataFrame | None = None,
    variant: str = VARIANT,
    metadata_extra: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, Path, dict[str, object]]:
    base_folder = _arm_folder(base_output_root, model, seed, targeted)
    output_folder = _arm_folder(output_root, model, seed, targeted)
    base_path = base_folder / "synth_generation.csv"
    base_metadata_path = base_folder / "generation_metadata.json"
    output_path = output_folder / "synth_generation.csv.gz"
    metadata_path = output_folder / "outcome_reprocessing_metadata.json"
    if not base_path.exists() or not base_metadata_path.exists():
        raise FileNotFoundError(f"Missing existing OULAD generation artifact: {base_path}")

    generation_metadata = json.loads(base_metadata_path.read_text(encoding="utf-8"))
    if sha256_file(base_path) != generation_metadata.get("synthetic_output_sha256"):
        raise RuntimeError(
            f"OULAD base artifact hash differs from generation metadata: {base_path}"
        )
    base = base_frame if base_frame is not None else pd.read_csv(base_path, low_memory=False)
    if base.columns.tolist() != split.train.columns.tolist():
        raise RuntimeError("OULAD base artifact schema differs from the current split.")
    if int(generation_metadata.get("synthetic_rows", -1)) != len(base):
        raise RuntimeError("OULAD base artifact row count differs from generation metadata.")
    before = _learner_outcomes(base)
    reprocessed, outcome_metadata = adapter.reassign_terminal_outcomes(
        base, split.train, seed, fitted=fitted
    )
    after = _learner_outcomes(reprocessed)
    non_outcome_columns = [column for column in base.columns if column not in OUTCOME_COLUMNS]
    if not base[non_outcome_columns].equals(reprocessed[non_outcome_columns]):
        raise RuntimeError("Outcome reprocessing changed non-outcome trajectory columns.")
    if reprocessed.columns.tolist() != base.columns.tolist():
        raise RuntimeError("Outcome reprocessing changed the synthetic schema.")

    _write_compressed_csv(output_path, reprocessed)
    metadata: dict[str, object] = {
        "variant": variant,
        "dataset": split.name,
        "model": model,
        "seed": int(seed),
        "arm": "tail_targeted" if targeted else "standard",
        "base_synthetic_path": str(base_path),
        "base_synthetic_sha256": sha256_file(base_path),
        "base_generation_metadata_path": str(base_metadata_path),
        "base_generation_metadata_sha256": sha256_file(base_metadata_path),
        "outcome_adapter_path": str(Path(adapter.__file__).resolve()),
        "outcome_adapter_sha256": sha256_file(Path(adapter.__file__).resolve()),
        "reprocessing_script_path": str(Path(__file__).resolve()),
        "reprocessing_script_sha256": sha256_file(Path(__file__).resolve()),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "rows": int(len(reprocessed)),
        "learners": int(reprocessed["learner_id"].nunique()),
        "schema": reprocessed.columns.tolist(),
        "non_outcome_columns_unchanged": True,
        "outcome_columns": list(OUTCOME_COLUMNS),
        "changed_learner_labels": {
            column: int((before[column].astype(str) != after[column].astype(str)).sum())
            for column in OUTCOME_COLUMNS
        },
        "dropout_labeler": outcome_metadata,
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    write_json(metadata_path, metadata)
    return reprocessed, output_path, metadata


def run_seed(
    split,
    model: str,
    seed: int,
    fitted,
    strict: bool,
    resume: bool,
) -> dict[str, object]:
    report_path = (
        REPORT_ROOT / model / f"seed_{seed}" / "generation_evaluation_report.json"
    )
    pipeline_report_path = (
        OUTPUT_ROOT / model / f"seed_{seed}" / "outcome_v2_pipeline_report.json"
    )
    if resume and report_path.exists() and pipeline_report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        run_report = json.loads(pipeline_report_path.read_text(encoding="utf-8"))
        if "publication" not in report:
            report["publication"] = build_publication_report(report, split)
            write_json(report_path, report)
        issues = final_report_issues(report)
        if report.get("outcome_postprocessing", {}).get("variant") != VARIANT:
            raise RuntimeError(f"Cannot resume an incompatible report: {report_path}")
        if strict and issues:
            raise RuntimeError(
                f"Cannot resume failed evaluation for {model} seed {seed}:\n- "
                + "\n- ".join(issues)
            )
        return {
            **run_report,
            "_metrics": flatten_numeric(report["publication"]["metrics"]),
            "_publication": report["publication"]["metrics"],
        }

    standard, standard_path, standard_metadata = reprocess_arm(
        split, model, seed, False, fitted
    )
    targeted, targeted_path, targeted_metadata = reprocess_arm(
        split, model, seed, True, fitted
    )
    report = run_general_evaluation(
        adapter,
        split,
        standard,
        standard_path,
        targeted,
        targeted_path,
        generation_seed=seed,
    )
    report["outcome_postprocessing"] = {
        "variant": VARIANT,
        "standard_metadata": str(
            _arm_folder(OUTPUT_ROOT, model, seed, False)
            / "outcome_reprocessing_metadata.json"
        ),
        "tail_targeted_metadata": str(
            _arm_folder(OUTPUT_ROOT, model, seed, True)
            / "outcome_reprocessing_metadata.json"
        ),
        "generator_retrained": False,
        "base_trajectory_columns_changed": False,
    }
    issues = final_report_issues(report)
    report["final_audit"] = {
        "status": "failed" if issues else "passed",
        "issues": issues,
        "strict_mode": bool(strict),
    }
    write_json(report_path, report)
    if strict and issues:
        raise RuntimeError(
            f"Behavioral-dropout evaluation failed for {model} seed {seed}:\n- "
            + "\n- ".join(issues)
        )

    run_report = {
        "variant": VARIANT,
        "dataset": split.name,
        "model": model,
        "seed": int(seed),
        "standard_synthetic": str(standard_path),
        "tail_targeted_synthetic": str(targeted_path),
        "standard_outcome_metadata": standard_metadata,
        "tail_targeted_outcome_metadata": targeted_metadata,
        "evaluation_report": str(report_path),
    }
    write_json(pipeline_report_path, run_report)
    return {
        **run_report,
        "_metrics": flatten_numeric(report["publication"]["metrics"]),
        "_publication": report["publication"]["metrics"],
    }


def run(
    models: list[str], seeds: list[int], strict: bool, resume: bool
) -> dict[str, object]:
    split = adapter.load_split()
    fitted = adapter.fit_dropout_behavior_model(split.train)
    summaries = {}
    for model in models:
        seed_reports = [
            run_seed(split, model, int(seed), fitted, strict, resume) for seed in seeds
        ]
        public_runs = [
            {key: value for key, value in report.items() if not key.startswith("_")}
            for report in seed_reports
        ]
        summary = {
            "variant": VARIANT,
            "dataset": split.name,
            "model": model,
            "seeds": [int(seed) for seed in seeds],
            "generator_retrained": False,
            "base_trajectory_columns_changed": False,
            "runs": public_runs,
            "metric_summary": summarize_metrics(seed_reports),
            "publication_status_summary": summarize_publication_statuses(seed_reports),
            "metric_summary_scope": "publication_only",
            "publication_schema_version": SCHEMA_VERSION,
        }
        summary_path = REPORT_ROOT / model / "multi_seed_summary.json"
        summary["summary_report"] = str(summary_path)
        write_json(summary_path, summary)
        summaries[model] = summary
    return {
        "variant": VARIANT,
        "dataset": split.name,
        "models": models,
        "seeds": seeds,
        "summaries": {
            model: summary["summary_report"] for model, summary in summaries.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Write diagnostic reports without failing on an incomplete audit.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed compatible reports and continue incomplete runs.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.models,
                args.seeds,
                strict=not args.no_strict,
                resume=args.resume,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
