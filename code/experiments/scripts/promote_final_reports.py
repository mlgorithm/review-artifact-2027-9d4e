#!/usr/bin/env python3
"""Promote audited local RQ1/RQ2 reports into tracked reference snapshots.

The experiment runner writes immutable working reports below ``experiments/runs``.
That workspace is intentionally git-ignored because it also contains manifests
and resumable local state. This script copies only completed evaluation reports
into ``experiments/reports/standard_vs_tail_targeted`` and rewrites repository-
local absolute paths as portable repository-relative paths.

No generator or evaluator is invoked and no synthetic artifact is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"
DEFAULT_SOURCE_ROOT = (
    EXPERIMENTS / "runs" / "standard_vs_tail_targeted" / "evaluation"
)
DEFAULT_TARGET_ROOT = EXPERIMENTS / "reports" / "standard_vs_tail_targeted"
DEFAULT_DATASETS = (
    "assistments_2009_2010_skill_builder",
    "ednet_kt1",
    "oulad_weekly_engagement",
)
DEFAULT_MODELS = ("markov_ngram", "sequence_vae_timevae", "timegan")
DEFAULT_SEEDS = (20260703, 20260704, 20260705)
EXPECTED_EXPERIMENT_DESIGN = "standard_vs_tail_targeted"
EXPECTED_PUBLICATION_SCHEMA = "publication_metrics_v2"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required report: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _resolve_repo_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected a repository artifact path, found {value!r}.")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _portable_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _portable_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_paths(item) for item in value]
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                return _relative(candidate)
            except ValueError:
                return value
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == rendered:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def _has_finite_number(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_has_finite_number(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_finite_number(item) for item in value)
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_seed_report(
    report: dict[str, Any], dataset: str, model: str, seed: int, source_path: Path
) -> None:
    issues = []
    if report.get("experiment_design") != EXPECTED_EXPERIMENT_DESIGN:
        issues.append("experiment design differs")
    if int(report.get("generation_seed", -1)) != int(seed):
        issues.append("generation seed differs")
    if report.get("dataset") not in (None, dataset):
        issues.append("dataset differs")
    if report.get("model") not in (None, model):
        issues.append("model differs")
    if report.get("publication", {}).get("schema_version") != EXPECTED_PUBLICATION_SCHEMA:
        issues.append("publication schema differs")
    if report.get("publication", {}).get("status") != "completed":
        issues.append("publication report is incomplete")
    publication_metrics = report.get("publication", {}).get("metrics", {})
    if not isinstance(publication_metrics, dict) or not all(
        isinstance(publication_metrics.get(name), dict)
        and publication_metrics[name]
        and _has_finite_number(publication_metrics[name])
        for name in ("rq1", "rq2")
    ):
        issues.append("publication metric tree is incomplete")
    if (
        report.get("downstream_utility_models", {})
        .get("tail_learnability", {})
        .get("status")
        != "completed"
    ):
        issues.append("tail-learnability evaluation is incomplete")
    if report.get("downstream_utility_tasks", {}).get("status") != "completed":
        issues.append("learner-level outcome tasks are incomplete")
    audit = report.get("final_audit", {})
    if audit.get("status") != "passed" or audit.get("issues"):
        issues.append("strict final audit did not pass cleanly")
    if issues:
        raise ValueError(f"Invalid report {source_path}: " + "; ".join(issues))


def _validated_artifact_provenance(
    run: dict[str, Any],
    *,
    arm: str,
    model: str,
    seed: int,
) -> dict[str, Any]:
    if arm == "standard":
        path_key = "standard_synthetic"
        metadata_key = "generation_metadata"
    elif arm == "tail_targeted":
        path_key = "tail_targeted_synthetic"
        metadata_key = "tail_targeted_generation_metadata"
    else:  # pragma: no cover - internal caller contract
        raise ValueError(f"Unknown artifact arm: {arm}")

    artifact_path = _resolve_repo_path(run.get(path_key))
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Missing {arm} synthetic artifact: {artifact_path}")
    metadata = run.get(metadata_key)
    if not isinstance(metadata, dict):
        raise ValueError(f"Run is missing complete {arm} generation metadata.")

    issues = []
    if int(metadata.get("seed", -1)) != int(seed):
        issues.append("seed differs")
    if str(metadata.get("model")) != str(model):
        issues.append("model differs")
    if arm == "tail_targeted" and metadata.get("generation_arm") != "tail_targeted":
        issues.append("generation arm differs")
    artifact_hash = _sha256(artifact_path)
    if artifact_hash != metadata.get("synthetic_output_sha256"):
        issues.append("artifact SHA-256 differs from generation metadata")
    if issues:
        raise ValueError(
            f"Invalid {arm} provenance for {model}/seed_{seed}: "
            + "; ".join(issues)
        )
    return {"path": str(artifact_path), "metadata": deepcopy(metadata)}


def _source_run(
    source_root: Path, dataset: str, model: str, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = source_root / dataset / model / "multi_seed_summary.json"
    summary = _read_json(summary_path)
    issues = []
    if summary.get("experiment_design") != EXPECTED_EXPERIMENT_DESIGN:
        issues.append("experiment design differs")
    if summary.get("dataset") != dataset:
        issues.append("dataset differs")
    if summary.get("model") != model:
        issues.append("model differs")
    if issues:
        raise ValueError(f"Invalid source summary {summary_path}: " + "; ".join(issues))

    matching = [
        run
        for run in summary.get("runs", [])
        if isinstance(run, dict) and int(run.get("seed", -1)) == int(seed)
    ]
    if len(matching) != 1:
        raise ValueError(
            f"Expected exactly one run for seed {seed} in {summary_path}, found {len(matching)}."
        )
    run = matching[0]
    if run.get("dataset") != dataset or run.get("model") != model:
        raise ValueError(
            f"Run identity differs in {summary_path}: "
            f"dataset={run.get('dataset')!r}, model={run.get('model')!r}."
        )
    provenance = {
        "standard": _validated_artifact_provenance(
            run, arm="standard", model=model, seed=seed
        ),
        "tail_targeted": _validated_artifact_provenance(
            run, arm="tail_targeted", model=model, seed=seed
        ),
    }
    return run, provenance


def _normalize_task_metadata(report: dict[str, Any], dataset: str) -> list[str]:
    expected = (
        "next_week_engagement"
        if dataset == "oulad_weekly_engagement"
        else "next_response_correctness"
    )
    corrections = []
    targeted = report.get("tail_targeted_evaluation")
    blocks = [
        report.get("downstream_utility"),
        report.get("downstream_utility_models"),
        targeted.get("downstream_utility") if isinstance(targeted, dict) else None,
    ]
    for block in blocks:
        if isinstance(block, dict) and block.get("task") != expected:
            block["task"] = expected
            corrections.append("primary_downstream_task_label")
    return sorted(set(corrections))


def promote_seed(
    dataset: str,
    model: str,
    seed: int,
    source_root: Path,
    target_root: Path,
) -> Path:
    source_path = (
        source_root
        / dataset
        / model
        / f"seed_{seed}"
        / "generation_evaluation_report.json"
    )
    report = _read_json(source_path)
    _, artifact_provenance = _source_run(source_root, dataset, model, seed)
    _validate_seed_report(report, dataset, model, seed, source_path)
    promoted = deepcopy(report)
    promoted["dataset"] = dataset
    promoted["model"] = model
    promoted["artifact_provenance"] = artifact_provenance
    metadata_corrections = _normalize_task_metadata(promoted, dataset)
    promoted = _portable_paths(promoted)
    promoted["curated_snapshot"] = {
        "source_report": _relative(source_path),
        "source_report_sha256": _sha256(source_path),
        "promotion_script": _relative(Path(__file__)),
        "promotion_script_sha256": _sha256(Path(__file__)),
        "paths_portabilized": True,
        "generator_retrained": False,
        "evaluator_rerun": False,
        "metric_values_changed": False,
        "metadata_enriched_from_summary": True,
        "metadata_corrections": metadata_corrections,
    }
    target_path = (
        target_root
        / dataset
        / model
        / f"seed_{seed}"
        / "generation_evaluation_report.json"
    )
    _write_json(target_path, promoted)
    return target_path


def promote_summary(
    dataset: str,
    model: str,
    seeds: list[int],
    source_root: Path,
    target_root: Path,
) -> Path:
    source_path = source_root / dataset / model / "multi_seed_summary.json"
    summary = _read_json(source_path)
    issues = []
    if summary.get("experiment_design") != EXPECTED_EXPERIMENT_DESIGN:
        issues.append("experiment design differs")
    if summary.get("dataset") != dataset:
        issues.append("dataset differs")
    if summary.get("model") != model:
        issues.append("model differs")
    if [int(seed) for seed in summary.get("seeds", [])] != seeds:
        issues.append("seed list differs")
    if summary.get("publication_schema_version") != EXPECTED_PUBLICATION_SCHEMA:
        issues.append("publication schema differs")
    if summary.get("metric_summary_scope") != "publication_only":
        issues.append("metric summary is not publication-only")
    for seed in seeds:
        try:
            _source_run(source_root, dataset, model, seed)
        except (FileNotFoundError, ValueError) as exc:
            issues.append(f"seed {seed} artifact provenance is invalid: {exc}")
        target_seed = (
            target_root
            / dataset
            / model
            / f"seed_{seed}"
            / "generation_evaluation_report.json"
        )
        if _read_json(target_seed).get("final_audit", {}).get("status") != "passed":
            issues.append(f"promoted seed {seed} did not pass its audit")
    if issues:
        raise ValueError(f"Invalid summary {source_path}: " + "; ".join(issues))

    promoted = _portable_paths(deepcopy(summary))
    promoted["curated_snapshot"] = {
        "source_report": _relative(source_path),
        "source_report_sha256": _sha256(source_path),
        "promotion_script": _relative(Path(__file__)),
        "promotion_script_sha256": _sha256(Path(__file__)),
        "paths_portabilized": True,
        "generator_retrained": False,
        "evaluator_rerun": False,
        "metric_values_changed": False,
        "targeted_provenance_embedded": True,
    }
    promoted["summary_report"] = _relative(
        target_root / dataset / model / "multi_seed_summary.json"
    )
    target_path = target_root / dataset / model / "multi_seed_summary.json"
    _write_json(target_path, promoted)
    return target_path


def run(
    datasets: list[str],
    models: list[str],
    seeds: list[int],
    source_root: Path,
    target_root: Path,
) -> dict[str, Any]:
    seed_reports = []
    summaries = []
    for dataset in datasets:
        for model in models:
            for seed in seeds:
                seed_reports.append(
                    promote_seed(dataset, model, seed, source_root, target_root)
                )
            summaries.append(
                promote_summary(dataset, model, seeds, source_root, target_root)
            )
    return {
        "experiment_design": EXPECTED_EXPERIMENT_DESIGN,
        "publication_schema_version": EXPECTED_PUBLICATION_SCHEMA,
        "source_root": _relative(source_root),
        "target_root": _relative(target_root),
        "datasets": datasets,
        "models": models,
        "seeds": seeds,
        "seed_reports": len(seed_reports),
        "multi_seed_summaries": len(summaries),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        datasets=list(args.datasets),
        models=list(args.models),
        seeds=[int(seed) for seed in args.seeds],
        source_root=args.source_root.resolve(),
        target_root=args.target_root.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
