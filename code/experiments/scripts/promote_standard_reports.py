#!/usr/bin/env python3
"""Promote validated standard-arm RQ1 results into active report snapshots.

The paired standard-versus-targeted reports contain the authoritative standard-
arm RQ1 results.  This script extracts only RQ1, verifies that each report
refers to the unchanged active standard artifact, and writes a clean,
provenance-linked snapshot under ``reports/standard``.

No generator or evaluator is run, and no synthetic artifact is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"
DEFAULT_SOURCE_ROOT = EXPERIMENTS / "reports" / "standard_vs_tail_targeted"
DEFAULT_TARGET_ROOT = EXPERIMENTS / "reports" / "standard"

DATASETS = {
    "assistments": (
        "assistments_2009_2010_skill_builder",
        "assistments_2009_2010_skill_builder",
    ),
    "ednet": ("ednet_kt1", "ednet_kt1"),
    "oulad_weekly": (
        "oulad_weekly_engagement",
        "oulad_weekly_engagement_behavioral_dropout_v2",
    ),
}
DEFAULT_MODELS = ("markov_ngram", "sequence_vae_timevae", "timegan")
DEFAULT_SEEDS = (20260703, 20260704, 20260705)
SCHEMA_VERSION = "standard_rq1_report_v1"

RQ1_DIAGNOSTIC_KEYS = (
    "dataset_sizes",
    "evaluation_sampling",
    "primary_binary_signal",
    "primary_binary_signal_semantics",
    "global_fidelity",
    "temporal_fidelity",
    "cross_signal_dependence",
    "subgroup_fidelity",
    "tail_fidelity",
    "distinguishability",
    "diversity_coverage",
)
RQ1_EXCLUSION_KEYS = (
    "distinguishability",
    "diversity_uniqueness_rates",
    "omitted_standard_tail_groups",
    "raw_diagnostic_aggregates",
    "raw_test_fidelity_duplicates",
    "sequence_length_model_ranking",
    "short_and_long_trajectory_tails",
)


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


def _portable_paths(value: Any) -> Any:
    """Rewrite repository-local absolute paths without changing other values."""
    if isinstance(value, dict):
        return {key: _portable_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_paths(item) for item in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute():
            try:
                return _relative(path)
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


def _active_standard_paths(dataset: str, model: str, seed: int) -> dict[str, Path]:
    seed_dir = (
        EXPERIMENTS
        / "outputs"
        / "standard"
        / dataset
        / model
        / f"seed_{seed}"
    )
    return {
        "csv": seed_dir / "synth_generation.csv",
        "metadata": seed_dir / "generation_metadata.json",
        "provenance": seed_dir / "standard_provenance.json",
    }


def _validate_standard_artifact(paths: dict[str, Path], seed: int) -> dict[str, Any]:
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing active standard artifact: {path}")

    metadata = _read_json(paths["metadata"])
    artifact_sha256 = _sha256(paths["csv"])
    recorded_sha256 = metadata.get("synthetic_output_sha256")
    if recorded_sha256 != artifact_sha256:
        raise ValueError(
            f"Standard artifact hash mismatch for {paths['csv']}: "
            f"metadata={recorded_sha256}, actual={artifact_sha256}"
        )
    if int(metadata.get("seed", -1)) != int(seed):
        raise ValueError(f"Standard metadata seed mismatch for {paths['metadata']}")

    return {
        "synthetic_csv": _relative(paths["csv"]),
        "synthetic_csv_sha256": artifact_sha256,
        "generation_metadata": _relative(paths["metadata"]),
        "generation_metadata_sha256": _sha256(paths["metadata"]),
        "standard_provenance": _relative(paths["provenance"]),
        "standard_provenance_sha256": _sha256(paths["provenance"]),
    }


def _rq1_publication(publication: dict[str, Any]) -> dict[str, Any]:
    metrics = publication.get("metrics", {})
    rq1 = metrics.get("rq1") if isinstance(metrics, dict) else None
    if not isinstance(rq1, dict) or not rq1:
        raise ValueError("Source report has no publication.metrics.rq1 block")

    exclusions = publication.get("excluded_from_publication", {})
    if not isinstance(exclusions, dict):
        exclusions = {}
    rq_mapping = publication.get("research_question_mapping", {})
    if not isinstance(rq_mapping, dict):
        rq_mapping = {}

    direction_contract = publication.get("direction_contract", {})
    if not isinstance(direction_contract, dict):
        direction_contract = {}

    return {
        "schema_version": publication.get("schema_version"),
        "status": publication.get("status"),
        "direction_contract": {
            "fidelity_errors": deepcopy(
                direction_contract.get("fidelity_errors", "lower_is_better")
            ),
            "detectability_auc": "closer_to_0.5_is_better",
        },
        "research_question_mapping": {"rq1": deepcopy(rq_mapping.get("rq1"))},
        "selection_metadata": deepcopy(publication.get("selection_metadata", {})),
        "metrics": {"rq1": deepcopy(rq1)},
        "excluded_from_publication": {
            key: deepcopy(exclusions[key])
            for key in RQ1_EXCLUSION_KEYS
            if key in exclusions
        },
    }


def promote_seed(
    dataset: str,
    source_dataset_slug: str,
    target_dataset_slug: str,
    model: str,
    seed: int,
    source_root: Path,
    target_root: Path,
) -> Path:
    source_path = (
        source_root
        / source_dataset_slug
        / model
        / f"seed_{seed}"
        / "generation_evaluation_report.json"
    )
    source = _read_json(source_path)
    source_audit = source.get("final_audit", {})
    if not isinstance(source_audit, dict) or source_audit.get("status") != "passed":
        raise ValueError(f"Source report did not pass its strict audit: {source_path}")
    if int(source.get("generation_seed", -1)) != int(seed):
        raise ValueError(f"Generation seed mismatch in {source_path}")

    artifact = _validate_standard_artifact(
        _active_standard_paths(dataset, model, seed), seed
    )
    diagnostics = {
        key: deepcopy(source[key]) for key in RQ1_DIAGNOSTIC_KEYS if key in source
    }
    publication = _rq1_publication(source.get("publication", {}))

    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_arm": "standard",
        "research_question": {
            "id": "RQ1",
            "scope": (
                "Preservation of average learner behaviour and rare educational "
                "pathways by unmodified synthetic trajectory generators."
            ),
        },
        "dataset": target_dataset_slug,
        "source_dataset": source_dataset_slug,
        "dataset_key": dataset,
        "model": model,
        "generation_seed": int(seed),
        "evaluation_seed": source.get("evaluation_seed"),
        "standard_artifact": artifact,
        "source_snapshot": {
            "report": _relative(source_path),
            "report_sha256": _sha256(source_path),
            "source_final_audit": deepcopy(source_audit),
        },
        "publication": publication,
        "diagnostics": diagnostics,
        "final_audit": {
            "status": "passed",
            "issues": [],
            "scope": "standard_arm_rq1_only",
            "checks": [
                "source strict audit passed",
                "publication.metrics.rq1 is present",
                "active standard artifact hash matches generation metadata",
                "generation seed matches",
            ],
        },
    }
    target_path = (
        target_root
        / target_dataset_slug
        / model
        / f"seed_{seed}"
        / "generation_evaluation_report.json"
    )
    _write_json(target_path, report)
    return target_path


def promote_summary(
    dataset: str,
    source_dataset_slug: str,
    target_dataset_slug: str,
    model: str,
    seeds: list[int],
    source_root: Path,
    target_root: Path,
) -> Path:
    source_path = source_root / source_dataset_slug / model / "multi_seed_summary.json"
    source = _read_json(source_path)
    source_metrics = source.get("metric_summary", {})
    if not isinstance(source_metrics, dict):
        raise ValueError(f"Missing metric_summary in {source_path}")
    rq1_metrics = {
        key: deepcopy(value)
        for key, value in source_metrics.items()
        if str(key).startswith("rq1.")
    }
    if not rq1_metrics:
        raise ValueError(f"No RQ1 metrics found in {source_path}")

    report_paths = [
        target_root
        / target_dataset_slug
        / model
        / f"seed_{seed}"
        / "generation_evaluation_report.json"
        for seed in seeds
    ]
    for path in report_paths:
        report = _read_json(path)
        if report.get("final_audit", {}).get("status") != "passed":
            raise ValueError(f"Promoted seed report failed audit: {path}")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_arm": "standard",
        "research_question": "RQ1",
        "dataset": target_dataset_slug,
        "source_dataset": source_dataset_slug,
        "dataset_key": dataset,
        "model": model,
        "seeds": [int(seed) for seed in seeds],
        "metric_summary_scope": "publication_rq1_only",
        "publication_schema_version": source.get("publication_schema_version"),
        "metric_summary": rq1_metrics,
        "publication_reports": [_relative(path) for path in report_paths],
        "data_provenance": _portable_paths(
            deepcopy(source.get("data_provenance", {}))
        ),
        "source_snapshot": {
            "summary_report": _relative(source_path),
            "summary_report_sha256": _sha256(source_path),
        },
        "final_audit": {
            "status": "passed",
            "issues": [],
            "scope": "standard_arm_rq1_only",
            "seed_report_count": len(report_paths),
            "rq1_metric_count": len(rq1_metrics),
        },
    }
    target_path = target_root / target_dataset_slug / model / "multi_seed_summary.json"
    _write_json(target_path, summary)
    return target_path


def run(
    datasets: list[str],
    models: list[str],
    seeds: list[int],
    source_root: Path,
    target_root: Path,
) -> dict[str, Any]:
    promoted_seed_reports = []
    promoted_summaries = []
    for dataset in datasets:
        source_dataset_slug, target_dataset_slug = DATASETS[dataset]
        for model in models:
            for seed in seeds:
                promoted_seed_reports.append(
                    promote_seed(
                        dataset,
                        source_dataset_slug,
                        target_dataset_slug,
                        model,
                        int(seed),
                        source_root,
                        target_root,
                    )
                )
            promoted_summaries.append(
                promote_summary(
                    dataset,
                    source_dataset_slug,
                    target_dataset_slug,
                    model,
                    seeds,
                    source_root,
                    target_root,
                )
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "source_root": _relative(source_root),
        "target_root": _relative(target_root),
        "datasets": datasets,
        "models": models,
        "seeds": seeds,
        "seed_reports": len(promoted_seed_reports),
        "multi_seed_summaries": len(promoted_summaries),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create active RQ1-only snapshots from validated paired reports."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=list(DATASETS),
    )
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
