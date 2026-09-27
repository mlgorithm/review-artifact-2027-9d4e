#!/usr/bin/env python3
"""Regenerate only corrected OULAD tail-targeted arms and reevaluate them.

The standard generator trajectories are reused after provenance validation. Both
arms receive the same current behavioral-dropout-v2 outcome postprocessor, while
only the targeted arm is refit on the proposal-aligned tail-selection-v2 input.
Outputs are written to a new versioned tree; legacy targeted artifacts are never
overwritten.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import pipeline as pipeline_module
from data.dataset_scripts.oulad_weekly import dataset as adapter
from evaluation import general_evaluation as evaluation_module
from evaluation import metrics as metrics_module
from evaluation import publication as publication_module
from scripts import reprocess_oulad_outcomes as outcome_script
from synthetic_generation.run_generator import (
    generator_source_provenance,
    load_generator_script,
)


VARIANT = "behavioral_dropout_v2_tail_selection_v2"
BASE_STANDARD_ROOT = EXPERIMENTS / "outputs" / "oulad_weekly"
TARGET_BASE_ROOT = EXPERIMENTS / "outputs" / "oulad_weekly_tail_selection_v2"
OUTPUT_ROOT = (
    EXPERIMENTS
    / "outputs"
    / "oulad_weekly_behavioral_dropout_v2_tail_selection_v2"
)
REPORT_ROOT = (
    EXPERIMENTS
    / "reports"
    / "oulad_weekly_engagement_behavioral_dropout_v2_tail_selection_v2"
)
DEFAULT_MODELS = ("markov_ngram", "sequence_vae_timevae", "timegan")
DEFAULT_SEEDS = (20260703, 20260704, 20260705)
DEFAULT_TAIL_OVERSAMPLE = pipeline_module.DEFAULT_TAIL_OVERSAMPLE


def _arm_folder(root: Path, model: str, seed: int, targeted: bool) -> Path:
    folder = root / model / f"seed_{seed}"
    return folder / "tail_targeted" if targeted else folder


def _source_fingerprint(model: str) -> dict[str, object]:
    generator = load_generator_script(model)
    _, generator_bundle = generator_source_provenance(generator)
    source_paths = {
        "target_only_script": Path(__file__).resolve(),
        "pipeline": Path(pipeline_module.__file__).resolve(),
        "oulad_adapter": Path(adapter.__file__).resolve(),
        "outcome_reprocessing": Path(outcome_script.__file__).resolve(),
        "general_evaluation": Path(evaluation_module.__file__).resolve(),
        "evaluation_metrics": Path(metrics_module.__file__).resolve(),
        "publication_selection": Path(publication_module.__file__).resolve(),
    }
    return {
        "variant": VARIANT,
        "publication_schema_version": publication_module.SCHEMA_VERSION,
        "tail_selection_schema_version": pipeline_module.TAIL_SELECTION_SCHEMA_VERSION,
        "generator_source_bundle_sha256": generator_bundle,
        "source_files_sha256": {
            name: pipeline_module.sha256_file(path)
            for name, path in sorted(source_paths.items())
        },
    }


def _validate_standard_base(split, model: str, seed: int) -> dict[str, object]:
    folder = _arm_folder(BASE_STANDARD_ROOT, model, seed, False)
    validated = pipeline_module._validate_standard_artifact_provenance(
        model, split, folder, seed
    )
    if validated is None:
        raise FileNotFoundError(
            f"Missing standard OULAD generator artifact for {model}/seed_{seed}: {folder}"
        )
    path, metadata, pipeline_report = validated
    if metadata.get("columns") != split.train.columns.tolist():
        raise RuntimeError(
            f"Standard OULAD schema differs for {model}/seed_{seed}."
        )
    if int(metadata.get("synthetic_rows", -1)) != len(split.train):
        raise RuntimeError(
            f"Standard OULAD row budget differs for {model}/seed_{seed}."
        )
    learner = split.columns.get("learner", "learner_id")
    expected_learners = int(split.train[learner].nunique())
    if int(metadata.get("synthetic_learners", -1)) != expected_learners:
        raise RuntimeError(
            f"Standard OULAD learner budget differs for {model}/seed_{seed}."
        )
    return {
        "path": str(path),
        "sha256": metadata["synthetic_output_sha256"],
        "generation_metadata": str(folder / "generation_metadata.json"),
        "generation_metadata_sha256": pipeline_module.sha256_file(
            folder / "generation_metadata.json"
        ),
        "pipeline_report": str(folder / "pipeline_report.json"),
        "pipeline_report_sha256": pipeline_module.sha256_file(
            folder / "pipeline_report.json"
        ),
        "data_provenance": pipeline_report.get("data_provenance", {}),
    }


def _resume_result(
    split,
    model: str,
    seed: int,
    oversample: int,
    strict: bool,
) -> dict[str, object] | None:
    report_path = (
        REPORT_ROOT / model / f"seed_{seed}" / "generation_evaluation_report.json"
    )
    run_report_path = (
        OUTPUT_ROOT / model / f"seed_{seed}" / "target_only_pipeline_report.json"
    )
    present = [report_path.exists(), run_report_path.exists()]
    if not any(present):
        return None
    if not all(present):
        raise RuntimeError(
            f"Cannot resume incomplete OULAD target-only outputs for {model}/seed_{seed}."
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
    reasons = []
    if run_report.get("dataset") != split.name:
        reasons.append("dataset name differs")
    if run_report.get("model") != model or int(run_report.get("seed", -1)) != int(seed):
        reasons.append("model or seed differs")
    if int(run_report.get("tail_oversample", -1)) != int(oversample):
        reasons.append("tail-oversampling factor differs")
    if run_report.get("reproducibility_fingerprint") != _source_fingerprint(model):
        reasons.append("source or schema fingerprint changed")
    if run_report.get("data_provenance") != split.metadata.get("artifact_metadata", {}):
        reasons.append("processed dataset provenance changed")
    if report.get("final_audit", {}).get("status") != "passed":
        reasons.append("prior final audit did not pass")
    contract = report.get("tail_targeting_contract", {})
    if contract.get("schema_version") != pipeline_module.TAIL_SELECTION_SCHEMA_VERSION:
        reasons.append("tail-selection schema differs")
    current_standard = _validate_standard_base(split, model, seed)
    if run_report.get("standard_base_validation") != current_standard:
        reasons.append("standard base artifact or provenance changed")
    for label, artifact in run_report.get("artifacts", {}).items():
        path = Path(str(artifact.get("path", "")))
        expected = artifact.get("sha256")
        if not path.exists() or not expected or pipeline_module.sha256_file(path) != expected:
            reasons.append(f"{label} artifact is missing or changed")
    if reasons:
        raise RuntimeError(
            f"Cannot resume incompatible OULAD target-only run {model}/seed_{seed}: "
            + "; ".join(reasons)
        )

    issues = pipeline_module.final_report_issues(report)
    if strict and issues:
        raise RuntimeError(
            f"Cannot resume failed OULAD evaluation for {model}/seed_{seed}:\n- "
            + "\n- ".join(issues)
        )
    return {
        **run_report,
        "resumed": True,
        "_metrics": pipeline_module.flatten_numeric(
            report["publication"]["metrics"]
        ),
        "_publication": report["publication"]["metrics"],
    }


def _target_training(split, oversample: int):
    targeted, contract = pipeline_module.tail_oversampled_generation_train(
        split, oversample, return_metadata=True
    )
    if targeted is None:
        raise RuntimeError(
            "Corrected OULAD tail selection produced no targeted training set: "
            + json.dumps(contract, sort_keys=True)
        )
    return targeted, contract


def run_seed(
    split,
    model: str,
    seed: int,
    fitted,
    oversample: int,
    strict: bool,
    resume: bool,
) -> dict[str, object]:
    if resume:
        resumed = _resume_result(split, model, seed, oversample, strict)
        if resumed is not None:
            return resumed

    standard_base = _validate_standard_base(split, model, seed)
    tail_train, tail_contract = _target_training(split, oversample)
    learner = split.columns.get("learner", "learner_id")
    target_base_folder = _arm_folder(TARGET_BASE_ROOT, model, seed, True)
    targeted_raw, targeted_raw_path, target_generation_metadata = (
        pipeline_module._generate_synthetic(
            model,
            split,
            tail_train,
            target_base_folder,
            seed,
            adapter,
            generation_shape={
                "target_rows": int(len(split.train)),
                "learner_count": int(split.train[learner].nunique()),
            },
        )
    )
    del tail_train
    gc.collect()

    target_generation_metadata["tail_selection"] = tail_contract
    pipeline_module.write_json(
        target_base_folder / "generation_metadata.json", target_generation_metadata
    )

    targeted, targeted_path, targeted_outcome_metadata = outcome_script.reprocess_arm(
        split,
        model,
        seed,
        True,
        fitted,
        base_output_root=TARGET_BASE_ROOT,
        output_root=OUTPUT_ROOT,
        base_frame=targeted_raw,
        variant=VARIANT,
        metadata_extra={
            "standard_generator_retrained": False,
            "target_generator_retrained": True,
            "tail_selection": tail_contract,
        },
    )
    del targeted_raw
    gc.collect()

    standard, standard_path, standard_outcome_metadata = outcome_script.reprocess_arm(
        split,
        model,
        seed,
        False,
        fitted,
        base_output_root=BASE_STANDARD_ROOT,
        output_root=OUTPUT_ROOT,
        variant=VARIANT,
        metadata_extra={
            "standard_generator_retrained": False,
            "target_generator_retrained": True,
            "standard_base_validation": standard_base,
        },
    )

    report = evaluation_module.run_general_evaluation(
        adapter,
        split,
        standard,
        standard_path,
        targeted,
        targeted_path,
        generation_seed=seed,
    )
    report["tail_targeting_contract"] = tail_contract
    report["outcome_postprocessing"] = {
        "variant": VARIANT,
        "base_outcome_variant": outcome_script.VARIANT,
        "standard_generator_retrained": False,
        "target_generator_retrained": True,
        "base_trajectory_columns_changed": False,
        "standard_metadata": str(
            _arm_folder(OUTPUT_ROOT, model, seed, False)
            / "outcome_reprocessing_metadata.json"
        ),
        "tail_targeted_metadata": str(
            _arm_folder(OUTPUT_ROOT, model, seed, True)
            / "outcome_reprocessing_metadata.json"
        ),
    }
    report["target_only_regeneration"] = {
        "variant": VARIANT,
        "standard_base_validation": standard_base,
        "target_raw_artifact": str(targeted_raw_path),
        "target_generation_metadata": str(
            target_base_folder / "generation_metadata.json"
        ),
        "reproducibility_fingerprint": _source_fingerprint(model),
    }

    issues = pipeline_module.final_report_issues(report)
    report["final_audit"] = {
        "status": "failed" if issues else "passed",
        "issues": issues,
        "strict_mode": bool(strict),
    }
    report_path = (
        REPORT_ROOT / model / f"seed_{seed}" / "generation_evaluation_report.json"
    )
    pipeline_module.write_json(report_path, report)

    fingerprint = _source_fingerprint(model)
    run_report = {
        "variant": VARIANT,
        "dataset": split.name,
        "model": model,
        "seed": int(seed),
        "standard_generator_retrained": False,
        "target_generator_retrained": True,
        "tail_oversample": int(oversample),
        "tail_targeting_contract": tail_contract,
        "standard_base_validation": standard_base,
        "standard_outcome_metadata": standard_outcome_metadata,
        "tail_targeted_generation_metadata": target_generation_metadata,
        "tail_targeted_outcome_metadata": targeted_outcome_metadata,
        "data_provenance": dict(split.metadata.get("artifact_metadata", {})),
        "reproducibility_fingerprint": fingerprint,
        "artifacts": {
            "standard_behavioral_v2": {
                "path": str(standard_path),
                "sha256": pipeline_module.sha256_file(standard_path),
            },
            "tail_targeted_raw": {
                "path": str(targeted_raw_path),
                "sha256": pipeline_module.sha256_file(targeted_raw_path),
            },
            "tail_targeted_behavioral_v2": {
                "path": str(targeted_path),
                "sha256": pipeline_module.sha256_file(targeted_path),
            },
        },
        "evaluation_report": str(report_path),
    }
    run_report_path = (
        OUTPUT_ROOT / model / f"seed_{seed}" / "target_only_pipeline_report.json"
    )
    pipeline_module.write_json(run_report_path, run_report)

    if strict and issues:
        raise RuntimeError(
            f"OULAD target-only evaluation failed for {model}/seed_{seed}:\n- "
            + "\n- ".join(issues)
        )
    return {
        **run_report,
        "resumed": False,
        "_metrics": pipeline_module.flatten_numeric(
            report["publication"]["metrics"]
        ),
        "_publication": report["publication"]["metrics"],
    }


def preflight(
    models: list[str], seeds: list[int], oversample: int
) -> dict[str, object]:
    split = adapter.load_split()
    tail_train, tail_contract = _target_training(split, oversample)
    del tail_train
    gc.collect()
    standards = {
        f"{model}/seed_{seed}": _validate_standard_base(split, model, int(seed))
        for model in models
        for seed in seeds
    }
    return {
        "status": "passed",
        "variant": VARIANT,
        "generator_invoked": False,
        "artifacts_written": False,
        "tail_targeting_contract": tail_contract,
        "validated_standard_artifacts": standards,
    }


def run(
    models: list[str],
    seeds: list[int],
    oversample: int,
    strict: bool,
    resume: bool,
) -> dict[str, object]:
    split = adapter.load_split()
    fitted = adapter.fit_dropout_behavior_model(split.train)
    summaries = {}
    for model in models:
        seed_reports = [
            run_seed(
                split,
                model,
                int(seed),
                fitted,
                oversample,
                strict,
                resume,
            )
            for seed in seeds
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
            "standard_generator_retrained": False,
            "target_generator_retrained": True,
            "tail_oversample": int(oversample),
            "runs": public_runs,
            "metric_summary": pipeline_module.summarize_metrics(seed_reports),
            "publication_status_summary": (
                pipeline_module.summarize_publication_statuses(seed_reports)
            ),
            "metric_summary_scope": "publication_only",
            "publication_schema_version": publication_module.SCHEMA_VERSION,
        }
        summary_path = REPORT_ROOT / model / "multi_seed_summary.json"
        summary["summary_report"] = str(summary_path)
        pipeline_module.write_json(summary_path, summary)
        summaries[model] = str(summary_path)
    return {
        "variant": VARIANT,
        "dataset": split.name,
        "models": models,
        "seeds": seeds,
        "standard_generator_retrained": False,
        "target_generator_retrained": True,
        "summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--tail-oversample",
        type=int,
        default=DEFAULT_TAIL_OVERSAMPLE,
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate inputs and report corrected tail selection without writing artifacts.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only completed runs whose code, data, and artifact hashes still match.",
    )
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        help="Fail after writing diagnostics when the required final audit is incomplete (default).",
    )
    strict_group.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Write diagnostic reports without failing on an incomplete final audit.",
    )
    parser.set_defaults(strict=True)
    args = parser.parse_args()
    if args.tail_oversample <= 1:
        parser.error("--tail-oversample must be greater than 1 for a targeted run.")
    payload = (
        preflight(args.models, args.seeds, args.tail_oversample)
        if args.preflight_only
        else run(
            args.models,
            args.seeds,
            args.tail_oversample,
            strict=args.strict,
            resume=args.resume,
        )
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
