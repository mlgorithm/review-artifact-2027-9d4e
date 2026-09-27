#!/usr/bin/env python3
"""Refresh publication-only metrics from existing synthetic artifacts.

No generator is invoked and no synthetic CSV is modified.  The script
recomputes the corrected fixed-prefix subgroup block for every dataset and, for
OULAD, reruns downstream/outcome evaluation without independently sampled
demographics.  It then attaches the curated publication contract and rebuilds
each multi-seed summary from publication metrics only.
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

from data.dataset_scripts.registry import get_dataset
from evaluation.evaluate import privacy_risk_comparison, subgroup_fidelity
from evaluation.general_evaluation import (
    _core_evaluation_sample,
    downstream_report,
    evaluation_columns,
    records,
)
from evaluation.metrics import group_trajectories
from evaluation.outcome_tasks import explicit_outcome_task_report
from evaluation.publication import SCHEMA_VERSION, build_publication_report
from pipeline import (
    final_report_issues,
    flatten_numeric,
    sha256_file,
    summarize_metrics,
    summarize_publication_statuses,
    write_json,
)


DEFAULT_MODELS = ("markov_ngram", "sequence_vae_timevae", "timegan")
DEFAULT_SEEDS = (20260703, 20260704, 20260705)
DATASETS = {
    "assistments": {
        "output_root": EXPERIMENTS / "outputs" / "assistments",
        "report_root": EXPERIMENTS / "reports" / "assistments_2009_2010_skill_builder",
        "filename": "synth_generation.csv",
        "refresh_downstream": False,
    },
    "ednet": {
        "output_root": EXPERIMENTS / "outputs" / "ednet",
        "report_root": EXPERIMENTS / "reports" / "ednet_kt1",
        "filename": "synth_generation.csv",
        "refresh_downstream": False,
    },
    "oulad_weekly": {
        "output_root": EXPERIMENTS / "outputs" / "oulad_weekly_behavioral_dropout_v2",
        "report_root": (
            EXPERIMENTS
            / "reports"
            / "oulad_weekly_engagement_behavioral_dropout_v2"
        ),
        "filename": "synth_generation.csv.gz",
        "refresh_downstream": True,
    },
}


def _artifact_path(config: dict[str, object], model: str, seed: int, targeted: bool) -> Path:
    folder = Path(config["output_root"]) / model / f"seed_{seed}"
    if targeted:
        folder = folder / "tail_targeted"
    return folder / str(config["filename"])


def _subgroup_block(split, synthetic: pd.DataFrame) -> dict[str, object]:
    columns = evaluation_columns(split)
    learner = columns["learner"]
    order = columns["order"]
    correct = columns["correct"]
    hint = columns.get("hint")
    hint = str(hint) if hint and str(hint) in split.train.columns else None
    real_eval = _core_evaluation_sample(split.train, learner, split.seed)
    synthetic_eval = _core_evaluation_sample(synthetic, learner, split.seed + 2)
    keep = [column for column in (learner, order, correct, hint) if column]
    real_trajectories = group_trajectories(records(real_eval[keep]), learner, order)
    synthetic_trajectories = group_trajectories(
        records(synthetic_eval[keep]), learner, order
    )
    return subgroup_fidelity(
        real_trajectories,
        synthetic_trajectories,
        correct,
        hint,
        subgroup_col=None,
    )


def _refresh_fingerprint(adapter) -> dict[str, object]:
    script_path = Path(__file__).resolve()
    publication_path = Path(sys.modules[build_publication_report.__module__].__file__).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_retrained": False,
        "synthetic_artifacts_changed": False,
        "script_path": str(script_path),
        "script_sha256": sha256_file(script_path),
        "publication_module_path": str(publication_path),
        "publication_module_sha256": sha256_file(publication_path),
        "adapter_path": str(Path(adapter.__file__).resolve()),
        "adapter_sha256": sha256_file(Path(adapter.__file__).resolve()),
    }


def _sanitize_stored_membership_metrics(report: dict[str, object]) -> None:
    """Repair legacy direction-flipped attacks without inventing new CIs.

    Legacy reports stored ``abs(AUC - .5)`` and also stored non-tail advantages
    without their raw AUC.  The former can be recomputed; the latter cannot.
    Bootstrap attack CIs likewise cannot be reconstructed from the aggregate
    report, so they are removed rather than silently retained under a new
    definition.
    """
    bundle = report.get("privacy_memorization")
    if not isinstance(bundle, dict):
        return
    for arm_name in ("standard", "tail_targeted"):
        arm = bundle.get(arm_name)
        if not isinstance(arm, dict):
            continue

        def repair(payload: dict[str, object]) -> None:
            raw = payload.get("membership_inference_auc")
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                raw = float(raw)
                payload["membership_inference_attack_auc"] = max(0.5, raw)
                payload["membership_inference_attack_advantage"] = max(
                    0.0, 2.0 * (raw - 0.5)
                )
            payload.pop("membership_inference_attack_advantage_ci", None)

        repair(arm)
        groups = arm.get("by_tail_group", {}).get("groups", {})
        if isinstance(groups, dict):
            for group in groups.values():
                if not isinstance(group, dict):
                    continue
                repair(group)
                # Legacy reports did not retain the non-tail raw AUC, so the
                # old direction-flipped advantage cannot be repaired.
                group.pop("nontail_membership_inference_attack_advantage", None)
                gap = group.get("privacy_tail_gap")
                if isinstance(gap, dict):
                    gap.pop("membership_attack_advantage_gap", None)

    standard = bundle.get("standard")
    targeted = bundle.get("tail_targeted")
    if isinstance(standard, dict) and isinstance(targeted, dict):
        bundle["comparison"] = privacy_risk_comparison(standard, targeted)
    bundle["membership_refresh"] = {
        "score_orientation": "closer_to_synthetic_means_member",
        "attack_advantage": "max(0, 2*(raw_auc-0.5))",
        "legacy_bootstrap_ci_removed": True,
        "reason": "Aggregate legacy reports do not retain bootstrap scores needed for a valid recomputation.",
    }


def refresh_seed(
    dataset_name: str,
    adapter,
    split,
    config: dict[str, object],
    model: str,
    seed: int,
    strict: bool,
    resume: bool,
    publication_only: bool,
) -> dict[str, object]:
    report_path = (
        Path(config["report_root"])
        / model
        / f"seed_{seed}"
        / "generation_evaluation_report.json"
    )
    if not report_path.exists():
        raise FileNotFoundError(f"Missing evaluation report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    fingerprint = _refresh_fingerprint(adapter)
    existing_refresh = report.get("publication_refresh", {})
    if (
        resume
        and isinstance(existing_refresh, dict)
        and existing_refresh.get("script_sha256") == fingerprint["script_sha256"]
        and existing_refresh.get("publication_module_sha256")
        == fingerprint["publication_module_sha256"]
        and report.get("publication", {}).get("schema_version") == SCHEMA_VERSION
        and bool(existing_refresh.get("membership_orientation_sanitized"))
    ):
        return report

    standard_path = _artifact_path(config, model, seed, False)
    targeted_path = _artifact_path(config, model, seed, True)
    if not standard_path.exists() or not targeted_path.exists():
        raise FileNotFoundError(
            f"Missing standard or targeted synthetic artifact for {model} seed {seed}."
        )
    if not publication_only:
        standard = pd.read_csv(standard_path, low_memory=False)
        targeted = pd.read_csv(targeted_path, low_memory=False)

        report["subgroup_fidelity"] = _subgroup_block(split, standard)
        targeted_evaluation = report.setdefault("tail_targeted_evaluation", {})
        targeted_evaluation["subgroup_fidelity"] = _subgroup_block(split, targeted)

        if bool(config["refresh_downstream"]):
            report["downstream_utility_models"] = downstream_report(
                adapter,
                split,
                standard,
                standard_path,
                targeted,
                targeted_path,
            )
            report["downstream_utility_tasks"] = explicit_outcome_task_report(
                split, standard, targeted
            )

    _sanitize_stored_membership_metrics(report)

    fingerprint.update(
        {
            "dataset": dataset_name,
            "model": model,
            "seed": int(seed),
            "standard_artifact": str(standard_path),
            "standard_artifact_sha256": sha256_file(standard_path),
            "tail_targeted_artifact": str(targeted_path),
            "tail_targeted_artifact_sha256": sha256_file(targeted_path),
            "subgroup_definition": "real_train_fitted_prefix_then_future_behavior",
            "oulad_demographic_features_removed": bool(config["refresh_downstream"]),
            "publication_only_rebuild": bool(publication_only),
            "membership_orientation_sanitized": True,
        }
    )
    report["publication_refresh"] = fingerprint
    report["publication"] = build_publication_report(report, split)
    issues = final_report_issues(report)
    report["final_audit"] = {
        "status": "failed" if issues else "passed",
        "issues": issues,
        "strict_mode": bool(strict),
    }
    write_json(report_path, report)
    if strict and issues:
        raise RuntimeError(
            f"Publication refresh audit failed for {dataset_name}/{model}/{seed}:\n- "
            + "\n- ".join(issues)
        )
    return report


def rebuild_summary(
    config: dict[str, object], model: str, seeds: list[int], reports: list[dict[str, object]]
) -> Path:
    summary_path = Path(config["report_root"]) / model / "multi_seed_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metric_runs = [
        {
            "seed": int(seed),
            "_metrics": flatten_numeric(report["publication"]["metrics"]),
            "_publication": report["publication"]["metrics"],
        }
        for seed, report in zip(seeds, reports)
    ]
    summary["seeds"] = [int(seed) for seed in seeds]
    summary["metric_summary"] = summarize_metrics(metric_runs)
    summary["publication_status_summary"] = summarize_publication_statuses(metric_runs)
    summary["metric_summary_scope"] = "publication_only"
    summary["publication_schema_version"] = SCHEMA_VERSION
    summary["publication_reports"] = [
        str(
            Path(config["report_root"])
            / model
            / f"seed_{seed}"
            / "generation_evaluation_report.json"
        )
        for seed in seeds
    ]
    write_json(summary_path, summary)
    return summary_path


def run(
    datasets: list[str],
    models: list[str],
    seeds: list[int],
    strict: bool,
    resume: bool,
    publication_only: bool,
) -> dict[str, object]:
    summaries = {}
    for dataset_name in datasets:
        config = DATASETS[dataset_name]
        adapter = get_dataset(dataset_name)
        split = adapter.load_split()
        for model in models:
            reports = [
                refresh_seed(
                    dataset_name,
                    adapter,
                    split,
                    config,
                    model,
                    int(seed),
                    strict,
                    resume,
                    publication_only,
                )
                for seed in seeds
            ]
            summaries[f"{dataset_name}/{model}"] = str(
                rebuild_summary(config, model, seeds, reports)
            )
    return {
        "publication_schema_version": SCHEMA_VERSION,
        "generator_retrained": False,
        "synthetic_artifacts_changed": False,
        "summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=list(DATASETS))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--publication-only",
        action="store_true",
        help="Rebuild only the curated subtree/summary after raw corrected blocks already exist.",
    )
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.datasets,
                args.models,
                args.seeds,
                strict=not args.no_strict,
                resume=args.resume,
                publication_only=args.publication_only,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
