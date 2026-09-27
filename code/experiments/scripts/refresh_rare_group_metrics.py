#!/usr/bin/env python3
"""Refresh corrected trajectory-shape and rare-group metrics without generation.

The command reuses the immutable standard and tail-targeted synthetic CSVs.  It
recomputes the metric blocks affected by the full-real-train tail reference and
the interpolated ten-position curve, rebuilds the curated publication subtree,
and refreshes each multi-seed summary.  No generator is imported or invoked and
no synthetic artifact is modified.
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
from evaluation.evaluate import tail_fidelity, temporal_fidelity
from evaluation.general_evaluation import (
    _core_evaluation_sample,
    _sampling_metadata,
    _tail_signal_columns,
    _trajectories,
    evaluation_columns,
)
from evaluation.metrics import tail_labels, tail_reference
from evaluation.publication import SCHEMA_VERSION, build_publication_report
from pipeline import (
    TAIL_REFERENCE_AUDIT_KEYS,
    final_report_issues,
    flatten_numeric,
    sha256_file,
    summarize_metrics,
    summarize_publication_statuses,
    write_json,
)


DEFAULT_DATASETS = ("assistments", "ednet", "oulad_weekly")
DEFAULT_MODELS = ("markov_ngram", "sequence_vae_timevae", "timegan")
DEFAULT_SEEDS = (20260703, 20260704, 20260705)
REPORTS_ROOT = EXPERIMENTS / "reports" / "standard_vs_tail_targeted"


def _artifact_path(dataset: str, model: str, seed: int, arm: str) -> Path:
    return (
        EXPERIMENTS
        / "outputs"
        / arm
        / dataset
        / model
        / f"seed_{seed}"
        / "synth_generation.csv"
    )


def _tail_inputs(split, frame: pd.DataFrame):
    columns = evaluation_columns(split)
    signals = _tail_signal_columns(split)
    learner = columns["learner"]
    order = columns["order"]
    values = [
        signals[key]
        for key in (
            "skill",
            "correct",
            "hint",
            "attempt",
            "response_time",
            "gap",
            "dropout",
        )
        if signals[key]
    ]
    return _trajectories(frame, learner, order, values), signals


def _tail_context(split):
    real_train, signals = _tail_inputs(split, split.train)
    real_test, _ = _tail_inputs(split, split.test)
    arguments = (
        signals["skill"],
        signals["correct"],
        signals["hint"],
        signals["attempt"],
        signals["response_time"],
        signals["gap"],
        signals["dropout"],
    )
    reference = tail_reference(real_train, *arguments)
    real_train_labels = tail_labels(
        real_train,
        arguments[0],
        arguments[1],
        arguments[2],
        reference,
        *arguments[3:],
    )
    real_test_labels = tail_labels(
        real_test,
        arguments[0],
        arguments[1],
        arguments[2],
        reference,
        *arguments[3:],
    )
    return (
        real_train,
        real_test,
        arguments,
        reference,
        real_train_labels,
        real_test_labels,
    )


def _tail_blocks(split, synthetic: pd.DataFrame, seed: int, context):
    (
        real_train,
        real_test,
        arguments,
        reference,
        real_train_labels,
        real_test_labels,
    ) = context
    generated, _ = _tail_inputs(split, synthetic)
    generated_labels = tail_labels(
        generated,
        arguments[0],
        arguments[1],
        arguments[2],
        reference,
        *arguments[3:],
    )
    against_train = tail_fidelity(
        real_train,
        generated,
        *arguments,
        reference=reference,
        seed=seed,
        reference_scope="complete_real_train",
        real_labels=real_train_labels,
        synth_labels=generated_labels,
    )
    against_test = tail_fidelity(
        real_test,
        generated,
        *arguments,
        reference=reference,
        seed=seed + 1,
        reference_scope="complete_real_train",
        real_labels=real_test_labels,
        synth_labels=generated_labels,
    )
    return against_train, against_test


def _temporal_context(split):
    columns = evaluation_columns(split)
    signals = _tail_signal_columns(split)
    learner = columns["learner"]
    order = columns["order"]
    values = [
        value
        for value in (
            signals["skill"],
            signals["correct"],
            signals["hint"],
            signals["gap"],
        )
        if value
    ]
    real_train = _core_evaluation_sample(split.train, learner, split.seed)
    real_test = _core_evaluation_sample(split.test, learner, split.seed + 1)
    train_traj = _trajectories(real_train, learner, order, values)
    test_traj = _trajectories(real_test, learner, order, values)
    arguments = (
        signals["skill"],
        signals["correct"],
        signals["hint"],
        signals["gap"],
    )
    return learner, order, values, train_traj, test_traj, arguments


def _temporal_blocks(split, synthetic: pd.DataFrame, context):
    learner, order, values, train_traj, test_traj, arguments = context
    generated = _core_evaluation_sample(synthetic, learner, split.seed + 2)
    synth_traj = _trajectories(generated, learner, order, values)
    return (
        temporal_fidelity(train_traj, synth_traj, *arguments),
        temporal_fidelity(test_traj, synth_traj, *arguments),
    )


def _update_sampling(report: dict[str, object], split, standard, targeted) -> None:
    learner = evaluation_columns(split)["learner"]
    sampling = report.setdefault("evaluation_sampling", {})
    if isinstance(sampling, dict):
        scope = str(sampling.get("scope", ""))
        sampling["scope"] = scope.replace("_tail", "").replace("tail_", "")
        sampling["tail_fidelity"] = {
            "scope": "complete_frames",
            "reference": "complete_real_train",
            "real_train": _sampling_metadata(split.train, split.train, learner),
            "real_test": _sampling_metadata(split.test, split.test, learner),
            "synthetic": _sampling_metadata(standard, standard, learner),
        }
        sampling["note"] = (
            "Rare-group fidelity, primary downstream models, and explicit outcome "
            "tasks use complete frames; memory-heavy diagnostics use the recorded "
            "whole-learner samples."
        )
    targeted_evaluation = report.get("tail_targeted_evaluation", {})
    if isinstance(targeted_evaluation, dict):
        targeted_sampling = targeted_evaluation.setdefault("evaluation_sampling", {})
        if isinstance(targeted_sampling, dict):
            targeted_sampling["tail_fidelity"] = {
                "scope": "complete_frames",
                "reference": "complete_real_train",
                "real_train": _sampling_metadata(split.train, split.train, learner),
                "real_test": _sampling_metadata(split.test, split.test, learner),
                "synthetic": _sampling_metadata(targeted, targeted, learner),
            }


def refresh_seed(
    dataset: str,
    split,
    model: str,
    seed: int,
    strict: bool,
    tail_context,
    temporal_context,
):
    report_path = (
        REPORTS_ROOT
        / split.name
        / model
        / f"seed_{seed}"
        / "generation_evaluation_report.json"
    )
    standard_path = _artifact_path(dataset, model, seed, "standard")
    targeted_path = _artifact_path(dataset, model, seed, "tail_targeted")
    for path in (report_path, standard_path, targeted_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    standard = pd.read_csv(standard_path, low_memory=False)
    targeted = pd.read_csv(targeted_path, low_memory=False)

    report["temporal_fidelity"], report["temporal_fidelity_vs_test"] = (
        _temporal_blocks(split, standard, temporal_context)
    )
    report["tail_fidelity"], report["tail_fidelity_vs_test"] = _tail_blocks(
        split, standard, seed, tail_context
    )
    targeted_evaluation = report.setdefault("tail_targeted_evaluation", {})
    if not isinstance(targeted_evaluation, dict):
        raise TypeError("tail_targeted_evaluation must be a mapping")
    (
        targeted_evaluation["temporal_fidelity"],
        targeted_evaluation["temporal_fidelity_vs_test"],
    ) = _temporal_blocks(split, targeted, temporal_context)
    (
        targeted_evaluation["tail_fidelity"],
        targeted_evaluation["tail_fidelity_vs_test"],
    ) = _tail_blocks(split, targeted, seed, tail_context)
    _update_sampling(report, split, standard, targeted)

    # These reports predate serialization of the fitted targeting reference.
    # Backfill the exact complete-real-train scalar cutoffs and transition-model
    # digest used by both the refreshed evaluation and the original deterministic
    # targeting selection.  The large transition map itself remains omitted.
    reference = tail_context[3]
    targeting_contract = report.get("tail_targeting_contract", {})
    if not isinstance(targeting_contract, dict):
        raise TypeError("tail_targeting_contract must be a mapping")
    targeting_contract["tail_reference"] = {
        key: reference.get(key)
        for key in TAIL_REFERENCE_AUDIT_KEYS
        if key in reference
    }

    report["rare_group_metric_refresh"] = {
        "generator_retrained": False,
        "synthetic_artifacts_changed": False,
        "standard_artifact": str(standard_path.relative_to(ROOT)),
        "standard_artifact_sha256": sha256_file(standard_path),
        "tail_targeted_artifact": str(targeted_path.relative_to(ROOT)),
        "tail_targeted_artifact_sha256": sha256_file(targeted_path),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "tail_reference_scope": "complete_real_train",
        "trajectory_curve": "learner_equal_ten_position_linear_interpolation",
        "rapid_guessing_max_primary_signal_rate": 0.5,
        "publication_rare_group_prevalence_ceiling": 0.10,
    }
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
            f"Refreshed report audit failed for {dataset}/{model}/{seed}:\n- "
            + "\n- ".join(issues)
        )
    return report


def rebuild_summary(split, model: str, seeds: list[int], reports):
    path = REPORTS_ROOT / split.name / model / "multi_seed_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    runs = [
        {
            "seed": int(seed),
            "_metrics": flatten_numeric(report["publication"]["metrics"]),
            "_publication": report["publication"]["metrics"],
        }
        for seed, report in zip(seeds, reports)
    ]
    summary["metric_summary"] = summarize_metrics(runs)
    summary["publication_status_summary"] = summarize_publication_statuses(runs)
    summary["publication_schema_version"] = SCHEMA_VERSION
    summary["rare_group_metric_refresh"] = {
        "generator_retrained": False,
        "synthetic_artifacts_changed": False,
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "tail_reference_scope": "complete_real_train",
        "trajectory_curve": "learner_equal_ten_position_linear_interpolation",
    }
    write_json(path, summary)
    return path


def run(datasets, models, seeds, strict=True):
    summaries = {}
    for dataset in datasets:
        split = get_dataset(dataset).load_split()
        tail_context = _tail_context(split)
        temporal_context = _temporal_context(split)
        for model in models:
            reports = []
            for seed in seeds:
                print(
                    f"[rare-group refresh] {dataset}/{model}/seed_{seed}",
                    file=sys.stderr,
                    flush=True,
                )
                reports.append(
                    refresh_seed(
                        dataset,
                        split,
                        model,
                        int(seed),
                        strict,
                        tail_context,
                        temporal_context,
                    )
                )
            summaries[f"{dataset}/{model}"] = str(
                rebuild_summary(split, model, list(seeds), reports).relative_to(ROOT)
            )
    return {
        "generator_retrained": False,
        "synthetic_artifacts_changed": False,
        "summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=DEFAULT_DATASETS, default=list(DEFAULT_DATASETS)
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.datasets,
                args.models,
                args.seeds,
                strict=not args.no_strict,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
