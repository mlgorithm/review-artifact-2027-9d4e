#!/usr/bin/env python3
"""Generic synthetic-data experiment pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from data.dataset_scripts.registry import get_dataset
from evaluation.general_evaluation import run_general_evaluation
from evaluation.metrics import group_trajectories, tail_labels, tail_reference
from evaluation.publication import SCHEMA_VERSION
from synthetic_generation.run_generator import (
    generator_source_provenance,
    load_generator_script,
    run_generator,
)


DEFAULT_SEEDS = (20260703, 20260704, 20260705)
DEFAULT_TAIL_OVERSAMPLE = 3
EXPERIMENT_DESIGN = "standard_vs_tail_targeted"
TAIL_SELECTION_CONTRACT = "maximum_10_percent_per_target_group"
MAX_TARGET_GROUP_PREVALENCE = 0.10
TAIL_REFERENCE_AUDIT_KEYS = (
    "quantile",
    "maximum_percentile_tail_prevalence",
    "rapid_guessing_max_primary_signal_rate",
    "short_cutoff",
    "short_cutoff_inclusive",
    "long_cutoff",
    "long_cutoff_inclusive",
    "low_correctness_cutoff",
    "low_correctness_cutoff_inclusive",
    "high_hint_cutoff",
    "high_hint_cutoff_inclusive",
    "fast_response_cutoff",
    "fast_response_cutoff_inclusive",
    "high_gap_cutoff",
    "high_gap_cutoff_inclusive",
    "rare_path_cutoff",
    "rare_path_cutoff_inclusive",
    "transition_floor_logp",
    "transition_model_sha256",
    "transition_vocabulary_size",
    "transition_observation_count",
)
INTERACTION_TARGET_TAIL_GROUPS = frozenset(
    {
        "persistent_failure",
        "persistent_misconception",
        "recovery",
        "late_failure",
        "rare_skill_path",
        "high_hint_use",
        "rapid_guessing",
        "long_inactivity",
    }
)
WEEKLY_TARGET_TAIL_GROUPS = frozenset(
    {
        "persistent_failure",
        "recovery",
        "late_failure",
        "rare_skill_path",
        "long_inactivity",
    }
)
STANDARD_OUTPUTS_ROOT = ROOT / "experiments" / "outputs" / "standard"
TAIL_TARGETED_OUTPUTS_ROOT = ROOT / "experiments" / "outputs" / "tail_targeted"
LOCAL_RUN_ROOT = ROOT / "experiments" / "runs" / EXPERIMENT_DESIGN
REPORTS_ROOT = LOCAL_RUN_ROOT / "evaluation"
REFERENCE_REPORTS_ROOT = ROOT / "experiments" / "reports" / EXPERIMENT_DESIGN


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_name(seed: int) -> str:
    return f"seed_{seed}"


def report_parts(split, model_name: str, include_seed: int | None = None) -> list[str]:
    parts = [split.name, model_name]
    if include_seed is not None:
        parts.append(seed_name(include_seed))
    return parts


def standard_synthetic_root(split) -> Path:
    """Canonical RQ1 output root for unmodified generator training."""
    return STANDARD_OUTPUTS_ROOT / split.synthetic_root.name


def tail_targeted_synthetic_root(split) -> Path:
    """Canonical RQ2 output root for tail-oversampled generator training."""
    return TAIL_TARGETED_OUTPUTS_ROOT / split.synthetic_root.name


def experiment_seed_paths(split, model_name: str, seed: int) -> dict[str, Path]:
    """Return every canonical local path for one immutable experiment seed."""
    run_root_parts = [model_name]
    if split.label:
        run_root_parts.append(split.label)
    run_root_parts.append(seed_name(seed))
    standard_seed_dir = standard_synthetic_root(split).joinpath(*run_root_parts)
    target_seed_dir = tail_targeted_synthetic_root(split).joinpath(*run_root_parts)
    evaluation_path = (
        REPORTS_ROOT.joinpath(*report_parts(split, model_name, seed))
        / "generation_evaluation_report.json"
    )
    pipeline_report_path = (
        LOCAL_RUN_ROOT.joinpath("pipeline", *report_parts(split, model_name, seed))
        / "pipeline_report.json"
    )
    return {
        "standard_seed_dir": standard_seed_dir,
        "target_seed_dir": target_seed_dir,
        "standard_csv": standard_seed_dir / "synth_generation.csv",
        "standard_metadata": standard_seed_dir / "generation_metadata.json",
        "target_csv": target_seed_dir / "synth_generation.csv",
        "target_metadata": target_seed_dir / "generation_metadata.json",
        "pipeline_report": pipeline_report_path,
        "evaluation_report": evaluation_path,
    }


def occupied_experiment_seed_paths(
    split,
    model_name: str,
    seed: int,
    *,
    allow_standard_reuse: bool = False,
    allow_target_reuse: bool = False,
) -> list[Path]:
    """Return existing paths that make a seed unsafe to overwrite."""
    paths = experiment_seed_paths(split, model_name, seed)
    protected = [
        paths["standard_csv"],
        paths["standard_metadata"],
        paths["pipeline_report"],
        paths["target_csv"],
        paths["target_metadata"],
        paths["evaluation_report"],
    ]
    if allow_standard_reuse:
        protected = [path for path in protected if path not in {
            paths["standard_csv"],
            paths["standard_metadata"],
            paths["pipeline_report"],
        }]
    if allow_target_reuse:
        protected = [path for path in protected if path not in {
            paths["target_csv"], paths["target_metadata"]
        }]
    return [path for path in protected if path.exists()]


def _standard_artifact_candidate_dirs(split, run_root_parts: Sequence[str]) -> tuple[Path, ...]:
    """Search the canonical RQ1 root, then historical validated standard roots."""
    candidates = [standard_synthetic_root(split).joinpath(*run_root_parts)]
    candidates.append(split.synthetic_root.joinpath(*run_root_parts))
    return tuple(dict.fromkeys(candidates))


def _standard_artifact_namespace(split, model_seed_dir: Path) -> str:
    if model_seed_dir.is_relative_to(standard_synthetic_root(split)):
        return "standard"
    if model_seed_dir.is_relative_to(split.synthetic_root):
        return "historical_unversioned"
    return "historical_unknown"


def _generation_drop_columns(split) -> list[str]:
    """Functionally derived value columns withheld from generation.

    Learner/order columns remain present as structural boundaries. Generators must
    use them for grouping but never model or reproduce the real identifier values.
    """
    return list(getattr(split, "derived_columns", ()))


def generation_train(split) -> pd.DataFrame:
    return split.train.drop(columns=_generation_drop_columns(split), errors="ignore").copy()


def _generation_records(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    keep = [column for column in columns if column in frame.columns]
    return frame[keep].astype(object).where(frame[keep].notna(), "").astype(str).to_dict("records")


def _detect_tail_signal(frame: pd.DataFrame, keywords: tuple[str, ...], exclude: tuple[str, ...] = ()) -> str | None:
    for column in frame.columns:
        name = str(column).lower()
        if any(token in name for token in exclude):
            continue
        if any(token in name for token in keywords):
            return str(column)
    return None


def _target_tail_groups(split) -> frozenset[str]:
    configured = getattr(split, "metadata", {}).get("target_tail_groups")
    if configured:
        return frozenset(str(name) for name in configured)
    semantics = str(
        getattr(split, "metadata", {}).get("primary_signal_semantics", "")
    )
    return (
        WEEKLY_TARGET_TAIL_GROUPS
        if semantics == "weekly_engagement"
        else INTERACTION_TARGET_TAIL_GROUPS
    )


def tail_oversampled_generation_train(
    split, oversample: int, *, return_metadata: bool = False
) -> pd.DataFrame | None | tuple[pd.DataFrame | None, dict[str, object]]:
    """Generation-ready train frame with tail learners up-weighted.

    Duplicates the (contiguous, order-preserving) rows of learners in any tail
    group ``oversample-1`` extra times so the generator sees rare pathways more
    often. This is the tail-targeted variant used for the RQ2 arm; returns None
    when there are no tail learners or oversampling is disabled.
    """
    if oversample is None or oversample <= 1:
        result = (
            None,
            {"selection_contract": TAIL_SELECTION_CONTRACT, "status": "disabled"},
        )
        return result if return_metadata else None
    columns = dict(split.columns)
    learner = columns.get("learner", "learner_id")
    order = columns.get("order", "order")
    if learner not in split.train.columns:
        result = (
            None,
            {
                "selection_contract": TAIL_SELECTION_CONTRACT,
                "status": "unavailable",
                "reason": f"missing learner column: {learner}",
            },
        )
        return result if return_metadata else None

    def present(name):
        value = columns.get(name)
        return value if value and value in split.train.columns else None

    skill = columns.get("skill", "skill_id")
    correct = columns.get("correct", "correct")
    hint = present("hint")
    attempt = present("attempts")
    response_time = present("response_time") or _detect_tail_signal(
        split.train, ("response", "duration", "elapsed"), ("gap",)
    )
    gap = present("gap") or _detect_tail_signal(split.train, ("gap",))
    dropout = present("dropout")
    value_cols = [
        column
        for column in [skill, correct, hint, attempt, response_time, gap, dropout]
        if column
    ]
    keep = [column for column in [learner, order, *value_cols] if column in split.train.columns]

    trajectories = group_trajectories(_generation_records(split.train, keep), learner, order)
    reference = tail_reference(
        trajectories, skill, correct, hint, attempt, response_time, gap, dropout
    )
    labels = tail_labels(
        trajectories, skill, correct, hint, reference, attempt, response_time, gap, dropout
    )
    observed_group_names = {
        name for learner_groups in labels.values() for name in learner_groups
    }
    semantically_eligible_group_names = _target_tail_groups(split)
    candidate_group_names = sorted(
        observed_group_names & semantically_eligible_group_names
    )
    group_counts = {
        name: int(sum(bool(groups.get(name)) for groups in labels.values()))
        for name in candidate_group_names
    }
    maximum_target_prevalence = min(
        MAX_TARGET_GROUP_PREVALENCE,
        float(
            reference.get(
                "maximum_percentile_tail_prevalence",
                MAX_TARGET_GROUP_PREVALENCE,
            )
        ),
    )
    group_names = [
        name
        for name in candidate_group_names
        if group_counts[name] > 0
        and (
            not labels
            or group_counts[name] / len(labels) <= maximum_target_prevalence + 1e-12
        )
    ]
    excluded_group_names = sorted(
        observed_group_names - semantically_eligible_group_names
    )
    excluded_target_groups = {
        name: (
            "empty_group"
            if group_counts[name] == 0
            else (
                f"learner_prevalence_{group_counts[name] / len(labels):.6f}_exceeds_"
                f"target_ceiling_{maximum_target_prevalence:.6f}"
            )
        )
        for name in candidate_group_names
        if name not in group_names
    }
    tail_ids = {
        str(learner_id)
        for learner_id, learner_groups in labels.items()
        if any(bool(learner_groups.get(name)) for name in group_names)
    }
    selection_metadata = {
        "selection_contract": TAIL_SELECTION_CONTRACT,
        "status": "completed" if tail_ids else "empty",
        "selection_policy": "proposal_aligned_generator_attributable_tails",
        "oversample": int(oversample),
        "quantile": reference.get("quantile"),
        "maximum_percentile_tail_prevalence": reference.get(
            "maximum_percentile_tail_prevalence"
        ),
        "maximum_target_group_prevalence": maximum_target_prevalence,
        "unavailable_percentile_tail_groups": reference.get(
            "unavailable_percentile_tail_groups", {}
        ),
        "tail_reference": {
            key: reference.get(key)
            for key in TAIL_REFERENCE_AUDIT_KEYS
            if key in reference
        },
        "eligible_tail_groups": sorted(semantically_eligible_group_names),
        "candidate_tail_groups": candidate_group_names,
        "implemented_tail_groups": group_names,
        "excluded_target_groups": excluded_target_groups,
        "excluded_observed_tail_groups": {
            name: (
                "shared_trajectory_length_sampler"
                if name in {"short_trajectory", "long_trajectory"}
                else (
                    "terminal_outcome_assigned_after_generation"
                    if name == "dropout"
                    else "not_meaningful_for_dataset_targeting_semantics"
                )
            )
            for name in excluded_group_names
        },
        "selected_learners_by_group": {
            name: group_counts[name] for name in group_names
        },
        "candidate_learners_by_group": group_counts,
        "base_learners": int(len(labels)),
        "selected_tail_learners": int(len(tail_ids)),
        "selected_tail_learner_fraction": (
            float(len(tail_ids) / len(labels)) if labels else 0.0
        ),
        "tail_learner_set_sha256": hashlib.sha256(
            "\n".join(sorted(tail_ids)).encode("utf-8")
        ).hexdigest(),
    }
    if not tail_ids:
        result = (None, selection_metadata)
        return result if return_metadata else None

    base = split.train
    if order in base.columns:
        base = base.sort_values([learner, order])
    tail_rows = base[base[learner].astype(str).isin(tail_ids)]
    selection_metadata["base_rows"] = int(len(base))
    selection_metadata["selected_tail_rows"] = int(len(tail_rows))
    if tail_rows.empty:
        selection_metadata["status"] = "empty"
        result = (None, selection_metadata)
        return result if return_metadata else None
    copies = [base]
    for repeat in range(1, oversample):
        repeated = tail_rows.copy()
        # A repeated learner is a separate training example, not one trajectory
        # whose length has silently been multiplied by the oversampling factor.
        repeated[learner] = repeated[learner].astype(str) + f"__tail_repeat_{repeat}"
        copies.append(repeated)
    oversampled = pd.concat(copies, ignore_index=True)
    result_frame = oversampled.drop(
        columns=_generation_drop_columns(split), errors="ignore"
    ).copy()
    selection_metadata["oversampled_train_rows"] = int(len(result_frame))
    result = (result_frame, selection_metadata)
    return result if return_metadata else result_frame


def normalize_synthetic_frame(
    split, synthetic: pd.DataFrame, adapter=None, seed: int = 0
) -> tuple[pd.DataFrame, list[str]]:
    normalized = synthetic.copy()
    added_columns = []
    missing_structural = [column for column in split.ignore_columns if column not in normalized.columns]
    if missing_structural:
        raise ValueError(
            "Synthetic output must provide independently generated trajectory boundaries; "
            f"missing structural columns: {missing_structural}"
        )

    learner_col = split.columns.get("learner", "learner_id")
    order_col = split.columns.get("order", "order")
    if learner_col in normalized.columns and learner_col in split.train.columns:
        real_ids = set(split.train[learner_col].astype(str))
        leaked_ids = real_ids & set(normalized[learner_col].astype(str))
        if leaked_ids:
            raise ValueError(f"Synthetic output reuses {len(leaked_ids)} real learner identifiers.")
    if learner_col in normalized.columns and order_col in normalized.columns:
        ordered = normalized.sort_values([learner_col, order_col]).reset_index(drop=True)
        expected_order = ordered.groupby(learner_col, sort=False).cumcount()
        actual_order = pd.to_numeric(ordered[order_col], errors="coerce")
        if actual_order.isna().any() or not actual_order.reset_index(drop=True).equals(expected_order):
            raise ValueError("Synthetic order must be contiguous and zero-based within every learner.")
        normalized = ordered

    # Rebuild the functionally-determined columns that were withheld from
    # generation, using the real train split as the lookup. This guarantees the
    # synthetic rows are internally consistent (no skill_id that contradicts its
    # module_id, no question paired with the wrong part).
    derived = tuple(getattr(split, "derived_columns", ()))
    if derived and adapter is not None and hasattr(adapter, "rebuild_derived_columns"):
        normalized = adapter.rebuild_derived_columns(normalized, split.train)
        added_columns.extend(column for column in derived if column in normalized.columns)

    # Dataset adapters may enforce additional *declared* invariants that a
    # column-independent decoder cannot guarantee (for example learner-level
    # outcome consistency or a gap bin derived from the generated activity
    # sequence). This hook must never consult real test data.
    if adapter is not None and hasattr(adapter, "normalize_generated_frame"):
        normalized = adapter.normalize_generated_frame(normalized, split.train, seed=seed)
        added_columns.extend(
            column
            for column in derived
            if column in normalized.columns and column not in added_columns
        )

    missing_columns = [column for column in split.train.columns if column not in normalized.columns]
    if missing_columns:
        raise ValueError(
            "Synthetic output is missing required generation columns after pipeline normalization: "
            f"{missing_columns}"
        )
    return normalized[split.train.columns.tolist()], added_columns


# Report subtrees/leaves that are counts, sizes, or absolute levels rather than
# quality metrics; averaging these across seeds is meaningless, so they are kept
# out of the aggregated metric_summary.
_NON_METRIC_KEYS = (
    "dataset_sizes",
    "tail_prevalence_real",
    "tail_prevalence_synthetic",
    "rows",
    "real",
    "synthetic",
    "train_class_counts",
    "test_class_counts",
)
_NON_METRIC_SUFFIXES = (
    "_rows",
    "_learners",
    "_count",
    "_sample_size",
    "_set_size",
    ".seed",
    "_seed",
)
_NON_METRIC_LEAVES = {
    "ci_low",
    "ci_high",
    "n_boot",
    "near_duplicate_threshold",
    "positive_rate",
    "prefix_length",
    "quantile",
}


def _is_metric_key(key: str) -> bool:
    parts = key.split(".")
    if any(part in _NON_METRIC_KEYS for part in parts):
        return False
    if any(part.endswith("_ci") for part in parts):
        return False
    leaf = parts[-1]
    if leaf in _NON_METRIC_LEAVES or leaf.endswith(("_cutoff", "_real", "_synthetic")):
        return False
    return not key.endswith(_NON_METRIC_SUFFIXES)


def flatten_numeric(payload: object, prefix: str = "") -> dict[str, float]:
    if isinstance(payload, dict):
        flattened: dict[str, float] = {}
        for key, value in payload.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_numeric(value, child_prefix))
        return flattened
    if isinstance(payload, (int, float)) and not isinstance(payload, bool):
        value = float(payload)
        # Skip non-finite values (NaN/inf): a single one would otherwise turn the
        # aggregated mean/std/min/max for that metric into NaN across all seeds.
        if not math.isfinite(value):
            return {}
        return {prefix: value} if _is_metric_key(prefix) else {}
    return {}


def summarize_metrics(seed_reports: Sequence[dict[str, object]]) -> dict[str, dict[str, float]]:
    by_metric: dict[str, list[float]] = {}
    for report in seed_reports:
        for metric, value in dict(report.get("_metrics", {})).items():
            by_metric.setdefault(metric, []).append(float(value))

    summary = {}
    for metric, values in sorted(by_metric.items()):
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        summary[metric] = {
            "n": len(values),
            "missing_runs": len(seed_reports) - len(values),
            "mean": mean_value,
            "std": variance**0.5,
            "min": min(values),
            "max": max(values),
        }
    return summary


def summarize_publication_statuses(
    seed_reports: Sequence[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Aggregate explicit non-numeric benchmark outcomes across seeds.

    Numeric summaries cannot represent a collapsed synthetic training arm. This
    companion summary keeps every publication ``status`` visible and associates
    it with the seeds that produced it.
    """
    by_path: dict[str, dict[str, list[int]]] = {}

    def collect(value: object, seed: int, prefix: str = "") -> None:
        if isinstance(value, dict):
            status = value.get("status")
            if isinstance(status, str):
                by_path.setdefault(prefix, {}).setdefault(status, []).append(seed)
            for key, child in value.items():
                if key == "status":
                    continue
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                collect(child, seed, child_prefix)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                collect(child, seed, f"{prefix}[{index}]")

    for report in seed_reports:
        seed = int(report.get("seed", -1))
        collect(report.get("_publication", {}), seed)

    return {
        path: {
            "counts": {
                status: len(seeds) for status, seeds in sorted(statuses.items())
            },
            "seeds_by_status": {
                status: sorted(seeds) for status, seeds in sorted(statuses.items())
            },
        }
        for path, statuses in sorted(by_path.items())
        if path
    }


def _non_singleton_batch_size(example_count: int, requested_batch_size: int) -> int:
    """Keep the requested batch cap while avoiding a final one-row minibatch.

    SynthCity TimeGAN's moment and recurrent calculations are not defined for a
    singleton minibatch. Most training-set sizes never hit this edge case. If
    they do, choose the largest smaller batch size whose remainder is not one.
    This preserves full-data training and the configured batch-update budget.
    """
    if example_count < 2:
        raise ValueError("TimeGAN requires at least two trajectory windows.")
    if requested_batch_size < 2:
        raise ValueError("TimeGAN batch size must be at least two.")
    capped = min(example_count, requested_batch_size)
    if example_count <= capped or example_count % capped != 1:
        return requested_batch_size
    for candidate in range(capped - 1, 1, -1):
        if example_count % candidate != 1:
            return candidate
    raise ValueError(
        "Unable to choose a TimeGAN minibatch size without a singleton batch."
    )


def _timegan_batch_safety_adjustment(
    model_name: str, train_frame: pd.DataFrame
) -> dict[str, object] | None:
    """Return a recorded TimeGAN batch override only for singleton remainders."""
    if model_name != "timegan":
        return None
    window = int(os.environ.get("TIMEGAN_WINDOW", "20"))
    requested = int(os.environ.get("TIMEGAN_BATCH", "200"))
    if window < 2:
        raise ValueError("TIMEGAN_WINDOW must be at least two.")
    if "learner_id" not in train_frame.columns:
        raise ValueError("TimeGAN input is missing structural learner_id.")
    lengths = train_frame.groupby("learner_id", sort=False).size()
    training_windows = int(((lengths + window - 1) // window).sum())
    effective = _non_singleton_batch_size(training_windows, requested)
    if effective == requested:
        return None
    return {
        "reason": "avoid_external_synthcity_timegan_singleton_minibatch",
        "requested_batch_size": requested,
        "effective_batch_size": effective,
        "training_windows": training_windows,
        "window": window,
        "full_training_data_retained": True,
    }


def _generate_synthetic(
    model_name,
    split,
    train_frame,
    output_dir,
    seed,
    adapter=None,
    generation_shape: dict[str, int] | None = None,
    generation_arm: str = "standard",
):
    """Run a generator and persist final normalized output provenance."""
    environment_adjustment = _timegan_batch_safety_adjustment(model_name, train_frame)
    prior_timegan_batch = os.environ.get("TIMEGAN_BATCH")
    if environment_adjustment is not None:
        os.environ["TIMEGAN_BATCH"] = str(environment_adjustment["effective_batch_size"])
    try:
        metadata = run_generator(model_name, train_frame, output_dir, seed, generation_shape)
    finally:
        if environment_adjustment is not None:
            if prior_timegan_batch is None:
                os.environ.pop("TIMEGAN_BATCH", None)
            else:
                os.environ["TIMEGAN_BATCH"] = prior_timegan_batch
    if environment_adjustment is not None:
        metadata.setdefault("orchestration_adjustments", {})[
            "timegan_batch_safety"
        ] = environment_adjustment
    path = output_dir / "synth_generation.csv"
    synthetic = pd.read_csv(path, low_memory=False)
    synthetic, added_columns = normalize_synthetic_frame(split, synthetic, adapter, seed=seed)
    temporary = path.with_name(f".{path.name}.tmp")
    synthetic.to_csv(temporary, index=False)
    temporary.replace(path)

    pipeline_path = Path(__file__).resolve()
    learner_col = split.columns.get("learner", "learner_id")
    metadata.update(
        {
            "experiment_design": EXPERIMENT_DESIGN,
            "generation_arm": generation_arm,
            "output": str(path),
            "columns": synthetic.columns.tolist(),
            "synthetic_rows": int(len(synthetic)),
            "pipeline_added_columns": added_columns,
            "adapter_normalized_invariants": bool(
                adapter is not None and hasattr(adapter, "normalize_generated_frame")
            ),
            "pipeline_normalized_output": str(path),
            "pipeline_script_path": str(pipeline_path),
            "pipeline_script_sha256": sha256_file(pipeline_path),
            "synthetic_output_sha256": sha256_file(path),
            "data_provenance": dict(
                getattr(split, "metadata", {}).get("artifact_metadata", {})
            ),
        }
    )
    if learner_col in synthetic.columns:
        metadata["synthetic_learners"] = int(synthetic[learner_col].nunique())
    write_json(output_dir / "generation_metadata.json", metadata)
    return synthetic, path, metadata


def _load_partial_generated_artifact(
    model_name: str,
    split,
    output_dir: Path,
    seed: int,
    adapter,
    generation_arm: str,
) -> tuple[pd.DataFrame, Path, dict[str, object]] | None:
    """Load a completed generation phase after an interrupted local run."""
    path = output_dir / "synth_generation.csv"
    metadata_path = output_dir / "generation_metadata.json"
    present = (path.exists(), metadata_path.exists())
    if not any(present):
        return None
    if not all(present):
        raise ValueError(
            f"Cannot resume incomplete {generation_arm} generation; expected both "
            "the synthetic CSV and generation metadata."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    issues = []
    if metadata.get("experiment_design") != EXPERIMENT_DESIGN:
        issues.append("experiment design differs")
    if metadata.get("generation_arm") != generation_arm:
        issues.append("generation arm differs")
    if int(metadata.get("seed", -1)) != int(seed):
        issues.append("generation seed differs")
    if str(metadata.get("model")) != str(model_name):
        issues.append("generator model differs")
    if sha256_file(path) != metadata.get("synthetic_output_sha256"):
        issues.append("synthetic artifact hash differs from generation metadata")

    try:
        generator_module = load_generator_script(model_name)
        generator_path = Path(generator_module.__file__).resolve()
        _, current_source_bundle = generator_source_provenance(generator_module)
        if metadata.get("generator_script_sha256") != sha256_file(generator_path):
            issues.append("generator script hash differs")
        if metadata.get("generator_source_bundle_sha256") != current_source_bundle:
            issues.append("generator source-bundle hash differs")
    except (ImportError, ValueError) as exc:
        issues.append(f"current generator source cannot be loaded: {exc}")

    current_provenance = dict(
        getattr(split, "metadata", {}).get("artifact_metadata", {}) or {}
    )
    prior_provenance = dict(metadata.get("data_provenance", {}) or {})
    for key in ("train_sha256", "test_sha256", "source_sha256"):
        current = current_provenance.get(key)
        if current is not None and current != prior_provenance.get(key):
            issues.append(f"dataset provenance differs for {key}")
    if issues:
        raise ValueError(
            f"Existing {generation_arm} generation phase is not resumable: "
            + "; ".join(issues)
        )

    synthetic = pd.read_csv(path, low_memory=False)
    if synthetic.columns.tolist() != split.train.columns.tolist():
        raise ValueError(
            f"Existing {generation_arm} artifact columns differ from the dataset contract."
        )
    if int(metadata.get("synthetic_rows", -1)) != len(synthetic):
        raise ValueError(
            f"Existing {generation_arm} artifact row count differs from its metadata."
        )
    normalized, _ = normalize_synthetic_frame(split, synthetic, adapter, seed=seed)
    if not normalized.equals(synthetic):
        raise ValueError(
            f"Existing {generation_arm} artifact differs under the normalization contract."
        )
    return synthetic, path, metadata


def _validate_standard_artifact_provenance(
    model_name: str,
    split,
    model_seed_dir: Path,
    seed: int,
) -> tuple[Path, dict[str, object], dict[str, object]] | None:
    """Validate an existing standard artifact without loading its full CSV.

    Missing artifacts return ``None`` so a multi-seed command can reuse completed
    seeds and generate new ones. Present-but-invalid artifacts raise: an explicit
    reuse request must never silently accept or overwrite a mismatched result.
    """
    path = model_seed_dir / "synth_generation.csv"
    metadata_path = model_seed_dir / "generation_metadata.json"
    provenance_path = model_seed_dir / "standard_provenance.json"
    legacy_pipeline_path = model_seed_dir / "pipeline_report.json"
    present = [path.exists(), metadata_path.exists()]
    if not any(present) and not provenance_path.exists() and not legacy_pipeline_path.exists():
        return None
    if not all(present):
        raise ValueError(
            "Cannot reuse an incomplete standard artifact; expected synthetic CSV, "
            "and generation metadata."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    provenance_report_path = (
        provenance_path if provenance_path.exists() else legacy_pipeline_path
    )
    prior_pipeline = (
        json.loads(provenance_report_path.read_text(encoding="utf-8"))
        if provenance_report_path.exists()
        else {}
    )
    issues = []
    if int(metadata.get("seed", -1)) != int(seed):
        issues.append("generation seed differs")
    if str(metadata.get("model")) != str(model_name):
        issues.append("generator model differs")
    if sha256_file(path) != metadata.get("synthetic_output_sha256"):
        issues.append("synthetic artifact hash differs from generation metadata")

    try:
        generator_module = load_generator_script(model_name)
        generator_path = Path(generator_module.__file__).resolve()
        _, current_source_bundle = generator_source_provenance(generator_module)
        if metadata.get("generator_script_sha256") != sha256_file(generator_path):
            issues.append("generator script hash differs")
        if metadata.get("generator_source_bundle_sha256") != current_source_bundle:
            issues.append("generator source-bundle hash differs")
    except (ImportError, ValueError) as exc:
        issues.append(f"current generator source cannot be loaded: {exc}")

    current_provenance = dict(split.metadata.get("artifact_metadata", {}) or {})
    prior_provenance = dict(
        metadata.get("data_provenance")
        or prior_pipeline.get("data_provenance", {})
        or {}
    )
    for key in ("train_sha256", "test_sha256", "source_sha256"):
        current = current_provenance.get(key)
        prior = prior_provenance.get(key)
        if current is not None and current != prior:
            issues.append(f"dataset provenance differs for {key}")
    if prior_pipeline:
        if str(prior_pipeline.get("dataset")) != str(split.name):
            issues.append("dataset name differs")
        if str(prior_pipeline.get("model")) != str(model_name):
            issues.append("standard-provenance model differs")
        if int(prior_pipeline.get("seed", -1)) != int(seed):
            issues.append("standard-provenance seed differs")
    if issues:
        raise ValueError("Existing standard artifact is not reusable: " + "; ".join(issues))

    return path, metadata, prior_pipeline


def _load_reusable_standard(
    model_name: str,
    split,
    model_seed_dir: Path,
    seed: int,
    adapter=None,
) -> tuple[pd.DataFrame, Path, dict[str, object]] | None:
    """Load a provenance-matched standard artifact without retraining it."""
    validated = _validate_standard_artifact_provenance(
        model_name, split, model_seed_dir, seed
    )
    if validated is None:
        return None
    path, metadata, _ = validated

    synthetic = pd.read_csv(path, low_memory=False)
    expected_columns = split.train.columns.tolist()
    if synthetic.columns.tolist() != expected_columns:
        raise ValueError("Existing standard artifact columns differ from the current dataset contract.")
    if int(metadata.get("synthetic_rows", -1)) != len(synthetic):
        raise ValueError("Existing standard artifact row count differs from generation metadata.")

    # Reapply current invariant logic in memory and require equality. This
    # rejects artifacts produced under an obsolete postprocessing contract
    # (notably pre-v2 OULAD outcomes) without changing a valid artifact.
    normalized, _ = normalize_synthetic_frame(split, synthetic, adapter, seed=seed)
    if not normalized.equals(synthetic):
        raise ValueError(
            "Existing standard artifact differs under the current normalization contract."
        )
    reused_metadata = dict(metadata)
    reused_metadata["standard_artifact_reuse"] = {
        "reused": True,
        "validation": "artifact, generator source, dataset provenance, and normalization matched",
        "artifact_sha256": metadata.get("synthetic_output_sha256"),
    }
    return synthetic, path, reused_metadata


def _load_standard_for_run(
    model_name: str,
    split,
    standard_seed_dir: Path,
    run_root_parts: Sequence[str],
    seed: int,
    adapter,
    *,
    reuse_standard_if_valid: bool,
    resume_incomplete: bool,
) -> tuple[tuple[pd.DataFrame, Path, dict[str, object]] | None, Path | None]:
    """Load a standard artifact using the correct contract precedence.

    Canonical RQ1 artifacts can predate the RQ2 ``experiment_design`` and
    ``generation_arm`` metadata fields while still satisfying the stricter RQ1
    provenance and normalization validator.  During an interrupted RQ2 resume,
    explicit standard reuse must therefore be attempted before treating the
    artifact as a partially generated RQ2 phase.
    """
    if reuse_standard_if_valid:
        for candidate_dir in _standard_artifact_candidate_dirs(split, run_root_parts):
            reusable = _load_reusable_standard(
                model_name, split, candidate_dir, seed, adapter
            )
            if reusable is not None:
                return reusable, candidate_dir

    if resume_incomplete:
        reusable = _load_partial_generated_artifact(
            model_name,
            split,
            standard_seed_dir,
            seed,
            adapter,
            "standard",
        )
        if reusable is not None:
            return reusable, standard_seed_dir

    return None, None


def final_report_issues(report: dict[str, object], require_tail_targeted: bool = True) -> list[str]:
    """Return fail-fast issues that invalidate a final experiment cell."""
    issues: list[str] = []
    required = (
        "global_fidelity",
        "temporal_fidelity",
        "tail_fidelity",
        "cross_signal_dependence",
        "subgroup_fidelity",
        "privacy_memorization",
        "diversity_coverage",
        "downstream_utility_models",
        "downstream_utility_tasks",
        "distinguishability",
        "publication",
    )
    for key in required:
        if key not in report:
            issues.append(f"missing report block: {key}")
    downstream = report.get("downstream_utility_models", {})
    if not isinstance(downstream, dict) or downstream.get("status") != "completed":
        issues.append("primary downstream model evaluation did not complete")
    elif not isinstance(downstream.get("tail_learnability"), dict) or downstream[
        "tail_learnability"
    ].get("status") != "completed":
        issues.append("tail-learnability evaluation did not complete")
    outcome_tasks = report.get("downstream_utility_tasks", {})
    if not isinstance(outcome_tasks, dict) or outcome_tasks.get("status") != "completed":
        issues.append("learner-level outcome task evaluation did not complete")
    detection = report.get("distinguishability", {})
    if not isinstance(detection, dict) or detection.get("status") != "completed":
        issues.append("real-vs-synthetic distinguishability did not complete")
    publication = report.get("publication", {})
    if not isinstance(publication, dict) or publication.get("status") != "completed":
        issues.append("publication metric selection did not complete")
    else:
        if publication.get("schema_version") != SCHEMA_VERSION:
            issues.append("publication metric schema differs from the locked schema")
        publication_metrics = publication.get("metrics", {})
        if not isinstance(publication_metrics, dict):
            issues.append("publication metric tree is malformed")
        else:
            for research_question in ("rq1", "rq2"):
                metrics = publication_metrics.get(research_question)
                if (
                    not isinstance(metrics, dict)
                    or not metrics
                    or not flatten_numeric(metrics, research_question)
                ):
                    issues.append(
                        f"publication metric tree is empty for {research_question.upper()}"
                    )
    if require_tail_targeted:
        if not report.get("tail_targeted_size_match", {}).get("matched"):
            issues.append("tail-targeted output is absent or not size matched")
        if "tail_targeted_evaluation" not in report:
            issues.append("full tail-targeted evaluation is missing")
        privacy = report.get("privacy_memorization", {})
        if not isinstance(privacy, dict) or not {
            "standard", "tail_targeted", "comparison"
        }.issubset(privacy):
            issues.append("standard-vs-tail-targeted privacy comparison is incomplete")

    def find_failed(value: object, path: str = "report"):
        if isinstance(value, dict):
            if value.get("status") == "failed":
                issues.append(f"failed metric/model block: {path} ({value.get('reason', 'no reason')})")
            for key, child in value.items():
                find_failed(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                find_failed(child, f"{path}[{index}]")

    find_failed(report)
    try:
        json.dumps(report, allow_nan=False)
    except (TypeError, ValueError) as exc:
        issues.append(f"report is not strict JSON: {exc}")
    return issues


def run_seed(
    adapter,
    split,
    model_name: str,
    seed: int,
    tail_oversample: int = DEFAULT_TAIL_OVERSAMPLE,
    strict: bool = False,
    reuse_standard_if_valid: bool = False,
    resume_incomplete: bool = False,
) -> dict[str, object]:
    paths = experiment_seed_paths(split, model_name, seed)
    standard_seed_dir = paths["standard_seed_dir"]
    target_seed_dir = paths["target_seed_dir"]
    evaluation_path = paths["evaluation_report"]
    run_root_parts = list(
        standard_seed_dir.relative_to(standard_synthetic_root(split)).parts
    )
    occupied = occupied_experiment_seed_paths(
        split,
        model_name,
        seed,
        allow_standard_reuse=reuse_standard_if_valid or resume_incomplete,
        allow_target_reuse=resume_incomplete,
    )
    if occupied:
        rendered = "\n- ".join(str(path) for path in occupied)
        raise FileExistsError(
            f"Refusing to overwrite an existing {EXPERIMENT_DESIGN} cell:\n- "
            f"{rendered}\nResume the same run or archive the old cell before changing the design."
        )

    base_generation_train = generation_train(split)
    reusable, reuse_source_dir = _load_standard_for_run(
        model_name,
        split,
        standard_seed_dir,
        run_root_parts,
        seed,
        adapter,
        reuse_standard_if_valid=reuse_standard_if_valid,
        resume_incomplete=resume_incomplete,
    )
    if reusable is None:
        synthetic, synthetic_path, generation_metadata = _generate_synthetic(
            model_name, split, base_generation_train, standard_seed_dir, seed, adapter
        )
        standard_was_reused = False
        standard_reuse_namespace = None
    else:
        synthetic, synthetic_path, generation_metadata = reusable
        standard_was_reused = True
        standard_reuse_namespace = _standard_artifact_namespace(
            split, reuse_source_dir
        )
        generation_metadata.setdefault("standard_artifact_reuse", {})[
            "source_namespace"
        ] = standard_reuse_namespace

    # Tail-targeted arm (RQ2): re-run the SAME generator on a tail-oversampled
    # training set. Additive -- a failure here must not break the main run.
    tail_targeted = None
    tail_targeted_path = None
    tail_targeted_generation_metadata = None
    tail_selection_metadata: dict[str, object] = {
        "selection_contract": TAIL_SELECTION_CONTRACT,
        "status": "not_computed",
    }
    try:
        tail_train, tail_selection_metadata = tail_oversampled_generation_train(
            split, tail_oversample, return_metadata=True
        )
        if tail_train is not None:
            resumed_target = (
                _load_partial_generated_artifact(
                    model_name,
                    split,
                    target_seed_dir,
                    seed,
                    adapter,
                    "tail_targeted",
                )
                if resume_incomplete
                else None
            )
            if resumed_target is None:
                tail_targeted, tail_targeted_path, tail_metadata = _generate_synthetic(
                    model_name,
                    split,
                    tail_train,
                    target_seed_dir,
                    seed,
                    adapter,
                    generation_shape={
                        "target_rows": int(len(base_generation_train)),
                        "learner_count": int(
                            base_generation_train[
                                split.columns.get("learner", "learner_id")
                            ].nunique()
                        ),
                    },
                    generation_arm="tail_targeted",
                )
            else:
                tail_targeted, tail_targeted_path, tail_metadata = resumed_target
            generation_metadata["tail_targeted"] = {
                "oversample": tail_oversample,
                "train_rows": int(len(tail_train)),
                "tail_selection": tail_selection_metadata,
                **{key: tail_metadata[key] for key in ("synthetic_rows",) if key in tail_metadata},
            }
            tail_metadata["experiment_design"] = EXPERIMENT_DESIGN
            tail_metadata["tail_selection"] = tail_selection_metadata
            tail_targeted_generation_metadata = dict(tail_metadata)
            write_json(target_seed_dir / "generation_metadata.json", tail_metadata)
    except Exception as exc:  # noqa: BLE001 - additive arm, never fatal
        generation_metadata["tail_targeted_error"] = str(exc)
        if strict:
            raise RuntimeError(f"Tail-targeted generation failed: {exc}") from exc

    if strict and tail_targeted is None:
        raise RuntimeError(
            "Final experiment requires a size-matched tail-targeted arm, but no tail training set was produced."
        )

    evaluation_report = run_general_evaluation(
        adapter,
        split,
        synthetic,
        synthetic_path,
        tail_targeted,
        tail_targeted_path,
        generation_seed=seed,
    )
    evaluation_report["experiment_design"] = EXPERIMENT_DESIGN
    evaluation_report["dataset"] = split.name
    evaluation_report["model"] = model_name
    evaluation_report["tail_targeting_contract"] = tail_selection_metadata
    evaluation_report["artifact_provenance"] = {
        "standard": {
            "path": str(synthetic_path),
            "metadata": generation_metadata,
        },
        "tail_targeted": {
            "path": str(tail_targeted_path) if tail_targeted_path else None,
            "metadata": tail_targeted_generation_metadata,
        },
    }
    issues = final_report_issues(evaluation_report)
    evaluation_report["final_audit"] = {
        "status": "failed" if issues else "passed",
        "issues": issues,
        "strict_mode": bool(strict),
    }
    write_json(evaluation_path, evaluation_report)
    if strict and issues:
        raise RuntimeError(
            "Final evaluation audit failed; the diagnostic report was saved to "
            f"{evaluation_path}:\n- " + "\n- ".join(issues)
        )

    pipeline_report = {
        "experiment_design": EXPERIMENT_DESIGN,
        "dataset": split.name,
        "model": model_name,
        "seed": seed,
        "train": str(split.train_path),
        "test": str(split.test_path),
        "ignore_columns": list(split.ignore_columns),
        "standard_synthetic": str(synthetic_path),
        "tail_targeted_synthetic": str(tail_targeted_path) if tail_targeted_path else None,
        "generation_metadata": generation_metadata,
        "tail_targeted_generation_metadata": tail_targeted_generation_metadata,
        "standard_artifact_reused": standard_was_reused,
        "standard_artifact_reuse_namespace": standard_reuse_namespace,
        "data_provenance": dict(split.metadata.get("artifact_metadata", {})),
        "evaluation_report": str(evaluation_path),
    }
    write_json(paths["pipeline_report"], pipeline_report)
    publication_metrics = evaluation_report.get("publication", {}).get("metrics", {})
    return {
        **pipeline_report,
        "_metrics": flatten_numeric(publication_metrics),
        "_publication": publication_metrics,
    }


def load_completed_seed(
    adapter,
    split,
    model_name: str,
    seed: int,
) -> dict[str, object] | None:
    """Validate and load a completed seed for an explicit resume operation."""
    paths = experiment_seed_paths(split, model_name, seed)
    required = (
        paths["standard_csv"],
        paths["standard_metadata"],
        paths["target_csv"],
        paths["target_metadata"],
        paths["pipeline_report"],
        paths["evaluation_report"],
    )
    present = [path.exists() for path in required]
    if not any(present):
        return None
    if not all(present):
        # Standard and tail generation phases are resumable when their CSV and
        # metadata were atomically completed. Evaluation/pipeline artifacts are
        # final-phase files and must either both be complete or be absent.
        if not paths["pipeline_report"].exists() and not paths["evaluation_report"].exists():
            return None
        missing = [str(path) for path, exists in zip(required, present) if not exists]
        raise ValueError(
            "Cannot resume an incomplete experiment seed. Missing:\n- "
            + "\n- ".join(missing)
        )

    validated = _validate_standard_artifact_provenance(
        model_name, split, paths["standard_seed_dir"], seed
    )
    if validated is None:  # pragma: no cover - guarded by required paths above
        raise ValueError("Completed seed is missing its standard artifact.")
    # The standard artifact may carry an RQ1-only provenance record. Resume the
    # completed RQ2 cell from its explicit pipeline report, not from that older
    # standard-arm record.
    prior_pipeline = json.loads(paths["pipeline_report"].read_text(encoding="utf-8"))
    pipeline_identity_issues = []
    if prior_pipeline.get("experiment_design") != EXPERIMENT_DESIGN:
        pipeline_identity_issues.append("experiment design differs")
    if prior_pipeline.get("dataset") != split.name:
        pipeline_identity_issues.append("dataset differs")
    if prior_pipeline.get("model") != model_name:
        pipeline_identity_issues.append("model differs")
    if int(prior_pipeline.get("seed", -1)) != int(seed):
        pipeline_identity_issues.append("seed differs")
    if pipeline_identity_issues:
        raise ValueError(
            "Existing pipeline report is not resumable: "
            + "; ".join(pipeline_identity_issues)
        )

    target_metadata = json.loads(paths["target_metadata"].read_text(encoding="utf-8"))
    target_issues = []
    if int(target_metadata.get("seed", -1)) != int(seed):
        target_issues.append("generation seed differs")
    if str(target_metadata.get("model")) != str(model_name):
        target_issues.append("generator model differs")
    if target_metadata.get("experiment_design") != EXPERIMENT_DESIGN:
        target_issues.append("experiment design differs")
    if sha256_file(paths["target_csv"]) != target_metadata.get("synthetic_output_sha256"):
        target_issues.append("synthetic artifact hash differs from generation metadata")
    if target_issues:
        raise ValueError(
            "Existing tail-targeted artifact is not resumable: "
            + "; ".join(target_issues)
        )

    evaluation_report = json.loads(paths["evaluation_report"].read_text(encoding="utf-8"))
    issues = final_report_issues(evaluation_report)
    if evaluation_report.get("experiment_design") != EXPERIMENT_DESIGN:
        issues.append("evaluation-report experiment design differs")
    if evaluation_report.get("final_audit", {}).get("status") != "passed":
        issues.append("evaluation report did not pass its final audit")
    if issues:
        raise ValueError(
            "Existing completed seed is not resumable:\n- " + "\n- ".join(issues)
        )
    resumed = dict(prior_pipeline)
    resumed["evaluation_report"] = str(paths["evaluation_report"])
    resumed["tail_targeted_generation_metadata"] = target_metadata
    publication_metrics = evaluation_report.get("publication", {}).get("metrics", {})
    resumed["_metrics"] = flatten_numeric(publication_metrics)
    resumed["_publication"] = publication_metrics
    # Internal control-flow marker only. Keep promoted summaries invariant to
    # whether a valid seed was freshly completed or resumed.
    resumed["_resumed_completed_seed"] = True
    return resumed


def run_pipeline(
    dataset_name: str,
    model_name: str,
    seeds: Sequence[int],
    preprocess: bool,
    tail_oversample: int = DEFAULT_TAIL_OVERSAMPLE,
    strict: bool = False,
    reuse_standard_if_valid: bool = False,
    resume_existing: bool = False,
) -> dict[str, object]:
    adapter = get_dataset(dataset_name)
    if preprocess:
        dataset_package = adapter.__name__.rsplit(".", 1)[0]
        preprocess_module = __import__(f"{dataset_package}.preprocess", fromlist=["run_preprocessing"])
        preprocess_module.run_preprocessing()

    split = adapter.load_split()
    seed_reports = []
    for seed in seeds:
        seed = int(seed)
        completed = (
            load_completed_seed(adapter, split, model_name, seed)
            if resume_existing
            else None
        )
        if completed is not None:
            seed_reports.append(completed)
            continue
        seed_reports.append(
            run_seed(
                adapter,
                split,
                model_name,
                seed,
                tail_oversample,
                strict=strict,
                reuse_standard_if_valid=reuse_standard_if_valid,
                resume_incomplete=resume_existing,
            )
        )
    public_runs = [
        {key: value for key, value in report.items() if not key.startswith("_")}
        for report in seed_reports
    ]
    summary = {
        "experiment_design": EXPERIMENT_DESIGN,
        "dataset": split.name,
        "model": model_name,
        "seeds": [int(seed) for seed in seeds],
        "train": str(split.train_path),
        "test": str(split.test_path),
        "ignore_columns": list(split.ignore_columns),
        "data_provenance": dict(split.metadata.get("artifact_metadata", {})),
        "runs": public_runs,
        "metric_summary": summarize_metrics(seed_reports),
        "publication_status_summary": summarize_publication_statuses(seed_reports),
        "metric_summary_scope": "publication_only",
        "publication_schema_version": SCHEMA_VERSION,
    }
    summary_path = REPORTS_ROOT.joinpath(*report_parts(split, model_name)) / "multi_seed_summary.json"
    summary["summary_report"] = str(summary_path)
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the generic synthetic-data experiment pipeline.")
    parser.add_argument("--dataset", default="assistments")
    parser.add_argument("--model", default="block_bootstrap", help="Generator script name under synthetic_generation/generator_scripts.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="One or more random seeds. Defaults to three fixed seeds.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any required final report block or tail-targeted arm is incomplete.",
    )
    parser.add_argument("--preprocess", action="store_true", help="Run dataset preprocessing before generation.")
    parser.add_argument(
        "--tail-oversample",
        type=int,
        default=DEFAULT_TAIL_OVERSAMPLE,
        help="Oversampling factor for the tail-targeted generation arm (RQ2). 1 or 0 disables it.",
    )
    parser.add_argument(
        "--reuse-standard-if-valid",
        action="store_true",
        help=(
            "Reuse an existing standard synthetic artifact only when its hash, generator source, "
            "dataset provenance, and normalization contract match; regenerate the targeted arm."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and skip already completed seeds in this immutable variant.",
    )
    args = parser.parse_args()

    report = run_pipeline(
        args.dataset,
        args.model,
        args.seeds,
        args.preprocess,
        args.tail_oversample,
        args.strict,
        args.reuse_standard_if_valid,
        args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
