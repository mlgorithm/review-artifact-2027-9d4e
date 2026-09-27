"""General synthetic-data evaluation called by the generic pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.data import EvaluationDataset, SupervisedSplit
from evaluation.downstream_models import (
    DegenerateSplitError,
    bootstrap_delta_ci,
    bootstrap_metric_ci,
    degenerate_arm_overall,
    fit_downstream_scored,
    real_vs_synthetic_detection,
    tail_slice_metrics,
)
from evaluation.evaluate import DEFAULT_COLUMNS, evaluate, privacy_risk_comparison
from evaluation.learner_tasks import learner_level_task_report
from evaluation.outcome_tasks import explicit_outcome_task_report
from evaluation.metrics import group_trajectories, tail_labels, tail_reference
from evaluation.publication import build_publication_report


# Threshold-free ranking metrics used for publication tail-learnability gaps.
# Fixed-threshold scores remain raw diagnostics but are not compared across arms.
_TAIL_DELTA_METRICS = ("auroc", "auprc")
_MIN_RELIABLE_TAIL_LEARNERS = 10
_CORE_EVALUATION_MAX_LEARNERS = 4_000
_CORE_EVALUATION_MAX_ROWS = 120_000


def records(frame):
    return frame.astype(object).where(frame.notna(), "").astype(str).to_dict(orient="records")


def _core_evaluation_sample(
    frame: pd.DataFrame,
    learner_column: str,
    seed: int,
    max_learners: int = _CORE_EVALUATION_MAX_LEARNERS,
    max_rows: int = _CORE_EVALUATION_MAX_ROWS,
) -> pd.DataFrame:
    """Deterministic whole-learner sample for memory-heavy fidelity/privacy blocks."""
    sizes = frame.groupby(learner_column, sort=False).size()
    if len(sizes) <= max_learners and len(frame) <= max_rows:
        return frame
    identifiers = np.asarray(sorted(sizes.index.astype(str)), dtype=object)
    rng = np.random.RandomState(seed)
    shuffled = identifiers[rng.permutation(len(identifiers))]
    size_by_string = {str(key): int(value) for key, value in sizes.items()}
    selected: list[str] = []
    rows = 0
    for identifier in shuffled:
        count = size_by_string[str(identifier)]
        if selected and (len(selected) >= max_learners or rows + count > max_rows):
            continue
        selected.append(str(identifier))
        rows += count
        if len(selected) >= max_learners or rows >= max_rows:
            break
    if not selected:
        selected = [str(shuffled[0])]
    return frame[frame[learner_column].astype(str).isin(set(selected))].copy()


def _sampling_metadata(original: pd.DataFrame, sampled: pd.DataFrame, learner: str):
    return {
        "original_rows": int(len(original)),
        "evaluated_rows": int(len(sampled)),
        "original_learners": int(original[learner].nunique()),
        "evaluated_learners": int(sampled[learner].nunique()),
        "whole_learner_sample": True,
        "sampling_applied": bool(
            len(original) != len(sampled)
            or original[learner].nunique() != sampled[learner].nunique()
        ),
    }


def evaluation_columns(split) -> dict[str, str]:
    columns = dict(DEFAULT_COLUMNS)
    columns.update({str(key): str(value) for key, value in split.columns.items()})
    return columns


def _primary_downstream_task(split) -> str:
    semantics = str(
        getattr(split, "metadata", {}).get("primary_signal_semantics", "")
    )
    return (
        "next_week_engagement"
        if semantics == "weekly_engagement"
        else "next_response_correctness"
    )


def _present(frame: pd.DataFrame, name: object) -> str | None:
    """Return ``name`` only if it is a real, non-empty column of ``frame``."""
    return str(name) if name and str(name) in frame.columns else None


def _detect_gap_column(frame: pd.DataFrame) -> str | None:
    for column in frame.columns:
        if "gap" in str(column).lower():
            return str(column)
    return None


def _detect_response_time_column(frame: pd.DataFrame) -> str | None:
    # response_time_bin / duration_bin / elapsed_time_bin; a gap column also
    # contains "time", so exclude it.
    for column in frame.columns:
        name = str(column).lower()
        if "gap" in name:
            continue
        if "response" in name or "duration" in name or "elapsed" in name:
            return str(column)
    return None


def _tail_signal_columns(split) -> dict[str, str | None]:
    """Resolve the optional tail signals (attempts/response-time/gap) present here."""
    columns = evaluation_columns(split)
    return {
        "skill": columns["skill"],
        "correct": columns["correct"],
        "hint": _present(split.train, columns.get("hint")),
        "attempt": _present(split.train, columns.get("attempts")),
        "response_time": _present(split.train, columns.get("response_time")) or _detect_response_time_column(split.train),
        "gap": _present(split.train, columns.get("gap")) or _detect_gap_column(split.train),
        "dropout": _present(split.train, columns.get("dropout")),
    }


def _trajectories(frame: pd.DataFrame, learner_col: str, order_col: str, value_cols: list[str]):
    """Group a generation frame into per-learner trajectories (subset of columns).

    Only the columns tail labelling needs are materialised, to keep the record
    conversion cheap on the large datasets.
    """
    keep = [column for column in [learner_col, order_col, *value_cols] if column and column in frame.columns]
    return group_trajectories(records(frame[keep]), learner_col, order_col)


def _tail_records(split, frame: pd.DataFrame) -> list[dict[str, str]]:
    """Materialize only columns required by full-frame rare-group fidelity."""
    columns = evaluation_columns(split)
    signals = _tail_signal_columns(split)
    keep = [
        column
        for column in (
            columns["learner"],
            columns["order"],
            signals["skill"],
            signals["correct"],
            signals["hint"],
            signals["attempt"],
            signals["response_time"],
            signals["gap"],
            signals["dropout"],
        )
        if column and column in frame.columns
    ]
    # Preserve first occurrence while removing aliases that resolve to the same
    # physical column.
    keep = list(dict.fromkeys(keep))
    return records(frame[keep])


def _test_learner_sequence(split, learner_col: str, order_col: str, correct_col: str) -> list[str]:
    """Learner id for each supervised test row, in supervised row order.

    Fallback used only when the dataset adapter cannot return learner ids aligned
    to its supervised rows. Mirrors the deterministic row filtering every adapter
    applies in ``generation_to_downstream_supervised`` (drop rows with missing
    learner/order/correct, sort by learner then order, drop each learner's first
    interaction), reconstructing the row->learner mapping so tail masks align
    positionally with the model scores.
    """
    frame = split.test[[learner_col, order_col, correct_col]].copy()
    frame[learner_col] = frame[learner_col].astype(str)
    frame[order_col] = pd.to_numeric(frame[order_col], errors="coerce")
    frame[correct_col] = pd.to_numeric(frame[correct_col], errors="coerce")
    frame = frame.dropna(subset=[learner_col, order_col, correct_col])
    frame = frame.sort_values([learner_col, order_col]).reset_index(drop=True)
    keep = frame.groupby(learner_col).cumcount() >= 1  # min_history=1, matches adapters
    return frame.loc[keep, learner_col].tolist()


def _supervised_with_ids(adapter, frame: pd.DataFrame):
    """Build the supervised table and, when the adapter supports it, the exact
    per-row learner ids. Preferring the adapter's own ids removes the fragile
    reconstruction of its filtering logic (a future adapter that filters on extra
    columns would silently mis-align tail slices otherwise)."""
    try:
        supervised, learner_ids = adapter.generation_to_downstream_supervised(frame, return_learner_ids=True)
        return supervised, [str(value) for value in learner_ids]
    except TypeError:
        return adapter.generation_to_downstream_supervised(frame), None


def _tail_masks(adapter, split, test_rows: int, test_learner_ids: list[str] | None):
    """Boolean masks (aligned to supervised test rows) for each tail group.

    Tail thresholds are fixed on the real *train* split (no leakage from test)
    and applied to label real *test* learners; every supervised test row inherits
    its learner's membership. Returns masks, counts, and aligned learner ids, or
    three ``None`` values when the mapping cannot be aligned.
    """
    columns = evaluation_columns(split)
    learner_col, order_col = columns["learner"], columns["order"]
    signals = _tail_signal_columns(split)
    skill_col, correct_col, hint_col = signals["skill"], signals["correct"], signals["hint"]
    attempt_col, response_time_col, gap_col = signals["attempt"], signals["response_time"], signals["gap"]
    dropout_col = signals["dropout"]
    value_cols = [skill_col, correct_col, hint_col, attempt_col, response_time_col, gap_col, dropout_col]

    reference = tail_reference(
        _trajectories(split.train, learner_col, order_col, value_cols),
        skill_col, correct_col, hint_col, attempt_col, response_time_col, gap_col,
        dropout_col,
    )
    test_labels = tail_labels(
        _trajectories(split.test, learner_col, order_col, value_cols),
        skill_col, correct_col, hint_col, reference, attempt_col, response_time_col, gap_col,
        dropout_col,
    )
    learner_seq = test_learner_ids
    if learner_seq is None:
        learner_seq = _test_learner_sequence(split, learner_col, order_col, correct_col)
    if len(learner_seq) != test_rows:
        # Adapter row filtering diverged from the id reconstruction; skip rather
        # than risk mis-aligned tail slices.
        return None, None, None

    group_names = sorted({name for labels in test_labels.values() for name in labels})
    masks: dict[str, np.ndarray] = {}
    counts: dict[str, dict[str, int]] = {}
    for name in group_names:
        mask = np.fromiter(
            (test_labels.get(learner, {}).get(name, False) for learner in learner_seq),
            dtype=bool,
            count=len(learner_seq),
        )
        masks[name] = mask
        counts[name] = {
            "test_rows": int(mask.sum()),
            "test_learners": int(sum(1 for labels in test_labels.values() if labels.get(name, False))),
        }
    return masks, counts, learner_seq


def _run_arm(
    dataset: EvaluationDataset,
    *,
    allow_model_collapse: bool = False,
) -> dict[str, object]:
    """Fit one downstream training arm, keeping per-model test scores for slicing.

    Degrades gracefully on a degenerate split (empty/single-class); genuine
    schema errors from ``validate`` still propagate.
    """
    try:
        dataset.validate()
        overall, scores = fit_downstream_scored(
            dataset.train.x,
            dataset.train.y,
            dataset.test.x,
            dataset.test.y,
            dataset.group_column,
            dataset.seed,
        )
        return {"overall": overall, "scores": scores, "y": dataset.test.y}
    except DegenerateSplitError as exc:
        return {
            "overall": degenerate_arm_overall(
                exc,
                dataset.train.y,
                dataset.test.y,
                allow_model_collapse=allow_model_collapse,
            ),
            "scores": None,
            "y": dataset.test.y,
        }


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _delta(real_metrics, other_metrics) -> dict[str, float]:
    per_metric = {}
    for metric in _TAIL_DELTA_METRICS:
        real_value, other_value = real_metrics.get(metric), other_metrics.get(metric)
        if _is_number(real_value) and _is_number(other_value):
            per_metric[metric] = float(real_value) - float(other_value)
    return per_metric


def _tail_learnability(
    real_arm,
    synthetic_arm,
    augmented_arm,
    tail_targeted_arm,
    tail_targeted_augmented_arm,
    masks,
    counts,
    learner_ids,
    seed,
) -> dict[str, object]:
    """Per-tail-group downstream metrics, tail-learnability gaps, and bootstrap CIs.

    delta_tail = M(real->real) - M(arm->real) per group/model/metric.
    delta_tail_augmented uses the real+synthetic arm; delta_tail_tail_targeted
    uses the tail-oversampled synthetic arm; delta_tail_tail_targeted_augmented
    uses real+tail-oversampled synthetic (RQ2: does tail-targeted augmentation
    recover rare-group performance?). Bootstrap CIs (resampling the tail slice)
    accompany the synthetic tail metrics and every delta so a rare-group result
    can be read with its uncertainty instead of a point estimate. Resampling is
    clustered by learner so correlated interactions remain together.
    """
    if masks is None:
        return {"status": "skipped", "reason": "Tail masks could not be aligned to supervised test rows."}
    if real_arm["scores"] is None or synthetic_arm["scores"] is None:
        return {"status": "skipped", "reason": "A downstream arm failed to fit (degenerate split)."}

    labels = np.asarray(list(real_arm["y"]), dtype=int)  # every arm is tested on the same real_test split
    learner_ids = np.asarray(learner_ids, dtype=str)
    optional_arms = {
        "augmented": {
            "arm": augmented_arm if augmented_arm is not None and augmented_arm["scores"] is not None else None,
            "train_key": "train_on_real_plus_synthetic_test_on_real",
            "delta_key": "delta_tail_augmented",
        },
        "tail_targeted": {
            "arm": tail_targeted_arm if tail_targeted_arm is not None and tail_targeted_arm["scores"] is not None else None,
            "train_key": "train_on_tail_targeted_synthetic_test_on_real",
            "delta_key": "delta_tail_tail_targeted",
        },
        "tail_targeted_augmented": {
            "arm": (
                tail_targeted_augmented_arm
                if tail_targeted_augmented_arm is not None and tail_targeted_augmented_arm["scores"] is not None
                else None
            ),
            "train_key": "train_on_real_plus_tail_targeted_synthetic_test_on_real",
            "delta_key": "delta_tail_tail_targeted_augmented",
        },
    }

    groups: dict[str, object] = {}
    for name, mask in masks.items():
        real_by_model = {m: tail_slice_metrics(labels, s, mask) for m, s in real_arm["scores"].items()}
        synthetic_by_model = {m: tail_slice_metrics(labels, s, mask) for m, s in synthetic_arm["scores"].items()}
        entry: dict[str, object] = {
            **counts[name],
            "exploratory": counts[name]["test_learners"] < _MIN_RELIABLE_TAIL_LEARNERS,
            "train_on_real_test_on_real": real_by_model,
            "train_on_synthetic_test_on_real": synthetic_by_model,
            "delta_tail": {},
            "delta_tail_ci": {},
            "synthetic_tail_metric_ci": {},
        }
        optional_by_model = {}
        for arm_name, spec in optional_arms.items():
            arm = spec["arm"]
            if arm is not None:
                optional_by_model[arm_name] = {m: tail_slice_metrics(labels, s, mask) for m, s in arm["scores"].items()}
                entry[spec["train_key"]] = optional_by_model[arm_name]
                entry[spec["delta_key"]] = {}

        sliced_labels = labels[mask]
        sliced_learners = learner_ids[mask]
        for model, real_metrics in real_by_model.items():
            if real_metrics.get("status") == "empty":
                continue
            real_scores = real_arm["scores"][model][mask]
            synthetic_scores = synthetic_arm["scores"][model][mask]

            synthetic_delta = _delta(real_metrics, synthetic_by_model[model])
            if synthetic_delta:
                entry["delta_tail"][model] = synthetic_delta
                entry["delta_tail_ci"][model] = bootstrap_delta_ci(
                    sliced_labels, real_scores, synthetic_scores, seed=seed, groups=sliced_learners
                )
            metric_ci = bootstrap_metric_ci(
                sliced_labels, synthetic_scores, seed=seed, groups=sliced_learners
            )
            if metric_ci:
                entry["synthetic_tail_metric_ci"][model] = metric_ci

            for arm_name, spec in optional_arms.items():
                arm = spec["arm"]
                if arm is None:
                    continue
                arm_delta = _delta(real_metrics, optional_by_model[arm_name][model])
                if arm_delta:
                    delta_key = spec["delta_key"]
                    entry[delta_key].setdefault(model, {})
                    entry[delta_key][model] = arm_delta
                    entry.setdefault(f"{delta_key}_ci", {})
                    entry[f"{delta_key}_ci"][model] = bootstrap_delta_ci(
                        sliced_labels,
                        real_scores,
                        arm["scores"][model][mask],
                        seed=seed,
                        groups=sliced_learners,
                    )
        groups[name] = entry
    return {
        "status": "completed",
        "tail_reference_split": "real_train",
        "bootstrap_unit": "learner",
        "delta_tail_definition": (
            "M(real->real) - M(arm->real) on higher-is-better metrics; >0 means that arm "
            "loses tail signal. delta_tail: synthetic; delta_tail_augmented: real+synthetic; "
            "delta_tail_tail_targeted: tail-oversampled synthetic; "
            "delta_tail_tail_targeted_augmented: real+tail-oversampled synthetic. "
            "*_ci are percentile bootstrap CIs."
        ),
        "groups": groups,
    }


def downstream_report(adapter, split, synthetic_frame, synthetic_path: Path, tail_targeted_frame=None, tail_targeted_path=None) -> dict[str, object]:
    if not hasattr(adapter, "generation_to_downstream_supervised"):
        return {
            "status": "skipped",
            "reason": "Dataset adapter does not define generation_to_downstream_supervised.",
        }

    primary_task = _primary_downstream_task(split)
    real_train = adapter.generation_to_downstream_supervised(split.train)
    real_test, real_test_ids = _supervised_with_ids(adapter, split.test)
    synthetic_train = adapter.generation_to_downstream_supervised(synthetic_frame)

    real_test_split = SupervisedSplit("real_test", real_test, split.target, split.test_path)

    def build(name, train_frame, path) -> EvaluationDataset:
        # Every arm trains on a different set but is tested on the shared real_test.
        return EvaluationDataset(
            task=primary_task,
            target=split.target,
            group_column=split.group_column,
            train=SupervisedSplit(name, train_frame, split.target, path),
            test=real_test_split,
            seed=split.seed,
        )

    real_dataset = build("real_train", real_train, split.train_path)
    synthetic_dataset = build("synthetic_train", synthetic_train, synthetic_path)
    # RQ2 augmentation arm: train on real + synthetic, test on real.
    augmented_train = pd.concat([real_train, synthetic_train], ignore_index=True)
    augmented_dataset = build("real_plus_synthetic", augmented_train, synthetic_path)

    real_arm = _run_arm(real_dataset)
    synthetic_arm = _run_arm(synthetic_dataset, allow_model_collapse=True)
    augmented_arm = _run_arm(augmented_dataset)

    # RQ2 tail-targeted arm: train on the tail-oversampled synthetic set only.
    tail_targeted_arm = None
    tail_targeted_rows = None
    tail_targeted_augmented_arm = None
    tail_targeted_augmented_rows = None
    if tail_targeted_frame is not None:
        tail_targeted_train = adapter.generation_to_downstream_supervised(tail_targeted_frame)
        tail_targeted_rows = len(tail_targeted_train)
        tail_targeted_arm = _run_arm(
            build("tail_targeted_synthetic", tail_targeted_train, tail_targeted_path or synthetic_path),
            allow_model_collapse=True,
        )
        tail_targeted_augmented_train = pd.concat([real_train, tail_targeted_train], ignore_index=True)
        tail_targeted_augmented_rows = len(tail_targeted_augmented_train)
        tail_targeted_augmented_arm = _run_arm(
            build(
                "real_plus_tail_targeted_synthetic",
                tail_targeted_augmented_train,
                tail_targeted_path or synthetic_path,
            )
        )

    try:
        masks, counts, tail_learner_ids = _tail_masks(adapter, split, len(real_test), real_test_ids)
        tail = _tail_learnability(
            real_arm,
            synthetic_arm,
            augmented_arm,
            tail_targeted_arm,
            tail_targeted_augmented_arm,
            masks,
            counts,
            tail_learner_ids,
            split.seed,
        )
    except Exception as exc:  # tail learnability is additive; never break the core report
        tail = {"status": "skipped", "reason": f"Tail-learnability computation failed: {exc}"}

    report = {
        "status": "completed",
        "task": primary_task,
        "target": split.target,
        "group_column": split.group_column,
        "evaluation_seed": split.seed,
        "ignored_generation_columns": list(split.ignore_columns),
        "feature_columns": real_train.drop(columns=[split.target]).columns.tolist(),
        "test_positive_rows": int(real_test[split.target].astype(int).sum()),
        "test_negative_rows": int(
            len(real_test) - real_test[split.target].astype(int).sum()
        ),
        "test_positive_rate": float(real_test[split.target].astype(int).mean()),
        "train_on_real_test_on_real": real_arm["overall"],
        "train_on_synthetic_test_on_real": synthetic_arm["overall"],
        "train_on_real_plus_synthetic_test_on_real": augmented_arm["overall"],
        "tail_learnability": tail,
        "rows": {
            "real_train": len(real_train),
            "synthetic_train": len(synthetic_train),
            "real_test": len(real_test),
        },
    }
    if tail_targeted_arm is not None:
        report["train_on_tail_targeted_synthetic_test_on_real"] = tail_targeted_arm["overall"]
        report["train_on_real_plus_tail_targeted_synthetic_test_on_real"] = tail_targeted_augmented_arm["overall"]
        report["rows"]["tail_targeted_synthetic_train"] = tail_targeted_rows
        report["rows"]["real_plus_tail_targeted_synthetic_train"] = tail_targeted_augmented_rows
    return report


def detection_report(split, synthetic_frame) -> dict[str, object]:
    """Real-vs-synthetic classifier AUC (Table-4 global distinguishability)."""
    ignore = set(split.ignore_columns)
    value_columns = [
        column
        for column in split.train.columns
        if column not in ignore and column in synthetic_frame.columns
    ]
    if not value_columns:
        return {"status": "skipped", "reason": "No shared value columns for detection."}
    try:
        learner_col = split.columns.get("learner", "learner_id")
        return real_vs_synthetic_detection(
            split.train[value_columns],
            synthetic_frame[value_columns],
            split.seed,
            real_groups=split.train[learner_col],
            synthetic_groups=synthetic_frame[learner_col],
        )
    except Exception as exc:  # additive metric; never break the core report
        return {"status": "skipped", "reason": f"Detection classifier failed: {exc}"}


def run_general_evaluation(
    adapter,
    split,
    synthetic_frame,
    synthetic_path: Path,
    tail_targeted_frame=None,
    tail_targeted_path=None,
    generation_seed: int | None = None,
) -> dict[str, object]:
    columns = evaluation_columns(split)
    split_metadata = dict(getattr(split, "metadata", {}) or {})
    learner_column = columns["learner"]
    real_train_eval = _core_evaluation_sample(
        split.train, learner_column, split.seed
    )
    real_test_eval = _core_evaluation_sample(
        split.test, learner_column, split.seed + 1
    )
    synthetic_eval = _core_evaluation_sample(
        synthetic_frame, learner_column, split.seed + 2
    )
    tail_real_train = _tail_records(split, split.train)
    tail_real_test = _tail_records(split, split.test)
    tail_synthetic = _tail_records(split, synthetic_frame)
    report = evaluate(
        real_train=records(real_train_eval),
        real_test=records(real_test_eval),
        synthetic=records(synthetic_eval),
        columns=columns,
        seed=split.seed,
        tail_real_train=tail_real_train,
        tail_real_test=tail_real_test,
        tail_synthetic=tail_synthetic,
    )
    report["evaluation_sampling"] = {
        "scope": "fidelity_temporal_dependence_subgroup_privacy_diversity_lightweight_utility",
        "real_train": _sampling_metadata(split.train, real_train_eval, learner_column),
        "real_test": _sampling_metadata(split.test, real_test_eval, learner_column),
        "synthetic": _sampling_metadata(synthetic_frame, synthetic_eval, learner_column),
        "tail_fidelity": {
            "scope": "complete_frames",
            "reference": "complete_real_train",
            "real_train": _sampling_metadata(split.train, split.train, learner_column),
            "real_test": _sampling_metadata(split.test, split.test, learner_column),
            "synthetic": _sampling_metadata(synthetic_frame, synthetic_frame, learner_column),
        },
        "max_learners_per_split": _CORE_EVALUATION_MAX_LEARNERS,
        "max_rows_per_split": _CORE_EVALUATION_MAX_ROWS,
        "note": (
            "Rare-group fidelity, primary downstream models, and explicit outcome tasks use "
            "the complete frames; only the listed memory-heavy diagnostics are sampled."
        ),
    }
    report["primary_binary_signal"] = columns["correct"]
    report["primary_binary_signal_semantics"] = split_metadata.get(
        "primary_signal_semantics", "response_correctness"
    )
    primary_task = _primary_downstream_task(split)
    if isinstance(report.get("downstream_utility"), dict):
        report["downstream_utility"]["task"] = primary_task
    report["generation_seed"] = generation_seed
    report["evaluation_seed"] = split.seed
    standard_privacy = report["privacy_memorization"]
    privacy_bundle = {"standard": standard_privacy}
    if tail_targeted_frame is not None:
        learner_col = columns["learner"]
        size_match = {
            "standard_rows": int(len(synthetic_frame)),
            "tail_targeted_rows": int(len(tail_targeted_frame)),
            "standard_learners": int(synthetic_frame[learner_col].nunique()),
            "tail_targeted_learners": int(tail_targeted_frame[learner_col].nunique()),
        }
        size_match["matched"] = (
            size_match["standard_rows"] == size_match["tail_targeted_rows"]
            and size_match["standard_learners"] == size_match["tail_targeted_learners"]
        )
        if not size_match["matched"]:
            raise ValueError(
                "Standard and tail-targeted synthetic outputs must have identical row and "
                f"learner budgets for a valid comparison: {size_match}."
            )
        report["tail_targeted_size_match"] = size_match

        # RQ2 asks whether targeting improves tail preservation, not only
        # downstream utility. Score the complete targeted set with the same real
        # splits, train-fitted cutoffs, and evaluation seed as the standard set.
        targeted_eval = _core_evaluation_sample(
            tail_targeted_frame, learner_column, split.seed + 2
        )
        tail_targeted_records = _tail_records(split, tail_targeted_frame)
        targeted_evaluation = evaluate(
            real_train=records(real_train_eval),
            real_test=records(real_test_eval),
            synthetic=records(targeted_eval),
            columns=columns,
            seed=split.seed,
            tail_real_train=tail_real_train,
            tail_real_test=tail_real_test,
            tail_synthetic=tail_targeted_records,
        )
        if isinstance(targeted_evaluation.get("downstream_utility"), dict):
            targeted_evaluation["downstream_utility"]["task"] = primary_task
        targeted_evaluation["evaluation_sampling"] = {
            "real_train": _sampling_metadata(split.train, real_train_eval, learner_column),
            "real_test": _sampling_metadata(split.test, real_test_eval, learner_column),
            "synthetic": _sampling_metadata(tail_targeted_frame, targeted_eval, learner_column),
            "tail_fidelity": {
                "scope": "complete_frames",
                "reference": "complete_real_train",
                "real_train": _sampling_metadata(split.train, split.train, learner_column),
                "real_test": _sampling_metadata(split.test, split.test, learner_column),
                "synthetic": _sampling_metadata(
                    tail_targeted_frame, tail_targeted_frame, learner_column
                ),
            },
        }
        targeted_privacy = targeted_evaluation.pop("privacy_memorization")
        targeted_evaluation["distinguishability"] = detection_report(
            split, tail_targeted_frame
        )
        report["tail_targeted_evaluation"] = targeted_evaluation
        privacy_bundle["tail_targeted"] = targeted_privacy
        privacy_bundle["comparison"] = privacy_risk_comparison(
            standard_privacy, targeted_privacy
        )
    report["privacy_memorization"] = privacy_bundle
    report["downstream_utility_models"] = downstream_report(
        adapter, split, synthetic_frame, synthetic_path, tail_targeted_frame, tail_targeted_path
    )
    report["distinguishability"] = detection_report(split, synthetic_frame)
    try:
        if split_metadata.get("outcome_task_specs"):
            report["downstream_utility_tasks"] = explicit_outcome_task_report(
                split, synthetic_frame, tail_targeted_frame
            )
        else:
            report["downstream_utility_tasks"] = learner_level_task_report(
                split, synthetic_frame, tail_targeted_frame
            )
    except Exception as exc:  # additive; never break the core report
        report["downstream_utility_tasks"] = {"status": "skipped", "reason": f"Learner-level tasks failed: {exc}"}
    report["publication"] = build_publication_report(report, split)
    return report
