"""Leakage-safe learner-level prediction of explicit terminal outcomes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.downstream_models import (
    DegenerateSplitError,
    bootstrap_delta_ci,
    bootstrap_metric_ci,
    degenerate_arm_overall,
    fit_downstream_scored,
)
from evaluation.metrics import binary_value, safe_float


_MIN_CLASS_LEARNERS = 10
_HIGHER_IS_BETTER = ("auroc", "auprc")
_OUTCOME_COLUMNS = {"dropout", "failure", "final_result"}


def _feature_frame(frame: pd.DataFrame, split, spec: dict[str, object]) -> pd.DataFrame:
    columns = dict(split.columns)
    learner = columns.get("learner", "learner_id")
    order = columns.get("order", "order")
    primary = columns.get("correct", split.target)
    skill = columns.get("skill")
    hint = columns.get("hint")
    attempts = columns.get("attempts")
    gap = columns.get("gap") or next(
        (str(column) for column in frame.columns if "gap" in str(column).lower()), None
    )
    prefix_length = int(split.metadata.get("outcome_prefix_length", 4))
    static_columns = [
        str(column)
        for column in split.metadata.get("outcome_static_features", [])
        if str(column) in frame.columns and str(column) not in _OUTCOME_COLUMNS
    ]
    target = str(spec["target"])
    if target not in frame.columns:
        return pd.DataFrame()

    ordered = frame.sort_values([learner, order])
    rows = []
    for _, trajectory in ordered.groupby(learner, sort=False):
        if len(trajectory) < prefix_length:
            continue
        prefix = trajectory.iloc[:prefix_length]
        if spec.get("eligible"):
            eligibility = str(spec["eligible"])
            if eligibility not in prefix.columns or not all(
                binary_value(record, eligibility)
                for record in prefix.astype(object).where(prefix.notna(), "").astype(str).to_dict("records")
            ):
                continue
        if spec.get("exclude_when"):
            excluded = str(spec["exclude_when"])
            if excluded in trajectory.columns and binary_value(
                trajectory.iloc[0].to_dict(), excluded
            ):
                continue

        prefix_records = prefix.astype(object).where(prefix.notna(), "").astype(str).to_dict("records")
        feature: dict[str, object] = {
            "prefix_primary_rate": float(
                np.mean([binary_value(record, primary) for record in prefix_records])
            ),
            "prefix_primary_last": binary_value(prefix_records[-1], primary),
            "first_skill": str(prefix_records[0].get(skill, "")) if skill else "",
            "last_skill": str(prefix_records[-1].get(skill, "")) if skill else "",
            "prefix_distinct_skills": float(
                len({str(record.get(skill, "")) for record in prefix_records})
            ) if skill else 0.0,
        }
        if hint and hint in prefix.columns:
            feature["prefix_hint_rate"] = float(
                np.mean([binary_value(record, hint) for record in prefix_records])
            )
        if attempts and attempts in prefix.columns:
            feature["prefix_attempt_mean"] = float(
                np.mean([safe_float(record.get(attempts), 0.0) for record in prefix_records])
            )
        if gap and gap in prefix.columns:
            feature["last_gap_bin"] = str(prefix_records[-1].get(gap, ""))
        for column in static_columns:
            feature[column] = str(prefix.iloc[0][column])
        feature["__target__"] = int(binary_value(trajectory.iloc[0].to_dict(), target))
        rows.append(feature)
    return pd.DataFrame(rows)


def _xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if frame.empty or "__target__" not in frame.columns:
        return pd.DataFrame(), pd.Series(dtype=int)
    return frame.drop(columns="__target__"), frame["__target__"].astype(int)


def _arm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    group_column: str,
    seed: int,
    *,
    allow_model_collapse: bool = False,
):
    x_train, y_train = _xy(train)
    x_test, y_test = _xy(test)
    try:
        overall, scores = fit_downstream_scored(
            x_train, y_train, x_test, y_test, group_column, seed
        )
        return {"overall": overall, "scores": scores, "y": y_test.to_numpy(dtype=int)}
    except DegenerateSplitError as exc:
        return {
            "overall": degenerate_arm_overall(
                exc,
                y_train,
                y_test,
                allow_model_collapse=allow_model_collapse,
            ),
            "scores": None,
            "y": y_test.to_numpy(dtype=int),
        }
    except ValueError as exc:
        return {
            "overall": {"status": "failed", "reason": str(exc)},
            "scores": None,
            "y": y_test.to_numpy(dtype=int),
        }


def _delta(real, other):
    result = {}
    for model, real_metrics in real.get("overall", {}).items():
        if not isinstance(real_metrics, dict):
            continue
        other_metrics = other.get("overall", {}).get(model, {})
        values = {
            metric: float(real_metrics[metric]) - float(other_metrics[metric])
            for metric in _HIGHER_IS_BETTER
            if isinstance(real_metrics.get(metric), (int, float))
            and isinstance(other_metrics.get(metric), (int, float))
        }
        if values:
            result[model] = values
    return result


def _intervals(arm, seed: int):
    if arm["scores"] is None:
        return {}
    groups = np.arange(len(arm["y"]), dtype=int)
    return {
        model: ci
        for model, scores in arm["scores"].items()
        if (ci := bootstrap_metric_ci(arm["y"], scores, seed=seed, groups=groups))
    }


def _delta_intervals(real, other, seed: int):
    if real["scores"] is None or other["scores"] is None:
        return {}
    groups = np.arange(len(real["y"]), dtype=int)
    return {
        model: ci
        for model, scores in real["scores"].items()
        if model in other["scores"]
        and (
            ci := bootstrap_delta_ci(
                real["y"], scores, other["scores"][model], seed=seed, groups=groups
            )
        )
    }


def explicit_outcome_task_report(
    split,
    synthetic_frame: pd.DataFrame,
    tail_targeted_frame: pd.DataFrame | None = None,
) -> dict[str, object]:
    specs = [dict(spec) for spec in split.metadata.get("outcome_task_specs", [])]
    if not specs:
        return {"status": "skipped", "reason": "No explicit outcome-task specifications."}

    tasks: dict[str, object] = {}
    for spec_index, spec in enumerate(specs):
        name = str(spec["name"])
        real_train = _feature_frame(split.train, split, spec)
        real_test = _feature_frame(split.test, split, spec)
        synthetic = _feature_frame(synthetic_frame, split, spec)
        _, train_y = _xy(real_train)
        _, test_y = _xy(real_test)
        counts = {
            "train_learners": int(len(train_y)),
            "test_learners": int(len(test_y)),
            "train_positive_learners": int(train_y.sum()),
            "test_positive_learners": int(test_y.sum()),
            "train_negative_learners": int(len(train_y) - train_y.sum()),
            "test_negative_learners": int(len(test_y) - test_y.sum()),
        }
        if min(counts.values()) < _MIN_CLASS_LEARNERS:
            tasks[name] = {
                "status": "exploratory",
                "target": str(spec["target"]),
                "reason": f"Need at least {_MIN_CLASS_LEARNERS} learners from each class in train and test.",
                **counts,
            }
            continue

        seed = int(split.seed) + spec_index
        group_column = next(
            (
                str(column)
                for column in split.metadata.get("outcome_static_features", [])
                if str(column) in real_train.columns
            ),
            "first_skill",
        )
        real_arm = _arm(real_train, real_test, group_column, seed)
        synthetic_arm = _arm(
            synthetic, real_test, group_column, seed, allow_model_collapse=True
        )
        augmented_arm = _arm(
            pd.concat([real_train, synthetic], ignore_index=True), real_test, group_column, seed
        )
        entry: dict[str, object] = {
            "status": "completed",
            "target": str(spec["target"]),
            "feature_window": f"first_{int(split.metadata.get('outcome_prefix_length', 4))}_events",
            "outcome_window": "terminal_outcome_after_feature_window",
            "outcome_columns_excluded_from_features": sorted(_OUTCOME_COLUMNS),
            "bootstrap_unit": "learner",
            **counts,
            "train_on_real_test_on_real": real_arm["overall"],
            "train_on_synthetic_test_on_real": synthetic_arm["overall"],
            "train_on_real_plus_synthetic_test_on_real": augmented_arm["overall"],
            "metric_ci": {
                "real": _intervals(real_arm, seed),
                "synthetic": _intervals(synthetic_arm, seed),
                "real_plus_synthetic": _intervals(augmented_arm, seed),
            },
            "delta": _delta(real_arm, synthetic_arm),
            "delta_ci": _delta_intervals(real_arm, synthetic_arm, seed),
            "delta_augmented": _delta(real_arm, augmented_arm),
            "delta_augmented_ci": _delta_intervals(real_arm, augmented_arm, seed),
        }
        if tail_targeted_frame is not None:
            targeted = _feature_frame(tail_targeted_frame, split, spec)
            targeted_arm = _arm(
                targeted,
                real_test,
                group_column,
                seed,
                allow_model_collapse=True,
            )
            entry["train_on_tail_targeted_synthetic_test_on_real"] = targeted_arm["overall"]
            entry["metric_ci"]["tail_targeted_synthetic"] = _intervals(targeted_arm, seed)
            entry["delta_tail_targeted"] = _delta(real_arm, targeted_arm)
            entry["delta_tail_targeted_ci"] = _delta_intervals(real_arm, targeted_arm, seed)
        tasks[name] = entry
    return {
        "status": "completed",
        "task_family": "fixed_prefix_to_explicit_terminal_outcome",
        "note": (
            "Terminal outcomes are used only as y or eligibility/exclusion criteria; repeated outcome "
            "columns never enter X. Every arm is tested on the same held-out real learners."
        ),
        "tasks": tasks,
    }
