"""Learner-level downstream tasks beyond next-response correctness.

RQ2 names dropout / failure / mastery / recovery prediction as learning-analytics
tasks. This module implements the two that are derivable from an interaction
stream alone: predicting future low correctness, or future high correctness after
an observed failure prefix, from a learner's EARLY behaviour. These future-only
outcomes are conceptually related to the whole-trajectory tail groups but are not
the same label definitions. Each task is trained real-only, synthetic-only, and
real+synthetic, and tested on held-out real learners, so the TSTR gap says whether
synthetic data preserves the early-signal -> outcome relationship.

Kept dependency-light and self-contained (no import from general_evaluation) so
it can be a leaf module in the evaluation import graph.
"""

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
from evaluation.evaluate import DEFAULT_COLUMNS
from evaluation.metrics import binary_value, group_trajectories, mean, safe_float, tail_reference


# Learner-level tasks and the exact future-only outcome each one predicts.
_TASKS = {
    "persistent_failure_prediction": {
        "predicted_outcome": "future_low_correctness",
        "outcome_definition": (
            "Mean correctness after the fixed prefix is at or below the "
            "training-fitted low-correctness cutoff."
        ),
        "eligibility_definition": "All learners with enough post-prefix interactions.",
        "relationship_to_tail_analysis": (
            "Related to persistent failure, but distinct from the whole-trajectory tail-group label."
        ),
    },
    "recovery_prediction": {
        "predicted_outcome": "future_high_correctness_after_prefix_failure",
        "outcome_definition": "Mean correctness after the fixed prefix is at least 0.65.",
        "eligibility_definition": "Correctness in the fixed prefix is at most 0.35.",
        "relationship_to_tail_analysis": (
            "Related to recovery, but distinct from the half-vs-half tail-group label."
        ),
    },
}
_PREFIX_LENGTH = 2
_MIN_FUTURE_LENGTH = 2
_MIN_LENGTH = _PREFIX_LENGTH + _MIN_FUTURE_LENGTH
# A rare-outcome task needs enough positives on BOTH sides for a meaningful
# estimate; below this the task is reported as exploratory rather than a metric.
_MIN_POSITIVE_LEARNERS = 10
_HIGHER_IS_BETTER = ("auroc", "auprc")
_LABEL_PREFIX = "__label__"
_ELIGIBLE_PREFIX = "__eligible__"


def _records(frame: pd.DataFrame) -> list[dict]:
    return frame.astype(object).where(frame.notna(), "").astype(str).to_dict(orient="records")


def _columns(split) -> dict[str, str]:
    columns = dict(DEFAULT_COLUMNS)
    columns.update({str(key): str(value) for key, value in split.columns.items()})
    return columns


def _present(frame: pd.DataFrame, name: object) -> str | None:
    return str(name) if name and str(name) in frame.columns else None


def _signals(split) -> dict[str, str | None]:
    columns = _columns(split)
    gap = _present(split.train, columns.get("gap"))
    if gap is None:
        gap = next((str(c) for c in split.train.columns if "gap" in str(c).lower()), None)
    response_time = _present(split.train, columns.get("response_time"))
    if response_time is None:
        response_time = next(
            (
                str(c)
                for c in split.train.columns
                if "gap" not in str(c).lower()
                and any(keyword in str(c).lower() for keyword in ("response", "duration", "elapsed"))
            ),
            None,
        )
    return {
        "learner": columns["learner"],
        "order": columns["order"],
        "skill": columns["skill"],
        "correct": columns["correct"],
        "hint": _present(split.train, columns.get("hint")),
        "attempt": _present(split.train, columns.get("attempts")),
        "response_time": response_time,
        "gap": gap,
    }


def _feature_frame(trajectories, signals, reference) -> pd.DataFrame:
    skill, correct, hint = signals["skill"], signals["correct"], signals["hint"]
    attempt, response_time, gap = signals["attempt"], signals["response_time"], signals["gap"]
    failure_cutoff = reference.get("low_correctness_cutoff")
    if failure_cutoff is None:
        failure_cutoff = 0.35
    rows = []
    for trajectory in trajectories.values():
        if len(trajectory) < _MIN_LENGTH:
            continue
        # A fixed prefix avoids using the final trajectory length to choose the
        # feature cutoff. Everything after it is outcome-only and never appears
        # in X, so the prediction represents a real early-warning setting.
        early = trajectory[:_PREFIX_LENGTH]
        future = trajectory[_PREFIX_LENGTH:]
        early_correct_rate = mean([binary_value(record, correct) for record in early])
        future_correct_rate = mean([binary_value(record, correct) for record in future])
        feature = {
            "early_correct_rate": early_correct_rate,
            "early_hint_rate": mean([binary_value(record, hint) for record in early]) if hint else 0.0,
            "early_attempt_mean": mean([safe_float(record.get(attempt), 0.0) for record in early]) if attempt else 0.0,
            "n_distinct_skills_early": float(len({str(record.get(skill, "")) for record in early})),
            "first_skill": str(early[0].get(skill, "")),
        }
        if response_time:
            feature["early_response_time_mean"] = mean(
                [safe_float(record.get(response_time), 0.0) for record in early]
            )
        feature[f"{_LABEL_PREFIX}persistent_failure_prediction"] = int(
            future_correct_rate <= float(failure_cutoff)
        )
        # Recovery is defined among learners who demonstrably struggle in the
        # observed prefix; the predicted label itself uses future responses only.
        feature[f"{_ELIGIBLE_PREFIX}recovery_prediction"] = int(early_correct_rate <= 0.35)
        feature[f"{_LABEL_PREFIX}recovery_prediction"] = int(future_correct_rate >= 0.65)
        rows.append(feature)
    return pd.DataFrame(rows)


def _split_xy(frame: pd.DataFrame, task: str):
    target = f"{_LABEL_PREFIX}{task}"
    eligible = f"{_ELIGIBLE_PREFIX}{task}"
    if frame.empty or target not in frame.columns:
        return pd.DataFrame(), pd.Series(dtype=int, name=target)
    selected = frame[frame[eligible].astype(bool)].copy() if eligible in frame.columns else frame.copy()
    hidden_columns = [
        column
        for column in selected.columns
        if column.startswith(_LABEL_PREFIX) or column.startswith(_ELIGIBLE_PREFIX)
    ]
    x = selected.drop(columns=hidden_columns)
    return x, selected[target].astype(int)


def _arm_metrics(
    train_frame,
    test_frame,
    task,
    group_column,
    seed,
    *,
    allow_model_collapse: bool = False,
):
    x_train, y_train = _split_xy(train_frame, task)
    x_test, y_test = _split_xy(test_frame, task)
    try:
        overall, scores = fit_downstream_scored(
            x_train, y_train, x_test, y_test, group_column, seed
        )
        return {
            "overall": overall,
            "scores": scores,
            "y": y_test.to_numpy(dtype=int),
        }
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


def _delta(real_overall, other_overall) -> dict[str, dict[str, float]]:
    if not isinstance(real_overall, dict) or not isinstance(other_overall, dict):
        return {}
    per_model = {}
    for model, real_metrics in real_overall.items():
        other_metrics = other_overall.get(model)
        if not isinstance(real_metrics, dict) or not isinstance(other_metrics, dict):
            continue
        deltas = {
            metric: float(real_metrics[metric]) - float(other_metrics[metric])
            for metric in _HIGHER_IS_BETTER
            if isinstance(real_metrics.get(metric), (int, float))
            and isinstance(other_metrics.get(metric), (int, float))
        }
        if deltas:
            per_model[model] = deltas
    return per_model


def _metric_ci(arm, seed: int) -> dict[str, object]:
    """Learner-bootstrap confidence intervals for a learner-level task arm."""
    if arm["scores"] is None or len(arm["y"]) == 0:
        return {}
    # Each feature row represents exactly one learner. Supplying a unique group
    # per row makes the bootstrap contract explicit and keeps it consistent with
    # the interaction-level tail analysis, where all rows for a learner move
    # together.
    learners = np.arange(len(arm["y"]), dtype=int)
    return {
        model: interval
        for model, scores in arm["scores"].items()
        if (
            interval := bootstrap_metric_ci(
                arm["y"], scores, seed=seed, groups=learners
            )
        )
    }


def _delta_ci(real_arm, other_arm, seed: int) -> dict[str, object]:
    """Paired learner-bootstrap CIs for M(real) - M(other)."""
    if real_arm["scores"] is None or other_arm["scores"] is None:
        return {}
    if len(real_arm["y"]) == 0 or len(real_arm["y"]) != len(other_arm["y"]):
        return {}
    learners = np.arange(len(real_arm["y"]), dtype=int)
    return {
        model: interval
        for model, real_scores in real_arm["scores"].items()
        if model in other_arm["scores"]
        and (
            interval := bootstrap_delta_ci(
                real_arm["y"],
                real_scores,
                other_arm["scores"][model],
                seed=seed,
                groups=learners,
            )
        )
    }


def learner_level_task_report(split, synthetic_frame: pd.DataFrame, tail_targeted_frame: pd.DataFrame | None = None) -> dict[str, object]:
    signals = _signals(split)
    learner_col, order_col = signals["learner"], signals["order"]
    value_cols = [c for c in (signals["skill"], signals["correct"], signals["hint"], signals["attempt"], signals["response_time"], signals["gap"]) if c]

    def traj(frame):
        keep = [c for c in [learner_col, order_col, *value_cols] if c in frame.columns]
        return group_trajectories(_records(frame[keep]), learner_col, order_col)

    train_traj = traj(split.train)
    reference = tail_reference(
        train_traj, signals["skill"], signals["correct"], signals["hint"],
        signals["attempt"], signals["response_time"], signals["gap"],
    )
    real_train_features = _feature_frame(train_traj, signals, reference)
    real_test_features = _feature_frame(traj(split.test), signals, reference)
    synthetic_features = _feature_frame(traj(synthetic_frame), signals, reference)

    group_column = "first_skill"
    seed = split.seed
    tasks: dict[str, object] = {}
    for task, task_definition in _TASKS.items():
        _, real_train_y = _split_xy(real_train_features, task)
        _, real_test_y = _split_xy(real_test_features, task)
        real_positive = int(real_train_y.sum())
        test_positive = int(real_test_y.sum())
        real_task_learners = len(real_train_y)
        test_task_learners = len(real_test_y)
        real_negative = real_task_learners - real_positive
        test_negative = test_task_learners - test_positive
        # Need enough positives on each side to fit/score reliably; otherwise
        # flag exploratory rather than emit a metric on a handful of learners.
        if (
            real_task_learners == 0
            or test_task_learners == 0
            or real_positive < _MIN_POSITIVE_LEARNERS
            or test_positive < _MIN_POSITIVE_LEARNERS
            or real_negative < _MIN_POSITIVE_LEARNERS
            or test_negative < _MIN_POSITIVE_LEARNERS
        ):
            tasks[task] = {
                "status": "exploratory",
                **task_definition,
                "reason": (
                    f"Fewer than {_MIN_POSITIVE_LEARNERS} learners from either class in "
                    "train or test; too sparse for a reliable estimate."
                ),
                "train_positive_learners": real_positive,
                "test_positive_learners": test_positive,
                "train_negative_learners": real_negative,
                "test_negative_learners": test_negative,
                "train_learners": int(real_task_learners),
                "test_learners": int(test_task_learners),
            }
            continue
        real_arm = _arm_metrics(real_train_features, real_test_features, task, group_column, seed)
        synthetic_arm = _arm_metrics(
            synthetic_features,
            real_test_features,
            task,
            group_column,
            seed,
            allow_model_collapse=True,
        )
        augmented = pd.concat([real_train_features, synthetic_features], ignore_index=True)
        augmented_arm = _arm_metrics(augmented, real_test_features, task, group_column, seed)
        entry = {
            "status": "completed",
            **task_definition,
            "train_positive_learners": real_positive,
            "test_positive_learners": test_positive,
            "train_negative_learners": real_negative,
            "test_negative_learners": test_negative,
            "train_learners": int(real_task_learners),
            "test_learners": int(test_task_learners),
            "bootstrap_unit": "learner",
            "train_on_real_test_on_real": real_arm["overall"],
            "train_on_synthetic_test_on_real": synthetic_arm["overall"],
            "train_on_real_plus_synthetic_test_on_real": augmented_arm["overall"],
            "metric_ci": {
                "real": _metric_ci(real_arm, seed),
                "synthetic": _metric_ci(synthetic_arm, seed),
                "real_plus_synthetic": _metric_ci(augmented_arm, seed),
            },
            "delta": _delta(real_arm["overall"], synthetic_arm["overall"]),
            "delta_ci": _delta_ci(real_arm, synthetic_arm, seed),
            "delta_augmented": _delta(real_arm["overall"], augmented_arm["overall"]),
            "delta_augmented_ci": _delta_ci(real_arm, augmented_arm, seed),
        }
        if tail_targeted_frame is not None:
            tail_features = _feature_frame(traj(tail_targeted_frame), signals, reference)
            tail_arm = _arm_metrics(
                tail_features,
                real_test_features,
                task,
                group_column,
                seed,
                allow_model_collapse=True,
            )
            entry["train_on_tail_targeted_synthetic_test_on_real"] = tail_arm["overall"]
            entry["metric_ci"]["tail_targeted_synthetic"] = _metric_ci(tail_arm, seed)
            entry["delta_tail_targeted"] = _delta(real_arm["overall"], tail_arm["overall"])
            entry["delta_tail_targeted_ci"] = _delta_ci(real_arm, tail_arm, seed)
        tasks[task] = entry

    return {
        "status": "completed",
        "note": (
            "Fixed-prefix early behaviour -> future-only outcomes; delta = M(real)-M(synthetic), "
            ">0 = synthetic loses signal. Recovery is evaluated only among learners whose observed "
            "prefix shows failure, and its label is determined exclusively by later responses. "
            "All *_ci fields are 95% percentile learner-bootstrap intervals; delta intervals use "
            "paired resamples of the common held-out real learners."
        ),
        "feature_window": f"first_{_PREFIX_LENGTH}_interactions",
        "outcome_window": "all_subsequent_interactions",
        "tasks": tasks,
    }
