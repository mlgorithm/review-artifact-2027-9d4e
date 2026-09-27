"""Adapter for the proposal-aligned weekly OULAD engagement benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data.dataset_scripts.common import (
    DatasetSplit,
    enforce_learner_static_columns,
    rebuild_from_key,
    validate_processed_artifacts,
)


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENTS = ROOT / "experiments"
DATASET_DIR = EXPERIMENTS / "data" / "datasets" / "oulad_weekly"
DATASET_NAME = "oulad_weekly_engagement"
OUTPUTS_DIR = EXPERIMENTS / "outputs" / "oulad_weekly"
REPORTS_DIR = EXPERIMENTS / "reports" / DATASET_NAME

TERMINAL_OUTCOME_COLUMNS = ("dropout", "failure", "final_result")
DERIVED_COLUMNS = (
    "module_id",
    "presentation_id",
    "engaged",
    "gap_bin",
    *TERMINAL_OUTCOME_COLUMNS,
)
LOOKUP_DERIVED_COLUMNS = ("module_id", "presentation_id")
STATIC_COLUMNS = (
    "course_id",
)
PROFILE_COLUMNS = (
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
    "previous_attempt_bin",
    "studied_credits_bin",
)
DROPOUT_PREFIX_LENGTH = 4
DROPOUT_BEHAVIOR_FEATURES = (
    "future_engagement_rate",
    "future_engagement_std",
)


def train_path() -> Path:
    return DATASET_DIR / "train.csv"


def test_path() -> Path:
    return DATASET_DIR / "test.csv"


def ignored_columns() -> tuple[str, ...]:
    return ("learner_id", "order")


def derived_columns() -> tuple[str, ...]:
    return DERIVED_COLUMNS


def _binary(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").fillna(0).ge(0.5).astype(int)


def _derive_gap(frame: pd.DataFrame) -> pd.Series:
    gaps = pd.Series(index=frame.index, dtype="object")
    for _, group in frame.groupby("learner_id", sort=False):
        last_active: int | None = None
        for index, row in group.sort_values("order").iterrows():
            order = int(row["order"])
            if last_active is None:
                label = "start"
            else:
                gap = max(1, order - last_active)
                label = "1w" if gap == 1 else ("2-3w" if gap <= 3 else "4w+")
            gaps.loc[index] = label
            if int(row["engaged"]) == 1:
                last_active = order
    return gaps


def rebuild_derived_columns(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    restored = rebuild_from_key(
        frame, reference, "course_id", LOOKUP_DERIVED_COLUMNS
    )
    restored["registered"] = _binary(restored["registered"])
    click_nonzero = restored["click_bin"].astype(str).ne("0")
    restored["engaged"] = (click_nonzero & restored["registered"].eq(1)).astype(int)
    restored.loc[restored["engaged"].eq(0), "dominant_activity"] = "none"
    restored.loc[restored["engaged"].eq(0), "active_days_bin"] = "0"
    active_missing = restored["engaged"].eq(1) & restored["active_days_bin"].astype(str).eq("0")
    restored.loc[active_missing, "active_days_bin"] = "1-2"
    restored["gap_bin"] = _derive_gap(restored)
    return restored


def _sample_profiles_by_course(
    frame: pd.DataFrame, reference: pd.DataFrame, seed: int
) -> pd.DataFrame:
    """Sample demographic marginals without copying complete real profiles."""
    restored = frame.copy()
    profiles = reference.groupby("learner_id", sort=False)[
        ["course_id", *PROFILE_COLUMNS]
    ].first()
    by_course = {
        str(course): group.reset_index(drop=True)
        for course, group in profiles.groupby("course_id", sort=False)
    }
    fallback = profiles.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    learner_courses = (
        restored.groupby("learner_id", sort=False)["course_id"].first().reset_index()
    )
    assignment_parts = []
    for course, learners in learner_courses.groupby("course_id", sort=False):
        candidates = by_course.get(course, fallback)
        selected = pd.DataFrame(
            {
                column: rng.choice(candidates[column].to_numpy(), size=len(learners), replace=True)
                for column in PROFILE_COLUMNS
            }
        )
        selected.insert(0, "learner_id", learners["learner_id"].astype(str).to_numpy())
        assignment_parts.append(selected)
    assignments = pd.concat(assignment_parts, ignore_index=True).set_index("learner_id")
    learner_values = restored["learner_id"].astype(str)
    for column in PROFILE_COLUMNS:
        restored[column] = learner_values.map(assignments[column]).to_numpy()
    return restored


def _dropout_behavior_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize post-prefix behavior without using outcomes or trajectory length.

    The downstream dropout task sees the first ``DROPOUT_PREFIX_LENGTH`` weeks.
    This labeler deliberately uses only later generated engagement, so downstream
    predictability exists only when a generator preserves early-to-late behavior.
    """
    required = {"learner_id", "order", "course_id", "engaged"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "OULAD behavioral dropout labeling is missing columns: "
            f"{sorted(missing)}"
        )

    values = frame[["learner_id", "order", "course_id", "engaged"]].copy()
    values["learner_id"] = values["learner_id"].astype(str)
    values["course_id"] = values["course_id"].astype(str)
    values["order"] = pd.to_numeric(values["order"], errors="coerce")
    values["engaged"] = _binary(values["engaged"])
    values = values.dropna(subset=["order"]).sort_values(
        ["learner_id", "order"]
    )
    values["__position__"] = values.groupby("learner_id", sort=False).cumcount()

    features = values.groupby("learner_id", sort=False).agg(
        course_id=("course_id", "first")
    )
    future = values[values["__position__"].ge(DROPOUT_PREFIX_LENGTH)]
    future_summary = future.groupby("learner_id", sort=False)["engaged"].agg(
        future_engagement_rate="mean",
        future_engagement_std="std",
    )
    features = features.join(future_summary)
    features[list(DROPOUT_BEHAVIOR_FEATURES)] = features[
        list(DROPOUT_BEHAVIOR_FEATURES)
    ].fillna(0.0)
    return features


def fit_dropout_behavior_model(
    reference: pd.DataFrame,
) -> tuple[Pipeline, dict[str, object]]:
    """Fit the frozen dropout-risk ranker on the real training split only."""
    if "dropout" not in reference.columns:
        raise ValueError("OULAD training reference is missing dropout labels.")
    features = _dropout_behavior_features(reference)
    targets = (
        reference.assign(learner_id=reference["learner_id"].astype(str))
        .groupby("learner_id", sort=False)["dropout"]
        .first()
        .reindex(features.index)
        .pipe(_binary)
    )
    if targets.nunique() < 2:
        raise ValueError("OULAD training dropout labels require both classes.")

    preprocessor = ColumnTransformer(
        [
            (
                "course",
                OneHotEncoder(handle_unknown="ignore"),
                ["course_id"],
            ),
            (
                "behavior",
                StandardScaler(),
                list(DROPOUT_BEHAVIOR_FEATURES),
            ),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=0,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    feature_columns = ["course_id", *DROPOUT_BEHAVIOR_FEATURES]
    model.fit(features[feature_columns], targets)
    transformed_feature_names = (
        model.named_steps["preprocessor"].get_feature_names_out().astype(str).tolist()
    )
    classifier = model.named_steps["classifier"]
    metadata = {
        "version": "behavioral_dropout_v2",
        "fit_split": "real_train_only",
        "prefix_length": DROPOUT_PREFIX_LENGTH,
        "risk_features": list(DROPOUT_BEHAVIOR_FEATURES),
        "categorical_features": ["course_id"],
        "explicitly_excluded": [
            "dropout",
            "failure",
            "final_result",
            "trajectory_length",
            "prediction_prefix_behavior",
        ],
        "classifier": "sklearn_logistic_regression_balanced",
        "training_learners": int(len(features)),
        "training_positive_learners": int(targets.sum()),
        "training_negative_learners": int((1 - targets).sum()),
        "intercept": classifier.intercept_.astype(float).tolist(),
        "transformed_feature_names": transformed_feature_names,
        "coefficients": classifier.coef_.astype(float).tolist(),
        "calibration": "per_course_real_train_prevalence_rank",
    }
    return model, metadata


def _assign_behavioral_dropout(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    seed: int,
    fitted: tuple[Pipeline, dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Assign dropout from generated future behavior at train-fitted course rates."""
    restored = frame.copy()
    model, model_metadata = fitted or fit_dropout_behavior_model(reference)
    features = _dropout_behavior_features(restored)
    feature_columns = ["course_id", *DROPOUT_BEHAVIOR_FEATURES]
    features["__risk__"] = model.predict_proba(features[feature_columns])[:, 1]

    reference_profiles = (
        reference.assign(learner_id=reference["learner_id"].astype(str))
        .groupby("learner_id", sort=False)[["course_id", "dropout"]]
        .first()
    )
    reference_profiles["course_id"] = reference_profiles["course_id"].astype(str)
    reference_profiles["dropout"] = _binary(reference_profiles["dropout"])
    global_rate = float(reference_profiles["dropout"].mean())
    course_rates = reference_profiles.groupby("course_id")["dropout"].mean().to_dict()

    rng = np.random.default_rng(seed)
    labels = pd.Series(0, index=features.index, dtype=int)
    assigned_by_course: dict[str, dict[str, object]] = {}
    for course, course_features in features.groupby("course_id", sort=False):
        course_key = str(course)
        rate = float(course_rates.get(course_key, global_rate))
        count = min(
            len(course_features),
            max(0, int(round(rate * len(course_features)))),
        )
        ranked = course_features.assign(__tie_break__=rng.random(len(course_features)))
        ranked = ranked.sort_values(
            ["__risk__", "__tie_break__"], ascending=[False, False]
        )
        if count:
            labels.loc[ranked.index[:count]] = 1
        assigned_by_course[course_key] = {
            "learners": int(len(course_features)),
            "training_prevalence": rate,
            "assigned_dropout_learners": int(count),
        }

    learner_ids = restored["learner_id"].astype(str)
    restored["dropout"] = learner_ids.map(labels).astype(int)
    metadata = {
        **model_metadata,
        "seed": int(seed),
        "synthetic_learners": int(len(features)),
        "synthetic_dropout_learners": int(labels.sum()),
        "synthetic_dropout_rate": float(labels.mean()),
        "by_course": assigned_by_course,
    }
    return restored, metadata


def reassign_terminal_outcomes(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    seed: int,
    fitted: tuple[Pipeline, dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Recompute only derived outcomes for an existing normalized trajectory."""
    restored, metadata = _assign_behavioral_dropout(
        frame, reference, seed, fitted=fitted
    )
    restored = _assign_terminal_outcomes(restored, reference, seed)
    metadata["outcome_columns_reassigned"] = list(TERMINAL_OUTCOME_COLUMNS)
    return restored, metadata


def _assign_terminal_outcomes(
    frame: pd.DataFrame, reference: pd.DataFrame, seed: int
) -> pd.DataFrame:
    """Derive outcomes from generated completion/activity and train aggregates."""
    restored = frame.copy()
    reference_profiles = reference.groupby("learner_id", sort=False)[
        ["course_id", "dropout", "failure", "final_result"]
    ].first()
    reference_profiles["dropout"] = _binary(reference_profiles["dropout"])
    reference_profiles["failure"] = _binary(reference_profiles["failure"])
    completed_reference = reference_profiles[reference_profiles["dropout"].eq(0)]
    successful_reference = completed_reference[completed_reference["failure"].eq(0)]

    global_failure_rate = float(completed_reference["failure"].mean()) if len(completed_reference) else 0.0
    global_distinction_rate = (
        float(successful_reference["final_result"].astype(str).eq("Distinction").mean())
        if len(successful_reference)
        else 0.0
    )
    failure_by_course = completed_reference.groupby("course_id")["failure"].mean().to_dict()
    distinction_by_course = (
        successful_reference.assign(
            distinction=successful_reference["final_result"].astype(str).eq("Distinction").astype(int)
        )
        .groupby("course_id")["distinction"]
        .mean()
        .to_dict()
    )

    summaries = restored.groupby("learner_id", sort=False).agg(
        course_id=("course_id", "first"),
        dropout=("dropout", "first"),
        engagement=("engaged", "mean"),
    )
    outcomes: dict[str, tuple[int, str]] = {}
    rng = np.random.default_rng(seed)
    for course, course_summaries in summaries.groupby("course_id", sort=False):
        course_key = str(course)
        for learner_id in course_summaries.index[course_summaries["dropout"].eq(1)]:
            outcomes[str(learner_id)] = (0, "Withdrawn")

        completed = course_summaries[course_summaries["dropout"].eq(0)].copy()
        if completed.empty:
            continue
        completed["__tie_break__"] = rng.random(len(completed))
        completed = completed.sort_values(["engagement", "__tie_break__"])
        failure_rate = float(failure_by_course.get(course, failure_by_course.get(course_key, global_failure_rate)))
        failure_count = min(len(completed), max(0, int(round(failure_rate * len(completed)))))
        failed_ids = set(completed.index[:failure_count].astype(str))
        successful = completed.iloc[failure_count:].copy()
        distinction_rate = float(
            distinction_by_course.get(course, distinction_by_course.get(course_key, global_distinction_rate))
        )
        distinction_count = min(
            len(successful), max(0, int(round(distinction_rate * len(successful))))
        )
        distinction_ids = set(successful.index[-distinction_count:].astype(str)) if distinction_count else set()
        for learner_id in completed.index.astype(str):
            if learner_id in failed_ids:
                outcomes[learner_id] = (1, "Fail")
            elif learner_id in distinction_ids:
                outcomes[learner_id] = (0, "Distinction")
            else:
                outcomes[learner_id] = (0, "Pass")

    learner_ids = restored["learner_id"].astype(str)
    restored["failure"] = learner_ids.map(lambda learner: outcomes[learner][0]).astype(int)
    restored["final_result"] = learner_ids.map(lambda learner: outcomes[learner][1])
    return restored


def normalize_generated_frame(
    frame: pd.DataFrame, reference: pd.DataFrame, seed: int = 0
) -> pd.DataFrame:
    """Derive terminal outcomes after generation and enforce consistency."""
    restored = enforce_learner_static_columns(frame, "learner_id", STATIC_COLUMNS)
    restored = _sample_profiles_by_course(restored, reference, seed)

    # Every retained weekly row is inside the learner's registered interval.
    # This is a row-wise invariant, so a 26k-group split/copy/concat adds memory
    # without changing the result. The vectorized assignment is equivalent and
    # keeps target-only OULAD regeneration within a practical memory envelope.
    restored["registered"] = 1
    # Static course_id is enforced above, so rebuild its dependent values after
    # the mode operation and derive activity/gap signals from coherent inputs
    # before outcome ranking. Outcome assignment must see the same engagement
    # values that are persisted in the final artifact.
    restored = rebuild_derived_columns(restored, reference)
    restored, _ = reassign_terminal_outcomes(restored, reference, seed)
    return restored


def load_split() -> DatasetSplit:
    metadata = validate_processed_artifacts(
        DATASET_DIR / "metadata.json",
        Path(__file__).with_name("preprocess.py"),
        train_path(),
        test_path(),
    )
    train = pd.read_csv(train_path(), low_memory=False)
    test = pd.read_csv(test_path(), low_memory=False)
    return DatasetSplit(
        name=DATASET_NAME,
        train=train,
        test=test,
        train_path=train_path(),
        test_path=test_path(),
        root=DATASET_DIR,
        synthetic_root=OUTPUTS_DIR,
        evaluation_root=REPORTS_DIR,
        columns={
            "learner": "learner_id",
            "order": "order",
            "skill": "dominant_activity",
            "correct": "engaged",
            "gap": "gap_bin",
            "dropout": "dropout",
        },
        ignore_columns=ignored_columns(),
        derived_columns=derived_columns(),
        target="engaged",
        group_column="course_id",
        seed=20260703,
        metadata={
            "primary_signal_semantics": "weekly_engagement",
            "outcome_task_specs": [
                {"name": "dropout_prediction", "target": "dropout", "eligible": "registered"},
                {"name": "course_failure_prediction", "target": "failure", "exclude_when": "dropout"},
            ],
            "outcome_prefix_length": DROPOUT_PREFIX_LENGTH,
            # Demographics are independently resampled after generation. They
            # therefore cannot be used as if the generator learned their joint
            # relationship with behavior or outcomes.
            "outcome_static_features": ["course_id"],
            "publication_postprocessed_columns": [
                "registered",
                *PROFILE_COLUMNS,
                *TERMINAL_OUTCOME_COLUMNS,
            ],
            # These are generator-modeled weekly behavior signals. Declare
            # their scientifically meaningful relationships explicitly so the
            # dependence metric does not mistake the absence of hints or
            # attempts for an absence of measurable feature dependence.
            "publication_dependence_pairs": [
                ["dominant_activity", "click_bin"],
                ["dominant_activity", "active_days_bin"],
                ["click_bin", "active_days_bin"],
            ],
            "publication_privacy_scope": "behavior_projection_exact_duplicates_only",
            "demographic_metric_policy": (
                "exclude_from_model_comparison_until_demographics_are jointly generated"
            ),
            "artifact_metadata": metadata,
        },
    )


def generation_to_downstream_supervised(
    frame: pd.DataFrame, min_history: int = 1, return_learner_ids: bool = False
):
    required = {"learner_id", "order", "course_id", "engaged"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OULAD weekly frame is missing columns: {sorted(missing)}")
    df = frame.copy()
    df["learner_id"] = df["learner_id"].astype(str)
    df["order"] = pd.to_numeric(df["order"], errors="coerce")
    df["engaged"] = pd.to_numeric(df["engaged"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["learner_id", "order", "engaged"])
    df = df.sort_values(["learner_id", "order"]).reset_index(drop=True)
    df["interaction_index"] = df.groupby("learner_id").cumcount()

    dynamic = [
        "dominant_activity",
        "engaged",
        "registered",
        "click_bin",
        "active_days_bin",
        "gap_bin",
    ]
    for column in dynamic:
        if column in df.columns:
            df[f"prev_{column}"] = df.groupby("learner_id")[column].shift(1)
    df["cum_engagement_count"] = (
        df.groupby("learner_id")["engaged"]
        .apply(lambda values: values.shift(1).expanding().sum())
        .reset_index(level=0, drop=True)
    )
    df["cum_interactions"] = df["interaction_index"]
    df["cum_engagement_rate"] = df["cum_engagement_count"] / df["cum_interactions"].where(
        df["cum_interactions"].ne(0)
    )
    df = df[df["interaction_index"] >= min_history].copy()

    feature_columns = [
        "course_id",
        "interaction_index",
        "prev_dominant_activity",
        "prev_engaged",
        "prev_registered",
        "prev_click_bin",
        "prev_active_days_bin",
        "prev_gap_bin",
        "cum_engagement_count",
        "cum_interactions",
        "cum_engagement_rate",
    ]
    for column in feature_columns:
        if column not in df.columns:
            df[column] = pd.NA
    supervised = df[[*feature_columns, "engaged"]]
    if return_learner_ids:
        return supervised, df["learner_id"].tolist()
    return supervised
