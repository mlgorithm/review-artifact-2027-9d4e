"""Downstream utility models for synthetic-data evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler


class DegenerateSplitError(ValueError):
    """Raised when a supervised split cannot train a classifier.

    Distinct from the plain ValueError raised by EvaluationDataset.validate()
    (a schema/config error): a DegenerateSplitError is data-dependent (empty
    split or single-class target) and callers may choose to record it as a
    failed evaluation cell and continue, rather than aborting the whole run.
    """


def degenerate_arm_overall(
    exc: DegenerateSplitError,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    allow_model_collapse: bool,
) -> dict[str, object]:
    """Describe an unfit arm without confusing model collapse with code failure."""
    def counts(values: pd.Series) -> dict[str, int]:
        return {
            str(label): int(count)
            for label, count in values.dropna().astype(int).value_counts().sort_index().items()
        }

    return {
        "status": "not_estimable_model_collapse" if allow_model_collapse else "failed",
        "reason": str(exc),
        "reason_category": (
            "synthetic_training_target_degenerate"
            if allow_model_collapse
            else "required_supervised_split_degenerate"
        ),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "train_class_counts": counts(y_train),
        "test_class_counts": counts(y_test),
    }


def classification_metrics(y_true: Iterable[int], y_score: Iterable[float]) -> Dict[str, float | None]:
    labels = np.asarray(list(y_true), dtype=int)
    scores = np.asarray(list(y_score), dtype=float)
    predictions = (scores >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "log_loss": float(log_loss(labels, np.clip(scores, 1e-12, 1 - 1e-12), labels=[0, 1])),
        "brier": float(brier_score_loss(labels, scores)),
    }
    two_classes = len(np.unique(labels)) == 2
    metrics["auroc"] = float(roc_auc_score(labels, scores)) if two_classes else None
    # A single-class test set has no ranking signal. Report None (serialized as
    # JSON null) rather than NaN: NaN makes the report file invalid JSON for
    # strict parsers (jq, JS, R) and, if averaged, poisons the multi-seed
    # summary. It must also not be the base rate, which for an all-positive set
    # reads as a perfect 1.0 score.
    metrics["auprc"] = float(average_precision_score(labels, scores)) if two_classes else None
    return metrics


@dataclass
class GroupMeanBaseline:
    group_column: str
    global_mean_: float = 0.0
    group_means_: Mapping[object, float] | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "GroupMeanBaseline":
        frame = pd.DataFrame({"group": x[self.group_column].astype(str), "target": y.astype(int)})
        self.global_mean_ = float(frame["target"].mean())
        self.group_means_ = frame.groupby("group")["target"].mean().to_dict()
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.group_means_ is None:
            raise RuntimeError("GroupMeanBaseline must be fitted before prediction.")
        return x[self.group_column].astype(str).map(self.group_means_).fillna(self.global_mean_).to_numpy()


def split_feature_types(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = []
    numeric = []
    for column in x.columns:
        if pd.api.types.is_numeric_dtype(x[column]):
            numeric.append(column)
        else:
            categorical.append(column)
    return categorical, numeric


def _categorical_strings(values) -> np.ndarray:
    """Normalize mixed CSV/object categories before sklearn encoding."""
    frame = pd.DataFrame(values).astype("string").fillna("__missing__")
    return frame.astype(str).to_numpy()


def logistic_regression_pipeline(x: pd.DataFrame, seed: int) -> Pipeline:
    categorical, numeric = split_feature_types(x)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    [
                        ("stringify", FunctionTransformer(_categorical_strings, validate=False)),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=50,
                                max_categories=100,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ],
        sparse_threshold=0.3,
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=300,
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        ]
    )


def gradient_boosting_pipeline(x: pd.DataFrame, seed: int) -> Pipeline:
    categorical, numeric = split_feature_types(x)
    categorical_mask = [True] * len(categorical) + [False] * len(numeric)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    [
                        ("stringify", FunctionTransformer(_categorical_strings, validate=False)),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                                min_frequency=50,
                                max_categories=254,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            ),
        ],
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_iter=120,
                    categorical_features=categorical_mask,
                    random_state=seed,
                ),
            ),
        ]
    )


def model_scores(model_name: str, model, x_test: pd.DataFrame) -> np.ndarray:
    if model_name == "group_mean":
        return model.predict_proba(x_test)
    # NumPy 2 / SciPy sparse matmul can emit transient overflow warnings even
    # when sklearn's stabilized probability result is finite. Silence only the
    # low-level arithmetic warnings, then enforce finiteness explicitly.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        scores = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError(f"{model_name} produced non-finite downstream probabilities.")
    return scores


def fit_downstream_scored(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    group_column: str,
    seed: int,
) -> tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    """Fit each downstream model once; return overall metrics and raw test scores.

    Returning the per-model test scores lets callers additionally evaluate tail
    slices (tail learnability) without refitting.
    """
    if len(y_train) == 0 or len(y_test) == 0:
        raise DegenerateSplitError(
            f"Degenerate supervised split: train rows={len(y_train)}, test rows={len(y_test)}. "
            "Check that `correct` is numeric/binary and that history filtering did not empty the frame."
        )
    if y_train.nunique() < 2:
        raise DegenerateSplitError(
            f"Downstream train target has a single class ({y_train.dropna().unique().tolist()}); "
            "need both correct=0 and correct=1 to fit a classifier."
        )
    models = {
        "group_mean": GroupMeanBaseline(group_column),
        "logistic_regression": logistic_regression_pipeline(x_train, seed),
        "hist_gradient_boosting": gradient_boosting_pipeline(x_train, seed),
    }
    overall: Dict[str, Dict[str, float]] = {}
    scores_by_model: Dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        scores = np.asarray(model_scores(name, model, x_test), dtype=float)
        scores_by_model[name] = scores
        overall[name] = classification_metrics(y_test, scores)
    return overall, scores_by_model


# Publication gaps use threshold-free ranking metrics only. Accuracy, F1,
# precision, and recall remain available in raw arm diagnostics, but a fixed
# 0.5 threshold is not comparable across differently calibrated training arms.
_CI_METRICS = ("auroc", "auprc")
_DEFAULT_N_BOOT = 200
_MAX_BOOTSTRAP_ROWS = 3000
_MAX_BOOTSTRAP_LEARNERS = 1000


def _bootstrap_arrays(
    *arrays: np.ndarray,
    seed: int,
    groups: np.ndarray | None = None,
    row_cap: int = _MAX_BOOTSTRAP_ROWS,
    learner_cap: int = _MAX_BOOTSTRAP_LEARNERS,
):
    """Prepare row or learner-cluster bootstrap arrays with a seeded cap."""
    n = len(arrays[0])
    rng = np.random.RandomState(seed)
    if any(len(array) != n for array in arrays):
        raise ValueError("Bootstrap arrays must have equal length.")
    if groups is not None:
        groups = np.asarray(groups).astype(str)
        if len(groups) != n:
            raise ValueError("Bootstrap learner groups must align with the score rows.")
        unique_groups = np.unique(groups)
        if len(unique_groups) > learner_cap:
            selected_groups = rng.choice(unique_groups, size=learner_cap, replace=False)
            keep = np.flatnonzero(np.isin(groups, selected_groups))
            arrays = tuple(array[keep] for array in arrays)
            groups = groups[keep]
        cluster_indices = [np.flatnonzero(groups == group) for group in np.unique(groups)]
        return arrays, cluster_indices, rng
    if n > row_cap:
        keep = rng.choice(n, size=row_cap, replace=False)
        arrays = tuple(array[keep] for array in arrays)
        n = row_cap
    return arrays, None, rng


def _bootstrap_resample_indices(
    n_rows: int,
    rng: np.random.RandomState,
    cluster_indices: list[np.ndarray] | None = None,
) -> np.ndarray:
    if cluster_indices is None:
        return rng.randint(0, n_rows, size=n_rows)
    selected = rng.randint(0, len(cluster_indices), size=len(cluster_indices))
    return np.concatenate([cluster_indices[index] for index in selected])


def _percentile_ci(values: list[float], alpha: float) -> Dict[str, float] | None:
    if not values:
        return None
    return {
        "ci_low": float(np.percentile(values, 100 * alpha / 2)),
        "ci_high": float(np.percentile(values, 100 * (1 - alpha / 2))),
        "n_boot": len(values),
    }


def bootstrap_metric_ci(
    y_true: Iterable[int],
    scores: np.ndarray,
    metric_keys: Iterable[str] = _CI_METRICS,
    n_boot: int = _DEFAULT_N_BOOT,
    seed: int = 0,
    alpha: float = 0.05,
    groups: Iterable[object] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Percentile bootstrap CIs, clustered by learner when groups are supplied."""
    labels = np.asarray(list(y_true), dtype=int)
    values = np.asarray(scores, dtype=float)
    if len(labels) == 0:
        return {}
    group_values = np.asarray(list(groups)) if groups is not None else None
    (labels, values), clusters, rng = _bootstrap_arrays(
        labels, values, seed=seed, groups=group_values
    )
    keys = list(metric_keys)
    samples: Dict[str, list[float]] = {key: [] for key in keys}
    for _ in range(n_boot):
        idx = _bootstrap_resample_indices(len(labels), rng, clusters)
        metrics = classification_metrics(labels[idx], values[idx])
        for key in keys:
            value = metrics.get(key)
            if value is not None:
                samples[key].append(float(value))
    return {key: ci for key in keys if (ci := _percentile_ci(samples[key], alpha))}


def bootstrap_delta_ci(
    y_true: Iterable[int],
    real_scores: np.ndarray,
    other_scores: np.ndarray,
    metric_keys: Iterable[str] = _CI_METRICS,
    n_boot: int = _DEFAULT_N_BOOT,
    seed: int = 0,
    alpha: float = 0.05,
    groups: Iterable[object] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Percentile bootstrap CIs for delta_tail = M(real) - M(other) on a slice.

    Paired resampling (the same resampled test rows score both arms) so the CI
    reflects the paired difference, which is what determines whether delta_tail is
    distinguishable from zero for a rare group.
    """
    labels = np.asarray(list(y_true), dtype=int)
    real = np.asarray(real_scores, dtype=float)
    other = np.asarray(other_scores, dtype=float)
    if len(labels) == 0:
        return {}
    group_values = np.asarray(list(groups)) if groups is not None else None
    (labels, real, other), clusters, rng = _bootstrap_arrays(
        labels, real, other, seed=seed, groups=group_values
    )
    keys = list(metric_keys)
    samples: Dict[str, list[float]] = {key: [] for key in keys}
    for _ in range(n_boot):
        idx = _bootstrap_resample_indices(len(labels), rng, clusters)
        sub_labels = labels[idx]
        real_metrics = classification_metrics(sub_labels, real[idx])
        other_metrics = classification_metrics(sub_labels, other[idx])
        for key in keys:
            real_value, other_value = real_metrics.get(key), other_metrics.get(key)
            if real_value is not None and other_value is not None:
                samples[key].append(float(real_value) - float(other_value))
    return {key: ci for key in keys if (ci := _percentile_ci(samples[key], alpha))}


def tail_slice_metrics(
    y_true: Iterable[int], scores: np.ndarray, mask: np.ndarray
) -> Dict[str, object]:
    """Classification metrics on the subset of test rows selected by ``mask``.

    ``mask`` is a boolean array aligned to the test rows (and therefore to
    ``scores``). Empty slices return a status marker; small/single-class slices
    are flagged so they can be treated as exploratory.
    """
    selected = int(np.count_nonzero(mask))
    if selected == 0:
        return {"status": "empty", "test_rows": 0}
    labels = np.asarray(list(y_true), dtype=int)[mask]
    sliced_scores = np.asarray(scores, dtype=float)[mask]
    metrics: Dict[str, object] = dict(classification_metrics(labels, sliced_scores))
    metrics["test_rows"] = selected
    metrics["positive_rate"] = float(labels.mean())
    metrics["single_class"] = bool(len(np.unique(labels)) < 2)
    return metrics


def fit_downstream_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    group_column: str,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    overall, _ = fit_downstream_scored(x_train, y_train, x_test, y_test, group_column, seed)
    return overall


def fit_downstream_dataset(dataset) -> Dict[str, Dict[str, float]]:
    dataset.validate()
    return fit_downstream_models(
        dataset.train.x,
        dataset.train.y,
        dataset.test.x,
        dataset.test.y,
        dataset.group_column,
        dataset.seed,
    )


def real_vs_synthetic_detection(
    real_values: pd.DataFrame,
    synthetic_values: pd.DataFrame,
    seed: int,
    test_fraction: float = 0.3,
    real_groups: Iterable[object] | None = None,
    synthetic_groups: Iterable[object] | None = None,
    max_rows_per_class: int = 100_000,
) -> Dict[str, object]:
    """Train a classifier to tell real rows from synthetic ones; report detection AUROC.

    This is the Table-4 "real-vs-synthetic classifier AUC": AUROC ~0.5 means the
    synthetic rows are indistinguishable from real (high global fidelity); AUROC
    near 1.0 means a model separates them trivially (low fidelity).
    """
    shared = [column for column in real_values.columns if column in synthetic_values.columns]
    if not shared:
        return {"status": "skipped", "reason": "No shared feature columns for detection."}
    rng = np.random.RandomState(seed)

    def group_masks(groups, rows):
        values = np.asarray(list(groups)) if groups is not None else np.arange(rows)
        if len(values) != rows:
            raise ValueError("Detection learner groups must align with feature rows.")
        unique = np.unique(values.astype(str))
        if len(unique) < 2:
            return None, None, 0, 0
        shuffled = unique[rng.permutation(len(unique))]
        test_groups = max(1, min(len(unique) - 1, int(round(len(unique) * test_fraction))))
        held_out = set(shuffled[:test_groups])
        test_mask = np.fromiter((str(value) in held_out for value in values), dtype=bool, count=rows)
        return ~test_mask, test_mask, len(unique) - test_groups, test_groups

    real_train, real_test, real_train_groups, real_test_groups = group_masks(real_groups, len(real_values))
    synth_train, synth_test, synth_train_groups, synth_test_groups = group_masks(
        synthetic_groups, len(synthetic_values)
    )
    if real_train is None or synth_train is None:
        return {"status": "skipped", "reason": "Need at least two real and synthetic learners."}
    # Split learners first, then cap interaction rows within each already-disjoint
    # partition. This keeps the detector affordable on million-row weekly panels
    # without allowing any learner to cross train/test. The cap is per class over
    # train+test and preserves the requested test fraction approximately.
    train_cap = max(1, int(round(max_rows_per_class * (1.0 - test_fraction))))
    test_cap = max(1, max_rows_per_class - train_cap)

    def selected(mask, cap):
        indices = np.flatnonzero(mask)
        if len(indices) > cap:
            indices = np.sort(rng.choice(indices, size=cap, replace=False))
        return indices

    real_train_idx = selected(real_train, train_cap)
    real_test_idx = selected(real_test, test_cap)
    synth_train_idx = selected(synth_train, train_cap)
    synth_test_idx = selected(synth_test, test_cap)

    def evaluated_groups(groups, indices, fallback):
        if groups is None:
            return fallback
        values = np.asarray(list(groups)).astype(str)
        return int(len(np.unique(values[indices])))

    real_train_groups = evaluated_groups(real_groups, real_train_idx, real_train_groups)
    real_test_groups = evaluated_groups(real_groups, real_test_idx, real_test_groups)
    synth_train_groups = evaluated_groups(synthetic_groups, synth_train_idx, synth_train_groups)
    synth_test_groups = evaluated_groups(synthetic_groups, synth_test_idx, synth_test_groups)
    x_train = pd.concat(
        [real_values.iloc[real_train_idx][shared], synthetic_values.iloc[synth_train_idx][shared]],
        ignore_index=True,
    )
    x_test = pd.concat(
        [real_values.iloc[real_test_idx][shared], synthetic_values.iloc[synth_test_idx][shared]],
        ignore_index=True,
    )
    y_train = np.concatenate(
        [np.ones(len(real_train_idx), dtype=int), np.zeros(len(synth_train_idx), dtype=int)]
    )
    y_test = np.concatenate(
        [np.ones(len(real_test_idx), dtype=int), np.zeros(len(synth_test_idx), dtype=int)]
    )
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"status": "skipped", "reason": "Degenerate detection split (single class)."}

    model = gradient_boosting_pipeline(x_train, seed)
    model.fit(x_train, pd.Series(y_train))
    scores = model.predict_proba(x_test)[:, 1]
    raw_auroc = float(roc_auc_score(y_test, scores))
    detectability_auc = max(raw_auroc, 1.0 - raw_auroc)
    return {
        "status": "completed",
        "raw_auroc": raw_auroc,
        "detectability_auc": detectability_auc,
        # Compatibility alias. It now has the orientation-invariant
        # detectability meaning; use raw_auroc only as a diagnostic.
        "auroc": detectability_auc,
        "accuracy": float(accuracy_score(y_test, (scores >= 0.5).astype(int))),
        "real_rows": int(len(real_values)),
        "synthetic_rows": int(len(synthetic_values)),
        "evaluated_real_rows": int(len(real_train_idx) + len(real_test_idx)),
        "evaluated_synthetic_rows": int(len(synth_train_idx) + len(synth_test_idx)),
        "max_rows_per_class": int(max_rows_per_class),
        "split_unit": "learner" if real_groups is not None and synthetic_groups is not None else "row",
        "real_train_learners": int(real_train_groups),
        "real_test_learners": int(real_test_groups),
        "synthetic_train_learners": int(synth_train_groups),
        "synthetic_test_learners": int(synth_test_groups),
        "note": (
            "detectability_auc=max(raw_auroc, 1-raw_auroc): 0.5 means chance-level "
            "separation (good fidelity); 1.0 means trivially separable (poor fidelity)."
        ),
    }
