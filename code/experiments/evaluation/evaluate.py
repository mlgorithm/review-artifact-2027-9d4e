#!/usr/bin/env python3
"""Run synthetic trajectory evaluation from CSV inputs."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    from .metrics import (
        binary_classification_metrics,
        binary_value,
        auc_score,
        correctness_rate,
        distribution,
        entropy,
        group_trajectories,
        histogram_total_variation,
        jensen_shannon,
        mean,
        ngram_distribution,
        per_group_absolute_error,
        predict_correctness,
        rate_by_group,
        safe_float,
        sequence_lengths,
        sequence_position_curve,
        skill_probability_model,
        streak_lengths,
        tail_labels,
        tail_prevalence,
        tail_reference,
        total_variation,
        trajectory_tokens,
        trajectory_value_tokens,
        transition_distribution,
        wasserstein_1d,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from metrics import (
        binary_classification_metrics,
        binary_value,
        auc_score,
        correctness_rate,
        distribution,
        entropy,
        group_trajectories,
        histogram_total_variation,
        jensen_shannon,
        mean,
        ngram_distribution,
        per_group_absolute_error,
        predict_correctness,
        rate_by_group,
        safe_float,
        sequence_lengths,
        sequence_position_curve,
        skill_probability_model,
        streak_lengths,
        tail_labels,
        tail_prevalence,
        tail_reference,
        total_variation,
        trajectory_tokens,
        trajectory_value_tokens,
        transition_distribution,
        wasserstein_1d,
    )


try:  # pragma: no cover - supports both package and direct-script execution.
    from .metrics import (
        autocorrelation,
        cross_correlation,
        membership_inference_scores,
        normalized_mutual_information,
    )
except ImportError:  # pragma: no cover
    from metrics import (
        autocorrelation,
        cross_correlation,
        membership_inference_scores,
        normalized_mutual_information,
    )

from itertools import combinations


DEFAULT_COLUMNS = {
    "learner": "learner_id",
    "order": "order",
    "skill": "skill",
    "correct": "correct",
    "hint": "hint",
    "attempts": "attempts",
    # Optional signals. `response_time` drives the Table-3 rapid-guessing tail;
    # `subgroup` names a static learner attribute (e.g. an OULAD demographic) for
    # subgroup fidelity. Absent columns are detected and simply skipped.
    "response_time": "response_time",
    "subgroup": "",
    "dropout": "",
}


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_config(path: Path | None) -> Mapping[str, object]:
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to load YAML config files.")
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return loaded


def column_config(config: Mapping[str, object]) -> Dict[str, str]:
    columns = dict(DEFAULT_COLUMNS)
    configured = config.get("columns", {})
    if isinstance(configured, Mapping):
        for key, value in configured.items():
            columns[str(key)] = str(value)
    return columns


def evaluate(
    real_train: Sequence[Dict[str, str]],
    real_test: Sequence[Dict[str, str]],
    synthetic: Sequence[Dict[str, str]],
    columns: Mapping[str, str],
    seed: int = 0,
    *,
    tail_real_train: Sequence[Dict[str, str]] | None = None,
    tail_real_test: Sequence[Dict[str, str]] | None = None,
    tail_synthetic: Sequence[Dict[str, str]] | None = None,
) -> Dict[str, object]:
    learner_col = columns["learner"]
    order_col = columns["order"]
    skill_col = columns["skill"]
    correct_col = columns["correct"]
    hint_col = _resolve_optional_column(real_train, columns.get("hint"))
    attempts_col = _resolve_optional_column(real_train, columns.get("attempts"))
    # Optional signals: kept only when the named column is actually present in the
    # data, so tail/subgroup metrics that need them are emitted for datasets that
    # have them and cleanly skipped for datasets that do not.
    response_time_col = _resolve_optional_column(
        real_train, columns.get("response_time")
    ) or _detect_response_time_column(real_train)
    subgroup_col = _resolve_optional_column(real_train, columns.get("subgroup"))
    dropout_col = _resolve_optional_column(real_train, columns.get("dropout"))

    train_traj = group_trajectories(real_train, learner_col, order_col)
    test_traj = group_trajectories(real_test, learner_col, order_col)
    synth_traj = group_trajectories(synthetic, learner_col, order_col)

    # Rare-group fidelity is inexpensive and support-sensitive, so callers may
    # supply the complete frames even when other memory-heavy diagnostic blocks
    # use deterministic samples.  This also lets RQ2 evaluate the exact same
    # full-real-train group definitions used to select targeted learners.
    complete_tail_inputs_provided = all(
        value is not None
        for value in (tail_real_train, tail_real_test, tail_synthetic)
    )
    tail_reference_scope = (
        "complete_real_train"
        if complete_tail_inputs_provided
        else "provided_real_train_input"
    )
    tail_real_train = real_train if tail_real_train is None else tail_real_train
    tail_real_test = real_test if tail_real_test is None else tail_real_test
    tail_synthetic = synthetic if tail_synthetic is None else tail_synthetic
    tail_hint_col = _resolve_optional_column(tail_real_train, columns.get("hint"))
    tail_attempts_col = _resolve_optional_column(
        tail_real_train, columns.get("attempts")
    )
    tail_response_time_col = _resolve_optional_column(
        tail_real_train, columns.get("response_time")
    ) or _detect_response_time_column(tail_real_train)
    tail_dropout_col = _resolve_optional_column(
        tail_real_train, columns.get("dropout")
    )
    tail_gap_col = _detect_gap_column(tail_real_train)
    tail_train_traj = group_trajectories(tail_real_train, learner_col, order_col)
    tail_test_traj = group_trajectories(tail_real_test, learner_col, order_col)
    tail_synth_traj = group_trajectories(tail_synthetic, learner_col, order_col)

    # Inter-event-time signal: the binned time-gap column, whose name varies by
    # dataset (timestamp_gap_bin, gap_bin, ...). Detect it heuristically; absent
    # (e.g. ASSISTments/OULAD) the inter-event-time distribution is simply skipped.
    gap_col = _detect_gap_column(real_train)
    value_columns = [
        column
        for column in real_train[0]
        if column not in {learner_col, order_col} and synthetic and column in synthetic[0]
    ] if real_train else []
    tail_cutoffs = tail_reference(
        tail_train_traj,
        skill_col,
        correct_col,
        tail_hint_col,
        tail_attempts_col,
        tail_response_time_col,
        tail_gap_col,
        tail_dropout_col,
    )
    tail_train_labels = tail_labels(
        tail_train_traj,
        skill_col,
        correct_col,
        tail_hint_col,
        tail_cutoffs,
        tail_attempts_col,
        tail_response_time_col,
        tail_gap_col,
        tail_dropout_col,
    )
    tail_test_labels = tail_labels(
        tail_test_traj,
        skill_col,
        correct_col,
        tail_hint_col,
        tail_cutoffs,
        tail_attempts_col,
        tail_response_time_col,
        tail_gap_col,
        tail_dropout_col,
    )
    tail_synth_labels = tail_labels(
        tail_synth_traj,
        skill_col,
        correct_col,
        tail_hint_col,
        tail_cutoffs,
        tail_attempts_col,
        tail_response_time_col,
        tail_gap_col,
        tail_dropout_col,
    )

    # Fidelity is scored against real_train (the reference distribution the
    # generator was fit on) AND, additionally, against the held-out real_test
    # split. Comparing to train measures how well the synthetic data reproduces
    # the training distribution; comparing to test measures generalization. A
    # generator that memorizes train scores near-perfectly on the `*_fidelity`
    # blocks but noticeably worse on the `*_fidelity_vs_test` blocks -- the gap
    # between the two is the memorization signal (read alongside
    # privacy_memorization).
    return {
        "global_fidelity": global_fidelity(
            real_train, synthetic, train_traj, synth_traj, skill_col, correct_col,
            hint_col, attempts_col, value_columns,
        ),
        "global_fidelity_vs_test": global_fidelity(
            real_test, synthetic, test_traj, synth_traj, skill_col, correct_col,
            hint_col, attempts_col, value_columns,
        ),
        "temporal_fidelity": temporal_fidelity(
            train_traj, synth_traj, skill_col, correct_col, hint_col, gap_col
        ),
        "temporal_fidelity_vs_test": temporal_fidelity(
            test_traj, synth_traj, skill_col, correct_col, hint_col, gap_col
        ),
        "tail_fidelity": tail_fidelity(
            tail_train_traj, tail_synth_traj, skill_col, correct_col, tail_hint_col,
            tail_attempts_col, tail_response_time_col, tail_gap_col, tail_dropout_col,
            reference=tail_cutoffs, seed=seed, reference_scope=tail_reference_scope,
            real_labels=tail_train_labels, synth_labels=tail_synth_labels,
        ),
        "tail_fidelity_vs_test": tail_fidelity(
            tail_test_traj, tail_synth_traj, skill_col, correct_col, tail_hint_col,
            tail_attempts_col, tail_response_time_col, tail_gap_col, tail_dropout_col,
            reference=tail_cutoffs, seed=seed + 1, reference_scope=tail_reference_scope,
            real_labels=tail_test_labels, synth_labels=tail_synth_labels,
        ),
        "cross_signal_dependence": cross_signal_dependence(
            real_train, synthetic, value_columns, seed=seed
        ),
        "subgroup_fidelity": subgroup_fidelity(
            train_traj, synth_traj, correct_col, hint_col, subgroup_col=subgroup_col
        ),
        "downstream_utility": downstream_utility(real_train, real_test, synthetic, skill_col, correct_col),
        "privacy_memorization": privacy_memorization(
            train_traj, synth_traj, skill_col, correct_col, hint_col, test_traj=test_traj,
            attempts_col=attempts_col, response_time_col=response_time_col, gap_col=gap_col,
            dropout_col=dropout_col,
            privacy_columns=value_columns,
            seed=seed,
        ),
        "diversity_coverage": diversity_coverage(
            train_traj, synth_traj, skill_col, correct_col, hint_col, value_columns
        ),
        "dataset_sizes": {
            "real_train_rows": len(real_train),
            "real_test_rows": len(real_test),
            "synthetic_rows": len(synthetic),
            "real_train_learners": len(train_traj),
            "real_test_learners": len(test_traj),
            "synthetic_learners": len(synth_traj),
        },
    }


def global_fidelity(
    real,
    synthetic,
    real_traj,
    synth_traj,
    skill_col,
    correct_col,
    hint_col,
    attempts_col,
    value_columns,
):
    real_skill_dist = distribution(row.get(skill_col, "") for row in real)
    synth_skill_dist = distribution(row.get(skill_col, "") for row in synthetic)
    # NOTE: `attempts_col` is a *binned* label column (e.g. attempt_bin with
    # values like "0", "1", "3-4", "5+"), not a raw count. `_safe_number`
    # coerces each label to a representative number (range labels -> midpoint,
    # "5+" -> 5), so the "attempt_count_*" metrics below are really distances
    # over bin midpoints, not raw attempt counts. Unrecognized tokens coerce to
    # 0.0 (see safe_float), so a schema/typo error degrades silently rather than
    # raising -- keep the source column clean.
    real_attempts = [_safe_number(row.get(attempts_col)) for row in real] if attempts_col else []
    synth_attempts = [_safe_number(row.get(attempts_col)) for row in synthetic] if attempts_col else []
    real_lengths = sequence_lengths(real_traj)
    synth_lengths = sequence_lengths(synth_traj)
    result = {
        "correctness_rate_error": abs(correctness_rate(real, correct_col) - correctness_rate(synthetic, correct_col)),
        "correctness_by_skill_mae": per_group_absolute_error(
            rate_by_group(real, skill_col, correct_col),
            rate_by_group(synthetic, skill_col, correct_col),
        ),
        "skill_frequency_js": jensen_shannon(real_skill_dist, synth_skill_dist),
        "skill_frequency_total_variation": total_variation(real_skill_dist, synth_skill_dist),
        "sequence_length_wasserstein": wasserstein_1d(real_lengths, synth_lengths),
        "sequence_length_histogram_tv": histogram_total_variation(real_lengths, synth_lengths),
        "all_value_marginals": marginal_fidelity(real, synthetic, value_columns),
    }
    if hint_col:
        result["hint_rate_error"] = abs(
            correctness_rate(real, hint_col) - correctness_rate(synthetic, hint_col)
        )
    if attempts_col:
        result["attempt_count_wasserstein"] = wasserstein_1d(real_attempts, synth_attempts)
        result["attempt_count_histogram_tv"] = histogram_total_variation(real_attempts, synth_attempts)
    return result


def marginal_fidelity(real, synthetic, columns):
    """Categorical marginal distances for every shared generated value column."""
    per_column = {}
    js_values = []
    tv_values = []
    for column in columns:
        real_distribution = distribution(record.get(column, "") for record in real)
        synthetic_distribution = distribution(record.get(column, "") for record in synthetic)
        js = jensen_shannon(real_distribution, synthetic_distribution)
        tv = total_variation(real_distribution, synthetic_distribution)
        per_column[column] = {
            "jensen_shannon_divergence": js,
            "total_variation": tv,
            "real_unique_count": len(real_distribution),
            "synthetic_unique_count": len(synthetic_distribution),
        }
        js_values.append(js)
        tv_values.append(tv)
    return {
        "by_column": per_column,
        "jensen_shannon_divergence_mean": mean(js_values),
        "total_variation_mean": mean(tv_values),
    }


def temporal_fidelity(real_traj, synth_traj, skill_col, correct_col, hint_col, gap_col=None):
    real_correct_transitions = transition_distribution(real_traj, correct_col)
    synth_correct_transitions = transition_distribution(synth_traj, correct_col)
    real_curve = sequence_position_curve(real_traj, correct_col)
    synth_curve = sequence_position_curve(synth_traj, correct_col)
    real_autocorr = autocorrelation(real_traj, correct_col)
    synth_autocorr = autocorrelation(synth_traj, correct_col)
    result = {
        "correctness_transition_js": jensen_shannon(real_correct_transitions, synth_correct_transitions),
        "correctness_transition_tv": total_variation(real_correct_transitions, synth_correct_transitions),
        "correct_streak_wasserstein": wasserstein_1d(
            streak_lengths(real_traj, correct_col, 1),
            streak_lengths(synth_traj, correct_col, 1),
        ),
        "failure_streak_wasserstein": wasserstein_1d(
            streak_lengths(real_traj, correct_col, 0),
            streak_lengths(synth_traj, correct_col, 0),
        ),
        "skill_trigram_js": jensen_shannon(
            ngram_distribution(real_traj, skill_col, n=3),
            ngram_distribution(synth_traj, skill_col, n=3),
        ),
        "correctness_curve_mae": per_group_absolute_error(real_curve, synth_curve),
        # Autocorrelation of the correctness signal (lags 1-3): does synthetic
        # data reproduce within-learner persistence of right/wrong answers?
        "correctness_autocorrelation_real": real_autocorr,
        "correctness_autocorrelation_synthetic": synth_autocorr,
        "correctness_autocorrelation_mae": per_group_absolute_error(real_autocorr, synth_autocorr),
    }
    # Single error metric for the cross-correlation coupling (lags 0,1), so it
    # aggregates into the multi-seed summary the way correctness_autocorrelation_mae
    # does instead of leaving the reader to diff the raw real/synth values.
    if hint_col:
        real_hint_transitions = transition_distribution(real_traj, hint_col)
        synth_hint_transitions = transition_distribution(synth_traj, hint_col)
        result.update(
            {
                "hint_transition_js": jensen_shannon(real_hint_transitions, synth_hint_transitions),
                "hint_transition_tv": total_variation(real_hint_transitions, synth_hint_transitions),
                "correct_hint_cross_correlation_real": cross_correlation(real_traj, correct_col, hint_col, 0),
                "correct_hint_cross_correlation_synthetic": cross_correlation(synth_traj, correct_col, hint_col, 0),
                "correct_hint_cross_correlation_lag1_real": cross_correlation(real_traj, correct_col, hint_col, 1),
                "correct_hint_cross_correlation_lag1_synthetic": cross_correlation(synth_traj, correct_col, hint_col, 1),
            }
        )
        result["correct_hint_cross_correlation_mae"] = mean(
            [
                abs(result["correct_hint_cross_correlation_real"] - result["correct_hint_cross_correlation_synthetic"]),
                abs(
                    result["correct_hint_cross_correlation_lag1_real"]
                    - result["correct_hint_cross_correlation_lag1_synthetic"]
                ),
            ]
        )
    if gap_col:
        real_gap = distribution(str(record.get(gap_col, "")) for traj in real_traj.values() for record in traj)
        synth_gap = distribution(str(record.get(gap_col, "")) for traj in synth_traj.values() for record in traj)
        result["inter_event_time_js"] = jensen_shannon(real_gap, synth_gap)
        result["inter_event_time_total_variation"] = total_variation(real_gap, synth_gap)
    return result


_TAIL_SHAPE_MIN_LEARNERS = 5
_TAIL_BOOTSTRAP_SAMPLES = 200
_TAIL_BOOTSTRAP_MAX_LEARNERS = 300


def _percentile_interval(values, alpha=0.05):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)

    def percentile(q):
        position = q * (len(ordered) - 1)
        low = int(position)
        high = min(len(ordered) - 1, low + 1)
        weight = position - low
        return ordered[low] * (1 - weight) + ordered[high] * weight

    return {
        "ci_low": percentile(alpha / 2),
        "ci_high": percentile(1 - alpha / 2),
        "n_boot": len(ordered),
    }


def _tail_prevalence_error_ci(real_labels, synth_labels, names, seed, n_boot=_TAIL_BOOTSTRAP_SAMPLES):
    real_n = len(real_labels)
    synth_n = len(synth_labels)
    if not real_n or not synth_n:
        return {}
    rng = np.random.RandomState(seed)
    intervals = {}
    for name in names:
        real_successes = sum(bool(labels.get(name, False)) for labels in real_labels.values())
        synth_successes = sum(bool(labels.get(name, False)) for labels in synth_labels.values())
        # For a binary group label, resampling learners with replacement gives a
        # binomial count with the empirical learner prevalence.  Sampling that
        # count directly is exactly the same marginal learner bootstrap and
        # avoids materializing O(n_boot * learners) dictionaries on full data.
        real_rates = rng.binomial(real_n, real_successes / real_n, size=n_boot) / real_n
        synth_rates = rng.binomial(synth_n, synth_successes / synth_n, size=n_boot) / synth_n
        intervals[name] = _percentile_interval(
            np.abs(real_rates - synth_rates).tolist()
        )
    return intervals


_TAIL_REFERENCE_REPORT_KEYS = (
    "quantile",
    "maximum_percentile_tail_prevalence",
    "rapid_guessing_max_primary_signal_rate",
    "unavailable_percentile_tail_groups",
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


def _tail_reference_report(reference: Mapping[str, object]) -> dict[str, object]:
    """Serializable audit record without the potentially huge transition map."""
    return {
        key: reference.get(key)
        for key in _TAIL_REFERENCE_REPORT_KEYS
        if key in reference
    } | {
        "rare_path_transition_model": "global_add_one_smoothed_log2_probability",
    }


def tail_fidelity(
    real_traj,
    synth_traj,
    skill_col,
    correct_col,
    hint_col,
    attempts_col=None,
    response_time_col=None,
    gap_col=None,
    dropout_col=None,
    reference=None,
    seed=0,
    reference_scope="provided_real_train_input",
    real_labels=None,
    synth_labels=None,
):
    # Fix tail-group thresholds on the real trajectories and reuse them when
    # labeling synthetic; recomputing per set would re-normalize each side to
    # its own distribution and mask real differences in length/rare-path tails.
    if reference is None:
        reference = tail_reference(
            real_traj, skill_col, correct_col, hint_col, attempts_col,
            response_time_col, gap_col, dropout_col
        )
    if real_labels is None:
        real_labels = tail_labels(
            real_traj, skill_col, correct_col, hint_col, reference, attempts_col,
            response_time_col, gap_col, dropout_col
        )
    if synth_labels is None:
        synth_labels = tail_labels(
            synth_traj, skill_col, correct_col, hint_col, reference, attempts_col,
            response_time_col, gap_col, dropout_col
        )
    real_prev = tail_prevalence(real_labels)
    synth_prev = tail_prevalence(synth_labels)
    prevalence_error = {
        name: abs(real_prev.get(name, 0.0) - synth_prev.get(name, 0.0))
        for name in sorted(set(real_prev) | set(synth_prev))
    }
    group_names = sorted(set(real_prev) | set(synth_prev))
    return {
        "tail_reference_split": "real_train",
        "tail_reference_scope": reference_scope,
        "tail_reference": _tail_reference_report(reference),
        "evaluated_real_learners": len(real_labels),
        "evaluated_synthetic_learners": len(synth_labels),
        "implemented_tail_groups": group_names,
        "unavailable_percentile_tail_groups": reference.get(
            "unavailable_percentile_tail_groups", {}
        ),
        "maximum_percentile_tail_prevalence": reference.get(
            "maximum_percentile_tail_prevalence"
        ),
        "unsupported_proposal_tail_groups": [] if dropout_col else ["dropout"],
        "operational_note": (
            "Late failure and recovery are correctness-trajectory definitions. "
            "Dropout is emitted only when an explicit completion/non-participation outcome is supplied."
        ),
        "tail_prevalence_real": real_prev,
        "tail_prevalence_synthetic": synth_prev,
        "tail_prevalence_error": prevalence_error,
        "tail_prevalence_mae": mean(list(prevalence_error.values())),
        "tail_prevalence_error_ci": _tail_prevalence_error_ci(
            real_labels, synth_labels, group_names, seed
        ),
        # Tail SHAPE (not just frequency): does the within-tail correctness curve
        # match? Covers the "tail shape loss" failure mode -- rare cases produced
        # but with unrealistic pre-event trajectories.
        "tail_shape": tail_shape_fidelity(
            real_traj, synth_traj, real_labels, synth_labels, correct_col, seed
        ),
    }


def tail_shape_fidelity(real_traj, synth_traj, real_labels, synth_labels, correct_col, seed=0):
    """Per-tail-group correctness-curve distance between real and synthetic tails.

    For each tail group we take the learners in that group and compare the
    position-normalized correctness curve (shape of the trajectory) of the real
    tail vs. the synthetic tail. Groups with too few learners on either side are
    marked exploratory and left out of the summary mean.
    """
    group_names = sorted({name for labels in real_labels.values() for name in labels})
    per_group: Dict[str, object] = {}
    reliable_errors: List[float] = []

    def position_matrix(trajectories):
        rows = []
        for trajectory in trajectories:
            curve = sequence_position_curve({"learner": trajectory}, correct_col)
            rows.append([curve[str(index)] for index in range(10)])
        return np.asarray(rows, dtype=float)

    for group_index, name in enumerate(group_names):
        real_ids = [learner for learner, labels in real_labels.items() if labels.get(name)]
        synth_ids = [learner for learner, labels in synth_labels.items() if labels.get(name)]
        real_group = {learner: real_traj[learner] for learner in real_ids if learner in real_traj}
        synth_group = {learner: synth_traj[learner] for learner in synth_ids if learner in synth_traj}
        exploratory = (
            len(real_group) < _TAIL_SHAPE_MIN_LEARNERS or len(synth_group) < _TAIL_SHAPE_MIN_LEARNERS
        )
        curve_mae = per_group_absolute_error(
            sequence_position_curve(real_group, correct_col),
            sequence_position_curve(synth_group, correct_col),
        )
        per_group[name] = {
            "correctness_curve_mae": curve_mae,
            "real_learners": len(real_group),
            "synthetic_learners": len(synth_group),
            "exploratory": exploratory,
        }
        if not exploratory:
            rng = random.Random(seed + group_index + 1)
            real_values = list(real_group.values())
            synth_values = list(synth_group.values())
            if len(real_values) > _TAIL_BOOTSTRAP_MAX_LEARNERS:
                real_values = rng.sample(real_values, _TAIL_BOOTSTRAP_MAX_LEARNERS)
            if len(synth_values) > _TAIL_BOOTSTRAP_MAX_LEARNERS:
                synth_values = rng.sample(synth_values, _TAIL_BOOTSTRAP_MAX_LEARNERS)
            real_matrix = position_matrix(real_values)
            synth_matrix = position_matrix(synth_values)
            bootstrap_errors = []
            for _ in range(_TAIL_BOOTSTRAP_SAMPLES):
                real_indices = [
                    rng.randrange(len(real_values)) for _ in range(len(real_values))
                ]
                synth_indices = [
                    rng.randrange(len(synth_values)) for _ in range(len(synth_values))
                ]
                bootstrap_errors.append(
                    float(
                        np.mean(
                            np.abs(
                                real_matrix[real_indices].mean(axis=0)
                                - synth_matrix[synth_indices].mean(axis=0)
                            )
                        )
                    )
                )
            per_group[name]["correctness_curve_mae_ci"] = _percentile_interval(
                bootstrap_errors
            )
        if not exploratory:
            reliable_errors.append(curve_mae)
    return {
        "by_tail_group": per_group,
        "correctness_curve_mae_mean": mean(reliable_errors) if reliable_errors else None,
    }


def _detect_gap_column(rows: Sequence[Dict[str, str]]) -> str | None:
    if not rows:
        return None
    for key in rows[0]:
        if "gap" in key.lower():
            return key
    return None


def _detect_response_time_column(rows: Sequence[Dict[str, str]]) -> str | None:
    # Response-time-on-task column names vary (response_time_bin, duration_bin,
    # elapsed_time_bin); a gap column also contains "time" so exclude it.
    if not rows:
        return None
    for key in rows[0]:
        name = key.lower()
        if "gap" in name:
            continue
        if "response" in name or "duration" in name or "elapsed" in name:
            return key
    return None


def _resolve_optional_column(rows: Sequence[Dict[str, str]], column: object) -> str | None:
    """Return ``column`` only if it is a non-empty name present in the data.

    Used for optional signals (response time, subgroup): a configured-but-absent
    column, or an empty placeholder, resolves to None so the dependent metric is
    skipped instead of silently reading missing values.
    """
    if not column or not rows:
        return None
    name = str(column)
    return name if name in rows[0] else None


def cross_signal_dependence(real, synthetic, columns, seed=0, max_rows=100_000):
    """Normalized mutual information between each pair of interaction signals.

    Checks whether the synthetic data preserves how signals co-vary (e.g.
    correctness x hint x attempts x skill), which marginal fidelity cannot see.
    """
    present = [
        column
        for column in columns
        if real and synthetic and column in real[0] and column in synthetic[0]
    ]
    sample_size = min(len(real), len(synthetic), max_rows)
    real_rng = random.Random(seed)
    synthetic_rng = random.Random(seed + 1)
    sampled_real = real_rng.sample(list(real), sample_size) if len(real) > sample_size else list(real)
    sampled_synthetic = (
        synthetic_rng.sample(list(synthetic), sample_size)
        if len(synthetic) > sample_size
        else list(synthetic)
    )
    pairwise = {}
    errors = []
    for left, right in combinations(present, 2):
        real_nmi = normalized_mutual_information(sampled_real, left, right)
        synth_nmi = normalized_mutual_information(sampled_synthetic, left, right)
        pairwise[f"{left}__{right}"] = {
            "real": real_nmi,
            "synthetic": synth_nmi,
            "abs_error": abs(real_nmi - synth_nmi),
        }
        errors.append(abs(real_nmi - synth_nmi))
    return {
        "columns": present,
        "real_sample_rows": len(sampled_real),
        "synthetic_sample_rows": len(sampled_synthetic),
        "pairwise_nmi": pairwise,
        "nmi_mae": mean(errors),
    }


_SUBGROUP_PREFIX_LENGTH = 2


def _performance_score(trajectory, correct_col):
    prefix = trajectory[:_SUBGROUP_PREFIX_LENGTH]
    return mean([binary_value(record, correct_col) for record in prefix])


def _performance_reference(trajectories, correct_col):
    values = sorted(_performance_score(trajectory, correct_col) for trajectory in trajectories.values())
    if not values:
        return {"low_cutoff": 0.0, "high_cutoff": 0.0, "prefix_length": _SUBGROUP_PREFIX_LENGTH}

    def quantile(q):
        position = q * (len(values) - 1)
        low = int(position)
        high = min(len(values) - 1, low + 1)
        weight = position - low
        return values[low] * (1 - weight) + values[high] * weight

    return {
        "low_cutoff": quantile(1 / 3),
        "high_cutoff": quantile(2 / 3),
        "prefix_length": _SUBGROUP_PREFIX_LENGTH,
    }


def _performance_subgroups(trajectories, correct_col, reference):
    """Apply real-train-fitted fixed-prefix performance cutoffs."""
    groups = {"low": [], "mid": [], "high": []}
    low_cutoff = float(reference["low_cutoff"])
    high_cutoff = float(reference["high_cutoff"])
    for learner, trajectory in trajectories.items():
        score = _performance_score(trajectory, correct_col)
        # Strict-low / inclusive-high tie handling keeps a point mass at perfect
        # early correctness in the high group instead of labelling everyone low
        # when both quantile cutoffs equal 1.0.
        label = "low" if score < low_cutoff else ("high" if score >= high_cutoff else "mid")
        groups[label].append(learner)
    return groups


def _attribute_subgroups(trajectories, subgroup_col):
    """Partition learners by a static learner attribute read from the first record."""
    groups: Dict[str, List[str]] = defaultdict(list)
    for learner, trajectory in trajectories.items():
        value = str(trajectory[0].get(subgroup_col, "")) if trajectory else ""
        groups[value].append(learner)
    return dict(groups)


def subgroup_fidelity(real_traj, synth_traj, correct_col, hint_col, subgroup_col=None):
    """Subgroup fidelity: does synthetic preserve per-subgroup behaviour?

    When ``subgroup_col`` names a static learner attribute present in the data
    (e.g. an OULAD demographic), learners are stratified by it -- this is the
    equity-relevant comparison. Otherwise we use low/mid/high fixed-prefix
    performance groups whose cut-points are fitted once on real train and reused
    on synthetic learners. The first two interactions define the group; behavior
    rates are evaluated only after that prefix so subgroup assignment and the
    measured behavior do not reuse the same events.
    """
    if subgroup_col:
        partition = "by_attribute"
        performance_reference = None
        real_groups = _attribute_subgroups(real_traj, subgroup_col)
        synth_groups = _attribute_subgroups(synth_traj, subgroup_col)
        group_labels = sorted(set(real_groups) | set(synth_groups))
    else:
        partition = "by_fixed_prefix_performance"
        performance_reference = _performance_reference(real_traj, correct_col)
        real_groups = _performance_subgroups(real_traj, correct_col, performance_reference)
        synth_groups = _performance_subgroups(synth_traj, correct_col, performance_reference)
        group_labels = ["low", "mid", "high"]

    def stratum_stats(trajectories, groups):
        total = sum(len(ids) for ids in groups.values()) or 1
        stats = {}
        for label in group_labels:
            ids = groups.get(label, [])
            rows = [
                record
                for learner in ids
                for record in trajectories[learner][_SUBGROUP_PREFIX_LENGTH:]
            ]
            correct_values = [binary_value(record, correct_col) for record in rows]
            hint_values = [binary_value(record, hint_col) for record in rows] if hint_col else []
            lengths = [float(len(trajectories[learner])) for learner in ids]
            entry = {
                "learner_share": len(ids) / total,
                "future_correctness_rate": mean(correct_values) if correct_values else None,
                "mean_length": mean(lengths) if lengths else None,
            }
            if hint_col:
                entry["future_hint_rate"] = mean(hint_values) if hint_values else None
            stats[label] = entry
        return stats

    real_stats = stratum_stats(real_traj, real_groups)
    synth_stats = stratum_stats(synth_traj, synth_groups)
    per_group = {}
    learner_share_errors = []
    correctness_errors = []
    hint_errors = []
    length_errors = []
    for label in group_labels:
        real_entry = real_stats.get(label, {})
        synth_entry = synth_stats.get(label, {})
        abs_error = {
            metric: abs(real_entry[metric] - synth_entry[metric])
            for metric in (
                "learner_share",
                "future_correctness_rate",
                "future_hint_rate",
                "mean_length",
            )
            if isinstance(real_entry.get(metric), (int, float))
            and isinstance(synth_entry.get(metric), (int, float))
        }
        per_group[label] = {"real": real_entry, "synthetic": synth_entry, "abs_error": abs_error}
        if "learner_share" in abs_error:
            learner_share_errors.append(abs_error["learner_share"])
        if "future_correctness_rate" in abs_error:
            correctness_errors.append(abs_error["future_correctness_rate"])
        if "future_hint_rate" in abs_error:
            hint_errors.append(abs_error["future_hint_rate"])
        if "mean_length" in abs_error:
            length_errors.append(abs_error["mean_length"])
    result = {
        "partition": partition,
        "subgroup_column": subgroup_col,
        "performance_reference": performance_reference,
        "behavior_window": f"events_after_first_{_SUBGROUP_PREFIX_LENGTH}",
        "by_subgroup": per_group,
        "learner_share_mae_across_subgroups": mean(learner_share_errors),
        "future_correctness_rate_mae_across_subgroups": mean(correctness_errors),
        "length_mae_across_subgroups": mean(length_errors),
    }
    if hint_col:
        result["future_hint_rate_mae_across_subgroups"] = mean(hint_errors)
    return result


def downstream_utility(real_train, real_test, synthetic, skill_col, correct_col):
    """Lightweight TRTR/TSTR sanity check using a per-skill mean-probability model.

    This is a fast, dependency-free smoke test of train-synthetic-test-real
    utility. The *primary* downstream evaluation -- real ML models (logistic
    regression, gradient boosting), the real+synthetic and tail-targeted
    augmentation arms, and per-tail-group learnability with delta_tail -- lives in
    the ``downstream_utility_models`` block produced by general_evaluation.
    """
    real_global, real_skill_model = skill_probability_model(real_train, skill_col, correct_col)
    synth_global, synth_skill_model = skill_probability_model(synthetic, skill_col, correct_col)
    labels = [binary_value(row, correct_col) for row in real_test]
    real_scores = predict_correctness(real_test, skill_col, real_global, real_skill_model)
    synth_scores = predict_correctness(real_test, skill_col, synth_global, synth_skill_model)
    return {
        "task": "next_response_correctness",
        "model": "skill_mean_probability_baseline",
        "role": "lightweight_sanity_baseline",
        "note": "Primary downstream metrics are in downstream_utility_models (ML models + augmentation + tail learnability).",
        "train_on_real_test_on_real": binary_classification_metrics(labels, real_scores),
        "train_on_synthetic_test_on_real": binary_classification_metrics(labels, synth_scores),
    }


def _nearest_distances(query_sequences, reference_sequences):
    """Vectorized exact nearest-neighbour Hamming-like trajectory distance."""
    if not reference_sequences:
        return []
    if not query_sequences:
        return []

    # Map arbitrary hashable event tokens to compact integers and pad with -1.
    # With the same sentinel on both sides, the vectorized mismatch count equals
    # aligned token mismatches plus the absolute length difference—the exact
    # definition of hamming_like_distance. Batching bounds peak memory.
    token_ids: dict[object, int] = {}

    def token_id(value: object) -> int:
        try:
            key = value if hash(value) is not None else value
        except TypeError:
            key = repr(value)
        if key not in token_ids:
            token_ids[key] = len(token_ids)
        return token_ids[key]

    max_length = max(
        1,
        max((len(sequence) for sequence in reference_sequences), default=0),
        max((len(sequence) for sequence in query_sequences), default=0),
    )
    reference_lengths = np.asarray([len(sequence) for sequence in reference_sequences], dtype=np.int32)
    reference_matrix = np.full((len(reference_sequences), max_length), -1, dtype=np.int32)
    for row, sequence in enumerate(reference_sequences):
        reference_matrix[row, : len(sequence)] = [token_id(value) for value in sequence]

    # Keep the temporary (batch, references, positions) boolean cube near 64 MB.
    batch_size = max(1, min(128, 64_000_000 // max(1, len(reference_sequences) * max_length)))
    distances: list[float] = []
    for start in range(0, len(query_sequences), batch_size):
        batch = query_sequences[start : start + batch_size]
        query_lengths = np.asarray([len(sequence) for sequence in batch], dtype=np.int32)
        query_matrix = np.full((len(batch), max_length), -1, dtype=np.int32)
        for row, sequence in enumerate(batch):
            query_matrix[row, : len(sequence)] = [token_id(value) for value in sequence]
        mismatch = np.count_nonzero(
            query_matrix[:, None, :] != reference_matrix[None, :, :], axis=2
        )
        denominator = np.maximum(query_lengths[:, None], reference_lengths[None, :])
        pairwise = np.divide(
            mismatch,
            denominator,
            out=np.zeros_like(mismatch, dtype=float),
            where=denominator > 0,
        )
        distances.extend(np.min(pairwise, axis=1).astype(float).tolist())
    return distances


def _bootstrap_mean_ci(values, seed, n_boot=200):
    if not values:
        return None
    rng = random.Random(seed)
    samples = [
        mean([values[rng.randrange(len(values))] for _ in values])
        for _ in range(n_boot)
    ]
    return _percentile_interval(samples)


def _membership_risk(member_sequences, nonmember_sequences, synthetic_sequences, seed):
    result = membership_inference_scores(
        member_sequences, nonmember_sequences, synthetic_sequences
    )
    if result is None:
        return {
            "raw_auc": None,
            "attack_auc": None,
            "attack_advantage": None,
            "attack_advantage_ci": None,
        }
    labels, scores = result
    raw_auc = auc_score(labels, scores)
    # The attack score has a declared orientation: closer to synthetic means
    # "member". An AUC below 0.5 is therefore no evidence of membership leakage;
    # do not flip it after seeing labels and call the inversion an attack.
    attack_auc = max(0.5, raw_auc)
    member_scores = scores[: len(member_sequences)]
    nonmember_scores = scores[len(member_sequences) :]
    rng = random.Random(seed)
    advantages = []
    for _ in range(200):
        sampled_member = [member_scores[rng.randrange(len(member_scores))] for _ in member_scores]
        sampled_nonmember = [
            nonmember_scores[rng.randrange(len(nonmember_scores))] for _ in nonmember_scores
        ]
        bootstrap_auc = auc_score(
            [1] * len(sampled_member) + [0] * len(sampled_nonmember),
            sampled_member + sampled_nonmember,
        )
        advantages.append(max(0.0, 2.0 * (bootstrap_auc - 0.5)))
    return {
        "raw_auc": raw_auc,
        "attack_auc": attack_auc,
        "attack_advantage": max(0.0, 2.0 * (raw_auc - 0.5)),
        "attack_advantage_ci": _percentile_interval(advantages),
    }


def privacy_memorization(
    real_traj,
    synth_traj,
    skill_col,
    correct_col,
    hint_col,
    max_nearest: int = 1000,
    seed: int = 0,
    test_traj=None,
    near_duplicate_threshold: float = 0.1,
    attempts_col=None,
    response_time_col=None,
    gap_col=None,
    dropout_col=None,
    min_tail_learners: int = 10,
    tail_reference_sample: int = 400,
    privacy_columns: Sequence[str] | None = None,
):
    rng = random.Random(seed)
    selected_columns = list(dict.fromkeys(column for column in (privacy_columns or []) if column))
    if not selected_columns:
        selected_columns = [column for column in (skill_col, correct_col, hint_col) if column]
    real_seq_by_learner = {
        learner: trajectory_value_tokens(traj, selected_columns) for learner, traj in real_traj.items()
    }
    real_sequences = list(real_seq_by_learner.values())
    synth_sequences = [trajectory_value_tokens(traj, selected_columns) for traj in synth_traj.values()]
    real_set = set(real_sequences)
    exact_duplicates = sum(1 for seq in synth_sequences if seq in real_set)

    projection_columns = [column for column in (skill_col, correct_col, hint_col) if column]
    projected_real = {
        trajectory_tokens(traj, skill_col, correct_col, hint_col) for traj in real_traj.values()
    }
    projected_synthetic = [
        trajectory_tokens(traj, skill_col, correct_col, hint_col) for traj in synth_traj.values()
    ]
    projected_duplicates = sum(sequence in projected_real for sequence in projected_synthetic)

    def _sample(sequences, cap=max_nearest):
        return rng.sample(sequences, cap) if len(sequences) > cap else list(sequences)

    # Nearest-neighbour search must cover the FULL real set: slicing the first
    # `max_nearest` real trajectories by insertion order (the previous behaviour)
    # hid near-duplicates of any real learner outside the window and reported them
    # as maximally private. We instead compare a representative, seeded sample of
    # synthetic trajectories against every real trajectory.
    sampled_synth = _sample(synth_sequences)
    nearest_distances = _nearest_distances(sampled_synth, real_sequences) if real_sequences else []
    has_distances = bool(nearest_distances)

    # Membership-inference audit: is a real *training* trajectory systematically
    # closer to the synthetic set than a held-out *test* trajectory? AUC ~0.5 is
    # safe; well above 0.5 means the generator leaks training membership.
    test_seq_by_learner: Dict[str, tuple] = {}
    membership = _membership_risk([], [], [], seed)
    membership_reference_size = 0
    if test_traj:
        test_seq_by_learner = {
            learner: trajectory_value_tokens(traj, selected_columns) for learner, traj in test_traj.items()
        }
        sampled_members = _sample(real_sequences)
        sampled_nonmembers = _sample(list(test_seq_by_learner.values()))
        membership_reference = _sample(synth_sequences)
        membership_reference_size = len(membership_reference)
        membership = _membership_risk(
            sampled_members,
            sampled_nonmembers,
            membership_reference,
            seed + 2,
        )

    exact_indicators = [1.0 if sequence in real_set else 0.0 for sequence in synth_sequences]
    near_duplicate_indicators = [
        1.0 if distance <= near_duplicate_threshold else 0.0 for distance in nearest_distances
    ]

    result = {
        "representation": "full_value_trajectory",
        "representation_columns": selected_columns,
        "exact_duplicate_rate": exact_duplicates / len(synth_sequences) if synth_sequences else 0.0,
        "exact_duplicate_count": exact_duplicates,
        "exact_duplicate_rate_ci": _bootstrap_mean_ci(exact_indicators, seed + 3),
        "real_set_size": len(real_sequences),
        "nearest_neighbor_synthetic_sample_size": len(sampled_synth),
        # Lower distance = more memorization. Report None (not 0.0) when no
        # comparison was possible, so an empty set is not confused with a leak.
        "mean_nearest_neighbor_distance": mean(nearest_distances) if has_distances else None,
        "mean_nearest_neighbor_distance_ci": _bootstrap_mean_ci(nearest_distances, seed + 4),
        "min_nearest_neighbor_distance": min(nearest_distances) if has_distances else None,
        # Fraction of synthetic trajectories within `near_duplicate_threshold` of
        # some real trajectory (near-copies, not just exact copies).
        "near_duplicate_rate": (
            mean(near_duplicate_indicators)
            if has_distances
            else None
        ),
        "near_duplicate_rate_ci": _bootstrap_mean_ci(near_duplicate_indicators, seed + 5),
        "near_duplicate_threshold": near_duplicate_threshold,
        "membership_inference_auc": membership["raw_auc"],
        "membership_inference_attack_auc": membership["attack_auc"],
        "membership_inference_attack_advantage": membership["attack_advantage"],
        "membership_inference_attack_advantage_ci": membership["attack_advantage_ci"],
        "membership_inference_reference_sample_size": membership_reference_size,
        "projected_pattern_audit": {
            "representation": "skill_correct_hint_projection",
            "representation_columns": projection_columns,
            "exact_duplicate_rate": (
                projected_duplicates / len(projected_synthetic) if projected_synthetic else 0.0
            ),
            "exact_duplicate_count": projected_duplicates,
        },
    }

    # --- Privacy risk separately for tail learners (privacy-tail conflict). ---
    # Rare trajectories are more identifiable, so a generator that preserves the
    # tail may expose it. Per tail group we ask the memorization question about
    # the *real tail learners*: how close does the synthetic set come to each rare
    # real learner (real_tail -> synthetic NN), and can membership be inferred for
    # tail members specifically. A non-tail baseline is reported for contrast.
    result["by_tail_group"] = _privacy_by_tail(
        real_traj, real_seq_by_learner, test_seq_by_learner, synth_sequences,
        skill_col, correct_col, hint_col, attempts_col, response_time_col, gap_col,
        dropout_col,
        test_traj, rng, seed, near_duplicate_threshold, min_tail_learners,
        tail_reference_sample, _sample,
    )
    return result


def _privacy_by_tail(
    real_traj, real_seq_by_learner, test_seq_by_learner, synth_sequences,
    skill_col, correct_col, hint_col, attempts_col, response_time_col, gap_col,
    dropout_col,
    test_traj, rng, seed, near_duplicate_threshold, min_tail_learners,
    tail_reference_sample, sample_fn,
):
    if not real_traj or not synth_sequences:
        return {"status": "skipped", "reason": "No real or synthetic trajectories for per-tail privacy."}
    reference = tail_reference(
        real_traj, skill_col, correct_col, hint_col, attempts_col,
        response_time_col, gap_col, dropout_col
    )
    real_labels = tail_labels(
        real_traj, skill_col, correct_col, hint_col, reference, attempts_col,
        response_time_col, gap_col, dropout_col
    )
    test_labels = {}
    if test_traj:
        test_labels = tail_labels(
            test_traj, skill_col, correct_col, hint_col, reference, attempts_col,
            response_time_col, gap_col, dropout_col
        )
    # Bounded synthetic reference: per-tail NN is an audit, not an exact search;
    # a seeded sample keeps it fast without changing the qualitative signal.
    synth_reference = (
        rng.sample(synth_sequences, tail_reference_sample)
        if len(synth_sequences) > tail_reference_sample
        else synth_sequences
    )
    group_names = sorted({name for labels in real_labels.values() for name in labels})
    by_group: Dict[str, object] = {}
    for group_index, name in enumerate(group_names):
        tail_seqs = [real_seq_by_learner[learner] for learner, labels in real_labels.items() if labels.get(name)]
        nontail_seqs = [
            real_seq_by_learner[learner] for learner, labels in real_labels.items() if not labels.get(name)
        ]
        nontail_query = sample_fn(nontail_seqs, min(200, len(nontail_seqs))) if nontail_seqs else []
        tail_nn = _nearest_distances(tail_seqs, synth_reference)
        nontail_nn = _nearest_distances(nontail_query, synth_reference)
        test_tail = [test_seq_by_learner[learner] for learner, labels in test_labels.items() if labels.get(name)]
        test_nontail = [
            test_seq_by_learner[learner]
            for learner, labels in test_labels.items()
            if not labels.get(name)
        ]
        test_nontail_query = (
            sample_fn(test_nontail, min(200, len(test_nontail))) if test_nontail else []
        )
        group_seed = seed + 100 + group_index * 10
        tail_membership = _membership_risk(
            tail_seqs, test_tail, synth_reference, group_seed
        )
        nontail_membership = _membership_risk(
            nontail_query, test_nontail_query, synth_reference, group_seed + 1
        )
        tail_near = [
            1.0 if distance <= near_duplicate_threshold else 0.0 for distance in tail_nn
        ]
        nontail_near = [
            1.0 if distance <= near_duplicate_threshold else 0.0 for distance in nontail_nn
        ]
        tail_near_rate = mean(tail_near) if tail_near else None
        nontail_near_rate = mean(nontail_near) if nontail_near else None
        tail_mean_distance = mean(tail_nn) if tail_nn else None
        nontail_mean_distance = mean(nontail_nn) if nontail_nn else None

        privacy_tail_gap = {}
        if tail_mean_distance is not None and nontail_mean_distance is not None:
            privacy_tail_gap["mean_nearest_exposure_gap"] = (
                nontail_mean_distance - tail_mean_distance
            )
        if tail_near_rate is not None and nontail_near_rate is not None:
            privacy_tail_gap["near_duplicate_rate_gap"] = tail_near_rate - nontail_near_rate
        if (
            tail_membership["attack_advantage"] is not None
            and nontail_membership["attack_advantage"] is not None
        ):
            privacy_tail_gap["membership_attack_advantage_gap"] = (
                tail_membership["attack_advantage"]
                - nontail_membership["attack_advantage"]
            )
        by_group[name] = {
            "real_tail_learners": len(tail_seqs),
            "test_tail_learners": len(test_tail),
            # real_tail -> synthetic NN: lower = the synthetic set sits closer to
            # rare real learners (higher exposure of the tail).
            "mean_nearest_synthetic_distance": tail_mean_distance,
            "mean_nearest_synthetic_distance_ci": _bootstrap_mean_ci(
                tail_nn, group_seed + 2
            ),
            "min_nearest_synthetic_distance": min(tail_nn) if tail_nn else None,
            "sampled_exact_match_rate": (
                mean([1.0 if distance == 0 else 0.0 for distance in tail_nn])
                if tail_nn
                else None
            ),
            "nontail_mean_nearest_synthetic_distance": nontail_mean_distance,
            "near_duplicate_rate": tail_near_rate,
            "near_duplicate_rate_ci": _bootstrap_mean_ci(tail_near, group_seed + 3),
            "nontail_near_duplicate_rate": nontail_near_rate,
            "membership_inference_auc": tail_membership["raw_auc"],
            "membership_inference_attack_auc": tail_membership["attack_auc"],
            "membership_inference_attack_advantage": tail_membership["attack_advantage"],
            "membership_inference_attack_advantage_ci": tail_membership["attack_advantage_ci"],
            "nontail_membership_inference_attack_advantage": nontail_membership["attack_advantage"],
            "privacy_tail_gap": privacy_tail_gap,
            "exploratory": (
                len(tail_seqs) < min_tail_learners
                or len(test_tail) < min_tail_learners
            ),
        }
    return {
        "status": "completed",
        "synthetic_reference_sample_size": len(synth_reference),
        "note": "real_tail->synthetic NN; lower distance / higher membership AUC than the non-tail baseline = privacy-tail conflict.",
        "groups": by_group,
    }


def privacy_risk_comparison(standard, tail_targeted):
    """Risk-coded tail-targeted minus standard privacy differences."""
    comparison = {}

    def difference(key):
        left = standard.get(key)
        right = tail_targeted.get(key)
        return float(right) - float(left) if isinstance(left, (int, float)) and isinstance(right, (int, float)) else None

    for key in (
        "exact_duplicate_rate",
        "near_duplicate_rate",
        "membership_inference_attack_advantage",
    ):
        value = difference(key)
        if value is not None:
            comparison[f"{key}_risk_delta"] = value
    standard_distance = standard.get("mean_nearest_neighbor_distance")
    targeted_distance = tail_targeted.get("mean_nearest_neighbor_distance")
    if isinstance(standard_distance, (int, float)) and isinstance(targeted_distance, (int, float)):
        comparison["mean_nearest_exposure_risk_delta"] = float(standard_distance) - float(
            targeted_distance
        )

    standard_groups = standard.get("by_tail_group", {}).get("groups", {})
    targeted_groups = tail_targeted.get("by_tail_group", {}).get("groups", {})
    by_group = {}
    for name in sorted(set(standard_groups) | set(targeted_groups)):
        standard_group = standard_groups.get(name, {})
        targeted_group = targeted_groups.get(name, {})
        entry = {}
        for key in ("near_duplicate_rate", "membership_inference_attack_advantage"):
            left = standard_group.get(key)
            right = targeted_group.get(key)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                entry[f"{key}_risk_delta"] = float(right) - float(left)
        left_distance = standard_group.get("mean_nearest_synthetic_distance")
        right_distance = targeted_group.get("mean_nearest_synthetic_distance")
        if isinstance(left_distance, (int, float)) and isinstance(right_distance, (int, float)):
            entry["mean_nearest_exposure_risk_delta"] = float(left_distance) - float(
                right_distance
            )
        by_group[name] = entry
    return {
        "direction": "Positive values mean tail-targeted generation increased empirical privacy risk.",
        "global": comparison,
        "by_tail_group": by_group,
    }


def diversity_coverage(
    real_traj,
    synth_traj,
    skill_col,
    correct_col,
    hint_col,
    value_columns=None,
):
    real_skill_ngrams = ngram_distribution(real_traj, skill_col, n=3)
    synth_skill_ngrams = ngram_distribution(synth_traj, skill_col, n=3)
    real_sequences = [trajectory_tokens(traj, skill_col, correct_col, hint_col) for traj in real_traj.values()]
    synth_sequences = [trajectory_tokens(traj, skill_col, correct_col, hint_col) for traj in synth_traj.values()]
    synth_set = set(synth_sequences)
    full_synth_sequences = [
        trajectory_value_tokens(trajectory, value_columns or [skill_col, correct_col])
        for trajectory in synth_traj.values()
    ]
    real_ngram_keys = set(real_skill_ngrams)
    synth_ngram_keys = set(synth_skill_ngrams)
    return {
        "unique_synthetic_full_trajectory_rate": (
            len(set(full_synth_sequences)) / len(full_synth_sequences) if full_synth_sequences else 0.0
        ),
        "unique_synthetic_projected_trajectory_rate": (
            len(synth_set) / len(synth_sequences) if synth_sequences else 0.0
        ),
        "unique_synthetic_skill_path_count": len(
            {
                tuple(str(record.get(skill_col, "")) for record in trajectory)
                for trajectory in synth_traj.values()
            }
        ),
        "skill_trigram_coverage": (
            len(real_ngram_keys & synth_ngram_keys) / len(real_ngram_keys) if real_ngram_keys else 0.0
        ),
        "skill_trigram_precision": (
            len(real_ngram_keys & synth_ngram_keys) / len(synth_ngram_keys) if synth_ngram_keys else 0.0
        ),
        "skill_trigram_jaccard": (
            len(real_ngram_keys & synth_ngram_keys) / len(real_ngram_keys | synth_ngram_keys)
            if real_ngram_keys or synth_ngram_keys
            else 1.0
        ),
        "synthetic_skill_trigram_entropy": entropy(synth_skill_ngrams),
        "real_skill_trigram_entropy": entropy(real_skill_ngrams),
        "skill_trigram_entropy_abs_error": abs(
            entropy(real_skill_ngrams) - entropy(synth_skill_ngrams)
        ),
    }


def _safe_number(value) -> float:
    return safe_float(value, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate synthetic educational trajectories.")
    parser.add_argument("--real-train", required=True, type=Path, help="CSV file for real training trajectories.")
    parser.add_argument("--real-test", required=True, type=Path, help="CSV file for held-out real trajectories.")
    parser.add_argument("--synthetic", required=True, type=Path, help="CSV file for synthetic trajectories.")
    parser.add_argument("--config", type=Path, help="Optional YAML config with column names.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    config = load_config(args.config)
    columns = column_config(config)
    report = evaluate(
        load_csv(args.real_train),
        load_csv(args.real_test),
        load_csv(args.synthetic),
        columns,
    )

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
