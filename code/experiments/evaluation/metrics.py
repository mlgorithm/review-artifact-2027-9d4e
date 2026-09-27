"""Metrics for synthetic educational trajectory evaluation.

The implementation is intentionally dependency-light so it can run in this
repository without a Python package setup. Inputs are lists of interaction
records loaded from CSV files.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


Record = Dict[str, str]
Trajectory = List[Record]


def _finite_or(value: float, default: float) -> float:
    # Reject non-finite parses (inf/-inf/nan). A string like "inf"/"1e999"/"nan"
    # is float-parseable but would poison downstream means/Wasserstein/histograms
    # (nan propagates silently; inf gives garbage or crashes binning).
    return value if math.isfinite(value) else default


def safe_float(value: object, default: float = 0.0) -> float:
    # Best-effort numeric coercion for CSV/bin-label values. Handles booleans,
    # duration suffixes ("15s"), range labels ("3-4" -> midpoint), and "5+".
    # Any value it cannot parse falls back to `default` (0.0) SILENTLY -- callers
    # that need to distinguish "genuinely zero" from "unparseable" must guard the
    # input themselves; this function never raises.
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return _finite_or(float(text), default)
    except ValueError:
        lowered = text.lower()
        if lowered in {"true", "yes", "y"}:
            return 1.0
        if lowered in {"false", "no", "n"}:
            return 0.0
        numeric_text = lowered.replace("seconds", "").replace("second", "").replace("secs", "")
        numeric_text = numeric_text.replace("sec", "").replace("s", "").strip()
        if numeric_text.endswith("+"):
            numeric_text = numeric_text[:-1].strip()
        if "-" in numeric_text:
            left, right = numeric_text.split("-", 1)
            try:
                return _finite_or((float(left.strip()) + float(right.strip())) / 2.0, default)
            except ValueError:
                return default
        try:
            return _finite_or(float(numeric_text), default)
        except ValueError:
            return default
    return default


def binary_value(record: Mapping[str, str], column: str) -> int:
    return 1 if safe_float(record.get(column), 0.0) >= 0.5 else 0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def total_variation(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def distribution(values: Iterable[object]) -> Dict[str, float]:
    counter = Counter(str(value) for value in values)
    total = sum(counter.values())
    if total == 0:
        return {}
    return {key: count / total for key, count in counter.items()}


def numeric_histogram(
    values: Sequence[float],
    bins: int = 10,
    low: float | None = None,
    high: float | None = None,
) -> Dict[str, float]:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return {}
    lo = min(values) if low is None else low
    hi = max(values) if high is None else high
    if lo == hi:
        # Degenerate range: emit a single zero-width bin using the same
        # "start-end" key scheme as the general case so a constant set stays
        # comparable to a range-keyed histogram computed over the same edges.
        return {f"{lo:g}-{hi:g}": 1.0}
    width = (hi - lo) / bins
    counts: Counter[str] = Counter()
    for value in values:
        index = min(bins - 1, max(0, int((value - lo) / width)))
        start = lo + index * width
        end = start + width
        counts[f"{start:g}-{end:g}"] += 1
    total = len(values)
    return {key: count / total for key, count in counts.items()}


def histogram_total_variation(
    left: Sequence[float],
    right: Sequence[float],
    bins: int = 10,
) -> float:
    """Total-variation distance between two numeric samples over shared bins.

    Bin edges are derived from the combined range so the two histograms use the
    same key scheme; binning each side over its own min/max produces
    non-aligning keys and systematically inflates the distance.
    """
    combined = list(left) + list(right)
    if not combined:
        return 0.0
    lo = min(combined)
    hi = max(combined)
    return total_variation(
        numeric_histogram(left, bins, low=lo, high=hi),
        numeric_histogram(right, bins, low=lo, high=hi),
    )


def jensen_shannon(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    midpoint = {key: 0.5 * (left.get(key, 0.0) + right.get(key, 0.0)) for key in keys}
    return 0.5 * _kl_divergence(left, midpoint) + 0.5 * _kl_divergence(right, midpoint)


def _kl_divergence(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    value = 0.0
    for key, probability in left.items():
        if probability <= 0:
            continue
        reference = right.get(key, 0.0)
        if reference <= 0:
            continue
        value += probability * math.log(probability / reference, 2)
    return value


def wasserstein_1d(left: Sequence[float], right: Sequence[float]) -> float | None:
    if not left and not right:
        # Both sides empty: nothing to compare, treat as identical (0 distance).
        return 0.0
    if not left or not right:
        # Exactly one side is empty: the distributions cannot be aligned, so the
        # distance is undefined. Report None (not 0.0, which reads as a perfect
        # match, and not a mean-magnitude surrogate, which is not a valid/bounded
        # distance and would be incomparable across dataset pairs if averaged).
        return None
    left_sorted = sorted(left)
    right_sorted = sorted(right)
    size = max(len(left_sorted), len(right_sorted))
    if size == 1:
        return abs(left_sorted[0] - right_sorted[0])
    total = 0.0
    for index in range(size):
        q = index / (size - 1)
        total += abs(_quantile(left_sorted, q) - _quantile(right_sorted, q))
    return total / size


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    position = q * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return sorted_values[low]
    weight = position - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def group_trajectories(
    records: Sequence[Record],
    learner_column: str,
    order_column: str,
) -> Dict[str, Trajectory]:
    grouped: Dict[str, Trajectory] = defaultdict(list)
    for record in records:
        grouped[str(record.get(learner_column, ""))].append(record)
    for learner_id, trajectory in grouped.items():
        grouped[learner_id] = sorted(
            trajectory,
            key=lambda row: (safe_float(row.get(order_column), 0.0), str(row.get(order_column, ""))),
        )
    return dict(grouped)


def sequence_lengths(trajectories: Mapping[str, Trajectory]) -> List[float]:
    return [float(len(trajectory)) for trajectory in trajectories.values()]


def correctness_rate(records: Sequence[Record], correct_column: str) -> float:
    return mean([binary_value(record, correct_column) for record in records])


def rate_by_group(records: Sequence[Record], group_column: str, value_column: str) -> Dict[str, float]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for record in records:
        grouped[str(record.get(group_column, ""))].append(binary_value(record, value_column))
    return {key: mean(values) for key, values in grouped.items()}


def per_group_absolute_error(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    return mean([abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys])


def transition_distribution(
    trajectories: Mapping[str, Trajectory],
    column: str,
    binary: bool = True,
) -> Dict[str, float]:
    transitions: List[str] = []
    for trajectory in trajectories.values():
        if len(trajectory) < 2:
            continue
        values = [
            str(binary_value(record, column)) if binary else str(record.get(column, ""))
            for record in trajectory
        ]
        transitions.extend(f"{left}->{right}" for left, right in zip(values, values[1:]))
    return distribution(transitions)


def streak_lengths(trajectories: Mapping[str, Trajectory], column: str, target: int) -> List[float]:
    lengths: List[float] = []
    for trajectory in trajectories.values():
        current = 0
        for record in trajectory:
            if binary_value(record, column) == target:
                current += 1
            else:
                if current:
                    lengths.append(float(current))
                current = 0
        if current:
            lengths.append(float(current))
    return lengths


def sequence_position_curve(
    trajectories: Mapping[str, Trajectory],
    value_column: str,
    buckets: int = 10,
) -> Dict[str, float]:
    """Learner-equal curve on a common relative-position grid.

    Every learner contributes one linearly interpolated value to every grid
    position.  The previous bucket-assignment implementation left interior bins
    empty for short trajectories and then treated a missing bin as a behavior
    rate of zero when curves were compared.  That mixed trajectory-length
    support with behavioral shape.  Interpolation makes the ten positions
    genuinely length-normalized while retaining equal learner weights.
    """
    if buckets <= 0:
        return {}
    bucket_values: Dict[str, List[float]] = defaultdict(list)
    for trajectory in trajectories.values():
        if not trajectory:
            continue
        values = [float(binary_value(record, value_column)) for record in trajectory]
        if len(values) == 1 or buckets == 1:
            interpolated = [values[0]] * buckets
        else:
            interpolated = []
            for bucket in range(buckets):
                relative_position = bucket / (buckets - 1)
                source_position = relative_position * (len(values) - 1)
                left = int(math.floor(source_position))
                right = min(len(values) - 1, left + 1)
                weight = source_position - left
                interpolated.append(
                    values[left] * (1.0 - weight) + values[right] * weight
                )
        for bucket, value in enumerate(interpolated):
            bucket_values[str(bucket)].append(value)
    return {bucket: mean(values) for bucket, values in bucket_values.items()}


def ngram_distribution(
    trajectories: Mapping[str, Trajectory],
    column: str,
    n: int = 3,
    binary: bool = False,
) -> Dict[str, float]:
    grams: List[str] = []
    for trajectory in trajectories.values():
        values = [
            str(binary_value(record, column)) if binary else str(record.get(column, ""))
            for record in trajectory
        ]
        if len(values) < n:
            continue
        grams.extend("|".join(values[index : index + n]) for index in range(len(values) - n + 1))
    return distribution(grams)


def trajectory_tokens(
    trajectory: Trajectory,
    skill_column: str,
    correct_column: str,
    hint_column: str | None,
) -> Tuple[str, ...]:
    tokens = []
    for record in trajectory:
        skill = str(record.get(skill_column, ""))
        correct = binary_value(record, correct_column)
        hint = binary_value(record, hint_column) if hint_column else 0
        tokens.append(f"{skill}:{correct}:{hint}")
    return tuple(tokens)


def trajectory_value_tokens(
    trajectory: Trajectory,
    columns: Sequence[str],
) -> Tuple[Tuple[str, ...], ...]:
    """Lossless event tuples for a trajectory over selected value columns."""
    return tuple(
        tuple(str(record.get(column, "")) for column in columns)
        for record in trajectory
    )


def hamming_like_distance(left: Sequence[object], right: Sequence[object]) -> float:
    if not left and not right:
        return 0.0
    max_len = max(len(left), len(right))
    mismatches = abs(len(left) - len(right))
    mismatches += sum(1 for lval, rval in zip(left, right) if lval != rval)
    return mismatches / max_len


def entropy(distribution_map: Mapping[str, float]) -> float:
    return -sum(prob * math.log(prob, 2) for prob in distribution_map.values() if prob > 0)


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    pairs = sorted((float(score), int(label)) for score, label in zip(scores, labels))
    positive_count = sum(1 for _, label in pairs if label == 1)
    negative_count = len(pairs) - positive_count
    if positive_count == 0 or negative_count == 0:
        return 0.5

    positive_rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        positive_rank_sum += average_rank * sum(1 for _, label in pairs[index:end] if label == 1)
        index = end

    return (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def binary_classification_metrics(
    labels: Sequence[int], scores: Sequence[float]
) -> Dict[str, float | None]:
    if not labels:
        return {
            "auc": None,
            "auroc": None,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "log_loss": 0.0,
        }
    predictions = [1 if score >= 0.5 else 0 for score in scores]
    tp = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 1)
    fp = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 1)
    fn = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 0)
    accuracy = mean([1.0 if label == pred else 0.0 for label, pred in zip(labels, predictions)])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    eps = 1e-12
    log_loss = -mean(
        [
            label * math.log(min(1 - eps, max(eps, score)))
            + (1 - label) * math.log(min(1 - eps, max(eps, 1 - score)))
            for label, score in zip(labels, scores)
        ]
    )
    two_classes = len(set(int(label) for label in labels)) == 2
    ranking_auc = auc_score(labels, scores) if two_classes else None
    return {
        # A ranking metric has no mathematical definition when the evaluated
        # labels contain only one class. Returning 0.5 would falsely turn an
        # unscorable slice into an apparently chance-level result.
        "auc": ranking_auc,
        "auroc": ranking_auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "log_loss": log_loss,
    }


def skill_probability_model(
    records: Sequence[Record],
    skill_column: str,
    correct_column: str,
) -> Tuple[float, Dict[str, float]]:
    global_rate = correctness_rate(records, correct_column)
    by_skill = rate_by_group(records, skill_column, correct_column)
    return global_rate, by_skill


def predict_correctness(
    records: Sequence[Record],
    skill_column: str,
    global_rate: float,
    by_skill: Mapping[str, float],
) -> List[float]:
    return [by_skill.get(str(record.get(skill_column, "")), global_rate) for record in records]


# Default tail size: the paper's Table 3 uses top/bottom 5% cut-points for the
# frequency-defined tails (long inactivity, high hint use, rapid guessing, ...).
DEFAULT_TAIL_QUANTILE = 0.05
# A discretized signal cannot always represent an exact 5% tail. We accept a
# non-empty boundary group up to twice the requested prevalence; beyond that the
# signal is too coarse to support a defensible percentile tail and is omitted.
MAX_PERCENTILE_TAIL_MULTIPLIER = 2.0
# "Low correctness" bar used by rapid_guessing (fast AND not doing well). Fixed,
# not a percentile: rapid guessing is defined by speed, correctness only gates it.
_LOW_CORRECTNESS = 0.5


def _learner_summary(
    trajectory: Trajectory,
    skill_column: str,
    correct_column: str,
    hint_column: str | None = None,
    attempt_column: str | None = None,
    response_time_column: str | None = None,
    gap_column: str | None = None,
) -> Dict[str, object]:
    """Per-learner scalar summaries used to place a learner in tail groups.

    Kept as one function so the reference (which fixes the percentile cut-points)
    and the labeller (which applies them) compute every quantity identically.
    """
    correct_values = [binary_value(record, correct_column) for record in trajectory]
    length = len(trajectory)
    half = length // 2
    first_half = correct_values[: max(1, half)]
    second_half = correct_values[half:] or first_half

    # hint requests per attempt (Table 3 "high hint use"): total hints / total
    # attempts, with each interaction counting as >=1 attempt so the denominator
    # is never zero for a non-empty trajectory.
    hint_per_attempt = None
    if hint_column:
        total_hints = sum(safe_float(record.get(hint_column), 0.0) for record in trajectory)
        if attempt_column:
            total_attempts = sum(max(1.0, safe_float(record.get(attempt_column), 0.0)) for record in trajectory)
        else:
            total_attempts = float(length)
        hint_per_attempt = total_hints / total_attempts if total_attempts else 0.0

    def _present_values(column: str | None) -> List[float]:
        if not column:
            return []
        parsed = (_duration_to_seconds(record.get(column, "")) for record in trajectory)
        return [value for value in parsed if value is not None]

    response_times = _present_values(response_time_column)
    gaps = _present_values(gap_column)

    by_skill: Dict[str, List[int]] = defaultdict(list)
    for record, value in zip(trajectory, correct_values):
        by_skill[str(record.get(skill_column, ""))].append(value)

    return {
        "length": float(length),
        "correctness": mean(correct_values),
        "first_half_correct": mean(first_half),
        "second_half_correct": mean(second_half),
        "hint_per_attempt": hint_per_attempt,
        "mean_response_time": mean(response_times) if response_times else None,
        "max_gap": max(gaps) if gaps else None,
        "skill_path": tuple(str(record.get(skill_column, "")) for record in trajectory),
        "persistent_misconception": any(
            len(values) >= 3 and mean(values) <= 0.25 for values in by_skill.values()
        ),
    }


_DURATION_UNIT_SECONDS = {
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
    "w": 7.0 * 86400.0,
}


def _duration_to_seconds(label: object) -> float | None:
    """Parse a binned duration/gap label into a representative magnitude (seconds).

    Handles unit suffixes (s/m/h/d), ranges ("5-60m" -> upper bound 60m), open
    bins ("7d+" -> 7d) and the "start" sentinel; returns None for empty or
    unparseable values. This replaces safe_float for time signals: safe_float
    only strips seconds, so "5-60m"/"1-24h"/"7d+" silently collapse to 0.0, which
    made the inactivity/response-time cut-points degenerate.
    """
    text = str(label).strip().lower()
    if text == "":
        return None
    if text == "start":
        return 0.0
    cleaned = text.rstrip("+")
    unit = 1.0
    if cleaned and cleaned[-1] in _DURATION_UNIT_SECONDS:
        unit = _DURATION_UNIT_SECONDS[cleaned[-1]]
        cleaned = cleaned[:-1]
    upper = cleaned.split("-")[-1].strip()
    try:
        return float(upper) * unit
    except ValueError:
        return None


def _skill_transitions(skill_path: Tuple[str, ...]) -> List[str]:
    return [f"{left}->{right}" for left, right in zip(skill_path, skill_path[1:])]


def _path_typicality(
    skill_path: Tuple[str, ...],
    transition_logp: Mapping[str, float],
    floor_logp: float,
) -> float | None:
    """Mean log-probability of a learner's skill transitions under the reference.

    A low value means the learner follows a low-frequency skill/item transition
    sequence (Table 3 "rare skill path"). Returns None for trajectories with no
    transition (length < 2), which cannot be placed on the typicality scale.
    """
    transitions = _skill_transitions(skill_path)
    if not transitions:
        return None
    return mean([transition_logp.get(transition, floor_logp) for transition in transitions])


def tail_reference(
    trajectories: Mapping[str, Trajectory],
    skill_column: str,
    correct_column: str = "correct",
    hint_column: str | None = None,
    attempt_column: str | None = None,
    response_time_column: str | None = None,
    gap_column: str | None = None,
    dropout_column: str | None = None,
    quantile: float = DEFAULT_TAIL_QUANTILE,
) -> Dict[str, object]:
    """Derive tail-group cut-points from a reference set (typically real_train).

    All cut-points are percentiles of the reference distribution (Table 3's
    top/bottom-5% definitions) and must be fixed here and reused when labelling
    other sets; recomputing per set re-normalises each side to its own
    distribution so the frequency/length tails could never detect a difference.
    Optional signals (hint-per-attempt, response time, inactivity gap) only get a
    cut-point when the corresponding column is supplied, which is how the labeller
    decides whether to emit those tail groups.
    """
    summaries = [
        _learner_summary(
            trajectory, skill_column, correct_column, hint_column,
            attempt_column, response_time_column, gap_column,
        )
        for trajectory in trajectories.values()
    ]

    def _cutoff(
        values: List[float], q: float, side: str
    ) -> tuple[float | None, bool, str | None]:
        finite = sorted(value for value in values if value is not None)
        if not finite:
            return None, True, "no_finite_values"
        if finite[0] == finite[-1]:
            return None, True, "constant_signal"
        cut = _quantile(finite, q)
        tail_fraction = q if side == "low" else 1.0 - q

        def selected(inclusive: bool) -> int:
            if side == "low":
                return sum(value <= cut if inclusive else value < cut for value in finite)
            return sum(value >= cut if inclusive else value > cut for value in finite)

        # Preserve the historical boundary convention whenever it yields a
        # defensibly small group. Only fall back to the opposite convention to
        # resolve an oversized tie; this keeps unaffected datasets bit-for-bit
        # stable while preventing a coarse terminal bin from becoming a "tail".
        preferred_inclusive = not (
            (side == "high" and cut <= finite[0])
            or (side == "low" and cut >= finite[-1])
        )
        maximum_rate = min(1.0, tail_fraction * MAX_PERCENTILE_TAIL_MULTIPLIER)
        candidates = []
        for inclusive in (preferred_inclusive, not preferred_inclusive):
            count = selected(inclusive)
            if count > 0:
                rate = count / len(finite)
                candidates.append((rate, inclusive))
                if rate <= maximum_rate + 1e-12:
                    return cut, inclusive, None
        if not candidates:
            return None, True, "empty_tail_under_both_boundary_conventions"
        selected_rate = min(rate for rate, _ in candidates)
        return (
            None,
            True,
            "coarse_tie_minimum_nonempty_prevalence_"
            f"{selected_rate:.6f}_exceeds_{maximum_rate:.6f}",
        )

    # Skill-transition model over the reference: a learner is on a "rare skill
    # path" when its transitions are collectively low-frequency, not merely when
    # its whole trajectory happens to be unique (nearly every long trajectory is).
    transition_counts: Counter = Counter()
    for summary in summaries:
        transition_counts.update(_skill_transitions(summary["skill_path"]))
    total_transitions = sum(transition_counts.values())
    vocabulary = len(transition_counts)
    # Add-one smoothed log-probabilities, with a floor for transitions unseen in
    # the reference (so a synthetic learner using a novel transition scores rare).
    denominator = total_transitions + vocabulary + 1
    transition_logp = {
        transition: math.log((count + 1) / denominator, 2) for transition, count in transition_counts.items()
    }
    transition_model_sha256 = hashlib.sha256(
        json.dumps(
            sorted(transition_logp.items()),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    floor_logp = math.log(1 / denominator, 2) if denominator > 0 else 0.0
    typicalities = [
        _path_typicality(summary["skill_path"], transition_logp, floor_logp) for summary in summaries
    ]

    lengths = [summary["length"] for summary in summaries]
    short_cutoff, short_inclusive, short_reason = _cutoff(lengths, quantile, "low")
    long_cutoff, long_inclusive, long_reason = _cutoff(lengths, 1.0 - quantile, "high")
    low_correctness_cutoff, low_correctness_inclusive, low_correctness_reason = _cutoff(
        [s["correctness"] for s in summaries], quantile, "low"
    )
    high_hint_cutoff, high_hint_inclusive, high_hint_reason = _cutoff(
        [s["hint_per_attempt"] for s in summaries], 1.0 - quantile, "high"
    )
    fast_response_cutoff, fast_response_inclusive, fast_response_reason = _cutoff(
        [s["mean_response_time"] for s in summaries], quantile, "low"
    )
    high_gap_cutoff, high_gap_inclusive, high_gap_reason = _cutoff(
        [s["max_gap"] for s in summaries], 1.0 - quantile, "high"
    )
    rare_path_cutoff, rare_path_inclusive, rare_path_reason = _cutoff(
        [t for t in typicalities if t is not None], quantile, "low"
    )
    unavailable = {
        name: reason
        for name, reason, available in (
            ("short_trajectory", short_reason, True),
            ("long_trajectory", long_reason, True),
            ("persistent_failure", low_correctness_reason, True),
            ("high_hint_use", high_hint_reason, hint_column is not None),
            ("rapid_guessing", fast_response_reason, response_time_column is not None),
            ("long_inactivity", high_gap_reason, gap_column is not None),
            ("rare_skill_path", rare_path_reason, True),
        )
        if available and reason is not None
    }
    return {
        "quantile": quantile,
        "maximum_percentile_tail_prevalence": quantile * MAX_PERCENTILE_TAIL_MULTIPLIER,
        "rapid_guessing_max_primary_signal_rate": _LOW_CORRECTNESS,
        "unavailable_percentile_tail_groups": unavailable,
        "short_cutoff": short_cutoff,
        "short_cutoff_inclusive": short_inclusive,
        "long_cutoff": long_cutoff,
        "long_cutoff_inclusive": long_inclusive,
        "low_correctness_cutoff": low_correctness_cutoff,
        "low_correctness_cutoff_inclusive": low_correctness_inclusive,
        "high_hint_cutoff": high_hint_cutoff,
        "high_hint_cutoff_inclusive": high_hint_inclusive,
        "fast_response_cutoff": fast_response_cutoff,
        "fast_response_cutoff_inclusive": fast_response_inclusive,
        "high_gap_cutoff": high_gap_cutoff,
        "high_gap_cutoff_inclusive": high_gap_inclusive,
        # Bottom-q path typicality = rarest skill-transition sequences.
        "rare_path_cutoff": rare_path_cutoff,
        "rare_path_cutoff_inclusive": rare_path_inclusive,
        "transition_logp": transition_logp,
        "transition_floor_logp": floor_logp,
        "transition_model_sha256": transition_model_sha256,
        "transition_vocabulary_size": vocabulary,
        "transition_observation_count": total_transitions,
        "has_hint": hint_column is not None,
        "has_response_time": response_time_column is not None,
        "has_gap": gap_column is not None,
        "has_dropout": dropout_column is not None,
    }


def tail_labels(
    trajectories: Mapping[str, Trajectory],
    skill_column: str,
    correct_column: str,
    hint_column: str,
    reference: Mapping[str, object] | None = None,
    attempt_column: str | None = None,
    response_time_column: str | None = None,
    gap_column: str | None = None,
    dropout_column: str | None = None,
) -> Dict[str, Dict[str, bool]]:
    """Assign each learner to the tail groups of Table 3, using fixed cut-points.

    Group set is stable within a call: frequency tails that need an absent signal
    (hint/response-time/gap column not supplied) are simply not emitted, rather
    than emitted as all-False.
    """
    if reference is None:
        reference = tail_reference(
            trajectories, skill_column, correct_column, hint_column,
            attempt_column, response_time_column, gap_column, dropout_column,
        )
    short_cutoff = reference.get("short_cutoff")
    long_cutoff = reference.get("long_cutoff")
    low_correct = reference.get("low_correctness_cutoff")
    high_hint = reference.get("high_hint_cutoff")
    fast_response = reference.get("fast_response_cutoff")
    high_gap = reference.get("high_gap_cutoff")
    rare_path_cutoff = reference.get("rare_path_cutoff")
    transition_logp = reference.get("transition_logp", {})
    floor_logp = float(reference.get("transition_floor_logp", 0.0))
    has_hint = bool(reference.get("has_hint")) and high_hint is not None
    has_response_time = bool(reference.get("has_response_time")) and fast_response is not None
    has_gap = bool(reference.get("has_gap")) and high_gap is not None
    has_dropout = bool(reference.get("has_dropout")) and dropout_column is not None

    def low_tail(value: float, cutoff: object, inclusive_key: str) -> bool:
        if cutoff is None:
            return False
        return value <= float(cutoff) if bool(reference.get(inclusive_key, True)) else value < float(cutoff)

    def high_tail(value: float, cutoff: object, inclusive_key: str) -> bool:
        if cutoff is None:
            return False
        return value >= float(cutoff) if bool(reference.get(inclusive_key, True)) else value > float(cutoff)

    stats: Dict[str, Dict[str, bool]] = {}
    for learner_id, trajectory in trajectories.items():
        summary = _learner_summary(
            trajectory, skill_column, correct_column, hint_column,
            attempt_column, response_time_column, gap_column,
        )
        length = summary["length"]
        correctness = summary["correctness"]
        first_half = summary["first_half_correct"]
        second_half = summary["second_half_correct"]
        labels: Dict[str, bool] = {
            # Repeated failure on the SAME skill despite practice (Table 3).
            "persistent_misconception": bool(summary["persistent_misconception"]),
            # Low early, high late -> recovered.
            "recovery": first_half <= 0.35 and second_half >= 0.65 and length >= 4,
            # High early, low late -> late failure (mirror of recovery).
            "late_failure": first_half >= 0.65 and second_half <= 0.35 and length >= 4,
        }
        if low_correct is not None:
            # Bottom-q overall correctness (a persistently failing learner).
            labels["persistent_failure"] = low_tail(
                correctness, low_correct, "low_correctness_cutoff_inclusive"
            ) and length >= 3
        if has_dropout:
            # Dropout is an explicit terminal outcome supplied by the dataset,
            # never inferred from a short trajectory. Static outcome columns are
            # repeated per event, so the learner mean is robust to malformed
            # single rows while adapter normalization keeps synthetic values
            # internally consistent.
            labels["dropout"] = mean(
                [binary_value(record, dropout_column) for record in trajectory]
            ) >= 0.5
        # Length tails only when the length distribution has spread (a degenerate
        # all-same-length cut-point would flag every learner short or long).
        if short_cutoff is not None:
            labels["short_trajectory"] = low_tail(
                length, short_cutoff, "short_cutoff_inclusive"
            )
        if long_cutoff is not None:
            labels["long_trajectory"] = high_tail(
                length, long_cutoff, "long_cutoff_inclusive"
            )
        if rare_path_cutoff is not None:
            typicality = _path_typicality(summary["skill_path"], transition_logp, floor_logp)
            labels["rare_skill_path"] = (
                typicality is not None
                and low_tail(typicality, rare_path_cutoff, "rare_path_cutoff_inclusive")
            )
        if has_hint:
            # Top-q hint requests per attempt.
            labels["high_hint_use"] = (
                summary["hint_per_attempt"] is not None
                and high_tail(summary["hint_per_attempt"], high_hint, "high_hint_cutoff_inclusive")
                and length >= 3
            )
        if has_response_time:
            # Bottom-q response time AND not doing well (Table 3 rapid guessing).
            mean_rt = summary["mean_response_time"]
            labels["rapid_guessing"] = (
                mean_rt is not None
                and low_tail(mean_rt, fast_response, "fast_response_cutoff_inclusive")
                and correctness <= _LOW_CORRECTNESS
                and length >= 3
            )
        if has_gap:
            # Top-q maximum inter-event gap (long inactivity).
            max_gap = summary["max_gap"]
            labels["long_inactivity"] = (
                max_gap is not None
                and high_tail(max_gap, high_gap, "high_gap_cutoff_inclusive")
                and length >= 3
            )
        stats[learner_id] = labels
    return stats


# ---------------------------------------------------------------------------
# Temporal dependence: autocorrelation and lagged cross-correlation.
# ---------------------------------------------------------------------------


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return 0.0
    return covariance / (var_x**0.5 * var_y**0.5)


def autocorrelation(
    trajectories: Mapping[str, Trajectory],
    column: str,
    lags: Sequence[int] = (1, 2, 3),
) -> Dict[str, float]:
    """Pearson autocorrelation of a binary signal at each lag, pooled over trajectories."""
    result: Dict[str, float] = {}
    for lag in lags:
        left: List[int] = []
        right: List[int] = []
        for trajectory in trajectories.values():
            values = [binary_value(record, column) for record in trajectory]
            for index in range(len(values) - lag):
                left.append(values[index])
                right.append(values[index + lag])
        result[str(lag)] = _pearson(left, right)
    return result


def cross_correlation(
    trajectories: Mapping[str, Trajectory],
    left_column: str,
    right_column: str,
    lag: int = 0,
) -> float:
    """Pearson correlation between two binary signals, right shifted by ``lag``."""
    left: List[int] = []
    right: List[int] = []
    for trajectory in trajectories.values():
        left_values = [binary_value(record, left_column) for record in trajectory]
        right_values = [binary_value(record, right_column) for record in trajectory]
        for index in range(len(trajectory)):
            shifted = index + lag
            if 0 <= shifted < len(trajectory):
                left.append(left_values[index])
                right.append(right_values[shifted])
    return _pearson(left, right)


# ---------------------------------------------------------------------------
# Cross-signal dependence: normalized mutual information between column pairs.
# ---------------------------------------------------------------------------


def _entropy_counts(counts: Mapping[object, int], total: int) -> float:
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log(count / total, 2) for count in counts.values() if count > 0)


def normalized_mutual_information(
    records: Sequence[Record],
    left_column: str,
    right_column: str,
) -> float:
    """NMI in [0, 1] between two categorical columns (geometric-mean normalized)."""
    joint: Counter = Counter()
    left_counts: Counter = Counter()
    right_counts: Counter = Counter()
    total = 0
    for record in records:
        left_value = str(record.get(left_column, ""))
        right_value = str(record.get(right_column, ""))
        joint[(left_value, right_value)] += 1
        left_counts[left_value] += 1
        right_counts[right_value] += 1
        total += 1
    if total == 0:
        return 0.0
    mutual = 0.0
    for (left_value, right_value), count in joint.items():
        p_xy = count / total
        p_x = left_counts[left_value] / total
        p_y = right_counts[right_value] / total
        mutual += p_xy * math.log(p_xy / (p_x * p_y), 2)
    entropy_x = _entropy_counts(left_counts, total)
    entropy_y = _entropy_counts(right_counts, total)
    if entropy_x <= 0 or entropy_y <= 0:
        return 0.0
    return max(0.0, mutual) / (entropy_x * entropy_y) ** 0.5


# ---------------------------------------------------------------------------
# Membership-inference audit (nearest-synthetic-neighbour attack).
# ---------------------------------------------------------------------------


def membership_inference_auc(
    member_sequences: Sequence[Sequence[object]],
    nonmember_sequences: Sequence[Sequence[object]],
    synthetic_sequences: Sequence[Sequence[object]],
) -> float | None:
    """AUC of a nearest-synthetic-neighbour membership-inference attack.

    Score = negative distance to the closest synthetic trajectory (closer =>
    more likely a training member). AUC ~0.5 means the generator does not leak
    membership; AUC well above 0.5 indicates training trajectories sit closer to
    the synthetic set than held-out ones (privacy risk).
    """
    result = membership_inference_scores(
        member_sequences, nonmember_sequences, synthetic_sequences
    )
    if result is None:
        return None
    labels, scores = result
    return auc_score(labels, scores)


def membership_inference_scores(
    member_sequences: Sequence[Sequence[object]],
    nonmember_sequences: Sequence[Sequence[object]],
    synthetic_sequences: Sequence[Sequence[object]],
) -> tuple[List[int], List[float]] | None:
    """Return labels and nearest-synthetic closeness scores for a MIA audit."""
    if not synthetic_sequences or not member_sequences or not nonmember_sequences:
        return None

    def closeness(sequence: Sequence[object]) -> float:
        return -min(hamming_like_distance(sequence, synth) for synth in synthetic_sequences)

    scores = [closeness(sequence) for sequence in member_sequences]
    scores += [closeness(sequence) for sequence in nonmember_sequences]
    labels = [1] * len(member_sequences) + [0] * len(nonmember_sequences)
    return labels, scores


def tail_prevalence(labels: Mapping[str, Mapping[str, bool]]) -> Dict[str, float]:
    if not labels:
        return {}
    names = sorted({name for learner_labels in labels.values() for name in learner_labels})
    return {
        name: mean([1.0 if learner_labels.get(name, False) else 0.0 for learner_labels in labels.values()])
        for name in names
    }
