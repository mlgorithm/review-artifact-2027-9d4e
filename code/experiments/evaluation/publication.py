"""Curated, research-question-aligned metrics for publication.

The raw evaluator intentionally retains extensive diagnostics.  This module is
the narrow contract for tables and statistical comparisons: it selects only
metrics that answer the proposal's RQ1/RQ2, uses one declared downstream model,
and omits quantities confounded by shared postprocessing or arbitrary decision
thresholds.
"""

from __future__ import annotations

from typing import Mapping


SCHEMA_VERSION = "publication_metrics_v2"
PRIMARY_DOWNSTREAM_MODEL = "hist_gradient_boosting"
PRIMARY_RANKING_METRIC = "auprc"
SECONDARY_RANKING_METRIC = "auroc"
RANKING_METRICS = (PRIMARY_RANKING_METRIC, SECONDARY_RANKING_METRIC)
MIN_PUBLICATION_TAIL_LEARNERS = 10
MAX_PUBLICATION_RARE_GROUP_PREVALENCE = 0.10

_PROPOSAL_TAIL_GROUPS = {
    "persistent_failure",
    "persistent_misconception",
    "recovery",
    "late_failure",
    "rare_skill_path",
    "high_hint_use",
    "rapid_guessing",
    "long_inactivity",
    "dropout",
}
_SHARED_LENGTH_GROUPS = {"short_trajectory", "long_trajectory"}
_WEEKLY_TAIL_NAMES = {
    "persistent_failure": "persistent_disengagement",
    "recovery": "reengagement",
    "late_failure": "late_disengagement",
    "rare_skill_path": "rare_activity_path",
    "long_inactivity": "long_inactivity",
}


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _ranking_metrics(payload: object) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    return {
        metric: value
        for metric in RANKING_METRICS
        if (value := _number(payload.get(metric))) is not None
    }


def _difference(left: Mapping[str, object], right: Mapping[str, object]) -> dict[str, float]:
    result = {}
    for metric in RANKING_METRICS:
        left_value = _number(left.get(metric))
        right_value = _number(right.get(metric))
        if left_value is not None and right_value is not None:
            result[metric] = left_value - right_value
    return result


def _gain(candidate: Mapping[str, object], reference: Mapping[str, object]) -> dict[str, float]:
    """Candidate minus reference; positive means the candidate improved."""
    return _difference(candidate, reference)


def _arm_metrics(block: Mapping[str, object], key: str) -> dict[str, float]:
    arm = block.get(key, {})
    if not isinstance(arm, Mapping):
        return {}
    return _ranking_metrics(arm.get(PRIMARY_DOWNSTREAM_MODEL, {}))


def _tail_policy(split) -> tuple[set[str], dict[str, str]]:
    semantics = str(getattr(split, "metadata", {}).get("primary_signal_semantics", ""))
    if semantics == "weekly_engagement":
        # A repeated incorrect answer on one skill is not meaningful when the
        # configured "skill" is a weekly activity category.  Dropout fidelity is
        # also excluded: its synthetic prevalence is calibrated by construction.
        return set(_WEEKLY_TAIL_NAMES), dict(_WEEKLY_TAIL_NAMES)
    return set(_PROPOSAL_TAIL_GROUPS - {"dropout"}), {
        name: name for name in _PROPOSAL_TAIL_GROUPS - {"dropout"}
    }


def _tail_fidelity(block: object, split) -> tuple[dict[str, object], dict[str, str]]:
    if not isinstance(block, Mapping):
        return {}, {}
    allowed, names = _tail_policy(split)
    prevalence = block.get("tail_prevalence_error", {})
    real_prevalence = block.get("tail_prevalence_real", {})
    intervals = block.get("tail_prevalence_error_ci", {})
    shape = block.get("tail_shape", {}).get("by_tail_group", {})
    if not isinstance(prevalence, Mapping) or not isinstance(shape, Mapping):
        return {}, {}

    retained: dict[str, object] = {}
    omitted: dict[str, str] = {}
    implemented = set(block.get("implemented_tail_groups", []))
    for raw_name in sorted(implemented):
        if raw_name in _SHARED_LENGTH_GROUPS:
            omitted[raw_name] = "Shared trajectory-length sampler; not attributable to the generator model."
            continue
        if raw_name not in allowed:
            reason = "Not a proposal-aligned tail for this dataset."
            if raw_name == "dropout":
                reason = "Synthetic dropout prevalence/shape is set by the common outcome postprocessor."
            elif raw_name == "persistent_misconception":
                reason = "Weekly activity categories do not define a pedagogical misconception."
            omitted[raw_name] = reason
            continue
        observed_real_prevalence = (
            _number(real_prevalence.get(raw_name))
            if isinstance(real_prevalence, Mapping)
            else None
        )
        if (
            observed_real_prevalence is not None
            and observed_real_prevalence
            > MAX_PUBLICATION_RARE_GROUP_PREVALENCE + 1e-12
        ):
            omitted[raw_name] = (
                f"Real-train learner prevalence {observed_real_prevalence:.6f} exceeds "
                f"the rare-group ceiling {MAX_PUBLICATION_RARE_GROUP_PREVALENCE:.6f}."
            )
            continue
        group = shape.get(raw_name, {})
        real_learners = int(group.get("real_learners", 0)) if isinstance(group, Mapping) else 0
        synthetic_learners = int(group.get("synthetic_learners", 0)) if isinstance(group, Mapping) else 0
        prevalence_error = _number(prevalence.get(raw_name))
        if prevalence_error is None:
            omitted[raw_name] = "Rare-group prevalence error is unavailable."
            continue
        published_name = names[raw_name]
        retained[published_name] = {
            "prevalence_error": prevalence_error,
            "prevalence_error_ci": intervals.get(raw_name),
            "real_learners": real_learners,
            "synthetic_learners": synthetic_learners,
            "source_tail_group": raw_name,
        }
        curve_error = (
            _number(group.get("correctness_curve_mae"))
            if isinstance(group, Mapping)
            else None
        )
        shape_estimable = (
            isinstance(group, Mapping)
            and not bool(group.get("exploratory", True))
            and real_learners >= MIN_PUBLICATION_TAIL_LEARNERS
            and synthetic_learners >= MIN_PUBLICATION_TAIL_LEARNERS
            and curve_error is not None
        )
        if shape_estimable:
            retained[published_name].update(
                {
                    "primary_signal_curve_mae": curve_error,
                    "primary_signal_curve_mae_ci": group.get(
                        "correctness_curve_mae_ci"
                    ),
                    "primary_signal_curve_mae_status": "estimable",
                }
            )
        else:
            retained[published_name]["primary_signal_curve_mae_status"] = (
                "not_estimable"
            )
            omitted[raw_name] = (
                "Rare-group prevalence is retained, but trajectory shape is not "
                f"estimable because fewer than {MIN_PUBLICATION_TAIL_LEARNERS} "
                "real or synthetic tail learners are available."
            )
    return retained, omitted


def _tail_targeting_gain(
    standard: Mapping[str, object], targeted: Mapping[str, object]
) -> dict[str, object]:
    gains = {}
    for name in sorted(set(standard) & set(targeted)):
        standard_group = standard[name]
        targeted_group = targeted[name]
        prevalence_standard = _number(standard_group.get("prevalence_error"))
        prevalence_targeted = _number(targeted_group.get("prevalence_error"))
        shape_standard = _number(standard_group.get("primary_signal_curve_mae"))
        shape_targeted = _number(targeted_group.get("primary_signal_curve_mae"))
        entry = {}
        if prevalence_standard is not None and prevalence_targeted is not None:
            entry["prevalence_error_reduction"] = prevalence_standard - prevalence_targeted
        if shape_standard is not None and shape_targeted is not None:
            entry["primary_signal_curve_mae_reduction"] = shape_standard - shape_targeted
        if entry:
            gains[name] = entry
    return gains


def _temporal_metrics(report: Mapping[str, object]) -> dict[str, float]:
    source = report.get("temporal_fidelity", {})
    if not isinstance(source, Mapping):
        return {}
    keys = (
        "correctness_transition_js",
        "skill_trigram_js",
        "correctness_curve_mae",
        "correctness_autocorrelation_mae",
        "correct_hint_cross_correlation_mae",
        "inter_event_time_js",
    )
    return {
        key: value
        for key in keys
        if (value := _number(source.get(key))) is not None
    }


def _modeled_marginal_metrics(report: Mapping[str, object], split) -> tuple[dict[str, object], list[str]]:
    source = (
        report.get("global_fidelity", {})
        .get("all_value_marginals", {})
        .get("by_column", {})
    )
    if not isinstance(source, Mapping):
        return {}, []
    metadata = dict(getattr(split, "metadata", {}) or {})
    excluded = set(getattr(split, "derived_columns", ()) or ())
    excluded.update(str(column) for column in metadata.get("publication_postprocessed_columns", []))
    retained = {
        str(column): float(entry["jensen_shannon_divergence"])
        for column, entry in source.items()
        if str(column) not in excluded
        and isinstance(entry, Mapping)
        and _number(entry.get("jensen_shannon_divergence")) is not None
    }
    return {
        "modeled_value_marginal_js_mean": _mean(list(retained.values())),
    }, sorted(retained)


def _detected_signal_columns(split) -> set[str]:
    # Dependence is prespecified over modeled behavioral/event signals. Do not
    # admit identifiers, grouping variables, outcomes, or other columns merely
    # because they happen to be present in the all-pairs diagnostic.
    columns = {
        str(dict(split.columns).get(role))
        for role in ("skill", "correct", "hint", "attempts", "response_time", "gap")
        if dict(split.columns).get(role)
    }
    for column in split.train.columns:
        name = str(column).lower()
        if "gap" in name or (
            "gap" not in name
            and any(token in name for token in ("response", "duration", "elapsed"))
        ):
            columns.add(str(column))
    metadata = dict(getattr(split, "metadata", {}) or {})
    columns.difference_update(str(column) for column in getattr(split, "derived_columns", ()) or ())
    columns.difference_update(
        str(column) for column in metadata.get("publication_postprocessed_columns", [])
    )
    return columns


def _dependence_metrics(report: Mapping[str, object], split) -> tuple[dict[str, object], list[str]]:
    pairwise = report.get("cross_signal_dependence", {}).get("pairwise_nmi", {})
    if not isinstance(pairwise, Mapping):
        return {}, []
    metadata = dict(getattr(split, "metadata", {}) or {})
    configured_pairs = metadata.get("publication_dependence_pairs")
    selected_pairs: set[frozenset[str]] | None = None
    if configured_pairs is not None:
        eligible_columns = {str(column) for column in split.train.columns}
        eligible_columns.difference_update(
            str(column) for column in getattr(split, "ignore_columns", ()) or ()
        )
        eligible_columns.difference_update(
            str(column) for column in getattr(split, "derived_columns", ()) or ()
        )
        eligible_columns.difference_update(
            str(column)
            for column in metadata.get("publication_postprocessed_columns", [])
        )
        selected_pairs = {
            frozenset((str(pair[0]), str(pair[1])))
            for pair in configured_pairs
            if isinstance(pair, (list, tuple))
            and len(pair) == 2
            and str(pair[0]) != str(pair[1])
            and str(pair[0]) in eligible_columns
            and str(pair[1]) in eligible_columns
        }
    allowed = _detected_signal_columns(split) if selected_pairs is None else set()
    errors = {}
    for pair, entry in pairwise.items():
        left, separator, right = str(pair).partition("__")
        if not separator:
            continue
        if selected_pairs is None:
            keep = left in allowed and right in allowed
        else:
            keep = frozenset((left, right)) in selected_pairs
        if not keep:
            continue
        value = _number(entry.get("abs_error")) if isinstance(entry, Mapping) else None
        if value is not None:
            errors[str(pair)] = value
    return {"prespecified_signal_nmi_mae": _mean(list(errors.values()))}, sorted(errors)


def _subgroup_metrics(report: Mapping[str, object]) -> tuple[dict[str, float], str | None]:
    source = report.get("subgroup_fidelity", {})
    if not isinstance(source, Mapping):
        return {}, "Subgroup block is unavailable."
    if source.get("partition") != "by_fixed_prefix_performance":
        return {}, (
            "Attribute subgroup output is excluded: independently resampled attributes do not "
            "measure relationships learned by the generator."
        )
    values = {}
    for output_key, source_key in (
        ("learner_share_mae", "learner_share_mae_across_subgroups"),
        ("future_primary_signal_rate_mae", "future_correctness_rate_mae_across_subgroups"),
        ("future_hint_rate_mae", "future_hint_rate_mae_across_subgroups"),
    ):
        value = _number(source.get(source_key))
        if value is not None:
            values[output_key] = value
    if not values:
        return {}, "Future-behavior subgroup aggregates are unavailable."
    return values, None


def _downstream_utility(report: Mapping[str, object]) -> dict[str, object]:
    block = report.get("downstream_utility_models", {})
    if not isinstance(block, Mapping) or block.get("status") != "completed":
        return {}
    arms = {
        "real_train": _arm_metrics(block, "train_on_real_test_on_real"),
        "synthetic_train": _arm_metrics(block, "train_on_synthetic_test_on_real"),
        "real_plus_synthetic": _arm_metrics(
            block, "train_on_real_plus_synthetic_test_on_real"
        ),
        "tail_targeted_synthetic": _arm_metrics(
            block, "train_on_tail_targeted_synthetic_test_on_real"
        ),
        "real_plus_tail_targeted_synthetic": _arm_metrics(
            block, "train_on_real_plus_tail_targeted_synthetic_test_on_real"
        ),
    }
    arms = {key: value for key, value in arms.items() if value}
    result: dict[str, object] = {"arms": arms}
    real = arms.get("real_train", {})
    synthetic = arms.get("synthetic_train", {})
    augmented = arms.get("real_plus_synthetic", {})
    targeted = arms.get("tail_targeted_synthetic", {})
    targeted_augmented = arms.get("real_plus_tail_targeted_synthetic", {})
    if real and synthetic:
        result["synthetic_utility_gap"] = _difference(real, synthetic)
    if real and augmented:
        result["real_plus_synthetic_gain_over_real"] = _gain(augmented, real)
    if real and targeted_augmented:
        result["targeted_augmentation_gain_over_real"] = _gain(
            targeted_augmented, real
        )
    if synthetic and targeted:
        result["tail_targeted_tstr_gain"] = _gain(targeted, synthetic)
    return result


def _tail_learnability(report: Mapping[str, object], split) -> tuple[dict[str, object], dict[str, str]]:
    groups = (
        report.get("downstream_utility_models", {})
        .get("tail_learnability", {})
        .get("groups", {})
    )
    if not isinstance(groups, Mapping):
        return {}, {}
    _, publication_names = _tail_policy(split)
    semantics = str(getattr(split, "metadata", {}).get("primary_signal_semantics", ""))
    if semantics == "weekly_engagement":
        # Unlike synthetic dropout fidelity, held-out-real dropout slicing is
        # valid: it uses the explicit real outcome only to select test learners.
        publication_names["dropout"] = "dropout"
    retained = {}
    omitted = {}
    train_prevalence = report.get("tail_fidelity", {}).get(
        "tail_prevalence_real", {}
    )
    for raw_name, entry in groups.items():
        raw_name = str(raw_name)
        if raw_name in _SHARED_LENGTH_GROUPS:
            omitted[raw_name] = "Shared length-sampler group; excluded from generator comparison."
            continue
        if raw_name not in publication_names:
            omitted[raw_name] = "Not a proposal-aligned tail for this dataset."
            continue
        observed_train_prevalence = (
            _number(train_prevalence.get(raw_name))
            if isinstance(train_prevalence, Mapping)
            else None
        )
        if (
            observed_train_prevalence is not None
            and observed_train_prevalence
            > MAX_PUBLICATION_RARE_GROUP_PREVALENCE + 1e-12
        ):
            omitted[raw_name] = (
                f"Real-train learner prevalence {observed_train_prevalence:.6f} exceeds "
                f"the rare-group ceiling {MAX_PUBLICATION_RARE_GROUP_PREVALENCE:.6f}."
            )
            continue
        if not isinstance(entry, Mapping) or bool(entry.get("exploratory", True)):
            omitted[raw_name] = "Held-out real tail has fewer than 10 learners."
            continue
        arms = {
            "real_train": _arm_metrics(entry, "train_on_real_test_on_real"),
            "synthetic_train": _arm_metrics(entry, "train_on_synthetic_test_on_real"),
            "real_plus_synthetic": _arm_metrics(
                entry, "train_on_real_plus_synthetic_test_on_real"
            ),
            "tail_targeted_synthetic": _arm_metrics(
                entry, "train_on_tail_targeted_synthetic_test_on_real"
            ),
            "real_plus_tail_targeted_synthetic": _arm_metrics(
                entry, "train_on_real_plus_tail_targeted_synthetic_test_on_real"
            ),
        }
        arms = {key: value for key, value in arms.items() if value}
        if not arms.get("real_train") or not arms.get("synthetic_train"):
            omitted[raw_name] = "AUPRC/AUROC is undefined for the held-out tail slice."
            continue
        published = {
            "test_learners": entry.get("test_learners"),
            "test_rows": entry.get("test_rows"),
            "arms": arms,
            "synthetic_utility_gap": _difference(
                arms["real_train"], arms["synthetic_train"]
            ),
        }
        if arms.get("real_plus_synthetic"):
            published["real_plus_synthetic_gain_over_real"] = _gain(
                arms["real_plus_synthetic"], arms["real_train"]
            )
        if arms.get("real_plus_tail_targeted_synthetic"):
            published["targeted_augmentation_gain_over_real"] = _gain(
                arms["real_plus_tail_targeted_synthetic"], arms["real_train"]
            )
        if arms.get("tail_targeted_synthetic"):
            published["tail_targeted_tstr_gain"] = _gain(
                arms["tail_targeted_synthetic"], arms["synthetic_train"]
            )
        retained[publication_names[raw_name]] = published
    return retained, omitted


def _outcome_tasks(report: Mapping[str, object], split) -> dict[str, object]:
    block = report.get("downstream_utility_tasks", {})
    if not isinstance(block, Mapping) or block.get("status") != "completed":
        return {}
    tasks = block.get("tasks", {})
    if not isinstance(tasks, Mapping):
        return {}
    published = {}
    weekly = str(getattr(split, "metadata", {}).get("primary_signal_semantics", "")) == "weekly_engagement"
    for name, entry in tasks.items():
        if not isinstance(entry, Mapping) or entry.get("status") != "completed":
            continue
        source_keys = {
            "real_train": "train_on_real_test_on_real",
            "synthetic_train": "train_on_synthetic_test_on_real",
            "real_plus_synthetic": "train_on_real_plus_synthetic_test_on_real",
            "tail_targeted_synthetic": "train_on_tail_targeted_synthetic_test_on_real",
        }
        arms = {
            arm_name: _arm_metrics(entry, source_key)
            for arm_name, source_key in source_keys.items()
        }
        arms = {key: value for key, value in arms.items() if value}
        if not arms.get("real_train"):
            continue
        non_estimable = {}
        for arm_name, source_key in source_keys.items():
            raw_arm = entry.get(source_key, {})
            if not isinstance(raw_arm, Mapping):
                continue
            status = raw_arm.get("status")
            if status == "not_estimable_model_collapse":
                class_counts = raw_arm.get("train_class_counts", {})
                non_estimable[arm_name] = {
                    "status": status,
                    "reason_category": raw_arm.get("reason_category"),
                    "reason": raw_arm.get("reason"),
                    "train_class_labels": (
                        sorted(str(label) for label in class_counts)
                        if isinstance(class_counts, Mapping)
                        else []
                    ),
                    "train_rows": raw_arm.get("train_rows"),
                }
        task = {
            "status": (
                "completed_with_model_collapse" if non_estimable else "completed"
            ),
            "arms": arms,
            "test_positive_learners": entry.get("test_positive_learners"),
            "test_negative_learners": entry.get("test_negative_learners"),
            "scope": "pipeline_construct" if weekly else "generated_future_behavior",
        }
        if non_estimable:
            task["non_estimable_arms"] = non_estimable
        if arms.get("synthetic_train"):
            task["synthetic_utility_gap"] = _difference(
                arms["real_train"], arms["synthetic_train"]
            )
        if arms.get("real_plus_synthetic"):
            task["real_plus_synthetic_gain_over_real"] = _gain(
                arms["real_plus_synthetic"], arms["real_train"]
            )
        published[str(name)] = task
    return published


def _conservative_membership_advantage(raw_auc: object) -> float | None:
    value = _number(raw_auc)
    return max(0.0, 2.0 * (value - 0.5)) if value is not None else None


def _privacy_arm(block: object, *, projection_only: bool) -> dict[str, float]:
    if not isinstance(block, Mapping):
        return {}
    if projection_only:
        projected = block.get("projected_pattern_audit", {})
        value = _number(projected.get("exact_duplicate_rate")) if isinstance(projected, Mapping) else None
        return {"behavior_projection_exact_duplicate_rate": value} if value is not None else {}
    result = {}
    for key in ("exact_duplicate_rate", "mean_nearest_neighbor_distance", "near_duplicate_rate"):
        value = _number(block.get(key))
        if value is not None:
            result[key] = value
    advantage = _conservative_membership_advantage(block.get("membership_inference_auc"))
    if advantage is not None:
        result["membership_inference_attack_advantage"] = advantage
    return result


def _privacy_metrics(report: Mapping[str, object], split) -> dict[str, object]:
    bundle = report.get("privacy_memorization", {})
    if not isinstance(bundle, Mapping):
        return {}
    scope = str(
        getattr(split, "metadata", {}).get("publication_privacy_scope", "full_trajectory")
    )
    projection_only = scope == "behavior_projection_exact_duplicates_only"
    standard = _privacy_arm(bundle.get("standard"), projection_only=projection_only)
    targeted = _privacy_arm(bundle.get("tail_targeted"), projection_only=projection_only)
    result: dict[str, object] = {
        "representation_scope": scope,
        "standard": standard,
        "tail_targeted": targeted,
    }
    comparison = {}
    for key in set(standard) & set(targeted):
        if key == "mean_nearest_neighbor_distance":
            # Lower distance means more exposure: standard - targeted is the
            # risk increase caused by targeting.
            comparison["mean_nearest_exposure_risk_delta"] = standard[key] - targeted[key]
        else:
            comparison[f"{key}_risk_delta"] = targeted[key] - standard[key]
    result["tail_targeted_minus_standard_risk"] = comparison
    if projection_only:
        result["note"] = (
            "Nearest-neighbour and membership metrics are omitted because independently "
            "resampled OULAD profiles/outcomes confound the full-record distance."
        )
    return result


def _detectability(report: Mapping[str, object], split) -> tuple[dict[str, float], str | None]:
    postprocessed = getattr(split, "metadata", {}).get("publication_postprocessed_columns", [])
    if postprocessed:
        return {}, "Detector includes independently postprocessed columns; excluded from model comparison."
    block = report.get("distinguishability", {})
    if not isinstance(block, Mapping) or block.get("status") != "completed":
        return {}, "Detector did not complete."
    raw = _number(block.get("raw_auroc"))
    if raw is None:
        raw = _number(block.get("auroc"))
    if raw is None:
        return {}, "Detector AUROC is unavailable."
    return {"detectability_auc": max(raw, 1.0 - raw)}, None


def build_publication_report(raw_report: Mapping[str, object], split) -> dict[str, object]:
    """Return the only metrics intended for paper tables and model ranking."""
    global_block = raw_report.get("global_fidelity", {})
    average = {}
    if isinstance(global_block, Mapping):
        for published_key, source_key in (
            ("primary_signal_rate_error", "correctness_rate_error"),
            ("skill_frequency_js", "skill_frequency_js"),
        ):
            value = _number(global_block.get(source_key))
            if value is not None:
                average[published_key] = value
    marginal, marginal_columns = _modeled_marginal_metrics(raw_report, split)
    average.update({key: value for key, value in marginal.items() if value is not None})

    dependence, dependence_pairs = _dependence_metrics(raw_report, split)
    dependence = {key: value for key, value in dependence.items() if value is not None}
    subgroup, subgroup_reason = _subgroup_metrics(raw_report)
    standard_tail, omitted_standard_tail = _tail_fidelity(
        raw_report.get("tail_fidelity"), split
    )
    targeted_report = raw_report.get("tail_targeted_evaluation", {})
    targeted_tail, omitted_targeted_tail = _tail_fidelity(
        targeted_report.get("tail_fidelity") if isinstance(targeted_report, Mapping) else None,
        split,
    )
    tail_learnability, omitted_tail_learnability = _tail_learnability(raw_report, split)
    detectability, detectability_reason = _detectability(raw_report, split)

    excluded = {
        "fixed_threshold_classification_scores": (
            "Accuracy, precision, recall, and F1 at an untuned 0.5 threshold are diagnostics only."
        ),
        "threshold_metric_deltas": "Only AUPRC/AUROC utility gaps are publication metrics.",
        "lightweight_downstream_utility": "Skill-mean smoke test; superseded by the declared HGB model.",
        "sequence_length_model_ranking": (
            "All generators use the same trajectory-length sampler, so length metrics are pipeline diagnostics."
        ),
        "short_and_long_trajectory_tails": (
            "Shared length sampler prevents attribution to a generator architecture."
        ),
        "diversity_uniqueness_rates": (
            "Novelty is not monotonic quality; retained only in the raw diagnostic report."
        ),
        "raw_membership_auc_inversion": (
            "An AUC below 0.5 is not inverted into privacy leakage; publication advantage is max(0, 2*(AUC-0.5))."
        ),
        "raw_test_fidelity_duplicates": (
            "Held-out real data is reserved for downstream generalization; primary fidelity is against real train."
        ),
        "raw_diagnostic_aggregates": (
            "Unfiltered all-column marginals/NMI, detector accuracy, and minimum-NN extremes are not table metrics."
        ),
        "omitted_standard_tail_groups": omitted_standard_tail,
        "omitted_targeted_tail_groups": omitted_targeted_tail,
        "omitted_tail_learnability_groups": omitted_tail_learnability,
    }
    standard_tail_block = raw_report.get("tail_fidelity", {})
    if isinstance(standard_tail_block, Mapping):
        unavailable = standard_tail_block.get(
            "unavailable_percentile_tail_groups", {}
        )
        if isinstance(unavailable, Mapping) and unavailable:
            excluded["unavailable_percentile_tail_groups"] = dict(unavailable)
    if subgroup_reason:
        excluded["subgroup_fidelity"] = subgroup_reason
    if detectability_reason:
        excluded["distinguishability"] = detectability_reason

    rq1 = {
        "average_fidelity": average,
        "temporal_fidelity": _temporal_metrics(raw_report),
        "cross_signal_dependence": dependence,
        "tail_fidelity": standard_tail,
        "subgroup_fidelity": subgroup,
        "detectability": detectability,
    }
    rq2 = {
        "downstream_utility": _downstream_utility(raw_report),
        "tail_learnability": tail_learnability,
        "tail_targeting_fidelity_gain": _tail_targeting_gain(
            standard_tail, targeted_tail
        ),
        "learner_level_outcome_tasks": _outcome_tasks(raw_report, split),
        "privacy_risk": _privacy_metrics(raw_report, split),
    }
    return {
        "status": "completed",
        "schema_version": SCHEMA_VERSION,
        "research_question_mapping": {
            "rq1": "Average, temporal, dependence, subgroup, and rare-pathway preservation.",
            "rq2": "Held-out-real utility, tail learnability, targeting gains, and privacy risk.",
        },
        "primary_downstream_model": PRIMARY_DOWNSTREAM_MODEL,
        "primary_ranking_metric": PRIMARY_RANKING_METRIC,
        "secondary_ranking_metric": SECONDARY_RANKING_METRIC,
        "direction_contract": {
            "fidelity_errors": "lower_is_better",
            "auprc_auroc": "higher_is_better",
            "utility_gap": "real_minus_candidate; lower_is_better",
            "augmentation_gain": "candidate_minus_reference; higher_is_better",
            "privacy_risk": "higher_is_more_risk_except_distance",
        },
        "metrics": {"rq1": rq1, "rq2": rq2},
        "selection_metadata": {
            "modeled_marginal_columns": marginal_columns,
            "prespecified_dependence_pairs": dependence_pairs,
            "raw_report_retained_for_diagnostics": True,
        },
        "excluded_from_publication": excluded,
    }
