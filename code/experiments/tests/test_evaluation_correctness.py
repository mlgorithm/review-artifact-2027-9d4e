from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import pipeline
from evaluation import evaluate
from evaluation import general_evaluation
from evaluation import publication as publication_module
from evaluation.downstream_models import (
    _bootstrap_arrays,
    _bootstrap_resample_indices,
    gradient_boosting_pipeline,
    real_vs_synthetic_detection,
    classification_metrics,
)
from evaluation.general_evaluation import _tail_signal_columns
from evaluation.metrics import (
    binary_classification_metrics,
    distribution,
    group_trajectories,
    jensen_shannon,
    normalized_mutual_information,
    sequence_position_curve,
    tail_labels,
    tail_reference,
    total_variation,
    wasserstein_1d,
)
from evaluation.publication import build_publication_report
from scripts.refresh_publication_reports import _sanitize_stored_membership_metrics
from synthetic_generation.generator_scripts.common import sample_trajectory_lengths


def trajectories(prefix: str, rates: list[float]) -> dict[str, list[dict[str, object]]]:
    rows = []
    for learner_index, rate in enumerate(rates):
        for order in range(10):
            rows.append(
                {
                    "learner_id": f"{prefix}_{learner_index}",
                    "order": order,
                    "skill_id": f"skill_{order % 2}",
                    "correct": int(order < rate * 10),
                }
            )
    return group_trajectories(rows, "learner_id", "order")


class EvaluationCorrectnessTests(unittest.TestCase):
    def test_distribution_distance_primitives_have_known_values(self) -> None:
        left = distribution(["a", "a"])
        right = distribution(["b", "b"])

        self.assertEqual(total_variation(left, right), 1.0)
        self.assertEqual(jensen_shannon(left, right), 1.0)
        self.assertEqual(wasserstein_1d([0.0, 2.0], [1.0, 3.0]), 1.0)
        records = [
            {"left": "a", "right": "a"},
            {"left": "b", "right": "b"},
        ]
        self.assertEqual(normalized_mutual_information(records, "left", "right"), 1.0)

    def test_sequence_position_curve_weights_learners_equally(self) -> None:
        trajectories_map = {
            "long": [{"correct": 1} for _ in range(100)],
            "short": [{"correct": 0} for _ in range(2)],
        }

        curve = sequence_position_curve(trajectories_map, "correct")

        self.assertEqual(set(curve), {str(index) for index in range(10)})
        self.assertTrue(all(value == 0.5 for value in curve.values()))

    def test_operational_tail_rules_use_declared_fixed_thresholds(self) -> None:
        def trajectory(
            values,
            *,
            skill="s",
            response_time="30s",
            hint=0,
            attempt=1,
            gap="0s",
        ):
            return [
                {
                    "skill": skill,
                    "correct": value,
                    "response_time": response_time,
                    "hint": hint,
                    "attempt": attempt,
                    "gap": gap,
                }
                for value in values
            ]

        trajectories_map = {
            "recovery": trajectory([0, 0, 1, 1]),
            "late_failure": trajectory([1, 1, 0, 0]),
            "misconception": trajectory([0, 0, 0, 1], skill="hard"),
            "not_misconception": trajectory([0, 0, 1], skill="mixed"),
            "rapid_at_half": trajectory([0, 1, 0, 1], response_time="5s"),
            "fast_but_accurate": trajectory([1, 1, 0, 1], response_time="5s"),
            "persistent_failure": trajectory([0, 0, 0]),
            "high_hint": trajectory([1, 1, 1], hint=3),
            "long_inactivity": trajectory([1, 1, 1], gap="200s"),
            "rare_path": trajectory([1, 1, 1], skill="novel"),
            "common_path": trajectory([1, 1, 1], skill="common"),
        }
        reference = {
            "short_cutoff": None,
            "long_cutoff": None,
            "low_correctness_cutoff": 0.0,
            "low_correctness_cutoff_inclusive": True,
            "high_hint_cutoff": 2.0,
            "high_hint_cutoff_inclusive": True,
            "fast_response_cutoff": 5.0,
            "fast_response_cutoff_inclusive": True,
            "high_gap_cutoff": 100.0,
            "high_gap_cutoff_inclusive": True,
            "rare_path_cutoff": -2.0,
            "rare_path_cutoff_inclusive": True,
            "transition_logp": {"common->common": -1.0},
            "transition_floor_logp": -5.0,
            "has_hint": True,
            "has_response_time": True,
            "has_gap": True,
            "has_dropout": False,
        }

        labels = tail_labels(
            trajectories_map,
            "skill",
            "correct",
            "hint",
            reference,
            attempt_column="attempt",
            response_time_column="response_time",
            gap_column="gap",
        )

        self.assertTrue(labels["recovery"]["recovery"])
        self.assertTrue(labels["late_failure"]["late_failure"])
        self.assertTrue(labels["misconception"]["persistent_misconception"])
        self.assertFalse(labels["not_misconception"]["persistent_misconception"])
        self.assertTrue(labels["rapid_at_half"]["rapid_guessing"])
        self.assertFalse(labels["fast_but_accurate"]["rapid_guessing"])
        self.assertTrue(labels["persistent_failure"]["persistent_failure"])
        self.assertTrue(labels["high_hint"]["high_hint_use"])
        self.assertTrue(labels["long_inactivity"]["long_inactivity"])
        self.assertTrue(labels["rare_path"]["rare_skill_path"])
        self.assertFalse(labels["common_path"]["rare_skill_path"])

    def test_quantile_tail_keeps_extreme_minority_when_cutoff_is_tied(self) -> None:
        trajectories_map = {}
        for learner in range(100):
            trajectories_map[str(learner)] = [
                {
                    "skill": "s",
                    "correct": int(learner != 0),
                    "hint": 3 if learner == 0 else 0,
                    "attempt": 1,
                }
                for _ in range(3)
            ]

        reference = tail_reference(
            trajectories_map, "skill", "correct", "hint", "attempt"
        )
        labels = tail_labels(
            trajectories_map,
            "skill",
            "correct",
            "hint",
            reference,
            "attempt",
        )

        self.assertFalse(reference["high_hint_cutoff_inclusive"])
        self.assertFalse(reference["low_correctness_cutoff_inclusive"])
        self.assertEqual(sum(group["high_hint_use"] for group in labels.values()), 1)
        self.assertEqual(sum(group["persistent_failure"] for group in labels.values()), 1)

    def test_coarse_percentile_bin_is_omitted_when_small_tail_is_impossible(self) -> None:
        trajectories_map = {}
        for learner in range(100):
            gap = "7d+" if learner < 75 else "1-7d"
            trajectories_map[str(learner)] = [
                {"skill": "s", "correct": 1, "gap": gap}
                for _ in range(3)
            ]

        reference = tail_reference(
            trajectories_map, "skill", "correct", gap_column="gap"
        )
        labels = tail_labels(
            trajectories_map,
            "skill",
            "correct",
            None,
            reference,
            gap_column="gap",
        )

        self.assertIsNone(reference["high_gap_cutoff"])
        unavailable = reference["unavailable_percentile_tail_groups"]
        self.assertIn("long_inactivity", unavailable)
        self.assertTrue(all("long_inactivity" not in group for group in labels.values()))

        split = SimpleNamespace(
            train=pd.DataFrame(), columns={}, derived_columns=(), metadata={}
        )
        publication = build_publication_report(
            {
                "tail_fidelity": {
                    "implemented_tail_groups": [],
                    "tail_prevalence_error": {},
                    "tail_shape": {"by_tail_group": {}},
                    "unavailable_percentile_tail_groups": unavailable,
                }
            },
            split,
        )
        self.assertEqual(
            publication["excluded_from_publication"][
                "unavailable_percentile_tail_groups"
            ],
            unavailable,
        )

    def test_single_class_ranking_metrics_are_undefined(self) -> None:
        metrics = classification_metrics([1, 1, 1], [0.8, 0.7, 0.9])
        lightweight = binary_classification_metrics([1, 1, 1], [0.8, 0.7, 0.9])

        self.assertIsNone(metrics["auroc"])
        self.assertIsNone(metrics["auprc"])
        self.assertIsNone(lightweight["auroc"])
        self.assertEqual(metrics["recall"], 1.0)

    def test_performance_subgroups_reuse_real_prefix_cutoffs(self) -> None:
        real = trajectories("real", [0.0, 0.0, 0.5, 0.5, 1.0, 1.0])
        synthetic = trajectories("synthetic", [1.0] * 6)

        report = evaluate.subgroup_fidelity(real, synthetic, "correct", None)

        self.assertEqual(report["partition"], "by_fixed_prefix_performance")
        self.assertEqual(report["performance_reference"]["prefix_length"], 2)
        self.assertNotEqual(
            report["by_subgroup"]["low"]["real"]["learner_share"],
            report["by_subgroup"]["low"]["synthetic"]["learner_share"],
        )

    def test_attribute_subgroups_remain_supported(self) -> None:
        real = trajectories("real", [0.0, 1.0])
        synthetic = trajectories("synthetic", [0.0, 1.0])
        for trajectory in real.values():
            for record in trajectory:
                record["cohort"] = "real_group"
        for trajectory in synthetic.values():
            for record in trajectory:
                record["cohort"] = "synthetic_group"

        report = evaluate.subgroup_fidelity(
            real, synthetic, "correct", None, subgroup_col="cohort"
        )

        self.assertEqual(report["partition"], "by_attribute")
        self.assertEqual(set(report["by_subgroup"]), {"real_group", "synthetic_group"})

    def test_privacy_exact_duplicates_use_full_value_trajectory(self) -> None:
        real = group_trajectories(
            [
                {"learner_id": "r1", "order": 0, "skill": "s", "correct": 1, "item": "a"},
                {"learner_id": "r1", "order": 1, "skill": "s", "correct": 0, "item": "b"},
            ],
            "learner_id",
            "order",
        )
        synthetic = group_trajectories(
            [
                {"learner_id": "s1", "order": 0, "skill": "s", "correct": 1, "item": "x"},
                {"learner_id": "s1", "order": 1, "skill": "s", "correct": 0, "item": "y"},
            ],
            "learner_id",
            "order",
        )

        report = evaluate.privacy_memorization(
            real,
            synthetic,
            "skill",
            "correct",
            None,
            privacy_columns=["skill", "correct", "item"],
        )

        self.assertEqual(report["representation"], "full_value_trajectory")
        self.assertEqual(report["exact_duplicate_rate"], 0.0)
        self.assertEqual(report["projected_pattern_audit"]["exact_duplicate_rate"], 1.0)
        self.assertIn("exact_duplicate_rate_ci", report)
        self.assertIn("membership_inference_attack_advantage", report)

    def test_privacy_comparison_uses_positive_as_more_risk(self) -> None:
        comparison = evaluate.privacy_risk_comparison(
            {
                "exact_duplicate_rate": 0.1,
                "near_duplicate_rate": 0.2,
                "membership_inference_attack_advantage": 0.1,
                "mean_nearest_neighbor_distance": 0.8,
                "by_tail_group": {"groups": {}},
            },
            {
                "exact_duplicate_rate": 0.3,
                "near_duplicate_rate": 0.4,
                "membership_inference_attack_advantage": 0.3,
                "mean_nearest_neighbor_distance": 0.5,
                "by_tail_group": {"groups": {}},
            },
        )

        self.assertAlmostEqual(
            comparison["global"]["exact_duplicate_rate_risk_delta"], 0.2
        )
        self.assertAlmostEqual(
            comparison["global"]["mean_nearest_exposure_risk_delta"], 0.3
        )

    def test_membership_attack_does_not_invert_below_chance_auc(self) -> None:
        report = evaluate._membership_risk(
            [("member",)],
            [("synthetic",)],
            [("synthetic",)],
            seed=3,
        )

        self.assertEqual(report["raw_auc"], 0.0)
        self.assertEqual(report["attack_auc"], 0.5)
        self.assertEqual(report["attack_advantage"], 0.0)

    def test_legacy_membership_refresh_drops_unrepairable_inverted_values(self) -> None:
        report = {
            "privacy_memorization": {
                "standard": {
                    "membership_inference_auc": 0.4,
                    "membership_inference_attack_auc": 0.6,
                    "membership_inference_attack_advantage": 0.2,
                    "membership_inference_attack_advantage_ci": {"ci_low": 0.1},
                    "by_tail_group": {
                        "groups": {
                            "rare": {
                                "membership_inference_auc": 0.45,
                                "membership_inference_attack_advantage": 0.1,
                                "nontail_membership_inference_attack_advantage": 0.2,
                                "privacy_tail_gap": {
                                    "membership_attack_advantage_gap": -0.1,
                                    "near_duplicate_rate_gap": 0.05,
                                },
                            }
                        }
                    },
                },
                "tail_targeted": {"membership_inference_auc": 0.55},
            }
        }

        _sanitize_stored_membership_metrics(report)

        standard = report["privacy_memorization"]["standard"]
        group = standard["by_tail_group"]["groups"]["rare"]
        self.assertEqual(standard["membership_inference_attack_auc"], 0.5)
        self.assertEqual(standard["membership_inference_attack_advantage"], 0.0)
        self.assertNotIn("membership_inference_attack_advantage_ci", standard)
        self.assertNotIn("nontail_membership_inference_attack_advantage", group)
        self.assertNotIn("membership_attack_advantage_gap", group["privacy_tail_gap"])
        self.assertEqual(group["privacy_tail_gap"]["near_duplicate_rate_gap"], 0.05)

    def test_publication_retains_synthetic_outcome_model_collapse(self) -> None:
        split = SimpleNamespace(
            train=pd.DataFrame(
                {"learner_id": ["r"], "order": [0], "skill": ["s"], "correct": [1]}
            ),
            columns={
                "learner": "learner_id",
                "order": "order",
                "skill": "skill",
                "correct": "correct",
            },
            derived_columns=(),
            metadata={},
        )
        raw = {
            "downstream_utility_tasks": {
                "status": "completed",
                "tasks": {
                    "persistent_failure_prediction": {
                        "status": "completed",
                        "test_positive_learners": 12,
                        "test_negative_learners": 20,
                        "train_on_real_test_on_real": {
                            "hist_gradient_boosting": {"auprc": 0.4, "auroc": 0.7}
                        },
                        "train_on_synthetic_test_on_real": {
                            "status": "not_estimable_model_collapse",
                            "reason_category": "synthetic_training_target_degenerate",
                            "reason": "single class",
                            "train_class_counts": {"0": 100},
                            "train_rows": 100,
                        },
                        "train_on_real_plus_synthetic_test_on_real": {
                            "hist_gradient_boosting": {"auprc": 0.42, "auroc": 0.71}
                        },
                        "train_on_tail_targeted_synthetic_test_on_real": {
                            "status": "not_estimable_model_collapse",
                            "reason_category": "synthetic_training_target_degenerate",
                            "reason": "single class",
                            "train_class_counts": {"0": 100},
                            "train_rows": 100,
                        },
                    }
                },
            }
        }

        publication = build_publication_report(raw, split)
        task = publication["metrics"]["rq2"]["learner_level_outcome_tasks"][
            "persistent_failure_prediction"
        ]

        self.assertEqual(task["status"], "completed_with_model_collapse")
        self.assertEqual(
            task["non_estimable_arms"]["synthetic_train"]["status"],
            "not_estimable_model_collapse",
        )
        self.assertEqual(task["arms"]["real_train"]["auprc"], 0.4)
        self.assertNotIn("synthetic_utility_gap", task)

    def test_publication_contract_drops_confounded_oulad_metrics(self) -> None:
        split = SimpleNamespace(
            name="oulad_weekly_engagement",
            train=pd.DataFrame(
                {
                    "learner_id": ["r"],
                    "order": [0],
                    "dominant_activity": ["resource"],
                    "engaged": [1],
                    "gap_bin": ["start"],
                    "course_id": ["AAA_2013J"],
                    "click_bin": ["1-5"],
                    "gender": ["F"],
                    "dropout": [0],
                }
            ),
            columns={
                "learner": "learner_id",
                "order": "order",
                "skill": "dominant_activity",
                "correct": "engaged",
                "gap": "gap_bin",
                "dropout": "dropout",
            },
            derived_columns=("dropout",),
            metadata={
                "primary_signal_semantics": "weekly_engagement",
                "publication_postprocessed_columns": ["gender", "dropout"],
                "publication_privacy_scope": "behavior_projection_exact_duplicates_only",
            },
        )
        shape = {
            "dropout": {
                "correctness_curve_mae": 0.1,
                "real_learners": 20,
                "synthetic_learners": 20,
                "exploratory": False,
            },
            "long_inactivity": {
                "correctness_curve_mae": 0.2,
                "real_learners": 20,
                "synthetic_learners": 20,
                "exploratory": False,
            },
            "long_trajectory": {
                "correctness_curve_mae": 0.3,
                "real_learners": 20,
                "synthetic_learners": 20,
                "exploratory": False,
            },
        }
        raw = {
            "global_fidelity": {
                "correctness_rate_error": 0.1,
                "skill_frequency_js": 0.2,
                "all_value_marginals": {
                    "by_column": {
                        "course_id": {"jensen_shannon_divergence": 0.1},
                        "click_bin": {"jensen_shannon_divergence": 0.3},
                        "gender": {"jensen_shannon_divergence": 0.9},
                        "dropout": {"jensen_shannon_divergence": 0.0},
                    }
                },
            },
            "temporal_fidelity": {"correctness_transition_js": 0.2},
            "cross_signal_dependence": {
                "pairwise_nmi": {
                    "dominant_activity__engaged": {"abs_error": 0.2},
                    "gender__engaged": {"abs_error": 0.9},
                    "dropout__engaged": {"abs_error": 0.8},
                }
            },
            "subgroup_fidelity": {"partition": "by_attribute"},
            "tail_fidelity": {
                "implemented_tail_groups": list(shape),
                "tail_prevalence_error": {
                    "dropout": 0.0,
                    "long_inactivity": 0.2,
                    "long_trajectory": 0.1,
                },
                "tail_prevalence_error_ci": {},
                "tail_shape": {"by_tail_group": shape},
            },
            "tail_targeted_evaluation": {},
            "downstream_utility_models": {
                "status": "completed",
                "train_on_real_test_on_real": {
                    "hist_gradient_boosting": {
                        "auprc": 0.8,
                        "auroc": 0.7,
                        "accuracy": 0.9,
                    }
                },
                "train_on_synthetic_test_on_real": {
                    "hist_gradient_boosting": {
                        "auprc": 0.6,
                        "auroc": 0.65,
                        "accuracy": 0.85,
                    }
                },
                "tail_learnability": {"groups": {}},
            },
            "downstream_utility_tasks": {"status": "completed", "tasks": {}},
            "privacy_memorization": {
                "standard": {
                    "membership_inference_auc": 0.9,
                    "projected_pattern_audit": {"exact_duplicate_rate": 0.1},
                },
                "tail_targeted": {
                    "membership_inference_auc": 0.9,
                    "projected_pattern_audit": {"exact_duplicate_rate": 0.2},
                },
                "comparison": {},
            },
            "distinguishability": {"status": "completed", "auroc": 0.2},
        }

        publication = build_publication_report(raw, split)
        metrics = publication["metrics"]

        self.assertEqual(
            metrics["rq1"]["average_fidelity"]["modeled_value_marginal_js_mean"],
            0.2,
        )
        self.assertEqual(
            metrics["rq1"]["cross_signal_dependence"]["prespecified_signal_nmi_mae"],
            0.2,
        )
        self.assertNotIn("dropout", metrics["rq1"]["tail_fidelity"])
        self.assertNotIn("long_trajectory", metrics["rq1"]["tail_fidelity"])
        self.assertIn("long_inactivity", metrics["rq1"]["tail_fidelity"])
        self.assertEqual(metrics["rq1"]["subgroup_fidelity"], {})
        self.assertEqual(metrics["rq1"]["detectability"], {})
        self.assertNotIn(
            "accuracy",
            metrics["rq2"]["downstream_utility"]["arms"]["real_train"],
        )
        self.assertEqual(
            set(metrics["rq2"]["privacy_risk"]["standard"]),
            {"behavior_projection_exact_duplicate_rate"},
        )

    def test_publication_uses_explicit_oulad_dependence_pairs(self) -> None:
        split = SimpleNamespace(
            train=pd.DataFrame(
                {
                    "dominant_activity": ["resource"],
                    "click_bin": ["1-5"],
                    "active_days_bin": ["1-2"],
                    "engaged": [1],
                    "gender": ["F"],
                }
            ),
            columns={"skill": "dominant_activity", "correct": "engaged"},
            derived_columns=("engaged",),
            metadata={
                "publication_postprocessed_columns": ["gender"],
                "publication_dependence_pairs": [
                    ["dominant_activity", "click_bin"],
                    ["dominant_activity", "active_days_bin"],
                    ["click_bin", "active_days_bin"],
                    ["dominant_activity", "engaged"],
                    ["dominant_activity", "gender"],
                ],
            },
        )
        raw = {
            "cross_signal_dependence": {
                "pairwise_nmi": {
                    "dominant_activity__click_bin": {"abs_error": 0.1},
                    "dominant_activity__active_days_bin": {"abs_error": 0.2},
                    "click_bin__active_days_bin": {"abs_error": 0.3},
                    "dominant_activity__engaged": {"abs_error": 0.0},
                    "dominant_activity__gender": {"abs_error": 0.0},
                }
            }
        }

        metric, pairs = publication_module._dependence_metrics(raw, split)

        self.assertAlmostEqual(metric["prespecified_signal_nmi_mae"], 0.2)
        self.assertEqual(
            pairs,
            [
                "click_bin__active_days_bin",
                "dominant_activity__active_days_bin",
                "dominant_activity__click_bin",
            ],
        )

    def test_publication_retains_prevalence_when_tail_shape_is_not_estimable(self) -> None:
        split = SimpleNamespace(
            train=pd.DataFrame({"learner_id": ["a"]}),
            columns={},
            derived_columns=(),
            metadata={},
        )
        publication = build_publication_report(
            {
                "tail_fidelity": {
                    "implemented_tail_groups": ["persistent_failure"],
                    "tail_prevalence_real": {"persistent_failure": 0.05},
                    "tail_prevalence_error": {"persistent_failure": 0.04},
                    "tail_prevalence_error_ci": {
                        "persistent_failure": {"ci_low": 0.02, "ci_high": 0.06}
                    },
                    "tail_shape": {
                        "by_tail_group": {
                            "persistent_failure": {
                                "correctness_curve_mae": 0.0,
                                "real_learners": 20,
                                "synthetic_learners": 1,
                                "exploratory": True,
                            }
                        }
                    },
                }
            },
            split,
        )

        group = publication["metrics"]["rq1"]["tail_fidelity"][
            "persistent_failure"
        ]
        self.assertEqual(group["prevalence_error"], 0.04)
        self.assertNotIn("primary_signal_curve_mae", group)
        self.assertEqual(group["primary_signal_curve_mae_status"], "not_estimable")

    def test_publication_excludes_groups_above_rare_prevalence_ceiling(self) -> None:
        split = SimpleNamespace(
            train=pd.DataFrame({"learner_id": ["a"]}),
            columns={},
            derived_columns=(),
            metadata={},
        )
        publication = build_publication_report(
            {
                "tail_fidelity": {
                    "implemented_tail_groups": ["persistent_misconception"],
                    "tail_prevalence_real": {"persistent_misconception": 0.30},
                    "tail_prevalence_error": {"persistent_misconception": 0.10},
                    "tail_shape": {
                        "by_tail_group": {
                            "persistent_misconception": {
                                "correctness_curve_mae": 0.10,
                                "real_learners": 30,
                                "synthetic_learners": 20,
                                "exploratory": False,
                            }
                        }
                    },
                }
            },
            split,
        )

        self.assertNotIn(
            "persistent_misconception",
            publication["metrics"]["rq1"]["tail_fidelity"],
        )
        reason = publication["excluded_from_publication"][
            "omitted_standard_tail_groups"
        ]["persistent_misconception"]
        self.assertIn("exceeds the rare-group ceiling", reason)

    def test_tail_targeted_arm_gets_full_size_matched_evaluation(self) -> None:
        frame = pd.DataFrame(
            {
                "learner_id": ["a", "a", "b", "b"],
                "order": [0, 1, 0, 1],
                "skill_id": ["s", "s", "s", "s"],
                "correct": [0, 1, 1, 0],
            }
        )
        standard = frame.assign(learner_id=["std_a", "std_a", "std_b", "std_b"])
        targeted = frame.assign(learner_id=["tail_a", "tail_a", "tail_b", "tail_b"])
        split = SimpleNamespace(
            train=frame,
            test=frame,
            columns={
                "learner": "learner_id",
                "order": "order",
                "skill": "skill_id",
                "correct": "correct",
            },
            ignore_columns=("learner_id", "order"),
            seed=3,
        )
        reports = [
            {"privacy_memorization": {"exact_duplicate_rate": 0.1}},
            {
                "privacy_memorization": {"exact_duplicate_rate": 0.2},
                "tail_fidelity": {"tail_prevalence_mae": 0.05},
            },
        ]
        with patch.object(general_evaluation, "evaluate", side_effect=reports) as evaluate_call, patch.object(
            general_evaluation, "detection_report", return_value={"status": "completed"}
        ), patch.object(general_evaluation, "downstream_report", return_value={"status": "completed"}), patch.object(
            general_evaluation, "learner_level_task_report", return_value={"status": "completed"}
        ):
            report = general_evaluation.run_general_evaluation(
                object(), split, standard, Path("standard.csv"), targeted, Path("targeted.csv")
            )

        self.assertEqual(evaluate_call.call_count, 2)
        standard_call = evaluate_call.call_args_list[0].kwargs
        targeted_call = evaluate_call.call_args_list[1].kwargs
        self.assertEqual(len(standard_call["tail_real_train"]), len(frame))
        self.assertEqual(len(standard_call["tail_real_test"]), len(frame))
        self.assertEqual(len(standard_call["tail_synthetic"]), len(standard))
        self.assertEqual(len(targeted_call["tail_synthetic"]), len(targeted))
        self.assertTrue(report["tail_targeted_size_match"]["matched"])
        self.assertEqual(
            report["tail_targeted_evaluation"]["tail_fidelity"]["tail_prevalence_mae"],
            0.05,
        )
        self.assertNotIn("privacy_memorization", report["tail_targeted_evaluation"])
        self.assertEqual(set(report["privacy_memorization"]), {"standard", "tail_targeted", "comparison"})

    def test_tail_targeted_arm_rejects_unmatched_output_budget(self) -> None:
        frame = pd.DataFrame(
            {"learner_id": ["a", "a"], "order": [0, 1], "skill_id": ["s", "s"], "correct": [0, 1]}
        )
        split = SimpleNamespace(
            train=frame,
            test=frame,
            columns={"learner": "learner_id", "order": "order", "skill": "skill_id", "correct": "correct"},
            ignore_columns=("learner_id", "order"),
            seed=3,
        )
        with patch.object(
            general_evaluation, "evaluate", return_value={"privacy_memorization": {}}
        ), self.assertRaisesRegex(ValueError, "identical row and learner budgets"):
            general_evaluation.run_general_evaluation(
                object(), split, frame, Path("standard.csv"), frame.iloc[:1], Path("targeted.csv")
            )

    def test_cluster_bootstrap_resamples_whole_learners(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 1])
        groups = np.asarray(["a", "a", "b", "b", "b"])
        (prepared,), clusters, rng = _bootstrap_arrays(labels, seed=4, groups=groups)

        indices = _bootstrap_resample_indices(len(prepared), rng, clusters)
        selected_a = int(np.count_nonzero(indices < 2))
        selected_b = int(np.count_nonzero(indices >= 2))

        self.assertEqual(selected_a % 2, 0)
        self.assertEqual(selected_b % 3, 0)

    def test_detection_split_is_learner_disjoint(self) -> None:
        real_groups = np.repeat(["r1", "r2", "r3", "r4"], 3)
        synth_groups = np.repeat(["s1", "s2", "s3", "s4"], 3)
        real = pd.DataFrame({"category": np.tile(["a", "b", "c"], 4), "value": 1.0})
        synthetic = pd.DataFrame({"category": np.tile(["a", "b", "d"], 4), "value": 0.0})

        report = real_vs_synthetic_detection(
            real,
            synthetic,
            seed=3,
            real_groups=real_groups,
            synthetic_groups=synth_groups,
        )

        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["split_unit"], "learner")
        self.assertEqual(report["real_train_learners"] + report["real_test_learners"], 4)
        self.assertEqual(report["synthetic_train_learners"] + report["synthetic_test_learners"], 4)

    def test_hist_gradient_boosting_treats_high_cardinality_as_categorical(self) -> None:
        x = pd.DataFrame(
            {
                "category": [f"category_{index}" for index in range(600)],
                "value": np.arange(600, dtype=float),
            }
        )
        y = pd.Series(np.arange(600) % 2)
        model = gradient_boosting_pipeline(x, seed=1)

        model.fit(x, y)
        scores = model.predict_proba(x.iloc[:5])[:, 1]

        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(model.named_steps["model"].is_categorical_[0])

        mixed = pd.concat([x.iloc[:10], x.iloc[:10].assign(category=np.arange(10, dtype=float))])
        mixed_model = gradient_boosting_pipeline(mixed, seed=2)
        mixed_model.fit(mixed, pd.Series(np.arange(len(mixed)) % 2))

    def test_tail_shape_override_keeps_standard_output_budget(self) -> None:
        rows = []
        for learner, length in (("tail_a", 8), ("tail_b", 6), ("other", 4)):
            rows.extend({"learner_id": learner, "order": order} for order in range(length))
        oversampled = pd.DataFrame(rows + [dict(row, learner_id=f"repeat_{row['learner_id']}") for row in rows[:8]])

        lengths, metadata = sample_trajectory_lengths(
            oversampled,
            seed=7,
            generation_shape={"target_rows": 18, "learner_count": 3},
        )

        self.assertEqual(sum(lengths), 18)
        self.assertEqual(len(lengths), 3)
        self.assertTrue(metadata["shape_override"])
        self.assertGreater(metadata["fitted_learner_count"], metadata["learner_count"])

    def test_tail_signal_resolver_finds_duration_and_gap(self) -> None:
        frame = pd.DataFrame(
            {
                "learner_id": ["a"],
                "order": [0],
                "skill_id": ["s"],
                "correct": [1],
                "duration_bin": ["5-15s"],
                "gap_bin": ["start"],
            }
        )
        split = SimpleNamespace(
            train=frame,
            columns={
                "learner": "learner_id",
                "order": "order",
                "skill": "skill_id",
                "correct": "correct",
            },
        )

        signals = _tail_signal_columns(split)

        self.assertEqual(signals["response_time"], "duration_bin")
        self.assertEqual(signals["gap"], "gap_bin")
        self.assertIsNone(signals["hint"])

        split.ignore_columns = ("learner_id", "order")
        split.derived_columns = ()
        with patch.object(pipeline, "tail_reference", return_value={}) as reference, patch.object(
            pipeline,
            "tail_labels",
            return_value={
                "a": {"long_inactivity": True},
                **{
                    f"non_tail_{index}": {"long_inactivity": False}
                    for index in range(19)
                },
            },
        ):
            targeted = pipeline.tail_oversampled_generation_train(split, oversample=2)

        self.assertIsNotNone(targeted)
        self.assertEqual(reference.call_args.args[5], "duration_bin")
        self.assertEqual(reference.call_args.args[6], "gap_bin")

    def test_unavailable_hint_and_attempt_metrics_are_omitted(self) -> None:
        def records(prefix):
            return [
                {
                    "learner_id": f"{prefix}_{learner}",
                    "order": order,
                    "skill_id": f"skill_{order % 2}",
                    "correct": (learner + order) % 2,
                }
                for learner in range(3)
                for order in range(3)
            ]

        report = evaluate.evaluate(
            records("train"),
            records("test"),
            records("synthetic"),
            {
                "learner": "learner_id",
                "order": "order",
                "skill": "skill_id",
                "correct": "correct",
                "hint": "hint",
                "attempts": "attempts",
            },
        )

        self.assertNotIn("hint_rate_error", report["global_fidelity"])
        self.assertNotIn("attempt_count_wasserstein", report["global_fidelity"])
        self.assertNotIn("hint_transition_js", report["temporal_fidelity"])
        self.assertNotIn("hint_rate_mae_across_subgroups", report["subgroup_fidelity"])

    def test_metric_summary_excludes_ci_metadata_and_reports_coverage(self) -> None:
        payload = {
            "score": 0.7,
            "generation_seed": 20260703,
            "delta_tail_ci": {"model": {"auroc": {"ci_low": 0.1, "ci_high": 0.9, "n_boot": 200}}},
            "privacy": {"near_duplicate_threshold": 0.1},
        }
        flattened = pipeline.flatten_numeric(payload)
        summary = pipeline.summarize_metrics(
            [{"_metrics": flattened}, {"_metrics": {}}, {"_metrics": {"score": 0.9}}]
        )

        self.assertEqual(flattened, {"score": 0.7})
        self.assertEqual(summary["score"]["n"], 2)
        self.assertEqual(summary["score"]["missing_runs"], 1)

    def test_publication_status_summary_retains_model_collapse_by_seed(self) -> None:
        path = "rq2.learner_level_outcome_tasks.failure.non_estimable_arms.synthetic_train"
        reports = [
            {
                "seed": 11,
                "_publication": {
                    "rq2": {
                        "learner_level_outcome_tasks": {
                            "failure": {
                                "status": "completed_with_model_collapse",
                                "non_estimable_arms": {
                                    "synthetic_train": {
                                        "status": "not_estimable_model_collapse"
                                    }
                                },
                            }
                        }
                    }
                },
            },
            {
                "seed": 22,
                "_publication": {
                    "rq2": {
                        "learner_level_outcome_tasks": {
                            "failure": {
                                "status": "completed_with_model_collapse",
                                "non_estimable_arms": {
                                    "synthetic_train": {
                                        "status": "not_estimable_model_collapse"
                                    }
                                },
                            }
                        }
                    }
                },
            },
        ]

        summary = pipeline.summarize_publication_statuses(reports)

        self.assertEqual(
            summary[path]["counts"]["not_estimable_model_collapse"], 2
        )
        self.assertEqual(
            summary[path]["seeds_by_status"]["not_estimable_model_collapse"],
            [11, 22],
        )


if __name__ == "__main__":
    unittest.main()
