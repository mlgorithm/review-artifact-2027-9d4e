from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

import pipeline
from scripts.run_final_experiments import neural_fairness_issues
from data.dataset_scripts.oulad_weekly import dataset as oulad_weekly_dataset
from data.dataset_scripts.oulad_weekly import preprocess as oulad_weekly
from evaluation.evaluate import _nearest_distances
from evaluation.general_evaluation import _core_evaluation_sample, _primary_downstream_task
from evaluation.metrics import group_trajectories, hamming_like_distance, tail_labels, tail_reference
from evaluation.outcome_tasks import _feature_frame, _xy


def _valid_strict_report() -> dict[str, object]:
    report = {
        key: {}
        for key in (
            "global_fidelity",
            "temporal_fidelity",
            "tail_fidelity",
            "cross_signal_dependence",
            "subgroup_fidelity",
            "diversity_coverage",
        )
    }
    report.update(
        {
            "privacy_memorization": {
                "standard": {},
                "tail_targeted": {},
                "comparison": {},
            },
            "downstream_utility_models": {
                "status": "completed",
                "tail_learnability": {"status": "completed"},
            },
            "downstream_utility_tasks": {"status": "completed", "tasks": {}},
            "distinguishability": {"status": "completed"},
            "publication": {
                "status": "completed",
                "schema_version": pipeline.SCHEMA_VERSION,
                "metrics": {"rq1": {"metric": 0.1}, "rq2": {"metric": 0.2}},
            },
            "tail_targeted_size_match": {"matched": True},
            "tail_targeted_evaluation": {},
        }
    )
    return report


class FinalReadinessTests(unittest.TestCase):
    def test_primary_task_name_matches_dataset_semantics(self) -> None:
        weekly = SimpleNamespace(
            metadata={"primary_signal_semantics": "weekly_engagement"}
        )
        interaction = SimpleNamespace(
            metadata={"primary_signal_semantics": "response_correctness"}
        )
        self.assertEqual(_primary_downstream_task(weekly), "next_week_engagement")
        self.assertEqual(
            _primary_downstream_task(interaction), "next_response_correctness"
        )

    def test_neural_fairness_guard_rejects_configuration_drift(self) -> None:
        environment = {
            "VAE_WINDOW": "20",
            "TIMEGAN_WINDOW": "20",
            "VAE_MAX_VOCAB": "100",
            "TIMEGAN_MAX_VOCAB": "100",
            "VAE_BATCH": "200",
            "TIMEGAN_BATCH": "200",
            "VAE_TARGET_STEPS": "4000",
            "TIMEGAN_TARGET_STEPS": "4000",
            "VAE_EPOCHS": "50",
            "TIMEGAN_N_ITER": "50",
        }
        self.assertEqual(neural_fairness_issues(environment), [])

        environment["VAE_MAX_VOCAB"] = "101"
        issues = neural_fairness_issues(environment)

        self.assertTrue(any("vocabulary cap differs" in issue for issue in issues))

    def test_oulad_weekly_withholds_terminal_outcomes_from_generation(self) -> None:
        train = pd.DataFrame(
            {
                "learner_id": ["real"],
                "order": [0],
                "course_id": ["AAA_2013J"],
                "engaged": [1],
                "dropout": [0],
                "failure": [0],
                "final_result": ["Pass"],
                "gender": ["F"],
            }
        )
        split = SimpleNamespace(
            train=train,
            derived_columns=oulad_weekly_dataset.derived_columns(),
        )

        generation_frame = pipeline.generation_train(split)

        self.assertFalse({"dropout", "failure", "final_result"} & set(generation_frame.columns))
        self.assertIn("gender", generation_frame.columns)

    def test_oulad_downstream_features_exclude_independently_sampled_profiles(self) -> None:
        frame = pd.DataFrame(
            {
                "learner_id": ["a", "a", "b", "b"],
                "order": [0, 1, 0, 1],
                "course_id": ["AAA_2013J"] * 4,
                "engaged": [0, 1, 1, 0],
                "registered": [1, 1, 1, 1],
                "gender": ["F", "F", "M", "M"],
                "age_band": ["0-35", "0-35", "35-55", "35-55"],
                "imd_band": ["0-10%", "0-10%", "90-100%", "90-100%"],
                "disability": ["N", "N", "Y", "Y"],
                "highest_education": ["A Level", "A Level", "HE", "HE"],
                "region": ["North", "North", "South", "South"],
                "studied_credits": [60, 60, 120, 120],
                "num_of_prev_attempts": [0, 0, 1, 1],
                "dropout": [0, 0, 1, 1],
                "failure": [0, 0, 0, 0],
                "final_result": ["Pass", "Pass", "Withdrawn", "Withdrawn"],
            }
        )

        supervised = oulad_weekly_dataset.generation_to_downstream_supervised(frame)

        excluded = set(oulad_weekly_dataset.PROFILE_COLUMNS) | {
            "dropout",
            "failure",
            "final_result",
        }
        self.assertFalse(excluded & set(supervised.columns))
        self.assertEqual(supervised["engaged"].tolist(), [1, 0])

    def test_oulad_weekly_derives_terminal_outcomes_after_generation(self) -> None:
        generated = pd.DataFrame(
            {
                "learner_id": ["short", "complete_low", "complete_high"],
                "course_id": ["AAA_2013J"] * 3,
                "dropout": [1, 0, 0],
                "engaged": [0, 0, 1],
            }
        )
        reference = pd.DataFrame(
            {
                "learner_id": ["r_fail", "r_pass"],
                "course_id": ["AAA_2013J"] * 2,
                "dropout": [0, 0],
                "failure": [1, 0],
                "final_result": ["Fail", "Pass"],
            }
        )

        restored = oulad_weekly_dataset._assign_terminal_outcomes(generated, reference, seed=1)
        outcomes = restored.set_index("learner_id")[["failure", "final_result"]]

        self.assertEqual(outcomes.loc["short"].to_dict(), {"failure": 0, "final_result": "Withdrawn"})
        self.assertEqual(outcomes.loc["complete_low"].to_dict(), {"failure": 1, "final_result": "Fail"})
        self.assertEqual(outcomes.loc["complete_high"].to_dict(), {"failure": 0, "final_result": "Pass"})

    def test_oulad_behavioral_dropout_uses_future_generated_behavior_only(self) -> None:
        rows = []
        for learner_index in range(8):
            withdrew = learner_index < 4
            for order in range(8):
                rows.append(
                    {
                        "learner_id": f"real_{learner_index}",
                        "order": order,
                        "course_id": "AAA_2013J",
                        "engaged": int(not withdrew) if order >= 4 else learner_index % 2,
                        "dropout": int(withdrew),
                        "failure": int(not withdrew and learner_index in {4, 5}),
                        "final_result": (
                            "Withdrawn"
                            if withdrew
                            else ("Fail" if learner_index in {4, 5} else "Pass")
                        ),
                    }
                )
        reference = pd.DataFrame(rows)
        fitted = oulad_weekly_dataset.fit_dropout_behavior_model(reference)

        synthetic_rows = []
        for learner, future_engagement, prefix_engagement in (
            ("synthetic_low", 0, 1),
            ("synthetic_high", 1, 0),
        ):
            for order in range(8):
                synthetic_rows.append(
                    {
                        "learner_id": learner,
                        "order": order,
                        "course_id": "AAA_2013J",
                        "engaged": future_engagement if order >= 4 else prefix_engagement,
                        "dropout": 0,
                        "failure": 0,
                        "final_result": "Pass",
                    }
                )
        synthetic = pd.DataFrame(synthetic_rows)
        reprocessed, metadata = oulad_weekly_dataset.reassign_terminal_outcomes(
            synthetic, reference, seed=7, fitted=fitted
        )
        outcomes = reprocessed.groupby("learner_id", sort=False)["dropout"].first()

        self.assertEqual(outcomes.to_dict(), {"synthetic_low": 1, "synthetic_high": 0})
        self.assertNotIn("trajectory_length", metadata["risk_features"])
        self.assertIn("trajectory_length", metadata["explicitly_excluded"])
        self.assertIn("prediction_prefix_behavior", metadata["explicitly_excluded"])
        self.assertTrue(
            synthetic.drop(columns=["dropout", "failure", "final_result"]).equals(
                reprocessed.drop(columns=["dropout", "failure", "final_result"])
            )
        )

    def test_oulad_behavioral_dropout_is_deterministic(self) -> None:
        rows = []
        for learner_index in range(6):
            for order in range(6):
                rows.append(
                    {
                        "learner_id": f"learner_{learner_index}",
                        "order": order,
                        "course_id": "AAA_2013J",
                        "engaged": int(learner_index >= 3 and order >= 4),
                        "dropout": int(learner_index < 3),
                        "failure": 0,
                        "final_result": "Withdrawn" if learner_index < 3 else "Pass",
                    }
                )
        frame = pd.DataFrame(rows)

        first, first_metadata = oulad_weekly_dataset.reassign_terminal_outcomes(
            frame, frame, seed=11
        )
        second, second_metadata = oulad_weekly_dataset.reassign_terminal_outcomes(
            frame, frame, seed=11
        )

        self.assertTrue(first.equals(second))
        self.assertEqual(first_metadata, second_metadata)

    def test_vectorized_nearest_distance_matches_scalar_definition(self) -> None:
        references = [("a", "b"), ("a", "c", "d"), tuple()]
        queries = [("a", "b"), ("x",), tuple(), ("a", "c", "x")]

        expected = [
            min(hamming_like_distance(query, reference) for reference in references)
            for query in queries
        ]

        self.assertEqual(_nearest_distances(queries, references), expected)

    def test_core_sample_never_splits_a_learner_trajectory(self) -> None:
        frame = pd.DataFrame(
            {
                "learner_id": np.repeat([f"l{index}" for index in range(20)], 5),
                "order": np.tile(np.arange(5), 20),
            }
        )
        sampled = _core_evaluation_sample(frame, "learner_id", seed=2, max_learners=7, max_rows=35)

        self.assertLessEqual(sampled["learner_id"].nunique(), 7)
        self.assertTrue(sampled.groupby("learner_id").size().eq(5).all())

    def test_dropout_tail_requires_and_uses_explicit_outcome(self) -> None:
        trajectories = group_trajectories(
            [
                {"learner_id": learner, "order": order, "skill": "s", "correct": 1, "dropout": outcome}
                for learner, outcome in (("completed", 0), ("withdrew", 1))
                for order in range(4)
            ],
            "learner_id",
            "order",
        )
        reference = tail_reference(
            trajectories, "skill", "correct", None, dropout_column="dropout"
        )
        labels = tail_labels(
            trajectories, "skill", "correct", None, reference, dropout_column="dropout"
        )

        self.assertFalse(labels["completed"]["dropout"])
        self.assertTrue(labels["withdrew"]["dropout"])

    def test_oulad_weekly_keeps_zero_weeks_before_but_not_after_withdrawal(self) -> None:
        enrollment = pd.DataFrame(
            {
                "learner_id": ["AAA_2013J_1"],
                "course_id": ["AAA_2013J"],
                "module_id": ["AAA"],
                "presentation_id": ["2013J"],
                "module_presentation_length": [35],
                "date_unregistration": [21],
                "final_result": ["Withdrawn"],
                "gender": ["F"],
                "region": ["Region"],
                "highest_education": ["A Level"],
                "imd_band": ["20-30%"],
                "age_band": ["0-35"],
                "disability": ["N"],
                "num_of_prev_attempts": [0],
                "studied_credits": [60],
            }
        )
        activity = pd.DataFrame(
            {
                "learner_id": ["AAA_2013J_1"],
                "order": [0],
                "weekly_clicks": [4],
                "active_days": [1],
                "dominant_activity": ["resource"],
            }
        )

        panel = oulad_weekly.build_weekly_panel(enrollment, activity)

        self.assertEqual(panel["order"].tolist(), [0, 1, 2])
        self.assertEqual(panel["engaged"].tolist(), [1, 0, 0])
        self.assertTrue(panel["dropout"].eq(1).all())

    def test_explicit_outcome_features_never_include_outcome_columns(self) -> None:
        def frame(outcome):
            return pd.DataFrame(
                {
                    "learner_id": ["l"] * 4,
                    "order": range(4),
                    "skill_id": ["s"] * 4,
                    "correct": [0, 1, 0, 1],
                    "dropout": [outcome] * 4,
                    "failure": [0] * 4,
                    "final_result": ["Withdrawn" if outcome else "Pass"] * 4,
                    "course_id": ["c"] * 4,
                }
            )

        split = SimpleNamespace(
            columns={"learner": "learner_id", "order": "order", "skill": "skill_id", "correct": "correct"},
            target="correct",
            metadata={"outcome_prefix_length": 4, "outcome_static_features": ["course_id"]},
        )
        completed = _feature_frame(frame(0), split, {"name": "dropout", "target": "dropout"})
        withdrew = _feature_frame(frame(1), split, {"name": "dropout", "target": "dropout"})
        completed_x, completed_y = _xy(completed)
        withdrew_x, withdrew_y = _xy(withdrew)

        self.assertTrue(completed_x.equals(withdrew_x))
        self.assertEqual(completed_y.tolist(), [0])
        self.assertEqual(withdrew_y.tolist(), [1])
        self.assertFalse({"dropout", "failure", "final_result"} & set(completed_x.columns))

    def test_strict_final_audit_rejects_missing_targeted_arm(self) -> None:
        issues = pipeline.final_report_issues({}, require_tail_targeted=True)
        self.assertTrue(any("tail-targeted" in issue for issue in issues))

    def test_strict_audit_accepts_explicit_synthetic_model_collapse(self) -> None:
        report = _valid_strict_report()
        report["downstream_utility_tasks"]["tasks"] = {
            "persistent_failure_prediction": {
                "train_on_synthetic_test_on_real": {
                    "status": "not_estimable_model_collapse",
                    "train_class_counts": {"0": 10},
                }
            }
        }

        self.assertEqual(pipeline.final_report_issues(report), [])

    def test_strict_audit_rejects_skipped_or_empty_publication_results(self) -> None:
        skipped_tail = _valid_strict_report()
        skipped_tail["downstream_utility_models"]["tail_learnability"] = {
            "status": "skipped"
        }
        self.assertTrue(
            any("tail-learnability" in issue for issue in pipeline.final_report_issues(skipped_tail))
        )

        skipped_tasks = _valid_strict_report()
        skipped_tasks["downstream_utility_tasks"] = {"status": "skipped"}
        self.assertTrue(
            any("outcome task" in issue for issue in pipeline.final_report_issues(skipped_tasks))
        )

        empty_publication = _valid_strict_report()
        empty_publication["publication"]["metrics"] = {"rq1": {}, "rq2": {}}
        issues = pipeline.final_report_issues(empty_publication)
        self.assertTrue(any("empty for RQ1" in issue for issue in issues))
        self.assertTrue(any("empty for RQ2" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
