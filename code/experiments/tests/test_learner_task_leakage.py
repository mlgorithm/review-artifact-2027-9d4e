from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from evaluation import learner_tasks


SIGNALS = {
    "skill": "skill_id",
    "correct": "correct",
    "hint": "hint_bin",
    "attempt": None,
    "response_time": None,
    "gap": None,
}
REFERENCE = {"low_correctness_cutoff": 0.25}


def trajectory(correctness: list[int]) -> list[dict[str, object]]:
    return [
        {"skill_id": f"skill_{index % 2}", "correct": value, "hint_bin": 0}
        for index, value in enumerate(correctness)
    ]


class LearnerTaskLeakageTests(unittest.TestCase):
    def test_synthetic_single_class_is_recorded_as_model_collapse(self) -> None:
        train = pd.DataFrame(
            {
                "first_skill": ["a", "b", "c"],
                "feature": [0.1, 0.2, 0.3],
                "__label__persistent_failure_prediction": [0, 0, 0],
            }
        )
        test = pd.DataFrame(
            {
                "first_skill": ["a", "b"],
                "feature": [0.1, 0.9],
                "__label__persistent_failure_prediction": [0, 1],
            }
        )

        result = learner_tasks._arm_metrics(
            train,
            test,
            "persistent_failure_prediction",
            "first_skill",
            seed=1,
            allow_model_collapse=True,
        )

        self.assertEqual(result["overall"]["status"], "not_estimable_model_collapse")
        self.assertEqual(result["overall"]["train_class_counts"], {"0": 3})
        self.assertEqual(result["overall"]["test_class_counts"], {"0": 1, "1": 1})

    def test_real_single_class_remains_a_strict_failure(self) -> None:
        frame = pd.DataFrame(
            {
                "first_skill": ["a", "b"],
                "feature": [0.1, 0.2],
                "__label__persistent_failure_prediction": [0, 0],
            }
        )

        result = learner_tasks._arm_metrics(
            frame,
            frame,
            "persistent_failure_prediction",
            "first_skill",
            seed=1,
        )

        self.assertEqual(result["overall"]["status"], "failed")

    def test_same_prefix_has_same_features_when_future_outcome_changes(self) -> None:
        failing = learner_tasks._feature_frame(
            {"learner": trajectory([0, 0, 0, 0, 0, 0])}, SIGNALS, REFERENCE
        )
        recovered = learner_tasks._feature_frame(
            {"learner": trajectory([0, 0, 1, 1, 1, 1])}, SIGNALS, REFERENCE
        )

        failing_x, failing_y = learner_tasks._split_xy(failing, "persistent_failure_prediction")
        recovered_x, recovered_y = learner_tasks._split_xy(recovered, "persistent_failure_prediction")

        self.assertTrue(failing_x.equals(recovered_x))
        self.assertEqual(failing_y.tolist(), [1])
        self.assertEqual(recovered_y.tolist(), [0])
        self.assertNotIn("n_early_interactions", failing_x.columns)

    def test_recovery_label_uses_future_with_early_failure_as_eligibility(self) -> None:
        frame = learner_tasks._feature_frame(
            {
                "recovered": trajectory([0, 0, 1, 1, 1, 1]),
                "not_recovered": trajectory([0, 0, 0, 0, 0, 0]),
                "not_eligible": trajectory([1, 1, 1, 1, 1, 1]),
            },
            SIGNALS,
            REFERENCE,
        )

        x, y = learner_tasks._split_xy(frame, "recovery_prediction")

        self.assertEqual(len(x), 2)
        self.assertEqual(sorted(y.tolist()), [0, 1])
        self.assertFalse(any(column.startswith("__") for column in x.columns))

    def test_report_names_future_outcomes_without_claiming_tail_group_identity(self) -> None:
        rows = []
        patterns = [[0, 0, 0, 0, 0, 0]] * 12 + [[0, 0, 1, 1, 1, 1]] * 12
        for learner_index, correctness in enumerate(patterns):
            for order, value in enumerate(correctness):
                rows.append(
                    {
                        "learner_id": f"learner_{learner_index}",
                        "order": order,
                        "skill_id": f"skill_{order % 2}",
                        "correct": value,
                        "hint_bin": 0,
                    }
                )
        frame = pd.DataFrame(rows)
        split = SimpleNamespace(
            train=frame,
            test=frame.copy(),
            columns={
                "learner": "learner_id",
                "order": "order",
                "skill": "skill_id",
                "correct": "correct",
                "hint": "hint_bin",
            },
            seed=1,
        )
        placeholder_metrics = {
            "overall": {
                "model": {"auroc": 0.5, "auprc": 0.5, "f1": 0.5, "accuracy": 0.5}
            },
            "scores": None,
            "y": pd.Series(dtype=int),
        }

        with patch.object(learner_tasks, "_arm_metrics", return_value=placeholder_metrics):
            report = learner_tasks.learner_level_task_report(split, frame)

        persistent = report["tasks"]["persistent_failure_prediction"]
        recovery = report["tasks"]["recovery_prediction"]
        self.assertNotIn("predicts_tail_group", persistent)
        self.assertNotIn("predicts_tail_group", recovery)
        self.assertEqual(persistent["predicted_outcome"], "future_low_correctness")
        self.assertEqual(
            recovery["predicted_outcome"], "future_high_correctness_after_prefix_failure"
        )
        self.assertIn("distinct", persistent["relationship_to_tail_analysis"])


if __name__ == "__main__":
    unittest.main()
