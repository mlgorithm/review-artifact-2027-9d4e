from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data.dataset_scripts.assistments import preprocess as assistments
from data.dataset_scripts.ednet import preprocess as ednet
from data.dataset_scripts.kddcup2010 import preprocess as kddcup2010
from data.dataset_scripts.oulad import preprocess as oulad
from data.dataset_scripts.common import validate_processed_artifacts


class AssistmentsPreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = assistments.load_config(assistments.DEFAULT_CONFIG)

    def test_split_assignment_does_not_depend_on_outcomes(self) -> None:
        first = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3, 4, 4],
                "correct": [0, 0, 0, 0, 1, 1, 1, 1],
            }
        )
        second = first.copy()
        second["correct"] = 1 - second["correct"]

        first_train, _ = assistments.split_by_learner(first, self.config)
        second_train, _ = assistments.split_by_learner(second, self.config)

        self.assertEqual(set(first_train["user_id"]), set(second_train["user_id"]))

    def test_supervised_export_excludes_learner_and_raw_order_ids(self) -> None:
        frame = pd.DataFrame(
            {
                "user_id": [1, 1, 1],
                "order_id": [10, 11, 12],
                "problem_id": [100, 101, 102],
                "correct": [0, 1, 1],
                "attempt_count": [1, 1, 2],
                "hint_count": [0, 1, 0],
                "hint_total": [0, 1, 1],
                "ms_first_response": [1000, 2000, 3000],
            }
        )

        supervised, features = assistments.build_xy_for_split(frame, self.config)

        self.assertNotIn("user_id", features)
        self.assertNotIn("order_id", features)
        self.assertNotIn("user_id", supervised.columns)
        self.assertNotIn("order_id", supervised.columns)
        self.assertIn("prev_correct", features)


class OuladPreprocessingTests(unittest.TestCase):
    def test_all_enrollments_for_student_receive_same_split(self) -> None:
        frame = pd.DataFrame(
            {
                "id_student": [10, 10, 10, 20, 20, 30],
                "code_module": ["AAA", "BBB", "CCC", "AAA", "BBB", "AAA"],
                "code_presentation": ["2013J", "2013J", "2014J", "2013J", "2014J", "2013J"],
            }
        )

        mask = oulad.student_train_mask(frame)

        self.assertTrue(mask.groupby(frame["id_student"]).nunique().eq(1).all())


class PipelineEntrypointTests(unittest.TestCase):
    def test_every_preprocessor_exposes_pipeline_entrypoint(self) -> None:
        for module in (assistments, ednet, kddcup2010, oulad):
            self.assertTrue(callable(getattr(module, "run_preprocessing", None)))

    def test_bounded_samples_do_not_depend_on_source_order(self) -> None:
        ednet_names = [f"KT1/u{index}.csv" for index in range(20)]
        self.assertEqual(
            set(ednet.select_learner_files(ednet_names, 7)),
            set(ednet.select_learner_files(reversed(ednet_names), 7)),
        )

        learner_ids = {f"student_{index}" for index in range(100)}
        first = kddcup2010.choose_learner_splits(learner_ids, 30, 10)
        second = kddcup2010.choose_learner_splits(set(reversed(sorted(learner_ids))), 30, 10)
        self.assertEqual(first, second)
        self.assertFalse(first[0] & first[1])

    def test_kdd_missing_learner_ids_are_not_coalesced(self) -> None:
        self.assertIsNone(kddcup2010.normalize_learner_id(None))
        self.assertIsNone(kddcup2010.normalize_learner_id(float("nan")))
        self.assertIsNone(kddcup2010.normalize_learner_id(""))

    def test_kdd_vectorized_cleaner_preserves_cross_chunk_state(self) -> None:
        rows = [
            {
                "Anon Student Id": "u1",
                "Correct First Attempt": 0,
                "Step Start Time": "2020-01-01 00:00:00",
                "Problem Name": "p1",
                "Step Name": "s1",
                "KC(KTracedSkills)": "k1~~k2",
                "KC(SubSkills)": None,
                "KC(Rules)": None,
                "KC(Default)": None,
                "Incorrects": 0,
                "Hints": 1,
                "Step Duration (sec)": 4,
            },
            {
                "Anon Student Id": "u1",
                "Correct First Attempt": 1,
                "Step Start Time": "2020-01-01 00:02:00",
                "Problem Name": "p2",
                "Step Name": "s2",
                "KC(KTracedSkills)": None,
                "KC(SubSkills)": "sub",
                "KC(Rules)": None,
                "KC(Default)": None,
                "Incorrects": 2,
                "Hints": 0,
                "Step Duration (sec)": 20,
            },
        ]
        frame = pd.DataFrame(rows)
        orders: dict[str, int] = {}
        previous: dict[str, object] = {}

        first = kddcup2010.clean_chunk(frame.iloc[:1], {"u1"}, set(), orders, previous)
        second = kddcup2010.clean_chunk(frame.iloc[1:], {"u1"}, set(), orders, previous)
        combined = pd.concat([first, second], ignore_index=True)

        self.assertEqual(combined["order"].tolist(), [0, 1])
        self.assertEqual(combined["gap_bin"].tolist(), ["start", "1-5m"])
        self.assertEqual(combined["skill_id"].tolist(), ["k1", "sub"])

    def test_kdd_missing_timestamp_does_not_erase_last_valid_time(self) -> None:
        rows = []
        for index, timestamp in enumerate(
            ["2020-01-01 00:00:00", None, "2020-01-01 00:10:00"]
        ):
            rows.append(
                {
                    "Anon Student Id": "u1",
                    "Correct First Attempt": index % 2,
                    "Step Start Time": timestamp,
                    "Problem Name": f"p{index}",
                    "Step Name": f"s{index}",
                    "KC(KTracedSkills)": "k1",
                    "KC(SubSkills)": None,
                    "KC(Rules)": None,
                    "KC(Default)": None,
                    "Incorrects": 0,
                    "Hints": 0,
                    "Step Duration (sec)": 10,
                }
            )

        cleaned = kddcup2010.clean_chunk(
            pd.DataFrame(rows), {"u1"}, set(), {}, {}
        )

        self.assertEqual(cleaned["gap_bin"].tolist(), ["start", "start", "5-60m"])

    def test_kdd_per_learner_cap_keeps_complete_early_prefix(self) -> None:
        rows = [
            {
                "Anon Student Id": "u1",
                "Correct First Attempt": index % 2,
                "Step Start Time": f"2020-01-01 00:0{index}:00",
                "Problem Name": f"p{index}",
                "Step Name": f"s{index}",
                "KC(KTracedSkills)": "k1",
                "KC(SubSkills)": None,
                "KC(Rules)": None,
                "KC(Default)": None,
                "Incorrects": 0,
                "Hints": 0,
                "Step Duration (sec)": 10,
            }
            for index in range(4)
        ]

        cleaned = kddcup2010.clean_chunk(
            pd.DataFrame(rows), {"u1"}, set(), {}, {}, max_interactions_per_learner=2
        )

        self.assertEqual(cleaned["order"].tolist(), [0, 1])

    def test_provenance_guard_rejects_changed_artifacts(self) -> None:
        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "preprocess.py"
            config = root / "config.yaml"
            train = root / "train.csv"
            test = root / "test.csv"
            metadata = root / "metadata.json"
            script.write_text("# preprocessor\n")
            config.write_text("seed: 1\n")
            train.write_text("x\n1\n")
            test.write_text("x\n2\n")
            metadata.write_text(
                json.dumps(
                    {
                        "preprocessing_script_sha256": digest(script),
                        "runtime": {"config_sha256": digest(config)},
                        "train_sha256": digest(train),
                        "test_sha256": digest(test),
                    }
                )
            )

            validate_processed_artifacts(metadata, script, train, test, config)
            train.write_text("x\nchanged\n")
            with self.assertRaises(RuntimeError):
                validate_processed_artifacts(metadata, script, train, test, config)


if __name__ == "__main__":
    unittest.main()
