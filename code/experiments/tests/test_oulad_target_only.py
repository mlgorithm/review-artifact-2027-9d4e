from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from scripts import regenerate_oulad_tail_targeted as target_only


def completed_evaluation_report() -> dict[str, object]:
    return {
        "global_fidelity": {},
        "temporal_fidelity": {},
        "tail_fidelity": {},
        "cross_signal_dependence": {},
        "subgroup_fidelity": {},
        "privacy_memorization": {
            "standard": {},
            "tail_targeted": {},
            "comparison": {},
        },
        "diversity_coverage": {},
        "downstream_utility_models": {
            "status": "completed",
            "tail_learnability": {"status": "completed"},
        },
        "downstream_utility_tasks": {"status": "completed", "tasks": {}},
        "distinguishability": {"status": "completed"},
        "publication": {
            "status": "completed",
            "schema_version": target_only.pipeline_module.SCHEMA_VERSION,
            "metrics": {
                "rq1": {"fidelity": {"example_distance": 0.1}},
                "rq2": {"tail": {"example_gap": 0.2}},
            },
        },
        "tail_targeted_size_match": {"matched": True},
        "tail_targeted_evaluation": {},
    }


class OuladTargetOnlyTests(unittest.TestCase):
    def test_run_seed_refits_only_targeted_generator(self) -> None:
        frame = pd.DataFrame(
            {
                "learner_id": ["learner_a", "learner_b"],
                "order": [0, 0],
                "engaged": [1, 0],
            }
        )
        split = SimpleNamespace(
            name="oulad_weekly_engagement",
            train=frame,
            columns={"learner": "learner_id", "order": "order"},
            metadata={"artifact_metadata": {"train_sha256": "train"}},
        )
        tail_contract = {
            "schema_version": "tail_selection_v2",
            "status": "completed",
            "selected_tail_learners": 1,
        }
        calls: list[bool] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_path = root / "raw.csv"
            standard_path = root / "standard.csv.gz"
            targeted_path = root / "targeted.csv.gz"

            def fake_generate(*args, **kwargs):
                frame.to_csv(raw_path, index=False)
                return frame.copy(), raw_path, {
                    "model": "markov_ngram",
                    "seed": 7,
                    "synthetic_rows": len(frame),
                    "synthetic_output_sha256": "hash",
                }

            def fake_reprocess(split_arg, model, seed, targeted, fitted, **kwargs):
                calls.append(targeted)
                path = targeted_path if targeted else standard_path
                path.write_text("artifact", encoding="utf-8")
                return frame.copy(), path, {"arm": "tail_targeted" if targeted else "standard"}

            with patch.object(target_only, "TARGET_BASE_ROOT", root / "raw_root"), patch.object(
                target_only, "OUTPUT_ROOT", root / "output_root"
            ), patch.object(target_only, "REPORT_ROOT", root / "report_root"), patch.object(
                target_only, "_validate_standard_base", return_value={"sha256": "standard"}
            ), patch.object(
                target_only, "_target_training", return_value=(frame.copy(), tail_contract)
            ), patch.object(
                target_only.pipeline_module, "_generate_synthetic", side_effect=fake_generate
            ) as generate, patch.object(
                target_only.outcome_script, "reprocess_arm", side_effect=fake_reprocess
            ), patch.object(
                target_only.evaluation_module,
                "run_general_evaluation",
                return_value=completed_evaluation_report(),
            ), patch.object(
                target_only, "_source_fingerprint", return_value={"fingerprint": "fixed"}
            ), patch.object(
                target_only.pipeline_module, "sha256_file", return_value="hash"
            ):
                result = target_only.run_seed(
                    split,
                    "markov_ngram",
                    7,
                    fitted=(object(), {}),
                    oversample=3,
                    strict=True,
                    resume=False,
                )

        self.assertEqual(generate.call_count, 1)
        self.assertEqual(calls, [True, False])
        self.assertFalse(result["standard_generator_retrained"])
        self.assertTrue(result["target_generator_retrained"])
        self.assertEqual(result["tail_targeting_contract"], tail_contract)

    def test_preflight_never_invokes_a_generator(self) -> None:
        split = SimpleNamespace()
        contract = {"schema_version": "tail_selection_v2", "status": "completed"}
        with patch.object(target_only.adapter, "load_split", return_value=split), patch.object(
            target_only, "_target_training", return_value=(pd.DataFrame(), contract)
        ), patch.object(
            target_only, "_validate_standard_base", return_value={"sha256": "ok"}
        ), patch.object(
            target_only.pipeline_module,
            "_generate_synthetic",
            side_effect=AssertionError("generator must not run during preflight"),
        ) as generate:
            result = target_only.preflight(["markov_ngram"], [7], 3)

        generate.assert_not_called()
        self.assertFalse(result["generator_invoked"])
        self.assertFalse(result["artifacts_written"])


if __name__ == "__main__":
    unittest.main()
