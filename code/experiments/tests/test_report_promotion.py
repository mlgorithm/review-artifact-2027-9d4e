from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import promote_final_reports


class FinalReportPromotionTests(unittest.TestCase):
    def test_promotes_only_audited_reports_and_portabilizes_paths(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=promote_final_reports.ROOT
        ) as temporary_directory:
            root = Path(temporary_directory)
            source_root = root / "runs"
            target_root = root / "reports"
            dataset = "dataset"
            model = "model"
            seed = 20260703
            seed_path = (
                source_root
                / dataset
                / model
                / f"seed_{seed}"
                / "generation_evaluation_report.json"
            )
            seed_path.parent.mkdir(parents=True)
            absolute_artifact = promote_final_reports.ROOT / "experiments" / "outputs" / "x.csv"
            standard_artifact = root / "standard" / "synth_generation.csv"
            targeted_artifact = root / "tail_targeted" / "synth_generation.csv"
            standard_artifact.parent.mkdir(parents=True)
            targeted_artifact.parent.mkdir(parents=True)
            standard_artifact.write_text("value\nstandard\n", encoding="utf-8")
            targeted_artifact.write_text("value\ntargeted\n", encoding="utf-8")
            standard_metadata = {
                "seed": seed,
                "model": model,
                "synthetic_output_sha256": promote_final_reports._sha256(
                    standard_artifact
                ),
            }
            targeted_metadata = {
                "seed": seed,
                "model": model,
                "generation_arm": "tail_targeted",
                "synthetic_output_sha256": promote_final_reports._sha256(
                    targeted_artifact
                ),
            }
            seed_path.write_text(
                json.dumps(
                    {
                        "experiment_design": "standard_vs_tail_targeted",
                        "generation_seed": seed,
                        "downstream_utility_models": {
                            "tail_learnability": {"status": "completed"}
                        },
                        "downstream_utility_tasks": {"status": "completed"},
                        "publication": {
                            "schema_version": "publication_metrics_v2",
                            "status": "completed",
                            "metrics": {
                                "rq1": {"metric": 0.1},
                                "rq2": {"metric": 0.2},
                            },
                        },
                        "final_audit": {"status": "passed", "issues": []},
                        "artifact": str(absolute_artifact),
                    }
                ),
                encoding="utf-8",
            )
            summary_path = source_root / dataset / model / "multi_seed_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "experiment_design": "standard_vs_tail_targeted",
                        "dataset": dataset,
                        "model": model,
                        "seeds": [seed],
                        "publication_schema_version": "publication_metrics_v2",
                        "metric_summary_scope": "publication_only",
                        "summary_report": str(summary_path),
                        "runs": [
                            {
                                "dataset": dataset,
                                "model": model,
                                "seed": seed,
                                "standard_synthetic": str(standard_artifact),
                                "tail_targeted_synthetic": str(targeted_artifact),
                                "generation_metadata": standard_metadata,
                                "tail_targeted_generation_metadata": targeted_metadata,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = promote_final_reports.run(
                [dataset], [model], [seed], source_root, target_root
            )

            promoted_seed = json.loads(
                (
                    target_root
                    / dataset
                    / model
                    / f"seed_{seed}"
                    / "generation_evaluation_report.json"
                ).read_text(encoding="utf-8")
            )
            promoted_summary = json.loads(
                (target_root / dataset / model / "multi_seed_summary.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(result["seed_reports"], 1)
        self.assertEqual(promoted_seed["artifact"], "experiments/outputs/x.csv")
        self.assertTrue(promoted_seed["curated_snapshot"]["paths_portabilized"])
        self.assertEqual(promoted_seed["dataset"], dataset)
        self.assertEqual(promoted_seed["model"], model)
        self.assertEqual(
            promoted_seed["artifact_provenance"]["tail_targeted"]["metadata"][
                "synthetic_output_sha256"
            ],
            targeted_metadata["synthetic_output_sha256"],
        )
        self.assertIn(
            "tail_targeted_generation_metadata", promoted_summary["runs"][0]
        )
        self.assertEqual(
            promoted_summary["summary_report"],
            promote_final_reports._relative(
                target_root / dataset / model / "multi_seed_summary.json"
            ),
        )

    def test_rejects_failed_seed_report(self) -> None:
        report = {
            "experiment_design": "standard_vs_tail_targeted",
            "generation_seed": 20260703,
            "publication": {
                "schema_version": "publication_metrics_v2",
                "status": "completed",
            },
            "final_audit": {"status": "failed", "issues": ["bad"]},
        }
        with self.assertRaisesRegex(ValueError, "strict final audit"):
            promote_final_reports._validate_seed_report(
                report, "dataset", "model", 20260703, Path("report.json")
            )

    def test_oulad_task_metadata_is_normalized_without_changing_metrics(self) -> None:
        report = {
            "downstream_utility": {"task": "next_response_correctness", "score": 0.5},
            "downstream_utility_models": {"task": "next_response_correctness"},
            "tail_targeted_evaluation": {
                "downstream_utility": {"task": "next_response_correctness"}
            },
        }
        corrections = promote_final_reports._normalize_task_metadata(
            report, "oulad_weekly_engagement"
        )

        self.assertEqual(corrections, ["primary_downstream_task_label"])
        self.assertEqual(report["downstream_utility"]["task"], "next_week_engagement")
        self.assertEqual(report["downstream_utility"]["score"], 0.5)


if __name__ == "__main__":
    unittest.main()
