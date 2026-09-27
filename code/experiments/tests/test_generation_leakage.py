from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import pipeline
from scripts import run_final_experiments
from synthetic_generation.generator_scripts import (
    block_bootstrap,
    markov_ngram,
    sequence_vae_timevae,
    timegan,
)
from synthetic_generation.generator_scripts.common import (
    fixed_step_budget,
    make_trajectory_windows,
    sample_trajectory_lengths,
)
from synthetic_generation.run_generator import load_generator_script, run_generator


def toy_train() -> pd.DataFrame:
    rows = []
    for learner_index, length in enumerate([2, 3, 5, 8, 13, 4, 7, 11]):
        for order in range(length):
            rows.append(
                {
                    "learner_id": f"real_{learner_index}",
                    "order": order,
                    "skill_id": f"skill_{order % 3}",
                    "correct": str((learner_index + order) % 2),
                    "hint_bin": str(order % 2),
                }
            )
    return pd.DataFrame(rows)


class BoundaryTests(unittest.TestCase):
    def test_final_runner_rejects_values_outside_locked_matrix(self) -> None:
        config = run_final_experiments.load_config(
            run_final_experiments.DEFAULT_CONFIG
        )
        base = {
            "datasets": ["assistments"],
            "models": ["markov_ngram"],
            "include_auxiliary": False,
            "seeds": None,
            "allow_fewer_seeds": False,
        }

        with self.assertRaisesRegex(ValueError, "outside the locked experiment config"):
            run_final_experiments.select_matrix(
                config, SimpleNamespace(**{**base, "datasets": ["kddcup2010"]})
            )
        with self.assertRaisesRegex(ValueError, "outside the locked experiment config"):
            run_final_experiments.select_matrix(
                config, SimpleNamespace(**{**base, "models": ["block_bootstrap"]})
            )
        with self.assertRaisesRegex(ValueError, "locked seeds"):
            run_final_experiments.select_matrix(
                config, SimpleNamespace(**{**base, "seeds": [1, 2, 3]})
            )

        _, _, smoke_seeds, _ = run_final_experiments.select_matrix(
            config,
            SimpleNamespace(
                **{**base, "seeds": [1], "allow_fewer_seeds": True}
            ),
        )
        self.assertEqual(smoke_seeds, [1])

    def test_final_runner_requires_one_dataset_model_cell(self) -> None:
        args = SimpleNamespace(
            config=run_final_experiments.DEFAULT_CONFIG,
            datasets=["assistments", "ednet"],
            models=["markov_ngram"],
            include_auxiliary=False,
            seeds=None,
            preflight_only=False,
            dry_run=False,
        )

        with self.assertRaisesRegex(ValueError, "exactly one dataset/model cell"):
            run_final_experiments.run(args)

    def test_standard_and_tail_targeted_use_explicit_sibling_roots(self) -> None:
        split = SimpleNamespace(
            synthetic_root=Path("/tmp/experiments/outputs/assistments")
        )
        parts = ["markov_ngram", "seed_20260703"]

        candidates = pipeline._standard_artifact_candidate_dirs(split, parts)

        self.assertEqual(
            candidates[0],
            pipeline.STANDARD_OUTPUTS_ROOT
            / "assistments"
            / "markov_ngram"
            / "seed_20260703",
        )
        self.assertEqual(candidates[1], split.synthetic_root.joinpath(*parts))
        self.assertEqual(pipeline.STANDARD_OUTPUTS_ROOT.name, "standard")
        self.assertEqual(pipeline.TAIL_TARGETED_OUTPUTS_ROOT.name, "tail_targeted")
        self.assertEqual(pipeline.LOCAL_RUN_ROOT.name, pipeline.EXPERIMENT_DESIGN)
        self.assertEqual(pipeline.REPORTS_ROOT, pipeline.LOCAL_RUN_ROOT / "evaluation")
        self.assertEqual(
            pipeline.REFERENCE_REPORTS_ROOT,
            pipeline.ROOT / "experiments" / "reports" / pipeline.EXPERIMENT_DESIGN,
        )
        self.assertNotEqual(pipeline.REPORTS_ROOT, pipeline.REFERENCE_REPORTS_ROOT)

    def test_local_seed_paths_do_not_collide_with_reference_reports(self) -> None:
        split = SimpleNamespace(
            name="assistments_test",
            label=None,
            synthetic_root=Path("/tmp/experiments/outputs/assistments"),
        )

        paths = pipeline.experiment_seed_paths(split, "markov_ngram", 20260703)

        self.assertTrue(
            paths["standard_csv"].is_relative_to(pipeline.STANDARD_OUTPUTS_ROOT)
        )
        self.assertTrue(
            paths["target_csv"].is_relative_to(pipeline.TAIL_TARGETED_OUTPUTS_ROOT)
        )
        self.assertNotEqual(paths["standard_csv"].parent, paths["target_csv"].parent)
        self.assertTrue(paths["evaluation_report"].is_relative_to(pipeline.REPORTS_ROOT))
        self.assertFalse(
            paths["evaluation_report"].is_relative_to(pipeline.REFERENCE_REPORTS_ROOT)
        )

    def test_standard_reuse_never_allows_target_overwrite(self) -> None:
        split = SimpleNamespace(
            name="assistments_test",
            label=None,
            synthetic_root=Path("/tmp/legacy/assistments"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(pipeline, "STANDARD_OUTPUTS_ROOT", root / "standard"),
                patch.object(
                    pipeline, "TAIL_TARGETED_OUTPUTS_ROOT", root / "tail_targeted"
                ),
                patch.object(pipeline, "REPORTS_ROOT", root / "evaluation"),
            ):
                paths = pipeline.experiment_seed_paths(
                    split, "markov_ngram", 20260703
                )
                paths["standard_csv"].parent.mkdir(parents=True)
                paths["standard_csv"].touch()
                paths["standard_metadata"].touch()

                self.assertEqual(
                    set(
                        pipeline.occupied_experiment_seed_paths(
                            split, "markov_ngram", 20260703
                        )
                    ),
                    {paths["standard_csv"], paths["standard_metadata"]},
                )
                self.assertEqual(
                    pipeline.occupied_experiment_seed_paths(
                        split,
                        "markov_ngram",
                        20260703,
                        allow_standard_reuse=True,
                    ),
                    [],
                )

                paths["target_csv"].parent.mkdir(parents=True)
                paths["target_csv"].touch()
                occupied = pipeline.occupied_experiment_seed_paths(
                    split,
                    "markov_ngram",
                    20260703,
                    allow_standard_reuse=True,
                )
                self.assertEqual(occupied, [paths["target_csv"]])

    def test_pipeline_resume_skips_only_validated_completed_seeds(self) -> None:
        split = SimpleNamespace(
            name="assistments_test",
            label=None,
            synthetic_root=Path("/tmp/experiments/outputs/assistments"),
            train_path=Path("/tmp/train.csv"),
            test_path=Path("/tmp/test.csv"),
            ignore_columns=("learner_id", "order"),
            metadata={"artifact_metadata": {}},
        )
        adapter = SimpleNamespace(load_split=lambda: split)
        completed = {
            "dataset": split.name,
            "model": "markov_ngram",
            "seed": 1,
            "_metrics": {},
            "_publication": {},
        }
        generated = {
            "dataset": split.name,
            "model": "markov_ngram",
            "seed": 2,
            "_metrics": {},
            "_publication": {},
        }

        with (
            patch.object(pipeline, "get_dataset", return_value=adapter),
            patch.object(pipeline, "load_completed_seed", side_effect=[completed, None]),
            patch.object(pipeline, "run_seed", return_value=generated) as run_seed,
            patch.object(pipeline, "write_json"),
        ):
            report = pipeline.run_pipeline(
                "assistments",
                "markov_ngram",
                [1, 2],
                preprocess=False,
                resume_existing=True,
            )

        self.assertEqual([run["seed"] for run in report["runs"]], [1, 2])
        run_seed.assert_called_once()
        self.assertEqual(run_seed.call_args.args[3], 2)
        self.assertTrue(run_seed.call_args.kwargs["resume_incomplete"])

    def test_completed_resume_uses_rq2_pipeline_report(self) -> None:
        """An RQ1 standard-provenance file must not replace the RQ2 run record."""
        split = SimpleNamespace(name="assistments", metadata={"artifact_metadata": {}})
        adapter = SimpleNamespace()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {
                "standard_seed_dir": root / "standard",
                "standard_csv": root / "standard" / "synth_generation.csv",
                "standard_metadata": root / "standard" / "generation_metadata.json",
                "target_csv": root / "target" / "synth_generation.csv",
                "target_metadata": root / "target" / "generation_metadata.json",
                "pipeline_report": root / "runs" / "pipeline_report.json",
                "evaluation_report": root / "runs" / "evaluation_report.json",
            }
            for path in paths.values():
                if path.suffix:
                    path.parent.mkdir(parents=True, exist_ok=True)
            paths["standard_csv"].write_text("correct\n1\n", encoding="utf-8")
            paths["standard_metadata"].write_text("{}", encoding="utf-8")
            paths["target_csv"].write_text("correct\n1\n", encoding="utf-8")
            target_metadata = {
                "seed": 20260703,
                "model": "markov_ngram",
                "experiment_design": pipeline.EXPERIMENT_DESIGN,
                "synthetic_output_sha256": "target-hash",
            }
            paths["target_metadata"].write_text(
                json.dumps(target_metadata), encoding="utf-8"
            )
            rq2_pipeline = {
                "experiment_design": pipeline.EXPERIMENT_DESIGN,
                "dataset": split.name,
                "model": "markov_ngram",
                "seed": 20260703,
                "tail_targeted_synthetic": str(paths["target_csv"]),
            }
            paths["pipeline_report"].write_text(
                json.dumps(rq2_pipeline), encoding="utf-8"
            )
            paths["evaluation_report"].write_text(
                json.dumps(
                    {
                        "experiment_design": pipeline.EXPERIMENT_DESIGN,
                        "final_audit": {"status": "passed"},
                        "publication": {"metrics": {"rq1": {"x": 1}, "rq2": {"x": 2}}},
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch.object(pipeline, "experiment_seed_paths", return_value=paths),
                patch.object(
                    pipeline,
                    "_validate_standard_artifact_provenance",
                    return_value=(
                        paths["standard_csv"],
                        {},
                        {"experiment_design": "standard", "seed": 20260703},
                    ),
                ),
                patch.object(pipeline, "sha256_file", return_value="target-hash"),
                patch.object(pipeline, "final_report_issues", return_value=[]),
            ):
                resumed = pipeline.load_completed_seed(
                    adapter, split, "markov_ngram", 20260703
                )

        self.assertEqual(resumed["experiment_design"], pipeline.EXPERIMENT_DESIGN)
        self.assertEqual(resumed["tail_targeted_generation_metadata"], target_metadata)
        self.assertTrue(resumed["_resumed_completed_seed"])

    def test_length_model_does_not_copy_empirical_length_list(self) -> None:
        train = toy_train()
        sampled, metadata = sample_trajectory_lengths(train, 20260703)
        observed = train.groupby("learner_id", sort=False).size().tolist()
        self.assertNotEqual(sorted(sampled), sorted(observed))
        self.assertEqual(len(sampled), len(observed))
        self.assertEqual(metadata["method"], "mean_calibrated_truncated_lognormal")

    def test_length_model_preserves_budget_for_extreme_tail(self) -> None:
        learner_ids = [f"small_{index}" for index in range(999)] + ["large"] * 10_000
        orders = [0] * 999 + list(range(10_000))
        train = pd.DataFrame({"learner_id": learner_ids, "order": orders})

        sampled, metadata = sample_trajectory_lengths(train, 20260703)

        self.assertEqual(len(sampled), 1_000)
        self.assertEqual(sum(sampled), len(train))
        self.assertEqual(metadata["synthetic_rows_from_length_model"], len(train))
        self.assertGreaterEqual(metadata["length_cap"] * len(sampled), len(train))

    def test_markov_contexts_reset_at_learner_boundaries(self) -> None:
        trajectories = [[("a",), ("b",)], [("c",), ("d",)]]
        tables, _ = markov_ngram._build_tables(trajectories, order=1)
        self.assertNotIn(("c",), tables[1].get((("b",),), {}))
        self.assertIn(("b",), tables[1].get((("a",),), {}))

    def test_neural_windows_never_mix_learners(self) -> None:
        first = np.asarray([[1], [1], [1]], dtype=np.int64)
        second = np.asarray([[2], [2]], dtype=np.int64)
        windows, masks = make_trajectory_windows([first, second], window=4)
        self.assertEqual(len(windows), 2)
        for window, mask in zip(windows, masks):
            self.assertEqual(len(set(window[mask, 0].tolist())), 1)

    def test_timegan_adapter_keeps_boundaries_and_marks_padding(self) -> None:
        frames = [
            pd.DataFrame({"course": [0, 0, 0], "correct": [0, 1, 0]}),
            pd.DataFrame({"course": [1, 1], "correct": [1, 0]}),
        ]
        static = timegan._infer_static_columns(frames)
        self.assertEqual(static, ["course"])
        static_data, temporal, horizons, masks = timegan._synthcity_training_data(
            frames, static, window=4
        )
        self.assertEqual(static_data["course"].tolist(), [0, 1])
        self.assertEqual([len(frame) for frame in temporal], [4, 4])
        self.assertEqual([frame[timegan.VALID_COLUMN].tolist() for frame in temporal], [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ])
        self.assertEqual(horizons, [list(range(4)), list(range(4))])
        self.assertEqual(masks.sum(axis=1).tolist(), [3, 2])

    def test_timegan_batch_safety_avoids_only_singleton_remainders(self) -> None:
        self.assertEqual(pipeline._non_singleton_batch_size(15_915, 200), 200)
        self.assertEqual(pipeline._non_singleton_batch_size(18_001, 200), 199)

        rows = [
            {"learner_id": f"learner_{index}", "order": 0}
            for index in range(18_001)
        ]
        with patch.dict(
            os.environ,
            {"TIMEGAN_WINDOW": "20", "TIMEGAN_BATCH": "200"},
            clear=False,
        ):
            adjustment = pipeline._timegan_batch_safety_adjustment(
                "timegan", pd.DataFrame(rows)
            )

        self.assertEqual(adjustment["training_windows"], 18_001)
        self.assertEqual(adjustment["requested_batch_size"], 200)
        self.assertEqual(adjustment["effective_batch_size"], 199)
        self.assertTrue(adjustment["full_training_data_retained"])
        self.assertIsNone(
            pipeline._timegan_batch_safety_adjustment(
                "markov_ngram", pd.DataFrame(rows)
            )
        )

    def test_pipeline_keeps_boundaries_and_never_reconstructs_them_from_real_lengths(self) -> None:
        train = toy_train().assign(part="derived")
        split = SimpleNamespace(
            train=train,
            ignore_columns=("learner_id", "order"),
            derived_columns=("part",),
            columns={"learner": "learner_id", "order": "order"},
        )

        generation_frame = pipeline.generation_train(split)

        self.assertIn("learner_id", generation_frame.columns)
        self.assertIn("order", generation_frame.columns)
        self.assertNotIn("part", generation_frame.columns)
        with self.assertRaises(ValueError):
            pipeline.normalize_synthetic_frame(
                split,
                generation_frame.drop(columns=["learner_id", "order"]),
            )

    def test_tail_oversampling_repeats_examples_without_merging_trajectories(self) -> None:
        rows = []
        # Two of 40 learners form a genuine 5% tail. Keeping this proportion
        # small is important: coarse 25% binary groups are intentionally no
        # longer mislabeled as percentile tails.
        for learner_index in range(40):
            for order in range(4):
                rows.append(
                    {
                        "learner_id": f"learner_{learner_index}",
                        "order": order,
                        "skill_id": f"skill_{order % 2}",
                        "correct": int(learner_index >= 2),
                        "hint_bin": int(learner_index < 2),
                        "attempt_bin": "1",
                    }
                )
        frame = pd.DataFrame(rows)
        split = SimpleNamespace(
            train=frame,
            ignore_columns=("learner_id", "order"),
            derived_columns=(),
            columns={
                "learner": "learner_id",
                "order": "order",
                "skill": "skill_id",
                "correct": "correct",
                "hint": "hint_bin",
                "attempts": "attempt_bin",
            },
        )

        oversampled, metadata = pipeline.tail_oversampled_generation_train(
            split, oversample=2, return_metadata=True
        )

        self.assertIsNotNone(oversampled)
        repeated_ids = oversampled.loc[
            oversampled["learner_id"].astype(str).str.contains("__tail_repeat_"), "learner_id"
        ]
        self.assertFalse(repeated_ids.empty)
        self.assertTrue(oversampled.groupby("learner_id").size().eq(4).all())
        self.assertEqual(metadata["selection_contract"], pipeline.TAIL_SELECTION_CONTRACT)
        self.assertGreater(metadata["selected_tail_learners"], 0)
        self.assertEqual(
            metadata["oversampled_train_rows"], len(oversampled)
        )

    def test_targeting_excludes_postprocessed_outcomes_and_nonsemantic_groups(self) -> None:
        rows = []
        for learner_index in range(20):
            for order in range(4):
                rows.append(
                    {
                        "learner_id": f"learner_{learner_index}",
                        "order": order,
                        "activity": "resource",
                        "engaged": 1,
                        "dropout": int(learner_index < 10),
                    }
                )
        split = SimpleNamespace(
            train=pd.DataFrame(rows),
            ignore_columns=("learner_id", "order"),
            derived_columns=("dropout",),
            columns={
                "learner": "learner_id",
                "order": "order",
                "skill": "activity",
                "correct": "engaged",
                "dropout": "dropout",
            },
            metadata={"primary_signal_semantics": "weekly_engagement"},
        )

        targeted, metadata = pipeline.tail_oversampled_generation_train(
            split, oversample=2, return_metadata=True
        )

        self.assertIsNone(targeted)
        self.assertEqual(metadata["selected_tail_learners"], 0)
        self.assertEqual(
            metadata["excluded_observed_tail_groups"]["dropout"],
            "terminal_outcome_assigned_after_generation",
        )
        self.assertNotIn("persistent_misconception", metadata["eligible_tail_groups"])
        self.assertNotIn("short_trajectory", metadata["eligible_tail_groups"])

    def test_targeting_excludes_semantic_groups_above_tail_prevalence_ceiling(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "learner_id": f"learner_{learner}",
                    "order": order,
                    "skill_id": "skill",
                    "correct": int(learner >= 10),
                }
                for learner in range(20)
                for order in range(6)
            ]
        )
        split = SimpleNamespace(
            train=frame,
            ignore_columns=("learner_id", "order"),
            derived_columns=(),
            columns={
                "learner": "learner_id",
                "order": "order",
                "skill": "skill_id",
                "correct": "correct",
            },
            metadata={},
        )

        targeted, metadata = pipeline.tail_oversampled_generation_train(
            split, oversample=3, return_metadata=True
        )

        self.assertIsNone(targeted)
        self.assertEqual(metadata["candidate_learners_by_group"]["persistent_misconception"], 10)
        self.assertNotIn("persistent_misconception", metadata["implemented_tail_groups"])
        self.assertIn("persistent_misconception", metadata["excluded_target_groups"])
        self.assertEqual(metadata["maximum_target_group_prevalence"], 0.10)


class GeneratorContractTests(unittest.TestCase):
    def assert_valid_synthetic(self, train: pd.DataFrame, synthetic: pd.DataFrame) -> None:
        self.assertEqual(synthetic.columns.tolist(), train.columns.tolist())
        self.assertFalse(set(train["learner_id"].astype(str)) & set(synthetic["learner_id"].astype(str)))
        expected_order = synthetic.groupby("learner_id", sort=False).cumcount().to_numpy()
        np.testing.assert_array_equal(synthetic["order"].to_numpy(), expected_order)
        observed_lengths = sorted(train.groupby("learner_id", sort=False).size().tolist())
        synthetic_lengths = sorted(synthetic.groupby("learner_id", sort=False).size().tolist())
        self.assertNotEqual(synthetic_lengths, observed_lengths)

    def test_lightweight_generators_obey_trajectory_contract(self) -> None:
        train = toy_train()
        for module in (markov_ngram, block_bootstrap):
            with self.subTest(model=module.MODEL_NAME), tempfile.TemporaryDirectory() as temp_dir:
                module.generate(train, Path(temp_dir), seed=20260703)
                synthetic = pd.read_csv(Path(temp_dir) / "synth_generation.csv")
                self.assert_valid_synthetic(train, synthetic)

    def test_selectable_generator_names_match_module_metadata(self) -> None:
        for model_name in (
            "markov_ngram",
            "sequence_vae_timevae",
            "timegan",
            "block_bootstrap",
        ):
            with self.subTest(model=model_name):
                module = load_generator_script(model_name)
                self.assertEqual(module.MODEL_NAME, model_name)

    def test_retired_rcgan_name_is_not_selectable(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported generator model"):
            load_generator_script("rcgan_crnngan")

    def test_generator_runner_records_source_provenance(self) -> None:
        train = toy_train()
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata = run_generator("markov_ngram", train, Path(temp_dir), seed=1)
            self.assertIn("generator_script_sha256", metadata)
            self.assertEqual(len(metadata["generator_script_sha256"]), 64)
            self.assertEqual(len(metadata["generator_source_bundle_sha256"]), 64)
            self.assertIn(
                "experiments/synthetic_generation/generator_scripts/common.py",
                metadata["generator_source_files_sha256"],
            )
            self.assertEqual(len(metadata["synthetic_output_sha256"]), 64)

    def test_pipeline_metadata_describes_final_normalized_csv(self) -> None:
        train = pd.DataFrame(
            {
                "learner_id": ["real"],
                "order": [0],
                "question_id": ["q1"],
                "skill_id": ["skill_1"],
                "correct": [1],
            }
        )
        split = SimpleNamespace(
            train=train,
            ignore_columns=("learner_id", "order"),
            derived_columns=("skill_id",),
            columns={"learner": "learner_id", "order": "order"},
        )
        generated = pd.DataFrame(
            {
                "learner_id": ["synthetic_000000"],
                "order": [0],
                "question_id": ["q1"],
                "correct": [0],
            }
        )

        class Adapter:
            @staticmethod
            def rebuild_derived_columns(frame, real_train):
                rebuilt = frame.copy()
                mapping = real_train.drop_duplicates("question_id").set_index("question_id")["skill_id"]
                rebuilt["skill_id"] = rebuilt["question_id"].map(mapping)
                return rebuilt

        def fake_run_generator(model_name, train_frame, output_dir, seed, generation_shape=None):
            output_dir.mkdir(parents=True, exist_ok=True)
            generated.to_csv(output_dir / "synth_generation.csv", index=False)
            return {"model": model_name, "seed": seed, "columns": generated.columns.tolist()}

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            pipeline, "run_generator", side_effect=fake_run_generator
        ):
            output_dir = Path(temp_dir)
            synthetic, path, metadata = pipeline._generate_synthetic(
                "toy", split, train.drop(columns="skill_id"), output_dir, 7, Adapter()
            )
            stored = json.loads((output_dir / "generation_metadata.json").read_text())

            self.assertEqual(stored, metadata)
            self.assertEqual(stored["columns"], train.columns.tolist())
            self.assertEqual(stored["synthetic_rows"], len(synthetic))
            self.assertEqual(stored["pipeline_added_columns"], ["skill_id"])
            self.assertEqual(
                stored["synthetic_output_sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            self.assertEqual(len(stored["pipeline_script_sha256"]), 64)

    def test_pipeline_reuses_only_a_provenance_matched_standard_artifact(self) -> None:
        real = toy_train()
        synthetic = real.copy()
        synthetic["learner_id"] = synthetic["learner_id"].str.replace(
            "real_", "synthetic_", regex=False
        )
        provenance = {
            "train_sha256": "train-hash",
            "test_sha256": "test-hash",
            "source_sha256": "source-hash",
        }
        split = SimpleNamespace(
            name="toy_dataset",
            train=real,
            ignore_columns=("learner_id", "order"),
            derived_columns=(),
            columns={"learner": "learner_id", "order": "order"},
            metadata={"artifact_metadata": provenance},
        )
        seed = 20260703
        model = "markov_ngram"
        generator = load_generator_script(model)
        source_files, source_bundle = pipeline.generator_source_provenance(generator)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            artifact = output_dir / "synth_generation.csv"
            synthetic.to_csv(artifact, index=False)
            metadata = {
                "model": model,
                "seed": seed,
                "synthetic_rows": len(synthetic),
                "synthetic_output_sha256": pipeline.sha256_file(artifact),
                "generator_script_sha256": pipeline.sha256_file(Path(generator.__file__)),
                "generator_source_files_sha256": source_files,
                "generator_source_bundle_sha256": source_bundle,
            }
            pipeline.write_json(output_dir / "generation_metadata.json", metadata)
            pipeline.write_json(
                output_dir / "pipeline_report.json",
                {
                    "dataset": split.name,
                    "model": model,
                    "seed": seed,
                    "data_provenance": provenance,
                },
            )

            reused = pipeline._load_reusable_standard(
                model, split, output_dir, seed
            )
            self.assertIsNotNone(reused)
            self.assertTrue(reused[2]["standard_artifact_reuse"]["reused"])

            metadata["generator_source_bundle_sha256"] = "obsolete"
            pipeline.write_json(output_dir / "generation_metadata.json", metadata)
            with self.assertRaisesRegex(ValueError, "source-bundle hash differs"):
                pipeline._load_reusable_standard(model, split, output_dir, seed)

    def test_resume_prefers_validated_rq1_standard_reuse_contract(self) -> None:
        split = SimpleNamespace()
        adapter = SimpleNamespace()
        standard_dir = Path("/tmp/standard/ednet/model/seed_20260703")
        historical_dir = Path("/tmp/historical/ednet/model/seed_20260703")
        reusable = (pd.DataFrame({"value": [1]}), standard_dir / "synth_generation.csv", {})

        with (
            patch.object(
                pipeline,
                "_standard_artifact_candidate_dirs",
                return_value=(standard_dir, historical_dir),
            ),
            patch.object(
                pipeline,
                "_load_reusable_standard",
                return_value=reusable,
            ) as load_standard,
            patch.object(
                pipeline,
                "_load_partial_generated_artifact",
                side_effect=AssertionError("partial RQ2 validator must not run"),
            ) as load_partial,
        ):
            loaded, source_dir = pipeline._load_standard_for_run(
                "sequence_vae_timevae",
                split,
                standard_dir,
                ("sequence_vae_timevae", "seed_20260703"),
                20260703,
                adapter,
                reuse_standard_if_valid=True,
                resume_incomplete=True,
            )

        self.assertIs(loaded, reusable)
        self.assertEqual(source_dir, standard_dir)
        load_standard.assert_called_once_with(
            "sequence_vae_timevae", split, standard_dir, 20260703, adapter
        )
        load_partial.assert_not_called()

    def test_external_timegan_adapter_obeys_trajectory_contract(self) -> None:
        train = toy_train()

        def fake_external(
            static_data,
            temporal_data,
            observation_times,
            needed_windows,
            seed,
            config,
            workspace,
        ):
            static = pd.concat(
                [static_data.iloc[[index % len(static_data)]] for index in range(needed_windows)],
                ignore_index=True,
            )
            temporal = [
                temporal_data[index % len(temporal_data)].copy()
                for index in range(needed_windows)
            ]
            return static, temporal, "FakeExternalTimeGAN"

        environment = {
            "TIMEGAN_WINDOW": "4",
            "TIMEGAN_MAX_VOCAB": "20",
            "TIMEGAN_N_ITER": "1",
            "TIMEGAN_TARGET_STEPS": "8",
            "TIMEGAN_BATCH": "4",
            "TIMEGAN_N_ITER_PRINT": "1",
            "TIMEGAN_GENERATOR_LAYERS": "1",
            "TIMEGAN_GENERATOR_HIDDEN": "8",
            "TIMEGAN_DISCRIMINATOR_LAYERS": "1",
            "TIMEGAN_DISCRIMINATOR_HIDDEN": "8",
            "TIMEGAN_DEVICE": "cpu",
        }
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ, environment, clear=False
        ), patch.object(timegan, "dependency_issues", return_value=[]), patch.object(
            timegan, "_fit_and_generate_external", side_effect=fake_external
        ), patch.object(timegan.importlib.metadata, "version", return_value="0.2.12"):
            timegan.generate(train, Path(temp_dir), seed=7)
            synthetic = pd.read_csv(Path(temp_dir) / "synth_generation.csv")
            metadata = json.loads((Path(temp_dir) / "generation_metadata.json").read_text())

        self.assert_valid_synthetic(train, synthetic)
        self.assertEqual(metadata["external_library"], "synthcity")
        self.assertEqual(metadata["external_plugin_class"], "FakeExternalTimeGAN")
        self.assertEqual(metadata["adapter_configuration"]["generator_n_units_hidden"], 8)
        self.assertEqual(metadata["adapter_configuration"]["discriminator_n_units_hidden"], 8)
        self.assertEqual(metadata["training_budget"]["target_batch_steps"], 8)

    def test_timegan_fixed_step_budget_scales_epochs_to_input_size(self) -> None:
        config = timegan.SynthCityTimeGANConfig(
            n_iter=50,
            target_steps=4000,
            batch_size=200,
        )

        assistments = fixed_step_budget(
            15_915, config.batch_size, config.target_steps, config.n_iter
        )
        ednet_targeted = fixed_step_budget(
            159_867, config.batch_size, config.target_steps, config.n_iter
        )

        self.assertEqual(assistments["effective_epochs"], 50)
        self.assertEqual(assistments["projected_batch_steps"], 4_000)
        self.assertEqual(ednet_targeted["effective_epochs"], 5)
        self.assertEqual(ednet_targeted["projected_batch_steps"], 4_000)

    @unittest.skipUnless(sequence_vae_timevae._TORCH_AVAILABLE, "PyTorch is not installed")
    def test_sequence_vae_obeys_trajectory_contract(self) -> None:
        train = toy_train()
        environment = {
            "VAE_EPOCHS": "1",
            "VAE_TARGET_STEPS": "8",
            "VAE_WINDOW": "4",
            "VAE_HIDDEN": "16",
            "VAE_LATENT": "4",
            "VAE_BATCH": "4",
            "VAE_MAX_VOCAB": "20",
            "VAE_DEVICE": "cpu",
        }
        with patch.dict(os.environ, environment, clear=False):
            with tempfile.TemporaryDirectory() as temp_dir:
                sequence_vae_timevae.generate(train, Path(temp_dir), seed=7)
                synthetic = pd.read_csv(Path(temp_dir) / "synth_generation.csv")
                metadata = json.loads((Path(temp_dir) / "generation_metadata.json").read_text())
                self.assert_valid_synthetic(train, synthetic)
                self.assertEqual(metadata["training_budget"]["target_batch_steps"], 8)


if __name__ == "__main__":
    unittest.main()
