#!/usr/bin/env python3
"""Migrate OULAD standard artifacts to behavioral-dropout-v2 outcomes.

The generator-produced behavioral trajectories remain unchanged.  Only the
derived ``dropout``, ``failure``, and ``final_result`` columns are recomputed
from the current, train-fitted OULAD outcome contract.  Original artifacts are
preserved byte-for-byte under ``outputs/archive`` before an active file is
replaced.

The migration is resumable: every cell is rebuilt from its archived original,
and a completed cell is accepted only after the pipeline's full standard reuse
validation succeeds.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from data.dataset_scripts.oulad_weekly import dataset as adapter
from pipeline import (
    EXPERIMENT_DESIGN,
    _load_reusable_standard,
    experiment_seed_paths,
    normalize_synthetic_frame,
    sha256_file,
)


VARIANT = "behavioral_dropout_v2"
STANDARD_ROOT = EXPERIMENTS / "outputs" / "standard" / "oulad_weekly"
ARCHIVE_ROOT = (
    EXPERIMENTS
    / "outputs"
    / "archive"
    / "oulad_standard_before_behavioral_dropout_v2"
)
REPORT_ROOT = (
    EXPERIMENTS
    / "reports"
    / "standard"
    / "oulad_weekly_engagement_behavioral_dropout_v2"
)
DEFAULT_MODELS = ("markov_ngram", "sequence_vae_timevae", "timegan")
DEFAULT_SEEDS = (20260703, 20260704, 20260705)
OUTCOME_COLUMNS = tuple(adapter.TERMINAL_OUTCOME_COLUMNS)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _learner_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("learner_id", sort=False)[list(OUTCOME_COLUMNS)]
        .first()
        .sort_index()
    )


def _cell_dirs(model: str, seed: int) -> tuple[Path, Path]:
    relative = Path(model) / f"seed_{seed}"
    return STANDARD_ROOT / relative, ARCHIVE_ROOT / relative


def _archive_original(active_dir: Path, archive_dir: Path) -> None:
    required = (
        "synth_generation.csv",
        "generation_metadata.json",
        "standard_provenance.json",
    )
    archive_dir.mkdir(parents=True, exist_ok=True)
    for name in required:
        source = active_dir / name
        target = archive_dir / name
        if target.exists():
            continue
        if not source.is_file():
            raise FileNotFoundError(f"Missing active OULAD standard file: {source}")
        temporary = target.with_name(f".{target.name}.tmp")
        shutil.copy2(source, temporary)
        temporary.replace(target)


def _validated_original(archive_dir: Path, model: str, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    csv_path = archive_dir / "synth_generation.csv"
    metadata_path = archive_dir / "generation_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("seed", -1)) != int(seed):
        raise RuntimeError(f"Archived OULAD seed mismatch: {metadata_path}")
    if str(metadata.get("model")) != model:
        raise RuntimeError(f"Archived OULAD model mismatch: {metadata_path}")
    actual_hash = sha256_file(csv_path)
    if metadata.get("synthetic_output_sha256") != actual_hash:
        raise RuntimeError(f"Archived OULAD artifact hash mismatch: {csv_path}")
    frame = pd.read_csv(csv_path, low_memory=False)
    if int(metadata.get("synthetic_rows", -1)) != len(frame):
        raise RuntimeError(f"Archived OULAD row-count mismatch: {csv_path}")
    return frame, metadata


def _completed_cell(split, model: str, seed: int) -> dict[str, Any] | None:
    active_dir, _ = _cell_dirs(model, seed)
    migration_path = active_dir / "outcome_migration_metadata.json"
    if not migration_path.exists():
        return None
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    if migration.get("variant") != VARIANT:
        raise RuntimeError(f"Incompatible OULAD outcome migration: {migration_path}")
    validated = _load_reusable_standard(
        model, split, active_dir, seed, adapter=adapter
    )
    if validated is None:
        raise RuntimeError(f"Completed migration is missing its active artifact: {active_dir}")
    if sha256_file(active_dir / "synth_generation.csv") != migration.get(
        "output_sha256"
    ):
        raise RuntimeError(f"Completed migration output hash changed: {active_dir}")
    return migration


def migrate_cell(split, fitted, model: str, seed: int) -> dict[str, Any]:
    completed = _completed_cell(split, model, seed)
    if completed is not None:
        return {**completed, "resumed": True}

    active_dir, archive_dir = _cell_dirs(model, seed)
    _archive_original(active_dir, archive_dir)
    original, original_metadata = _validated_original(archive_dir, model, seed)
    if original.columns.tolist() != split.train.columns.tolist():
        raise RuntimeError(f"OULAD standard schema mismatch: {active_dir}")

    before = _learner_outcomes(original)
    migrated, outcome_metadata = adapter.reassign_terminal_outcomes(
        original, split.train, seed, fitted=fitted
    )
    after = _learner_outcomes(migrated)
    non_outcome_columns = [
        column for column in original.columns if column not in OUTCOME_COLUMNS
    ]
    if not original[non_outcome_columns].equals(migrated[non_outcome_columns]):
        raise RuntimeError(
            f"OULAD migration changed generator-produced behavior: {model}/seed_{seed}"
        )
    normalized, _ = normalize_synthetic_frame(
        split, migrated, adapter=adapter, seed=seed
    )
    if not normalized.equals(migrated):
        raise RuntimeError(
            f"Migrated OULAD artifact is not stable under normalization: {model}/seed_{seed}"
        )

    active_csv = active_dir / "synth_generation.csv"
    _write_csv_atomic(active_csv, migrated)
    persisted = pd.read_csv(active_csv, low_memory=False)
    if not persisted.equals(migrated):
        raise RuntimeError(f"OULAD CSV round-trip changed values: {active_csv}")
    output_sha256 = sha256_file(active_csv)

    adapter_path = Path(adapter.__file__).resolve()
    script_path = Path(__file__).resolve()
    archive_csv = archive_dir / "synth_generation.csv"
    migration = {
        "variant": VARIANT,
        "dataset": split.name,
        "model": model,
        "seed": int(seed),
        "arm": "standard",
        "generator_retrained": False,
        "non_outcome_columns_unchanged": True,
        "outcome_columns": list(OUTCOME_COLUMNS),
        "changed_learner_labels": {
            column: int(
                before[column].astype(str).ne(after[column].astype(str)).sum()
            )
            for column in OUTCOME_COLUMNS
        },
        "archived_original": str(archive_csv.relative_to(ROOT)),
        "archived_original_sha256": sha256_file(archive_csv),
        "archived_generation_metadata": str(
            (archive_dir / "generation_metadata.json").relative_to(ROOT)
        ),
        "archived_standard_provenance": str(
            (archive_dir / "standard_provenance.json").relative_to(ROOT)
        ),
        "output": str(active_csv.relative_to(ROOT)),
        "output_sha256": output_sha256,
        "rows": int(len(migrated)),
        "learners": int(migrated["learner_id"].nunique()),
        "adapter": str(adapter_path.relative_to(ROOT)),
        "adapter_sha256": sha256_file(adapter_path),
        "migration_script": str(script_path.relative_to(ROOT)),
        "migration_script_sha256": sha256_file(script_path),
        "dropout_labeler": outcome_metadata,
    }

    generation_metadata = dict(original_metadata)
    generation_metadata.update(
        {
            "experiment_design": EXPERIMENT_DESIGN,
            "generation_arm": "standard",
            "output": str(active_csv),
            "pipeline_normalized_output": str(active_csv),
            "synthetic_output_sha256": output_sha256,
            "data_provenance": dict(split.metadata.get("artifact_metadata", {})),
            "outcome_postprocessing": {
                "variant": VARIANT,
                "generator_retrained": False,
                "non_outcome_columns_unchanged": True,
                "metadata": str(
                    (active_dir / "outcome_migration_metadata.json").relative_to(ROOT)
                ),
                "metadata_sha256_pending": True,
            },
        }
    )
    _write_json_atomic(active_dir / "generation_metadata.json", generation_metadata)

    original_provenance_path = archive_dir / "standard_provenance.json"
    provenance = json.loads(original_provenance_path.read_text(encoding="utf-8"))
    provenance.update(
        {
            "experiment_arm": "standard",
            "synthetic": str(active_csv),
            "generation_metadata": generation_metadata,
            "outcome_migration": migration,
            "evaluation_report": str(
                REPORT_ROOT
                / model
                / f"seed_{seed}"
                / "generation_evaluation_report.json"
            ),
        }
    )
    provenance.pop("tail_targeted_synthetic", None)
    _write_json_atomic(active_dir / "standard_provenance.json", provenance)
    _write_json_atomic(active_dir / "outcome_migration_metadata.json", migration)

    migration_sha256 = sha256_file(active_dir / "outcome_migration_metadata.json")
    generation_metadata["outcome_postprocessing"].pop(
        "metadata_sha256_pending", None
    )
    generation_metadata["outcome_postprocessing"][
        "metadata_sha256"
    ] = migration_sha256
    _write_json_atomic(active_dir / "generation_metadata.json", generation_metadata)
    provenance["generation_metadata"] = generation_metadata
    _write_json_atomic(active_dir / "standard_provenance.json", provenance)

    validated = _load_reusable_standard(
        model, split, active_dir, seed, adapter=adapter
    )
    if validated is None:
        raise RuntimeError(f"Migrated OULAD standard is not reusable: {active_dir}")
    return {**migration, "resumed": False}


def run(models: list[str], seeds: list[int]) -> dict[str, Any]:
    split = adapter.load_split()
    fitted = adapter.fit_dropout_behavior_model(split.train)
    cells = []
    for model in models:
        for seed in seeds:
            result = migrate_cell(split, fitted, model, int(seed))
            cells.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "resumed": bool(result["resumed"]),
                    "output_sha256": result["output_sha256"],
                    "changed_learner_labels": result["changed_learner_labels"],
                }
            )
            print(
                f"validated {model}/seed_{seed} "
                f"(resumed={result['resumed']})",
                flush=True,
            )
    return {
        "variant": VARIANT,
        "dataset": split.name,
        "generator_retrained": False,
        "models": models,
        "seeds": seeds,
        "validated_cells": len(cells),
        "cells": cells,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(list(args.models), [int(seed) for seed in args.seeds])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
