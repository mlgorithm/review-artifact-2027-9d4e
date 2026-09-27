"""Common dataset objects for the generic experiment pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    train_path: Path
    test_path: Path
    root: Path
    synthetic_root: Path
    evaluation_root: Path
    columns: Mapping[str, str]
    ignore_columns: tuple[str, ...]
    target: str
    group_column: str
    seed: int
    # Columns that are a deterministic function of another generated column, so
    # they are withheld from generation and rebuilt afterwards by the adapter's
    # `rebuild_derived_columns`. See `rebuild_from_key` for why.
    derived_columns: tuple[str, ...] = ()
    label: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_processed_artifacts(
    metadata_path: Path,
    preprocessing_script: Path,
    train_path: Path,
    test_path: Path,
    config_path: Path | None = None,
) -> Mapping[str, object]:
    """Refuse stale processed data that cannot be attributed to current code."""
    if not metadata_path.exists():
        raise RuntimeError(f"Missing preprocessing metadata: {metadata_path}. Run the pipeline with --preprocess.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    runtime = metadata.get("runtime", {}) if isinstance(metadata.get("runtime"), dict) else {}
    recorded_script_hash = metadata.get("preprocessing_script_sha256") or runtime.get(
        "preprocessing_script_sha256"
    )
    current_script_hash = _sha256_file(preprocessing_script)
    if recorded_script_hash != current_script_hash:
        raise RuntimeError(
            f"Processed data in {metadata_path.parent} was not produced by the current preprocessor. "
            "Run the pipeline with --preprocess before generation."
        )
    if config_path is not None:
        recorded_config_hash = runtime.get("config_sha256") or metadata.get("config_sha256")
        if recorded_config_hash != _sha256_file(config_path):
            raise RuntimeError(
                f"Processed data in {metadata_path.parent} was produced with a different config. "
                "Run the pipeline with --preprocess before generation."
            )
    for role, path in (("train", train_path), ("test", test_path)):
        recorded_hash = metadata.get(f"{role}_generation_sha256") or metadata.get(f"{role}_sha256")
        if not path.exists() or recorded_hash != _sha256_file(path):
            raise RuntimeError(
                f"Processed {role} artifact does not match its metadata: {path}. "
                "Run the pipeline with --preprocess before generation."
            )
    return metadata


def rebuild_from_key(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    key: str,
    derived: tuple[str, ...],
) -> pd.DataFrame:
    """Restore columns that are functionally determined by ``key``.

    Generators emit each column independently, so left to themselves they produce
    self-contradictory records: an OULAD row whose `skill_id` disagrees with its
    `module_id`, or an EdNet row whose `part` disagrees with its `question_id`.
    Such rows cannot exist in the real data, are trivially separable by a
    real-vs-synthetic classifier, and poison the downstream models. We therefore
    keep these columns out of generation and rebuild them here from the generated
    key, using a lookup taken from the real training split.
    """
    columns = [column for column in derived if column in reference.columns]
    if key not in frame.columns or not columns:
        return frame

    lookup = reference.drop_duplicates(key).set_index(key)[columns]
    restored = frame.copy()
    joined = restored[[key]].join(lookup, on=key)

    unresolved = joined[columns[0]].isna()
    if unresolved.any():
        # A key never seen in the reference cannot be resolved; fall back to the
        # most common key's row so the record stays internally consistent.
        fallback = lookup.loc[reference[key].mode().iloc[0]]
        for column in columns:
            joined.loc[unresolved, column] = fallback[column]

    for column in columns:
        restored[column] = joined[column].to_numpy()
    return restored


def enforce_learner_static_columns(
    frame: pd.DataFrame,
    learner_column: str,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Make declared learner-level columns constant within each trajectory.

    Static attributes and terminal outcomes are repeated on every real event so
    generators can learn their joint distribution with the trajectory. A generic
    event generator can nevertheless decode a different category at each step.
    That state is impossible in the source data and would leak the outcome time
    into downstream features. We therefore select the learner-wise modal value
    (ties resolved by first appearance) and repeat it across the synthetic
    trajectory. Only columns explicitly declared static by an adapter are changed.
    """
    present = [column for column in columns if column in frame.columns]
    if learner_column not in frame.columns or not present:
        return frame
    restored = frame.copy()
    for column in present:
        def stable_mode(values: pd.Series):
            counts = values.astype(str).value_counts(dropna=False)
            maximum = int(counts.max()) if len(counts) else 0
            candidates = set(counts[counts == maximum].index.astype(str))
            for value in values.astype(str):
                if value in candidates:
                    return value
            return ""

        by_learner = restored.groupby(learner_column, sort=False)[column].transform(stable_mode)
        restored[column] = by_learner.to_numpy()
    return restored
