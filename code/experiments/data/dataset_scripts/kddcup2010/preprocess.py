#!/usr/bin/env python3
"""Preprocess a KDD Cup 2010 Bridge-to-Algebra pilot split."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DATASET_DIR = ROOT / "experiments" / "data" / "datasets" / "kddcup2010"
RAW_FILE = (
    DATASET_DIR
    / "raw"
    / "KDD_Cup_2010"
    / "bridge_to_algebra_2008_2009"
    / "bridge_to_algebra_2008_2009_train.txt"
)
PROCESSED_DIR = DATASET_DIR
SEED = 20260703
DEFAULT_MAX_TRAIN_LEARNERS = 2000
DEFAULT_MAX_TEST_LEARNERS = 500
DEFAULT_MAX_INTERACTIONS_PER_LEARNER = 500
OUTPUT_COLUMNS = [
    "learner_id",
    "order",
    "problem_id",
    "step_id",
    "skill_id",
    "correct",
    "attempt_bin",
    "hint_bin",
    "duration_bin",
    "gap_bin",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_value(learner_id: str, seed: int = SEED) -> float:
    digest = hashlib.sha256(f"{seed}:{learner_id}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def normalize_learner_id(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    learner_id = str(value).strip()
    return learner_id if learner_id and learner_id.lower() != "nan" else None


def choose_learner_splits(
    learner_ids: set[str],
    max_train_learners: int,
    max_test_learners: int,
) -> tuple[set[str], set[str]]:
    """Choose order-independent learner samples, then keep them disjoint."""
    preferred_train = [learner for learner in learner_ids if stable_value(learner) < 0.8]
    preferred_test = [learner for learner in learner_ids if stable_value(learner) >= 0.8]
    train_ids = set(
        sorted(preferred_train, key=lambda learner: stable_value(learner, SEED + 1))[:max_train_learners]
    )
    test_ids = set(
        sorted(preferred_test, key=lambda learner: stable_value(learner, SEED + 1))[:max_test_learners]
    )
    return train_ids, test_ids


def discover_learner_ids(raw_file: Path, chunksize: int) -> set[str]:
    learner_ids: set[str] = set()
    for chunk in pd.read_csv(
        raw_file,
        sep="\t",
        usecols=["Anon Student Id"],
        chunksize=chunksize,
        low_memory=False,
    ):
        for value in chunk["Anon Student Id"]:
            learner_id = normalize_learner_id(value)
            if learner_id is not None:
                learner_ids.add(learner_id)
    return learner_ids


def choose_skill(row: Mapping[str, object]) -> str:
    for column in ["KC(KTracedSkills)", "KC(SubSkills)", "KC(Rules)", "KC(Default)"]:
        value = row.get(column)
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value).split("~~")[0].strip()
    return "unknown"


def bin_count(value: object, labels: tuple[str, ...] = ("0", "1", "2", "3+")) -> str:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        number = 0
    if number <= 0:
        return labels[0]
    if number == 1:
        return labels[1]
    if number == 2:
        return labels[2]
    return labels[3]


def bin_duration(value: object) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if seconds < 0:
        return "unknown"
    if seconds < 5:
        return "0-5s"
    if seconds < 15:
        return "5-15s"
    if seconds < 30:
        return "15-30s"
    if seconds < 60:
        return "30-60s"
    if seconds < 120:
        return "60-120s"
    return "120s+"


def bin_gap(previous_time, current_time) -> str:
    if previous_time is None or pd.isna(current_time):
        return "start"
    seconds = max(0.0, (current_time - previous_time).total_seconds())
    minutes = seconds / 60
    if minutes < 1:
        return "0-1m"
    if minutes < 5:
        return "1-5m"
    if minutes < 60:
        return "5-60m"
    if minutes < 24 * 60:
        return "1-24h"
    if minutes < 7 * 24 * 60:
        return "1-7d"
    return "7d+"


def clean_row(row: Mapping[str, object], order: int, previous_time) -> tuple[dict[str, object] | None, object]:
    learner_id = normalize_learner_id(row.get("Anon Student Id"))
    if learner_id is None:
        return None, previous_time

    correct = row.get("Correct First Attempt")
    try:
        correct_int = int(float(correct))
    except (TypeError, ValueError):
        return None, previous_time
    if correct_int not in {0, 1}:
        return None, previous_time

    start_time = pd.to_datetime(row.get("Step Start Time"), errors="coerce")
    gap = bin_gap(previous_time, start_time)
    previous_time = start_time if not pd.isna(start_time) else previous_time

    problem_id = str(row.get("Problem Name", "")).strip() or "unknown"
    step_id = str(row.get("Step Name", "")).strip() or "unknown"
    return (
        {
            "learner_id": learner_id,
            "order": order,
            "problem_id": problem_id,
            "step_id": step_id,
            "skill_id": choose_skill(row),
            "correct": correct_int,
            "attempt_bin": bin_count(row.get("Incorrects")),
            "hint_bin": bin_count(row.get("Hints")),
            "duration_bin": bin_duration(row.get("Step Duration (sec)")),
            "gap_bin": gap,
        },
        previous_time,
    )


def _binned_counts(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    return np.select(
        [values <= 0, values == 1, values == 2],
        ["0", "1", "2"],
        default="3+",
    )


def _binned_durations(series: pd.Series) -> np.ndarray:
    seconds = pd.to_numeric(series, errors="coerce")
    return np.select(
        [
            seconds.isna() | (seconds < 0),
            seconds < 5,
            seconds < 15,
            seconds < 30,
            seconds < 60,
            seconds < 120,
        ],
        ["unknown", "0-5s", "5-15s", "15-30s", "30-60s", "60-120s"],
        default="120s+",
    )


def clean_chunk(
    chunk: pd.DataFrame,
    train_ids: set[str],
    test_ids: set[str],
    learner_orders: dict[str, int],
    learner_previous_time: dict[str, object],
    max_interactions_per_learner: int | None = None,
) -> pd.DataFrame:
    """Vectorized row cleaning while preserving state across source chunks."""
    learner_ids = chunk["Anon Student Id"].astype("string").str.strip()
    selected = learner_ids.isin(train_ids | test_ids)
    frame = chunk.loc[selected].copy()
    frame["_learner_id"] = learner_ids.loc[selected].astype(str)
    frame["_correct"] = pd.to_numeric(frame["Correct First Attempt"], errors="coerce")
    frame = frame[frame["_correct"].isin([0, 1])].copy()
    if frame.empty:
        return pd.DataFrame(columns=[*OUTPUT_COLUMNS, "_split"])

    frame["_start_time"] = pd.to_datetime(frame["Step Start Time"], errors="coerce")
    local_order = frame.groupby("_learner_id", sort=False).cumcount()
    offsets = frame["_learner_id"].map(learner_orders).fillna(0).astype(int)
    if max_interactions_per_learner is not None:
        keep = (offsets + local_order) < int(max_interactions_per_learner)
        frame = frame.loc[keep].copy()
        local_order = local_order.loc[keep]
        offsets = offsets.loc[keep]
        if frame.empty:
            return pd.DataFrame(columns=[*OUTPUT_COLUMNS, "_split"])

    # A missing timestamp must not erase the learner's temporal state. Use the
    # last valid timestamp before the current row, then seed leading rows from
    # the previous source chunk. This matches the original stateful row cleaner
    # for sequences such as valid -> missing -> valid.
    last_valid = frame.groupby("_learner_id", sort=False)["_start_time"].ffill()
    previous = last_valid.groupby(frame["_learner_id"], sort=False).shift(1)
    prior_from_previous_chunk = pd.to_datetime(
        frame["_learner_id"].map(learner_previous_time),
        errors="coerce",
    )
    previous = previous.fillna(prior_from_previous_chunk)
    minutes = (frame["_start_time"] - previous).dt.total_seconds().clip(lower=0) / 60.0
    gap_bins = np.select(
        [
            previous.isna() | frame["_start_time"].isna(),
            minutes < 1,
            minutes < 5,
            minutes < 60,
            minutes < 24 * 60,
            minutes < 7 * 24 * 60,
        ],
        ["start", "0-1m", "1-5m", "5-60m", "1-24h", "1-7d"],
        default="7d+",
    )

    skills = pd.Series("unknown", index=frame.index, dtype="string")
    for column in ["KC(KTracedSkills)", "KC(SubSkills)", "KC(Rules)", "KC(Default)"]:
        if column not in frame.columns:
            continue
        candidate = frame[column].astype("string").fillna("").str.strip()
        use = skills.eq("unknown") & candidate.ne("")
        skills.loc[use] = candidate.loc[use].str.split("~~").str[0].str.strip()

    def identifiers(column: str) -> pd.Series:
        values = frame[column].astype("string").fillna("").str.strip()
        return values.mask(values.eq(""), "unknown")

    output = pd.DataFrame(
        {
            "learner_id": frame["_learner_id"].to_numpy(),
            "order": (offsets + local_order).to_numpy(),
            "problem_id": identifiers("Problem Name").to_numpy(),
            "step_id": identifiers("Step Name").to_numpy(),
            "skill_id": skills.to_numpy(),
            "correct": frame["_correct"].astype(int).to_numpy(),
            "attempt_bin": _binned_counts(frame["Incorrects"]),
            "hint_bin": _binned_counts(frame["Hints"]),
            "duration_bin": _binned_durations(frame["Step Duration (sec)"]),
            "gap_bin": gap_bins,
        }
    )
    output["_split"] = np.where(output["learner_id"].isin(train_ids), "train", "test")

    counts = frame["_learner_id"].value_counts()
    for learner_id, count in counts.items():
        learner_orders[str(learner_id)] = learner_orders.get(str(learner_id), 0) + int(count)
    valid_times = frame.dropna(subset=["_start_time"])
    if not valid_times.empty:
        learner_previous_time.update(
            valid_times.groupby("_learner_id", sort=False)["_start_time"].last().to_dict()
        )
    return output


def open_outputs(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "main": (output_dir / "main.csv").open("w", newline="", encoding="utf-8"),
        "train": (output_dir / "train.csv").open("w", newline="", encoding="utf-8"),
        "test": (output_dir / "test.csv").open("w", newline="", encoding="utf-8"),
    }
    header = pd.DataFrame(columns=OUTPUT_COLUMNS)
    for handle in files.values():
        header.to_csv(handle, index=False)
    return files


def close_outputs(files) -> None:
    for handle in files.values():
        handle.close()


def preprocess(
    raw_file: Path = RAW_FILE,
    output_dir: Path = PROCESSED_DIR,
    max_train_learners: int = DEFAULT_MAX_TRAIN_LEARNERS,
    max_test_learners: int = DEFAULT_MAX_TEST_LEARNERS,
    max_interactions_per_learner: int = DEFAULT_MAX_INTERACTIONS_PER_LEARNER,
    chunksize: int = 200_000,
) -> dict[str, object]:
    available_learner_ids = discover_learner_ids(raw_file, chunksize)
    train_ids, test_ids = choose_learner_splits(
        available_learner_ids, max_train_learners, max_test_learners
    )
    files = open_outputs(output_dir)
    learner_orders: dict[str, int] = {}
    learner_previous_time = {}
    written_train_ids: set[str] = set()
    written_test_ids: set[str] = set()
    stats = {
        "raw_rows_seen": 0,
        "clean_rows": 0,
        "train_rows": 0,
        "test_rows": 0,
        "max_train_learners": max_train_learners,
        "max_test_learners": max_test_learners,
        "max_interactions_per_learner": max_interactions_per_learner,
    }

    try:
        for chunk in pd.read_csv(raw_file, sep="\t", chunksize=chunksize, low_memory=False):
            stats["raw_rows_seen"] += len(chunk)
            cleaned = clean_chunk(
                chunk,
                train_ids,
                test_ids,
                learner_orders,
                learner_previous_time,
                max_interactions_per_learner,
            )
            if cleaned.empty:
                continue
            cleaned[OUTPUT_COLUMNS].to_csv(files["main"], index=False, header=False)
            stats["clean_rows"] += len(cleaned)
            for split_name, written_ids in (
                ("train", written_train_ids),
                ("test", written_test_ids),
            ):
                split_frame = cleaned[cleaned["_split"] == split_name]
                if split_frame.empty:
                    continue
                split_frame[OUTPUT_COLUMNS].to_csv(files[split_name], index=False, header=False)
                written_ids.update(split_frame["learner_id"].astype(str).unique())
                stats[f"{split_name}_rows"] += len(split_frame)
    finally:
        close_outputs(files)

    learner_overlap = train_ids & test_ids
    if learner_overlap:
        raise RuntimeError(f"Learner leakage across KDD splits: {len(learner_overlap)} overlapping ids.")

    metadata = {
        "dataset": "kddcup2010_bridge_to_algebra_2008_2009",
        "source_file": str(raw_file),
        "source_sha256": sha256_file(raw_file),
        "columns": OUTPUT_COLUMNS,
        "target": "correct",
        "split_seed": SEED,
        "split_unit": "learner_id",
        "learner_overlap": 0,
        "preprocessing_script_sha256": sha256_file(Path(__file__)),
        "selection_method": "two_pass_stable_hash_sample",
        "available_learners": len(available_learner_ids),
        "train_learners": len(written_train_ids),
        "test_learners": len(written_test_ids),
        "main_sha256": sha256_file(output_dir / "main.csv"),
        "train_sha256": sha256_file(output_dir / "train.csv"),
        "test_sha256": sha256_file(output_dir / "test.csv"),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
        },
        **stats,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def run_preprocessing() -> dict[str, object]:
    """Pipeline entry point using the canonical pilot-split defaults."""
    return preprocess()


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess KDD Cup 2010 Bridge-to-Algebra pilot split.")
    parser.add_argument("--raw-file", type=Path, default=RAW_FILE)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--max-train-learners", type=int, default=DEFAULT_MAX_TRAIN_LEARNERS)
    parser.add_argument("--max-test-learners", type=int, default=DEFAULT_MAX_TEST_LEARNERS)
    parser.add_argument(
        "--max-interactions-per-learner",
        type=int,
        default=DEFAULT_MAX_INTERACTIONS_PER_LEARNER,
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()
    metadata = preprocess(
        raw_file=args.raw_file,
        output_dir=args.output_dir,
        max_train_learners=args.max_train_learners,
        max_test_learners=args.max_test_learners,
        max_interactions_per_learner=args.max_interactions_per_learner,
        chunksize=args.chunksize,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
