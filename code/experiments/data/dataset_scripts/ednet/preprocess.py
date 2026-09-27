#!/usr/bin/env python3
"""Preprocess EdNet KT1 into canonical train/test trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import io
import json
import platform
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DATASET_DIR = ROOT / "experiments" / "data" / "datasets" / "ednet"
RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR
KT1_ZIP = RAW_DIR / "ednet_kt1.zip"
CONTENTS_ZIP = RAW_DIR / "contents.zip"
SEED = 20260703
TRAIN_RATIO = 0.8
DEFAULT_MAX_LEARNERS = 10_000
OUTPUT_COLUMNS = [
    "learner_id",
    "order",
    "question_id",
    "skill_id",
    "part",
    "correct",
    "elapsed_time_bin",
    "timestamp_gap_bin",
    "attempt_bin",
    "hint_bin",
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


def stable_train_assignment(learner_id: str, train_ratio: float = TRAIN_RATIO, seed: int = SEED) -> bool:
    return stable_value(learner_id, seed) < train_ratio


def bin_elapsed(milliseconds: int) -> str:
    seconds = milliseconds / 1000
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


def bin_gap(milliseconds: int | None) -> str:
    if milliseconds is None:
        return "start"
    minutes = max(0, milliseconds) / 60000
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


def primary_skill(tags: object) -> str:
    text = "" if pd.isna(tags) else str(tags)
    return text.split(";")[0].strip() if text else "unknown"


def load_question_metadata(contents_zip: Path) -> dict[str, dict[str, str]]:
    with zipfile.ZipFile(contents_zip) as archive:
        with archive.open("contents/questions.csv") as handle:
            questions = pd.read_csv(handle, dtype=str)
    questions["skill_id"] = questions["tags"].map(primary_skill)
    questions["part"] = questions["part"].fillna("unknown").astype(str)
    questions["correct_answer"] = questions["correct_answer"].fillna("").astype(str).str.strip()
    return questions.set_index("question_id")[["skill_id", "part", "correct_answer"]].to_dict("index")


def iter_learner_files(archive: zipfile.ZipFile) -> Iterable[str]:
    for name in archive.namelist():
        if name.startswith("KT1/") and name.endswith(".csv") and not name.endswith("/"):
            yield name


def select_learner_files(file_names: Iterable[str], max_learners: int | None) -> list[str]:
    """Select a deterministic hash sample, independent of ZIP member ordering."""
    names = list(file_names)
    if max_learners is None or max_learners >= len(names):
        return names
    return heapq.nsmallest(
        max_learners,
        names,
        key=lambda name: stable_value(Path(name).stem, SEED + 1),
    )


def clean_learner_rows(
    archive: zipfile.ZipFile,
    file_name: str,
    question_metadata: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    learner_id = Path(file_name).stem
    with archive.open(file_name) as raw_handle:
        text_handle = io.TextIOWrapper(raw_handle, encoding="utf-8")
        reader = csv.DictReader(text_handle)
        rows = []
        for row in reader:
            question_id = str(row.get("question_id", "")).strip()
            metadata = question_metadata.get(question_id)
            if metadata is None:
                continue
            user_answer = str(row.get("user_answer", "")).strip()
            correct_answer = str(metadata.get("correct_answer", "")).strip()
            if user_answer == "" or correct_answer == "":
                continue
            try:
                timestamp = int(float(row.get("timestamp", "")))
                solving_id = int(float(row.get("solving_id", "")))
                elapsed_time = int(float(row.get("elapsed_time", "")))
            except (TypeError, ValueError):
                continue
            if elapsed_time < 0:
                continue
            rows.append(
                {
                    "learner_id": learner_id,
                    "timestamp": timestamp,
                    "solving_id": solving_id,
                    "question_id": question_id,
                    "skill_id": str(metadata["skill_id"]),
                    "part": str(metadata["part"]),
                    "correct": 1 if user_answer == correct_answer else 0,
                    "elapsed_time_bin": bin_elapsed(elapsed_time),
                }
            )

    rows = sorted(rows, key=lambda item: (item["timestamp"], item["solving_id"], item["question_id"]))
    cleaned = []
    previous_timestamp = None
    for order, row in enumerate(rows):
        gap = None if previous_timestamp is None else int(row["timestamp"]) - previous_timestamp
        previous_timestamp = int(row["timestamp"])
        cleaned.append(
            {
                "learner_id": row["learner_id"],
                "order": order,
                "question_id": row["question_id"],
                "skill_id": row["skill_id"],
                "part": row["part"],
                "correct": row["correct"],
                "elapsed_time_bin": row["elapsed_time_bin"],
                "timestamp_gap_bin": bin_gap(gap),
                "attempt_bin": "1",
                "hint_bin": "0",
            }
        )
    return cleaned


def open_writers(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "main": (output_dir / "main.csv").open("w", newline="", encoding="utf-8"),
        "train": (output_dir / "train.csv").open("w", newline="", encoding="utf-8"),
        "test": (output_dir / "test.csv").open("w", newline="", encoding="utf-8"),
    }
    writers = {name: csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS) for name, handle in files.items()}
    for writer in writers.values():
        writer.writeheader()
    return files, writers


def close_files(files) -> None:
    for handle in files.values():
        handle.close()


def preprocess(
    kt1_zip: Path = KT1_ZIP,
    contents_zip: Path = CONTENTS_ZIP,
    output_dir: Path = PROCESSED_DIR,
    min_interactions: int = 2,
    max_learners: int | None = None,
) -> dict[str, object]:
    question_metadata = load_question_metadata(contents_zip)
    files, writers = open_writers(output_dir)

    stats = {
        "raw_learner_files": 0,
        "kept_learners": 0,
        "train_learners": 0,
        "test_learners": 0,
        "clean_rows": 0,
        "train_rows": 0,
        "test_rows": 0,
        "min_interactions": min_interactions,
    }
    train_learner_ids: set[str] = set()
    test_learner_ids: set[str] = set()

    try:
        with zipfile.ZipFile(kt1_zip) as archive:
            available_files = list(iter_learner_files(archive))
            selected_files = select_learner_files(available_files, max_learners)
            stats["available_learner_files"] = len(available_files)
            stats["selection_method"] = "stable_hash_sample"
            for file_name in selected_files:
                stats["raw_learner_files"] += 1

                rows = clean_learner_rows(archive, file_name, question_metadata)
                if len(rows) < min_interactions:
                    continue

                learner_id = rows[0]["learner_id"]
                is_train = stable_train_assignment(str(learner_id))
                split_name = "train" if is_train else "test"
                (train_learner_ids if is_train else test_learner_ids).add(str(learner_id))

                writers["main"].writerows(rows)
                writers[split_name].writerows(rows)

                stats["kept_learners"] += 1
                stats[f"{split_name}_learners"] += 1
                stats["clean_rows"] += len(rows)
                stats[f"{split_name}_rows"] += len(rows)
    finally:
        close_files(files)

    learner_overlap = train_learner_ids & test_learner_ids
    if learner_overlap:
        raise RuntimeError(f"Learner leakage across EdNet splits: {len(learner_overlap)} overlapping ids.")

    metadata = {
        "dataset": "ednet_kt1",
        "columns": OUTPUT_COLUMNS,
        "target": "correct",
        "split_seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "split_unit": "learner_id",
        "learner_overlap": 0,
        "max_raw_learner_files": max_learners,
        "question_metadata_rows": len(question_metadata),
        "kt1_zip": str(kt1_zip),
        "contents_zip": str(contents_zip),
        "kt1_zip_sha256": sha256_file(kt1_zip),
        "contents_zip_sha256": sha256_file(contents_zip),
        "preprocessing_script_sha256": sha256_file(Path(__file__)),
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
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def run_preprocessing() -> dict[str, object]:
    """Pipeline entry point using the same bounded sample as the CLI default."""
    return preprocess(max_learners=DEFAULT_MAX_LEARNERS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess EdNet KT1.")
    parser.add_argument("--kt1-zip", type=Path, default=KT1_ZIP)
    parser.add_argument("--contents-zip", type=Path, default=CONTENTS_ZIP)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--min-interactions", type=int, default=2)
    parser.add_argument("--max-learners", type=int, default=DEFAULT_MAX_LEARNERS)
    args = parser.parse_args()

    metadata = preprocess(
        kt1_zip=args.kt1_zip,
        contents_zip=args.contents_zip,
        output_dir=args.output_dir,
        min_interactions=args.min_interactions,
        max_learners=args.max_learners,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
