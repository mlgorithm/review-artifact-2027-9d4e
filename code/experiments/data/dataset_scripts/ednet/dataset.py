"""EdNet KT1 dataset adapter for the generic experiment pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.dataset_scripts.common import DatasetSplit, rebuild_from_key, validate_processed_artifacts


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENTS = ROOT / "experiments"
DATASET_DIR = EXPERIMENTS / "data" / "datasets" / "ednet"
DATASET_NAME = "ednet_kt1"
OUTPUTS_DIR = EXPERIMENTS / "outputs" / "ednet"
REPORTS_DIR = EXPERIMENTS / "reports" / DATASET_NAME

# Each EdNet question has exactly one primary skill (tag) and one part, so both
# are functionally determined by question_id. They are withheld from generation
# and rebuilt from the generated question_id, so a generator cannot emit a
# question paired with the wrong skill/part.
DERIVED_KEY = "question_id"
DERIVED_COLUMNS = ("skill_id", "part")


def train_path() -> Path:
    return DATASET_DIR / "train.csv"


def test_path() -> Path:
    return DATASET_DIR / "test.csv"


def ignored_columns() -> tuple[str, ...]:
    return ("learner_id", "order")


def derived_columns() -> tuple[str, ...]:
    return DERIVED_COLUMNS


def rebuild_derived_columns(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    return rebuild_from_key(frame, reference, DERIVED_KEY, DERIVED_COLUMNS)


def load_split() -> DatasetSplit:
    artifact_metadata = validate_processed_artifacts(
        DATASET_DIR / "metadata.json",
        Path(__file__).with_name("preprocess.py"),
        train_path(),
        test_path(),
    )
    train = pd.read_csv(train_path(), low_memory=False)
    test = pd.read_csv(test_path(), low_memory=False)
    columns = {
        "learner": "learner_id",
        "order": "order",
        "skill": "skill_id",
        "correct": "correct",
    }
    return DatasetSplit(
        name=DATASET_NAME,
        train=train,
        test=test,
        train_path=train_path(),
        test_path=test_path(),
        root=DATASET_DIR,
        synthetic_root=OUTPUTS_DIR,
        evaluation_root=REPORTS_DIR,
        columns=columns,
        ignore_columns=ignored_columns(),
        derived_columns=derived_columns(),
        target="correct",
        group_column="skill_id",
        seed=20260703,
        metadata={
            "source": "EdNet KT1",
            "artifact_metadata": artifact_metadata,
        },
    )


def generation_to_downstream_supervised(
    frame: pd.DataFrame, min_history: int = 1, return_learner_ids: bool = False
) -> pd.DataFrame:
    required = {"learner_id", "order", "skill_id", "correct"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"EdNet frame is missing columns: {sorted(missing)}")

    df = frame.copy()
    df["learner_id"] = df["learner_id"].astype(str)
    df["order"] = pd.to_numeric(df["order"], errors="coerce")
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["learner_id", "order", "correct"])
    df = df.sort_values(["learner_id", "order"]).reset_index(drop=True)
    df["interaction_index"] = df.groupby("learner_id").cumcount()

    for column in ["correct", "elapsed_time_bin", "timestamp_gap_bin", "part"]:
        if column in df.columns:
            df[f"prev_{column}"] = df.groupby("learner_id")[column].shift(1)

    df["cum_correct_count"] = (
        df.groupby("learner_id")["correct"]
        .apply(lambda series: series.shift(1).expanding().sum())
        .reset_index(level=0, drop=True)
    )
    df["cum_interactions"] = df["interaction_index"]
    # .where keeps float64; .replace(0, pd.NA) would upcast to object and get
    # mis-encoded as categorical downstream (see assistments adapter for detail).
    df["cum_correct_rate"] = df["cum_correct_count"] / df["cum_interactions"].where(df["cum_interactions"] != 0)
    df = df[df["interaction_index"] >= min_history].copy()

    columns = [
        "skill_id",
        "question_id",
        "part",
        "interaction_index",
        "prev_correct",
        "cum_correct_count",
        "cum_interactions",
        "cum_correct_rate",
        "prev_elapsed_time_bin",
        "prev_timestamp_gap_bin",
        "prev_part",
        "correct",
    ]
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    supervised = df[columns]
    if return_learner_ids:
        return supervised, df["learner_id"].tolist()
    return supervised
