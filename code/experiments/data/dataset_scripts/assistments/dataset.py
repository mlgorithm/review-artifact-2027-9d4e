"""ASSISTments dataset adapter for the generic experiment pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENTS = ROOT / "experiments"
DATASET_DIR = EXPERIMENTS / "data" / "datasets" / "assistments"
OUTPUTS_DIR = EXPERIMENTS / "outputs" / "assistments"
REPORTS_DIR = EXPERIMENTS / "reports" / "assistments_2009_2010_skill_builder"
sys.path.insert(0, str(EXPERIMENTS))

from data.dataset_scripts.common import DatasetSplit, validate_processed_artifacts


DATASET_NAME = "assistments_2009_2010_skill_builder"
CONFIG_PATH = EXPERIMENTS / "configs" / "assistments_preprocessing.yaml"
PIPELINE_CONFIG_PATH = EXPERIMENTS / "configs" / "assistments_pipeline.yaml"


def load_yaml(path: Path) -> Mapping[str, object]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def preprocessing_config(config_path: Path = CONFIG_PATH) -> Mapping[str, object]:
    return load_yaml(config_path)


def pipeline_config(config_path: Path = PIPELINE_CONFIG_PATH) -> Mapping[str, object]:
    return load_yaml(config_path)


def processed_dir(config: Mapping[str, object] | None = None) -> Path:
    return DATASET_DIR


def train_path(config: Mapping[str, object] | None = None) -> Path:
    return processed_dir(config) / "train.csv"


def test_path(config: Mapping[str, object] | None = None) -> Path:
    return processed_dir(config) / "test.csv"


def synthetic_root(config: Mapping[str, object] | None = None) -> Path:
    return OUTPUTS_DIR


def evaluation_root(config: Mapping[str, object] | None = None) -> Path:
    return REPORTS_DIR


def ignored_columns() -> tuple[str, ...]:
    return ("learner_id", "order")


def load_split() -> DatasetSplit:
    prep_config = preprocessing_config()
    pipe_config = pipeline_config()
    model_config = pipe_config.get("downstream_models", {})
    generation_columns = pipe_config.get("generation_columns", {})
    root = processed_dir(prep_config)
    artifact_metadata = validate_processed_artifacts(
        root / "metadata.json",
        Path(__file__).with_name("preprocess.py"),
        train_path(prep_config),
        test_path(prep_config),
        CONFIG_PATH,
    )
    train = pd.read_csv(train_path(prep_config), low_memory=False)
    test = pd.read_csv(test_path(prep_config), low_memory=False)
    return DatasetSplit(
        name=DATASET_NAME,
        train=train,
        test=test,
        train_path=train_path(prep_config),
        test_path=test_path(prep_config),
        root=root,
        synthetic_root=synthetic_root(prep_config),
        evaluation_root=evaluation_root(prep_config),
        columns=dict(generation_columns.get("columns", generation_columns)),
        ignore_columns=ignored_columns(),
        target=str(model_config.get("target", "correct")),
        group_column=str(model_config.get("group_column", "skill_id")),
        seed=int(model_config.get("seed", 20260703)),
        metadata={
            "preprocessing_config": str(CONFIG_PATH),
            "pipeline_config": str(PIPELINE_CONFIG_PATH),
            "artifact_metadata": artifact_metadata,
        },
    )


def generation_to_downstream_supervised(
    frame: pd.DataFrame, min_history: int = 1, return_learner_ids: bool = False
):
    """Build a leakage-aware downstream table from generation-schema rows.

    When ``return_learner_ids`` is True, also return the per-row learner id
    aligned to the supervised rows. The evaluator uses these ids to build tail
    masks that line up with the model scores, instead of reconstructing this
    function's row filtering (which would silently mis-align if the filtering
    logic ever changed).
    """
    required = {"learner_id", "order", "skill_id", "correct"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ASSISTments generation frame is missing columns: {sorted(missing)}")

    df = frame.copy()
    df["learner_id"] = df["learner_id"].astype(str)
    df["order"] = pd.to_numeric(df["order"], errors="coerce")
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["learner_id", "order", "correct"])
    df = df.sort_values(["learner_id", "order"]).reset_index(drop=True)
    df["interaction_index"] = df.groupby("learner_id").cumcount()

    for column in ["correct", "attempt_bin", "hint_bin", "response_time_bin", "opportunity_bin"]:
        if column in df.columns:
            df[f"prev_{column}"] = df.groupby("learner_id")[column].shift(1)

    df["cum_correct_count"] = (
        df.groupby("learner_id")["correct"]
        .apply(lambda series: series.shift(1).expanding().sum())
        .reset_index(level=0, drop=True)
    )
    df["cum_interactions"] = df["interaction_index"]
    # Use .where (not .replace(0, pd.NA)): injecting pd.NA into the int denominator
    # upcasts the result to object dtype, which the downstream feature typer then
    # mis-routes to the categorical encoder, destroying this continuous predictor.
    df["cum_correct_rate"] = df["cum_correct_count"] / df["cum_interactions"].where(df["cum_interactions"] != 0)
    df = df[df["interaction_index"] >= min_history].copy()

    columns = [
        "skill_id",
        "interaction_index",
        "prev_correct",
        "cum_correct_count",
        "cum_interactions",
        "cum_correct_rate",
        "prev_attempt_bin",
        "prev_hint_bin",
        "prev_response_time_bin",
        "prev_opportunity_bin",
        "correct",
    ]
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    supervised = df[columns]
    if return_learner_ids:
        return supervised, df["learner_id"].tolist()
    return supervised
