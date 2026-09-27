#!/usr/bin/env python3
"""Preprocess ASSISTments 2009-2010 skill-builder data.

The script has two stages:

1. clean-split: raw CSV -> cleaned learner-level train/test CSVs.
2. make-supervised: cleaned splits -> supervised train/test CSVs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Mapping, Tuple

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENTS = ROOT / "experiments"
DEFAULT_CONFIG = EXPERIMENTS / "configs" / "assistments_preprocessing.yaml"


def load_config(path: Path) -> Mapping[str, object]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_metadata(config_path: Path) -> Mapping[str, object]:
    return {
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "preprocessing_script_sha256": sha256_file(Path(__file__)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "pyyaml": yaml.__version__,
    }


def clean_dataframe(df: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    cleaning = config["cleaning"]
    required = list(cleaning["required_columns"])
    df = df.copy()

    df = df.dropna(subset=required)

    for column in cleaning.get("binary_columns", []):
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            df = df[df[column].isin([0, 1])]
            df[column] = df[column].astype(int)

    for column in cleaning.get("nonnegative_columns", []):
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            df.loc[df[column] < 0, column] = pd.NA

    duplicate_subset = [col for col in cleaning.get("drop_duplicate_subset", []) if col in df.columns]
    if duplicate_subset:
        df = df.drop_duplicates(subset=duplicate_subset, keep="first")

    learner_col = config["columns"]["learner_id"]
    order_col = config["columns"]["order"]
    df[order_col] = pd.to_numeric(df[order_col], errors="coerce")
    df = df.dropna(subset=[learner_col, order_col])
    df = df.sort_values([learner_col, order_col]).reset_index(drop=True)

    return df


def stable_train_assignment(
    learner_id: object,
    train_ratio: float,
    seed: int,
) -> bool:
    """Assign a learner without consulting any outcomes or trajectory statistics."""
    digest = hashlib.sha256(f"{seed}:{learner_id}".encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12)
    return value < train_ratio


def split_by_learner(df: pd.DataFrame, config: Mapping[str, object]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_config = config["split"]
    learner_col = split_config["learner_column"]
    train_ratio = float(split_config["train_ratio"])
    seed = int(split_config["seed"])

    learner_ids = df[learner_col].drop_duplicates()
    train_id_set = {
        learner_id
        for learner_id in learner_ids
        if stable_train_assignment(learner_id, train_ratio, seed)
    }
    train = df[df[learner_col].isin(train_id_set)].copy()
    test = df[~df[learner_col].isin(train_id_set)].copy()
    overlap = set(train[learner_col].unique()) & set(test[learner_col].unique())
    if overlap:
        raise RuntimeError(f"Learner leakage across ASSISTments splits: {len(overlap)} overlapping ids.")
    return train, test


def write_metadata(path: Path, metadata: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clean_split(config: Mapping[str, object], config_path: Path) -> None:
    raw_path = project_path(str(config["raw_path"]))
    processed_dir = project_path(str(config["processed_dir"]))
    processed_dir.mkdir(parents=True, exist_ok=True)
    encoding = str(config.get("encoding", "utf-8"))

    df = pd.read_csv(raw_path, low_memory=False, encoding=encoding)
    raw_rows = len(df)
    df = clean_dataframe(df, config)
    train, test = split_by_learner(df, config)

    clean_path = processed_dir / "assistments_clean.csv"
    train_path = processed_dir / "clean_train.csv"
    test_path = processed_dir / "clean_test.csv"

    df.to_csv(clean_path, index=False)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    learner_col = config["columns"]["learner_id"]
    target_col = str(config["target"])

    metadata = {
        "raw_rows": raw_rows,
        "clean_rows": len(df),
        "train_rows": len(train),
        "test_rows": len(test),
        "clean_learners": int(df[learner_col].nunique()),
        "train_learners": int(train[learner_col].nunique()),
        "test_learners": int(test[learner_col].nunique()),
        "train_correct_rate": float(train[target_col].mean()) if len(train) else 0.0,
        "test_correct_rate": float(test[target_col].mean()) if len(test) else 0.0,
        "target": target_col,
        "split_method": "stable_hash_by_learner",
        "split_unit": "learner_id",
        "learner_overlap": 0,
        "split_seed": int(config["split"]["seed"]),
        "raw_sha256": sha256_file(raw_path),
        "clean_sha256": sha256_file(clean_path),
        "train_sha256": sha256_file(train_path),
        "test_sha256": sha256_file(test_path),
        "runtime": runtime_metadata(config_path),
    }
    write_metadata(processed_dir / "clean_split_metadata.json", metadata)


def build_xy_for_split(
    df: pd.DataFrame,
    config: Mapping[str, object],
    feature_columns: list[str] | None = None,
) -> Tuple[pd.DataFrame, list[str]]:
    ignore_columns = set(config.get("ignore_columns", []))
    xy_config = config["xy"]
    target_col = str(xy_config["target"])
    learner_col = config["columns"]["learner_id"]
    order_col = config["columns"]["order"]
    excluded_feature_columns = ignore_columns | {learner_col, order_col}
    min_history = int(xy_config.get("min_history", 1))

    df = df.sort_values([learner_col, order_col]).reset_index(drop=True)
    df["interaction_index"] = df.groupby(learner_col).cumcount()

    # Shift answer-consequence columns so the current row never sees its own outcome process.
    history_columns = [col for col in xy_config.get("history_columns", []) if col in df.columns]
    for column in history_columns:
        df[f"prev_{column}"] = df.groupby(learner_col)[column].shift(1)
        df[f"cummean_prev_{column}"] = (
            df.groupby(learner_col)[column]
            .apply(lambda series: series.shift(1).expanding().mean())
            .reset_index(level=0, drop=True)
        )

    df["prev_correct"] = df.groupby(learner_col)[target_col].shift(1)
    df["cum_correct_count"] = (
        df.groupby(learner_col)[target_col]
        .apply(lambda series: series.shift(1).expanding().sum())
        .reset_index(level=0, drop=True)
    )
    df["cum_interactions"] = df["interaction_index"]
    df["cum_correct_rate"] = df["cum_correct_count"] / df["cum_interactions"].replace(0, pd.NA)

    df = df[df["interaction_index"] >= min_history].copy()

    passthrough = [col for col in xy_config.get("passthrough_current_columns", []) if col in df.columns]
    engineered = [
        "interaction_index",
        "prev_correct",
        "cum_correct_count",
        "cum_interactions",
        "cum_correct_rate",
    ]
    engineered.extend(f"prev_{col}" for col in history_columns)
    engineered.extend(f"cummean_prev_{col}" for col in history_columns)

    if feature_columns is None:
        feature_columns = [
            col
            for col in passthrough + engineered
            if col in df.columns and col != target_col and col not in excluded_feature_columns
        ]
    else:
        for column in feature_columns:
            if column not in df:
                df[column] = pd.NA

    supervised = df[feature_columns + [target_col]].copy()
    return supervised, feature_columns


def make_supervised(config: Mapping[str, object], config_path: Path) -> None:
    processed_dir = project_path(str(config["processed_dir"]))
    train_path = processed_dir / "clean_train.csv"
    test_path = processed_dir / "clean_test.csv"
    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)

    train_supervised, feature_columns = build_xy_for_split(train_df, config)
    train_supervised_path = processed_dir / "train_supervised.csv"
    train_supervised.to_csv(train_supervised_path, index=False)

    test_supervised, _ = build_xy_for_split(test_df, config, feature_columns)
    test_supervised_path = processed_dir / "test_supervised.csv"
    test_supervised.to_csv(test_supervised_path, index=False)

    xy_config = config["xy"]
    target_col = str(xy_config["target"])
    ignore_columns = set(config.get("ignore_columns", []))
    learner_col = str(config["columns"]["learner_id"])
    order_col = str(config["columns"]["order"])
    min_history = int(xy_config.get("min_history", 1))
    metadata = {
        "train_source": str(train_path),
        "test_source": str(test_path),
        "train_rows": len(train_supervised),
        "test_rows": len(test_supervised),
        "target": target_col,
        "feature_columns": feature_columns,
        "supervised_columns": feature_columns + [target_col],
        "ignored_columns": sorted(ignore_columns),
        "excluded_identifier_columns": sorted({learner_col, order_col}),
        "min_history": min_history,
        "train_source_sha256": sha256_file(train_path),
        "test_source_sha256": sha256_file(test_path),
        "train_supervised_sha256": sha256_file(train_supervised_path),
        "test_supervised_sha256": sha256_file(test_supervised_path),
        "runtime": runtime_metadata(config_path),
    }
    write_metadata(processed_dir / "xy_metadata.json", metadata)


def make_xy(config: Mapping[str, object], config_path: Path) -> None:
    make_supervised(config, config_path)


def bin_series(series: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    upper = float("inf")
    cut_bins = list(bins) + [upper]
    return pd.cut(numeric, bins=cut_bins, labels=labels, right=False, include_lowest=True).astype("string")


def build_generation_split(df: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    generation = config["generation"]
    source_columns = generation["source_columns"]
    output = pd.DataFrame()
    for output_name, source_name in source_columns.items():
        output[output_name] = df[source_name]

    for source_name, bin_config in generation.get("binning", {}).items():
        output_name = bin_config["output"]
        output[output_name] = bin_series(
            output[source_name],
            [float(value) for value in bin_config["bins"]],
            [str(value) for value in bin_config["labels"]],
        )
        if source_name != output_name and source_name in output:
            output = output.drop(columns=[source_name])

    output["learner_id"] = output["learner_id"].astype(str)
    output["order"] = pd.to_numeric(output["order"], errors="coerce")
    output["correct"] = pd.to_numeric(output["correct"], errors="coerce").astype("Int64")
    output = output.sort_values(["learner_id", "order"]).reset_index(drop=True)
    output["order"] = output.groupby("learner_id").cumcount()
    return output[list(generation["output_columns"])]


def make_generation(config: Mapping[str, object], config_path: Path) -> None:
    processed_dir = project_path(str(config["processed_dir"]))
    train_path = processed_dir / "clean_train.csv"
    test_path = processed_dir / "clean_test.csv"
    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)

    train_generation = build_generation_split(train_df, config)
    test_generation = build_generation_split(test_df, config)

    train_generation_path = processed_dir / "train.csv"
    test_generation_path = processed_dir / "test.csv"
    train_generation.to_csv(train_generation_path, index=False)
    test_generation.to_csv(test_generation_path, index=False)

    metadata = {
        "train_source": str(train_path),
        "test_source": str(test_path),
        "train_rows": len(train_generation),
        "test_rows": len(test_generation),
        "generation_columns": list(train_generation.columns),
        "train_source_sha256": sha256_file(train_path),
        "test_source_sha256": sha256_file(test_path),
        "train_generation_sha256": sha256_file(train_generation_path),
        "test_generation_sha256": sha256_file(test_generation_path),
        "runtime": runtime_metadata(config_path),
    }
    write_metadata(processed_dir / "metadata.json", metadata)


def run_preprocessing(config_path: Path = DEFAULT_CONFIG) -> None:
    config = load_config(config_path)
    clean_split(config, config_path)
    make_supervised(config, config_path)
    make_generation(config, config_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ASSISTments data.")
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["clean-split", "make-supervised", "make-generation", "make-xy", "all"],
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.command == "clean-split":
        clean_split(config, args.config)
    elif args.command in {"make-supervised", "make-xy"}:
        make_supervised(config, args.config)
    elif args.command == "make-generation":
        make_generation(config, args.config)
    elif args.command == "all":
        clean_split(config, args.config)
        make_supervised(config, args.config)
        make_generation(config, args.config)


if __name__ == "__main__":
    main()
