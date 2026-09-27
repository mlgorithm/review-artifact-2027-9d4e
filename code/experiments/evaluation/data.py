"""Data objects for downstream evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml


@dataclass(frozen=True)
class SupervisedSplit:
    name: str
    frame: pd.DataFrame
    target: str
    path: Path

    @property
    def rows(self) -> int:
        return len(self.frame)

    @property
    def x(self) -> pd.DataFrame:
        return self.frame.drop(columns=[self.target])

    @property
    def y(self) -> pd.Series:
        y = self.frame[self.target]
        y.name = self.target
        return y


@dataclass(frozen=True)
class EvaluationDataset:
    task: str
    target: str
    group_column: str
    train: SupervisedSplit
    test: SupervisedSplit
    seed: int

    def validate(self) -> None:
        if self.target not in self.train.frame.columns:
            raise ValueError(f"Target {self.target!r} is missing from train split.")
        if self.target not in self.test.frame.columns:
            raise ValueError(f"Target {self.target!r} is missing from test split.")
        if self.group_column not in self.train.x.columns:
            raise ValueError(f"Group column {self.group_column!r} is missing from train X.")
        if self.group_column not in self.test.x.columns:
            raise ValueError(f"Group column {self.group_column!r} is missing from test X.")

    def metadata(self) -> dict[str, object]:
        return {
            "task": self.task,
            "target": self.target,
            "group_column": self.group_column,
            "seed": self.seed,
            "train_rows": self.train.rows,
            "test_rows": self.test.rows,
            "train_path": str(self.train.path),
            "test_path": str(self.test.path),
        }


def load_config(path: Path | None) -> Mapping[str, object]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def downstream_config(config: Mapping[str, object]) -> Mapping[str, object]:
    value = config.get("downstream_models", {})
    return value if isinstance(value, Mapping) else {}


def load_split(name: str, path: Path, target: str) -> SupervisedSplit:
    frame = pd.read_csv(path, low_memory=False)
    if target not in frame.columns:
        raise ValueError(f"Target {target!r} is missing from {path}.")
    return SupervisedSplit(name=name, frame=frame, target=target, path=path)


def load_evaluation_dataset(
    *,
    train_path: Path,
    test_path: Path,
    config_path: Path | None,
    target_override: str | None = None,
    group_column_override: str | None = None,
    seed_override: int | None = None,
) -> EvaluationDataset:
    config = load_config(config_path)
    model_config = downstream_config(config)
    target = target_override or str(model_config.get("target", "correct"))
    group_column = group_column_override or str(model_config.get("group_column", "skill_id"))
    seed = seed_override if seed_override is not None else int(model_config.get("seed", 20260703))
    task = str(model_config.get("task", "next_response_correctness"))
    dataset = EvaluationDataset(
        task=task,
        target=target,
        group_column=group_column,
        train=load_split("train", train_path, target),
        test=load_split("test", test_path, target),
        seed=seed,
    )
    dataset.validate()
    return dataset
