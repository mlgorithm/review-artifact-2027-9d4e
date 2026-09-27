"""Block-bootstrap synthetic trajectory generator."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from synthetic_generation.generator_scripts.common import (
    sample_trajectory_lengths,
    validate_train_real,
    write_generation_output,
)


MODEL_NAME = "block_bootstrap"


@dataclass
class BlockBootstrapGenerator:
    seed: int = 20260703
    min_block_size: int = 2
    max_block_size: int = 20

    def fit(self, train_real: pd.DataFrame) -> "BlockBootstrapGenerator":
        validate_train_real(train_real)
        self.columns_ = train_real.columns.tolist()
        self.has_trajectory_columns_ = {"learner_id", "order"}.issubset(train_real.columns)
        self.value_columns_ = [
            column for column in self.columns_ if not self.has_trajectory_columns_ or column not in {"learner_id", "order"}
        ]
        if self.has_trajectory_columns_:
            sorted_train = train_real.sort_values(["learner_id", "order"]).reset_index(drop=True)
            self.trajectories_ = [
                group[self.value_columns_].astype(object).where(group[self.value_columns_].notna(), "").to_dict("records")
                for _, group in sorted_train.groupby("learner_id", sort=False)
                if len(group)
            ]
        else:
            self.trajectories_ = [
                train_real[self.value_columns_].astype(object).where(train_real[self.value_columns_].notna(), "").to_dict("records")
            ]
        self.lengths_ = [len(trajectory) for trajectory in self.trajectories_]
        if not self.trajectories_:
            raise ValueError("Cannot fit block bootstrap generator on empty training data.")
        return self

    def sample(self, lengths: list[int]) -> pd.DataFrame:
        rng = random.Random(self.seed)

        rows = []
        for learner_index, target_length in enumerate(lengths):
            synthetic_rows = self._sample_trajectory(rng, target_length)
            for order, values in enumerate(synthetic_rows):
                row = {"learner_id": f"synthetic_{learner_index:06d}", "order": order} if self.has_trajectory_columns_ else {}
                row.update(values)
                rows.append(row)

        return pd.DataFrame(rows, columns=self.columns_)

    def _sample_trajectory(self, rng: random.Random, target_length: int) -> list[dict[str, object]]:
        rows = []
        while len(rows) < target_length:
            trajectory = rng.choice(self.trajectories_)
            start = rng.randrange(len(trajectory))
            remaining = target_length - len(rows)
            max_available = len(trajectory) - start
            max_block = max(1, min(self.max_block_size, max_available, remaining))
            min_block = min(self.min_block_size, max_block)
            block_size = rng.randint(min_block, max_block)
            rows.extend(dict(row) for row in trajectory[start : start + block_size])
        return rows


def generate(
    train_real: pd.DataFrame,
    output_dir: Path,
    seed: int,
    generation_shape: dict[str, int] | None = None,
) -> dict[str, object]:
    """Generate synthetic data from the fixed real training split.

    Inputs:
        train_real: generation-ready real train trajectories. Learner/order
            columns define source blocks but their values are never copied.
        output_dir: directory where this function writes `synth_generation.csv`.
        seed: reproducibility seed selected by the pipeline.

    Returns:
        Metadata dictionary describing the generated synthetic data.
    """
    generator = BlockBootstrapGenerator(seed=seed).fit(train_real)
    lengths, length_metadata = sample_trajectory_lengths(train_real, seed, generation_shape)
    synthetic = generator.sample(lengths)
    return write_generation_output(
        model_name=MODEL_NAME,
        train_real=train_real,
        synthetic=synthetic,
        output_dir=output_dir,
        seed=seed,
        extra_metadata={
            "generator_script": __name__,
            "min_block_size": generator.min_block_size,
            "max_block_size": generator.max_block_size,
            "source_trajectories": len(generator.trajectories_),
            "trajectory_boundaries": "learner_isolated_source_blocks",
            "length_model": length_metadata,
        },
    )
