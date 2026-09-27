"""Shared helpers for standalone synthetic generator scripts."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


LEARNER_COLUMN = "learner_id"
ORDER_COLUMN = "order"
STRUCTURAL_COLUMNS = (LEARNER_COLUMN, ORDER_COLUMN)
OTHER_TOKEN = "<other>"


def fixed_step_budget(
    training_windows: int,
    batch_size: int,
    target_steps: int,
    max_epochs: int,
) -> dict[str, int]:
    """Convert a common neural batch-update budget into full-data epochs."""
    values = {
        "training_windows": training_windows,
        "batch_size": batch_size,
        "target_steps": target_steps,
        "max_epochs": max_epochs,
    }
    invalid = [name for name, value in values.items() if value < 1]
    if invalid:
        raise ValueError(f"Invalid fixed-step budget settings: {invalid}")
    batches_per_epoch = math.ceil(training_windows / batch_size)
    effective_epochs = min(max_epochs, max(1, target_steps // batches_per_epoch))
    return {
        "target_batch_steps": target_steps,
        "max_epochs": max_epochs,
        "training_windows": training_windows,
        "batch_size": batch_size,
        "batches_per_epoch": batches_per_epoch,
        "effective_epochs": effective_epochs,
        "projected_batch_steps": effective_epochs * batches_per_epoch,
    }


class CategoricalCodec:
    """Train-only categorical encoder shared by neural trajectory generators.

    The most frequent values receive their own category. Values outside the
    fixed vocabulary are represented by one ``<other>`` category during model
    training and decoded from their train-fitted empirical distribution. The
    codec never inspects validation or test data.
    """

    def __init__(self, max_vocab: int) -> None:
        if max_vocab < 1:
            raise ValueError("max_vocab must be positive.")
        self.max_vocab = max_vocab
        self.columns: list[str] = []
        self.itos: dict[str, list[str]] = {}
        self.stoi: dict[str, dict[str, int]] = {}
        self.rare_values: dict[str, list[str]] = {}
        self.rare_weights: dict[str, list[float]] = {}

    @staticmethod
    def _as_str(series: pd.Series) -> pd.Series:
        return series.astype(object).where(series.notna(), "").astype(str)

    def fit(self, frame: pd.DataFrame) -> "CategoricalCodec":
        if frame.empty or not len(frame.columns):
            raise ValueError("Cannot fit a categorical codec on an empty frame.")
        self.columns = list(frame.columns)
        for column in self.columns:
            counts = self._as_str(frame[column]).value_counts()
            top = counts.index[: self.max_vocab].tolist()
            vocab = [str(value) for value in top]
            rare = counts.index[self.max_vocab :].tolist()
            if rare:
                vocab.append(OTHER_TOKEN)
                self.rare_values[column] = [str(value) for value in rare]
                self.rare_weights[column] = counts.loc[rare].to_numpy(dtype=float).tolist()
            self.itos[column] = vocab
            self.stoi[column] = {token: index for index, token in enumerate(vocab)}
        return self

    @property
    def vocab_sizes(self) -> list[int]:
        return [len(self.itos[column]) for column in self.columns]

    def encode(self, frame: pd.DataFrame) -> np.ndarray:
        if list(frame.columns) != self.columns:
            frame = frame[self.columns]
        encoded_columns = []
        for column in self.columns:
            values = self._as_str(frame[column])
            mapping = self.stoi[column]
            other = mapping.get(OTHER_TOKEN, 0)
            encoded_columns.append(values.map(mapping).fillna(other).to_numpy(dtype=np.int64))
        return np.stack(encoded_columns, axis=1)

    def decode(self, indices: np.ndarray, rng: random.Random) -> pd.DataFrame:
        if indices.ndim != 2 or indices.shape[1] != len(self.columns):
            raise ValueError("Categorical indices have an incompatible shape.")
        data: dict[str, np.ndarray] = {}
        for position, column in enumerate(self.columns):
            vocab = np.asarray(self.itos[column], dtype=object)
            tokens = vocab[indices[:, position]].astype(object)
            rare = self.rare_values.get(column)
            if rare:
                mask = tokens == OTHER_TOKEN
                replace_count = int(mask.sum())
                if replace_count:
                    tokens[mask] = rng.choices(
                        rare,
                        weights=self.rare_weights[column],
                        k=replace_count,
                    )
            data[column] = tokens
        return pd.DataFrame(data, columns=self.columns)


def make_trajectory_windows(
    encoded_trajectories: list[np.ndarray],
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create right-padded windows without crossing learner boundaries."""
    if window < 2:
        raise ValueError("Neural trajectory windows must contain at least two timesteps.")
    windows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for trajectory in encoded_trajectories:
        if len(trajectory) == 0:
            continue
        for start in range(0, len(trajectory), window):
            chunk = trajectory[start : start + window]
            padded = np.zeros((window, trajectory.shape[1]), dtype=np.int64)
            mask = np.zeros(window, dtype=bool)
            padded[: len(chunk)] = chunk
            mask[: len(chunk)] = True
            windows.append(padded)
            masks.append(mask)
    if not windows:
        raise ValueError("Cannot create sequence windows from empty trajectories.")
    return np.stack(windows), np.stack(masks)


def _rebalance_lengths(
    sampled: np.ndarray,
    target_rows: int,
    cap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Adjust bounded positive lengths to exactly match an aggregate budget."""
    minimum_total = len(sampled)
    maximum_total = len(sampled) * cap
    if not minimum_total <= target_rows <= maximum_total:
        raise ValueError(
            "Trajectory row budget is infeasible under the sampling bounds: "
            f"target={target_rows}, feasible=[{minimum_total}, {maximum_total}]."
        )

    sampled = np.clip(np.asarray(sampled, dtype=int), 1, cap)
    difference = target_rows - int(sampled.sum())
    while difference != 0:
        increasing = difference > 0
        candidates = np.flatnonzero(sampled < cap) if increasing else np.flatnonzero(sampled > 1)
        if len(candidates) == 0:
            raise RuntimeError("Unable to rebalance sampled trajectory lengths to the row budget.")

        distance = cap - sampled[candidates] if increasing else sampled[candidates] - 1
        magnitude = abs(difference)
        even_step = min(magnitude // len(candidates), int(distance.min()))
        if even_step > 0:
            sampled[candidates] += even_step if increasing else -even_step
            difference += -even_step * len(candidates) if increasing else even_step * len(candidates)
            continue

        count = min(magnitude, len(candidates))
        selected = rng.choice(candidates, size=count, replace=False)
        sampled[selected] += 1 if increasing else -1
        difference += -count if increasing else count

    if int(sampled.sum()) != target_rows:
        raise RuntimeError("Sampled trajectory lengths do not match the requested row budget.")
    return sampled


def validate_train_real(train_real: pd.DataFrame) -> None:
    if train_real.empty:
        raise ValueError("Generator input train_real is empty.")
    missing = set(STRUCTURAL_COLUMNS) - set(train_real.columns)
    if missing:
        raise ValueError(
            "Trajectory generators require explicit learner boundaries; "
            f"missing structural columns: {sorted(missing)}."
        )


def trajectory_value_frames(train_real: pd.DataFrame) -> tuple[list[str], list[pd.DataFrame]]:
    """Return learner-isolated, ordered value frames for generator training."""
    validate_train_real(train_real)
    value_columns = [column for column in train_real.columns if column not in STRUCTURAL_COLUMNS]
    if not value_columns:
        raise ValueError("Generator input has no value columns after structural columns are removed.")
    ordered = train_real.sort_values(list(STRUCTURAL_COLUMNS)).reset_index(drop=True)
    trajectories = [
        group[value_columns].reset_index(drop=True)
        for _, group in ordered.groupby(LEARNER_COLUMN, sort=False)
        if len(group)
    ]
    if not trajectories:
        raise ValueError("Generator input contains no learner trajectories.")
    return value_columns, trajectories


def sample_trajectory_lengths(
    train_real: pd.DataFrame,
    seed: int,
    generation_shape: Mapping[str, int] | None = None,
) -> tuple[list[int], dict[str, object]]:
    """Sample lengths from a train-fitted parametric distribution.

    This deliberately does not resample the empirical list of learner lengths:
    doing so can reproduce rare individual lengths and makes length fidelity a
    pipeline artefact. A truncated log-normal retains the broad positive,
    right-skewed shape while emitting an independently sampled length per learner.
    ``generation_shape`` can fix the public output row/learner budget separately
    from the fitting population, which keeps targeted and standard arms size-matched.
    """
    validate_train_real(train_real)
    observed = train_real.groupby(LEARNER_COLUMN, sort=False).size().to_numpy(dtype=float)
    log_lengths = np.log(observed)
    log_mean = float(log_lengths.mean())
    log_std = float(log_lengths.std())
    fitted_learner_count = len(observed)
    requested_shape = dict(generation_shape or {})
    learner_count = int(requested_shape.get("learner_count", fitted_learner_count))
    rng = np.random.default_rng(int(seed) % (2**32))

    target_rows = int(requested_shape.get("target_rows", int(observed.sum())))
    if learner_count < 1 or target_rows < learner_count:
        raise ValueError(
            "Generation shape must contain at least one row per synthetic learner; "
            f"target_rows={target_rows}, learner_count={learner_count}."
        )
    minimum_feasible_cap = int(math.ceil(target_rows / learner_count))
    if log_std < 1e-8:
        sampled = np.full(learner_count, max(1, int(round(math.exp(log_mean)))), dtype=int)
        cap = max(int(sampled[0]), minimum_feasible_cap)
        method = "degenerate_constant"
    else:
        # A robust aggregate tail cap prevents a single numerical draw from
        # exploding the output while avoiding disclosure of the observed maximum.
        # It must nevertheless be large enough for learner_count * cap to reach
        # the public aggregate row budget.
        robust_cap = max(1, min(100_000, int(math.ceil(float(np.quantile(observed, 0.995)) * 1.5))))
        cap = max(robust_cap, minimum_feasible_cap)
        raw = rng.lognormal(log_mean, log_std, learner_count)
        # Preserve only the public aggregate row budget, never the individual
        # length list. This keeps downstream training-arm sizes comparable.
        raw *= target_rows / max(float(raw.sum()), 1.0)
        sampled = np.rint(raw).astype(int)
        method = "mean_calibrated_truncated_lognormal"

    sampled = _rebalance_lengths(sampled, target_rows, cap, rng)

    lengths = sampled.tolist()
    metadata = {
        "method": method,
        "learner_count": learner_count,
        "fitted_learner_count": fitted_learner_count,
        "fitted_log_mean": log_mean,
        "fitted_log_std": log_std,
        "length_cap": cap,
        "max_sampled_length": int(sampled.max()),
        "target_rows": target_rows,
        "synthetic_rows_from_length_model": int(sampled.sum()),
        "shape_override": bool(generation_shape),
    }
    return lengths, metadata


def attach_synthetic_identifiers(
    values: pd.DataFrame,
    lengths: list[int],
    output_columns: list[str],
) -> pd.DataFrame:
    """Attach fresh ids and within-learner order to generated value rows."""
    expected_rows = sum(lengths)
    if len(values) != expected_rows:
        raise ValueError(f"Generated value rows={len(values)} but sampled lengths require {expected_rows} rows.")
    learner_ids: list[str] = []
    orders: list[int] = []
    for learner_index, length in enumerate(lengths):
        learner_ids.extend([f"synthetic_{learner_index:06d}"] * int(length))
        orders.extend(range(int(length)))
    output = values.reset_index(drop=True).copy()
    output.insert(0, ORDER_COLUMN, orders)
    output.insert(0, LEARNER_COLUMN, learner_ids)
    return output[output_columns]


def trim_generated_windows(generated: np.ndarray, lengths: list[int], window: int) -> np.ndarray:
    """Trim independent generated windows into the sampled learner trajectories."""
    pieces: list[np.ndarray] = []
    cursor = 0
    for length in lengths:
        window_count = int(math.ceil(length / window))
        learner_windows = generated[cursor : cursor + window_count]
        if len(learner_windows) != window_count:
            raise ValueError("Generator returned too few windows for sampled trajectory lengths.")
        pieces.append(learner_windows.reshape(-1, generated.shape[-1])[:length])
        cursor += window_count
    if cursor != len(generated):
        raise ValueError("Generator returned unused windows; trajectory assembly is misaligned.")
    return np.concatenate(pieces, axis=0) if pieces else np.empty((0, generated.shape[-1]))


def write_generation_output(
    model_name: str,
    train_real: pd.DataFrame,
    synthetic: pd.DataFrame,
    output_dir: Path,
    seed: int,
    extra_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    validate_train_real(train_real)
    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic_path = output_dir / "synth_generation.csv"
    metadata_path = output_dir / "generation_metadata.json"
    synthetic.to_csv(synthetic_path, index=False)

    metadata: dict[str, object] = {
        "model": model_name,
        "seed": int(seed),
        "train_rows": int(len(train_real)),
        "synthetic_rows": int(len(synthetic)),
        "output": str(synthetic_path),
        "columns": synthetic.columns.tolist(),
    }
    if "learner_id" in train_real.columns:
        metadata["train_learners"] = int(train_real["learner_id"].nunique())
    if "learner_id" in synthetic.columns:
        metadata["synthetic_learners"] = int(synthetic["learner_id"].nunique())
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def not_implemented(model_name: str) -> None:
    raise NotImplementedError(
        f"{model_name} has the standard generator script contract, but the model-specific training and sampling "
        "logic still needs to be filled in by the model owner."
    )
