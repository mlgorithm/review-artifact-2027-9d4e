"""Thin adapter around SynthCity's external TimeGAN implementation.

The generator architecture, training objectives, optimization, encoding inside
the model, and sampling are owned by ``synthcity``. This module only translates
the repository's learner-trajectory contract to SynthCity's
``TimeSeriesDataLoader`` and translates generated sequences back.

The adapter uses learner-isolated fixed windows because the benchmark contains
very long trajectories and SynthCity trains a single fixed maximum window. A
train-only categorical vocabulary bounds high-cardinality educational IDs before
they reach SynthCity. No test data or evaluation metric is used.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import os
import random
import sys
import types
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from synthetic_generation.generator_scripts.common import (
    CategoricalCodec,
    attach_synthetic_identifiers,
    fixed_step_budget,
    make_trajectory_windows,
    sample_trajectory_lengths,
    trajectory_value_frames,
    trim_generated_windows,
    validate_train_real,
    write_generation_output,
)


MODEL_NAME = "timegan"
SYNTHCITY_VERSION = "0.2.12"
PLUGIN_NAME = "timegan"
VALID_COLUMN = "__adapter_valid_timestep__"
STATIC_PLACEHOLDER = "__adapter_static_placeholder__"
OUTCOME_PLACEHOLDER = "__adapter_outcome_placeholder__"
_XGBOOST_IMPORT_STUBBED = False


@dataclass(frozen=True)
class SynthCityTimeGANConfig:
    """Adapter settings plus documented SynthCity TimeGAN public arguments."""

    window: int = 20
    max_vocab: int = 100
    n_iter: int = 50
    target_steps: int = 4000
    batch_size: int = 200
    n_iter_print: int = 10
    generator_n_layers_hidden: int = 1
    generator_n_units_hidden: int = 50
    discriminator_n_layers_hidden: int = 1
    discriminator_n_units_hidden: int = 50
    encoder_max_clusters: int = 20
    mode: str = "RNN"
    device: str = "cpu"

    @classmethod
    def from_env(cls) -> "SynthCityTimeGANConfig":
        base = cls()
        config = cls(
            window=int(os.environ.get("TIMEGAN_WINDOW", base.window)),
            max_vocab=int(os.environ.get("TIMEGAN_MAX_VOCAB", base.max_vocab)),
            n_iter=int(os.environ.get("TIMEGAN_N_ITER", base.n_iter)),
            target_steps=int(os.environ.get("TIMEGAN_TARGET_STEPS", base.target_steps)),
            batch_size=int(os.environ.get("TIMEGAN_BATCH", base.batch_size)),
            n_iter_print=int(os.environ.get("TIMEGAN_N_ITER_PRINT", base.n_iter_print)),
            generator_n_layers_hidden=int(
                os.environ.get("TIMEGAN_GENERATOR_LAYERS", base.generator_n_layers_hidden)
            ),
            generator_n_units_hidden=int(
                os.environ.get("TIMEGAN_GENERATOR_HIDDEN", base.generator_n_units_hidden)
            ),
            discriminator_n_layers_hidden=int(
                os.environ.get(
                    "TIMEGAN_DISCRIMINATOR_LAYERS", base.discriminator_n_layers_hidden
                )
            ),
            discriminator_n_units_hidden=int(
                os.environ.get(
                    "TIMEGAN_DISCRIMINATOR_HIDDEN", base.discriminator_n_units_hidden
                )
            ),
            encoder_max_clusters=int(
                os.environ.get("TIMEGAN_ENCODER_MAX_CLUSTERS", base.encoder_max_clusters)
            ),
            mode=os.environ.get("TIMEGAN_MODE", base.mode),
            device=os.environ.get("TIMEGAN_DEVICE", base.device),
        )
        config.validate()
        return config

    def validate(self) -> None:
        values = {
            "window": self.window,
            "max_vocab": self.max_vocab,
            "n_iter": self.n_iter,
            "target_steps": self.target_steps,
            "batch_size": self.batch_size,
            "n_iter_print": self.n_iter_print,
            "generator_n_layers_hidden": self.generator_n_layers_hidden,
            "generator_n_units_hidden": self.generator_n_units_hidden,
            "discriminator_n_layers_hidden": self.discriminator_n_layers_hidden,
            "discriminator_n_units_hidden": self.discriminator_n_units_hidden,
            "encoder_max_clusters": self.encoder_max_clusters,
        }
        invalid = [name for name, value in values.items() if value < 1]
        if invalid or self.window < 2:
            raise ValueError(f"Invalid SynthCity TimeGAN settings: {invalid}")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError(
                "SynthCity 0.2.12 TimeGAN supports TIMEGAN_DEVICE=cpu or cuda; "
                f"received {self.device!r}."
            )


def dependency_issues() -> list[str]:
    """Return actionable external-library problems for experiment preflight."""
    if importlib.util.find_spec("synthcity") is None:
        return [
            f"timegan requires external dependency synthcity=={SYNTHCITY_VERSION}; "
            "install experiments/requirements.txt"
        ]
    try:
        installed = importlib.metadata.version("synthcity")
    except importlib.metadata.PackageNotFoundError:
        return ["the synthcity module is importable but package metadata is unavailable"]
    if installed != SYNTHCITY_VERSION:
        return [
            f"timegan requires synthcity=={SYNTHCITY_VERSION}, found synthcity=={installed}"
        ]
    try:
        opacus_version = importlib.metadata.version("opacus")
    except importlib.metadata.PackageNotFoundError:
        return ["SynthCity TimeGAN requires opacus==1.4.1 in the PyTorch 2.2 environment"]
    if opacus_version != "1.4.1":
        return [
            f"SynthCity TimeGAN requires opacus==1.4.1 with PyTorch 2.2, found {opacus_version}"
        ]
    return []


def _isolate_unused_xgboost_import() -> None:
    """Allow SynthCity TimeGAN to load when optional XGBoost cannot load.

    SynthCity imports its compression helper eagerly even though this adapter
    disables dataset compression. On macOS, XGBoost can fail solely because a
    system OpenMP runtime is absent. A minimal module placeholder isolates that
    unused feature; trying to instantiate either class still fails explicitly.
    """
    global _XGBOOST_IMPORT_STUBBED
    try:
        import xgboost  # noqa: F401

        return
    except Exception:
        for module_name in list(sys.modules):
            if module_name == "xgboost" or module_name.startswith("xgboost."):
                sys.modules.pop(module_name, None)

    class UnavailableXGBoost:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("XGBoost is unavailable and is not used by the TimeGAN adapter.")

    placeholder = types.ModuleType("xgboost")
    placeholder.XGBClassifier = UnavailableXGBoost
    placeholder.XGBRegressor = UnavailableXGBoost
    sys.modules["xgboost"] = placeholder
    _XGBOOST_IMPORT_STUBBED = True


def _load_synthcity() -> tuple[Any, Any, Any]:
    issues = dependency_issues()
    if issues:
        raise RuntimeError("; ".join(issues))
    _isolate_unused_xgboost_import()
    try:
        from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
        from synthcity.plugins.core.models.tabular_encoder import TimeSeriesTabularEncoder
        from synthcity.plugins.time_series.plugin_timegan import TimeGANPlugin
    except Exception as exc:
        raise RuntimeError(
            "SynthCity is installed but its TimeGAN dependencies could not be imported. "
            "Reinstall the pinned experiments/requirements.txt environment."
        ) from exc
    return TimeGANPlugin, TimeSeriesDataLoader, TimeSeriesTabularEncoder


def _infer_static_columns(value_frames: list[pd.DataFrame]) -> list[str]:
    """Identify fields constant inside every real training trajectory."""
    return [
        column
        for column in value_frames[0].columns
        if all(frame[column].nunique(dropna=False) <= 1 for frame in value_frames)
    ]


def _encoded_frames(
    value_frames: list[pd.DataFrame], codec: CategoricalCodec
) -> list[pd.DataFrame]:
    return [pd.DataFrame(codec.encode(frame), columns=codec.columns) for frame in value_frames]


def _synthcity_training_data(
    encoded_frames: list[pd.DataFrame],
    static_columns: list[str],
    window: int,
) -> tuple[pd.DataFrame, list[pd.DataFrame], list[list[int]], np.ndarray]:
    """Create equal-length, boundary-safe categorical windows for SynthCity."""
    value_columns = encoded_frames[0].columns.tolist()
    temporal_columns = [column for column in value_columns if column not in static_columns]
    arrays = [frame.to_numpy(dtype=np.int64) for frame in encoded_frames]
    windows, masks = make_trajectory_windows(arrays, window)

    static_rows: list[dict[str, int]] = []
    temporal_windows: list[pd.DataFrame] = []
    for encoded_window, mask in zip(windows, masks):
        first = encoded_window[0]
        if static_columns:
            static_rows.append(
                {
                    column: int(first[value_columns.index(column)])
                    for column in static_columns
                }
            )
        else:
            static_rows.append({STATIC_PLACEHOLDER: 0})

        temporal = {
            column: encoded_window[:, value_columns.index(column)].astype(np.int64)
            for column in temporal_columns
        }
        temporal[VALID_COLUMN] = mask.astype(np.int8)
        temporal_windows.append(pd.DataFrame(temporal))

    observation_times = [list(range(window)) for _ in temporal_windows]
    return pd.DataFrame(static_rows), temporal_windows, observation_times, masks


def _parse_code(value: object, column: str, upper_bound: int) -> int:
    """Validate SynthCity's decoded category as one of the adapter codes."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SynthCity generated invalid category {value!r} for {column}.") from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"SynthCity generated non-integral category {value!r} for {column}.")
    parsed = int(numeric)
    if not 0 <= parsed < upper_bound:
        raise ValueError(
            f"SynthCity category {parsed} for {column} is outside [0, {upper_bound})."
        )
    return parsed


def _generated_code_windows(
    generated_static: pd.DataFrame,
    generated_temporal: list[pd.DataFrame],
    value_columns: list[str],
    static_columns: list[str],
    vocab_sizes: list[int],
    window: int,
) -> np.ndarray:
    if len(generated_static) != len(generated_temporal):
        raise ValueError("SynthCity returned different static and temporal sample counts.")
    static_set = set(static_columns)
    result: list[np.ndarray] = []
    for sample_index, temporal in enumerate(generated_temporal):
        if len(temporal) < window:
            raise ValueError("SynthCity returned a trajectory shorter than the conditioned window.")
        encoded = np.zeros((window, len(value_columns)), dtype=np.int64)
        for column_index, (column, size) in enumerate(zip(value_columns, vocab_sizes)):
            if column in static_set:
                value = generated_static.iloc[sample_index][column]
                encoded[:, column_index] = _parse_code(value, column, size)
            else:
                encoded[:, column_index] = [
                    _parse_code(value, column, size)
                    for value in temporal.iloc[:window][column].tolist()
                ]
        result.append(encoded)
    if not result:
        raise ValueError("SynthCity returned no generated TimeGAN samples.")
    return np.stack(result)


def _fit_and_generate_external(
    static_data: pd.DataFrame,
    temporal_data: list[pd.DataFrame],
    observation_times: list[list[int]],
    needed_windows: int,
    seed: int,
    config: SynthCityTimeGANConfig,
    workspace: Path,
) -> tuple[pd.DataFrame, list[pd.DataFrame], str]:
    TimeGANPlugin, TimeSeriesDataLoader, TimeSeriesTabularEncoder = _load_synthcity()
    outcome = pd.DataFrame({OUTCOME_PLACEHOLDER: np.zeros(len(temporal_data), dtype=int)})
    discrete_columns = list(static_data.columns) + list(temporal_data[0].columns)
    encoder = TimeSeriesTabularEncoder(max_clusters=config.encoder_max_clusters).fit(
        static_data,
        temporal_data,
        observation_times,
        discrete_columns=discrete_columns,
    )
    # SynthCity 0.2.12 forwards ``discrete_columns`` to its static encoder but
    # accidentally drops the argument for the temporal encoder. Refit that
    # external encoder explicitly so high-cardinality code strings are handled
    # as categories rather than sent to a continuous Gaussian-mixture encoder.
    # This changes only the library's input typing; it does not replace TimeGAN.
    encoder.temporal_encoder.fit(
        pd.concat(temporal_data, ignore_index=True),
        discrete_columns=list(temporal_data[0].columns),
    )
    loader = TimeSeriesDataLoader(
        temporal_data=temporal_data,
        observation_times=observation_times,
        static_data=static_data,
        outcome=outcome,
        random_state=seed,
    )
    plugin = TimeGANPlugin(
        n_iter=config.n_iter,
        batch_size=config.batch_size,
        n_iter_print=config.n_iter_print,
        generator_n_layers_hidden=config.generator_n_layers_hidden,
        generator_n_units_hidden=config.generator_n_units_hidden,
        discriminator_n_layers_hidden=config.discriminator_n_layers_hidden,
        discriminator_n_units_hidden=config.discriminator_n_units_hidden,
        encoder_max_clusters=config.encoder_max_clusters,
        encoder=encoder,
        mode=config.mode,
        device=config.device,
        random_state=seed,
        workspace=workspace,
    )
    plugin.fit(loader)
    fixed_horizons = [list(range(config.window)) for _ in range(needed_windows)]
    # The public Plugin.generate wrapper may discard rows while enforcing its
    # packed-data schema and can consequently return fewer than ``count``
    # trajectories. Sampling the trained external TimeGAN covariance model gives
    # an exact sequence count; the adapter validates every decoded category and
    # the final row/learner budgets below.
    generated_static, generated_temporal, _ = plugin.cov_model.generate(
        needed_windows,
        observation_times=fixed_horizons,
    )
    if len(generated_temporal) != needed_windows:
        raise ValueError(
            "SynthCity TimeGAN returned an unexpected sequence count: "
            f"requested={needed_windows}, returned={len(generated_temporal)}."
        )
    return generated_static, generated_temporal, plugin.__class__.__name__


def generate(
    train_real: pd.DataFrame,
    output_dir: Path,
    seed: int,
    generation_shape: dict[str, int] | None = None,
) -> dict[str, object]:
    """Fit and sample SynthCity TimeGAN through the repository contract."""
    validate_train_real(train_real)
    output_dir.mkdir(parents=True, exist_ok=True)
    issues = dependency_issues()
    if issues:
        raise RuntimeError("; ".join(issues))

    config = SynthCityTimeGANConfig.from_env()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    output_columns = train_real.columns.tolist()
    value_columns, value_frames = trajectory_value_frames(train_real)
    codec = CategoricalCodec(config.max_vocab).fit(pd.concat(value_frames, ignore_index=True))
    encoded_frames = _encoded_frames(value_frames, codec)
    static_columns = _infer_static_columns(value_frames)
    static_data, temporal_data, observation_times, training_masks = _synthcity_training_data(
        encoded_frames, static_columns, config.window
    )
    training_budget = fixed_step_budget(
        len(temporal_data),
        config.batch_size,
        config.target_steps,
        config.n_iter,
    )
    effective_config = replace(
        config,
        n_iter=training_budget["effective_epochs"],
        n_iter_print=min(config.n_iter_print, training_budget["effective_epochs"]),
    )

    lengths, length_metadata = sample_trajectory_lengths(train_real, seed, generation_shape)
    needed_windows = sum(math.ceil(length / config.window) for length in lengths)
    generated_static, generated_temporal, plugin_class = _fit_and_generate_external(
        static_data,
        temporal_data,
        observation_times,
        needed_windows,
        seed,
        effective_config,
        output_dir / "synthcity_workspace",
    )
    generated_windows = _generated_code_windows(
        generated_static,
        generated_temporal,
        value_columns,
        static_columns,
        codec.vocab_sizes,
        effective_config.window,
    )
    generated_rows = trim_generated_windows(generated_windows, lengths, effective_config.window)
    values = codec.decode(generated_rows, random.Random(seed))[value_columns]
    synthetic = attach_synthetic_identifiers(values, lengths, output_columns)

    return write_generation_output(
        model_name=MODEL_NAME,
        train_real=train_real,
        synthetic=synthetic,
        output_dir=output_dir,
        seed=seed,
        extra_metadata={
            "generator_script": __name__,
            "framework": "synthcity",
            "external_library": "synthcity",
            "external_library_version": importlib.metadata.version("synthcity"),
            "external_plugin": PLUGIN_NAME,
            "external_plugin_class": plugin_class,
            "external_sampling_api": "TimeGANPlugin.cov_model.generate",
            "unused_xgboost_import_stubbed": _XGBOOST_IMPORT_STUBBED,
            "method": "synthcity_timegan_fixed_window_adapter",
            "adapter_configuration": asdict(effective_config),
            "training_budget": training_budget,
            "training_windows": len(temporal_data),
            "padded_training_rows": int((~training_masks).sum()),
            "static_columns_inferred_from_train": static_columns,
            "column_vocab_sizes": {
                column: len(codec.itos[column]) for column in codec.columns
            },
            "trajectory_boundaries": "learner_isolated_fixed_windows",
            "length_model": length_metadata,
        },
    )
