"""Sequence-VAE / TimeVAE synthetic trajectory generator.

The core *encoder-based* baseline of the benchmark: a variational autoencoder over
fixed-length vectorised learner trajectories (the "TimeVAE" recipe of Desai et
al., 2021, in MLP form). Fixed-length windows of the interaction stream are
one-hot encoded, an encoder maps each window to a Gaussian latent, and a decoder
reconstructs per-timestep per-column categorical logits. New trajectories are
produced by sampling the latent prior and decoding.

Design notes for this repo's pipeline contract:
  * ``learner_id``/``order`` define explicit trajectory boundaries. Fixed windows
    never cross learners; short final windows are zero-padded and masked out of
    reconstruction loss.
  * Categorical encoding (with vocab capping and rare-value decoding) is shared
    by all neural baselines so they see the same train-only representation and
    never emit placeholder tokens.

Env overrides: ``VAE_WINDOW``, ``VAE_HIDDEN``, ``VAE_LATENT``, ``VAE_EPOCHS``
(epoch ceiling), ``VAE_TARGET_STEPS``, ``VAE_BATCH``, ``VAE_LR``, ``VAE_BETA``
(KL weight), ``VAE_MAX_VOCAB``, ``VAE_SAMPLE_TEMP`` (decode temperature),
``VAE_DEVICE`` (cpu/cuda/mps).
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

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

try:
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except ImportError:  # torch is an optional heavy dependency; see requirements.txt
    _TORCH_AVAILABLE = False


MODEL_NAME = "sequence_vae_timevae"


@dataclass
class VaeConfig:
    window: int = 20
    hidden: int = 256
    latent: int = 16
    epochs: int = 50
    target_steps: int = 4000
    batch: int = 200
    lr: float = 1e-3
    beta: float = 1.0
    max_vocab: int = 100
    sample_temperature: float = 1.0
    device: str = "cpu"

    @classmethod
    def from_env(cls) -> "VaeConfig":
        base = cls()

        def _int(name: str, default: int) -> int:
            return int(os.environ.get(name, default))

        def _float(name: str, default: float) -> float:
            return float(os.environ.get(name, default))

        return cls(
            window=_int("VAE_WINDOW", base.window),
            hidden=_int("VAE_HIDDEN", base.hidden),
            latent=_int("VAE_LATENT", base.latent),
            epochs=_int("VAE_EPOCHS", base.epochs),
            target_steps=_int("VAE_TARGET_STEPS", base.target_steps),
            batch=_int("VAE_BATCH", base.batch),
            lr=_float("VAE_LR", base.lr),
            beta=_float("VAE_BETA", base.beta),
            max_vocab=_int("VAE_MAX_VOCAB", base.max_vocab),
            sample_temperature=_float("VAE_SAMPLE_TEMP", base.sample_temperature),
            device=os.environ.get("VAE_DEVICE", base.device),
        )


if _TORCH_AVAILABLE:

    class SequenceVAE(nn.Module):
        """MLP VAE over a flattened window of one-hot categorical timesteps."""

        def __init__(self, window: int, vocab_sizes: list[int], hidden: int, latent: int) -> None:
            super().__init__()
            self.window = window
            self.vocab_sizes = vocab_sizes
            self.total_vocab = sum(vocab_sizes)
            input_dim = window * self.total_vocab
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
            )
            self.to_mu = nn.Linear(hidden, latent)
            self.to_logvar = nn.Linear(hidden, latent)
            self.decoder = nn.Sequential(
                nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, input_dim),
            )

        def encode(self, x_flat: "torch.Tensor"):
            hidden = self.encoder(x_flat)
            return self.to_mu(hidden), self.to_logvar(hidden)

        def decode(self, z: "torch.Tensor") -> "torch.Tensor":
            batch = z.shape[0]
            return self.decoder(z).view(batch, self.window, self.total_vocab)

        def forward(self, x_flat: "torch.Tensor"):
            mu, logvar = self.encode(x_flat)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            return self.decode(z), mu, logvar


def _one_hot_windows(
    windows_int: "torch.Tensor",
    vocab_sizes: list[int],
    mask: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    parts = [
        nn.functional.one_hot(windows_int[:, :, column], num_classes=size).float()
        for column, size in enumerate(vocab_sizes)
    ]
    values = torch.cat(parts, dim=-1)
    return values if mask is None else values * mask.unsqueeze(-1).to(values.dtype)


def _column_cross_entropy(
    logits: "torch.Tensor",
    targets: "torch.Tensor",
    vocab_sizes: list[int],
    mask: "torch.Tensor",
) -> "torch.Tensor":
    """Per-column reconstruction loss over real, non-padding timesteps only."""
    total = logits.new_zeros(())
    offset = 0
    for column, size in enumerate(vocab_sizes):
        segment = logits[:, :, offset : offset + size].reshape(-1, size)
        target = targets[:, :, column].reshape(-1)
        losses = nn.functional.cross_entropy(segment, target, reduction="none").reshape(mask.shape)
        total = total + (losses * mask.to(losses.dtype)).sum()
        offset += size
    return total / mask.sum().clamp_min(1).to(total.dtype)


def _sample_categories(logits: "torch.Tensor", vocab_sizes: list[int], temperature: float) -> np.ndarray:
    """Temperature sampling per (timestep, column) from decoder logits."""
    columns = []
    offset = 0
    temp = max(1e-3, temperature)
    for size in vocab_sizes:
        segment = logits[:, :, offset : offset + size] / temp
        probabilities = nn.functional.softmax(segment, dim=-1)
        flat = probabilities.reshape(-1, size)
        sampled = torch.multinomial(flat, num_samples=1).reshape(probabilities.shape[0], probabilities.shape[1])
        columns.append(sampled)
        offset += size
    return torch.stack(columns, dim=-1).cpu().numpy()  # (batch, window, columns)


def _train_and_sample(
    training_windows: np.ndarray,
    training_masks: np.ndarray,
    vocab_sizes: list[int],
    needed_windows: int,
    config: VaeConfig,
) -> np.ndarray:
    device = torch.device(config.device)
    windows = torch.as_tensor(training_windows, dtype=torch.long)
    masks = torch.as_tensor(training_masks, dtype=torch.bool)
    window_count = windows.shape[0]
    model = SequenceVAE(config.window, vocab_sizes, config.hidden, config.latent).to(device)
    # PyTorch 2.2 imports its ONNX patcher while constructing an optimizer. If
    # Transformers happens to be installed, that unused path recursively scans
    # the entire package even though this VAE does not use ONNX or Transformers.
    # Make the optional import unavailable only for this constructor call; Adam
    # itself and every training operation remain the external PyTorch versions.
    transformers_was_loaded = "transformers" in sys.modules
    if not transformers_was_loaded:
        sys.modules["transformers"] = None
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    finally:
        if not transformers_was_loaded:
            sys.modules.pop("transformers", None)

    model.train()
    for _ in range(max(1, config.epochs)):
        order = torch.randperm(window_count)
        for start in range(0, window_count, config.batch):
            batch_order = order[start : start + config.batch]
            batch_int = windows[batch_order].to(device)
            batch_mask = masks[batch_order].to(device)
            x_flat = _one_hot_windows(batch_int, vocab_sizes, batch_mask).reshape(batch_int.shape[0], -1)
            logits, mu, logvar = model(x_flat)
            recon = _column_cross_entropy(logits, batch_int, vocab_sizes, batch_mask)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_int.shape[0]
            loss = recon + config.beta * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    chunk = max(1, min(512, needed_windows))
    produced: list[np.ndarray] = []
    remaining = needed_windows
    with torch.no_grad():
        while remaining > 0:
            batch_size = min(chunk, remaining)
            z = torch.randn(batch_size, config.latent, device=device)
            logits = model.decode(z)
            produced.append(_sample_categories(logits, vocab_sizes, config.sample_temperature))
            remaining -= batch_size
    return np.concatenate(produced, axis=0)[:needed_windows]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)


def generate(
    train_real: pd.DataFrame,
    output_dir: Path,
    seed: int,
    generation_shape: dict[str, int] | None = None,
) -> dict[str, object]:
    """Generate synthetic data from the fixed real training split.

    Inputs:
        train_real: generation-ready real train trajectories. Learner/order
            columns define boundaries but are not modelled.
        output_dir: directory where this function writes `synth_generation.csv`.
        seed: reproducibility seed selected by the pipeline.

    Returns:
        Metadata dictionary describing the generated synthetic data.
    """
    validate_train_real(train_real)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "sequence_vae_timevae requires PyTorch, which is not installed. Install it with "
            "`python3 -m pip install torch` (see experiments/requirements.txt)."
        )

    config = VaeConfig.from_env()
    _seed_everything(seed)

    output_columns = train_real.columns.tolist()
    value_columns, value_frames = trajectory_value_frames(train_real)
    value_rows = pd.concat(value_frames, ignore_index=True)
    codec = CategoricalCodec(config.max_vocab).fit(value_rows)
    encoded_trajectories = [codec.encode(frame) for frame in value_frames]
    training_windows, training_masks = make_trajectory_windows(encoded_trajectories, config.window)
    training_budget = fixed_step_budget(
        len(training_windows),
        config.batch,
        config.target_steps,
        config.epochs,
    )
    effective_config = replace(config, epochs=training_budget["effective_epochs"])
    vocab_sizes = codec.vocab_sizes
    lengths, length_metadata = sample_trajectory_lengths(train_real, seed, generation_shape)
    needed_windows = sum(int(np.ceil(length / config.window)) for length in lengths)

    generated_windows = _train_and_sample(
        training_windows, training_masks, vocab_sizes, needed_windows, effective_config
    )
    generated_rows = trim_generated_windows(generated_windows, lengths, config.window)
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
            "framework": "pytorch",
            "method": "sequence_vae_timevae_mlp",
            "torch_version": torch.__version__,
            "device": effective_config.device,
            "window": effective_config.window,
            "latent_dim": effective_config.latent,
            "hidden_dim": effective_config.hidden,
            "epochs": effective_config.epochs,
            "beta": effective_config.beta,
            "max_vocab": effective_config.max_vocab,
            "training_budget": training_budget,
            "training_windows": int(len(training_windows)),
            "padded_training_rows": int((~training_masks).sum()),
            "column_vocab_sizes": {column: len(codec.itos[column]) for column in codec.columns},
            "trajectory_boundaries": "learner_isolated_masked_windows",
            "length_model": length_metadata,
        },
    )
