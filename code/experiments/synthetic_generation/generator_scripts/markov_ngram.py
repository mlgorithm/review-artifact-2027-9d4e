"""Markov / n-gram synthetic trajectory generator.

The core *statistical* baseline of the benchmark: an interpretable order-k Markov
chain over the interaction stream with stupid backoff. Each interaction row is
treated as one categorical token (the tuple of its value columns); the model
learns, for every context of up to ``order`` preceding rows, the distribution of
the next row, and falls back to shorter contexts (finally the unigram) when a
context is unseen or too rare.

Design notes for this repo's pipeline contract:
  * ``learner_id``/``order`` are structural columns. Contexts are learned within
    each learner only and reset at every boundary; the identifier values are never
    modelled or copied to output.
  * Emitting whole real row tuples at backoff preserves cross-signal structure
    (correct x hint x attempts x skill) exactly; the higher-order contexts add
    the temporal transition structure where the data supports it.
  * ``min_context`` requires a context to have been seen at least that many times
    before it is used, which curbs verbatim reproduction of long training
    subsequences (a memorisation/privacy risk for pure n-gram models).

Env overrides for quick experimentation: ``MARKOV_ORDER`` (context length, default
2), ``MARKOV_MIN_CONTEXT`` (minimum context count to use an order, default 2).
"""

from __future__ import annotations

import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from synthetic_generation.generator_scripts.common import (
    attach_synthetic_identifiers,
    sample_trajectory_lengths,
    trajectory_value_frames,
    validate_train_real,
    write_generation_output,
)


MODEL_NAME = "markov_ngram"


def _encode_trajectory(frame: pd.DataFrame) -> list[tuple[str, ...]]:
    encoded = frame.astype(object).where(frame.notna(), "").astype(str)
    return list(encoded.itertuples(index=False, name=None))


def _build_tables(trajectories: list[list[tuple[str, ...]]], order: int):
    """Context (length 1..order) -> Counter over next row, plus the unigram."""
    tables: list[dict[tuple, Counter]] = [defaultdict(Counter) for _ in range(order + 1)]
    unigram: Counter = Counter()
    for rows in trajectories:
        for index, row in enumerate(rows):
            unigram[row] += 1
            for context_length in range(1, order + 1):
                if index >= context_length:
                    context = tuple(rows[index - context_length : index])
                    tables[context_length][context][row] += 1
    return tables, unigram


def _sample_from(counter: Counter, rng: random.Random):
    items = list(counter.keys())
    weights = list(counter.values())
    return rng.choices(items, weights=weights, k=1)[0]


def _generate_trajectory(tables, unigram, order, min_context, n_rows, rng):
    unigram_items = list(unigram.keys())
    unigram_weights = list(unigram.values())
    output: list[tuple[str, ...]] = []
    context: list[tuple[str, ...]] = []
    used_backoff = 0
    for _ in range(n_rows):
        next_row = None
        for context_length in range(min(order, len(context)), 0, -1):
            distribution = tables[context_length].get(tuple(context[-context_length:]))
            if distribution and sum(distribution.values()) >= min_context:
                next_row = _sample_from(distribution, rng)
                break
        if next_row is None:
            used_backoff += 1
            next_row = _sample_from_indexable(unigram_items, unigram_weights, rng)
        output.append(next_row)
        context.append(next_row)
        if len(context) > order:
            context.pop(0)
    return output, used_backoff


def _sample_from_indexable(items, weights, rng: random.Random):
    return rng.choices(items, weights=weights, k=1)[0]


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

    order = max(1, int(os.environ.get("MARKOV_ORDER", 2)))
    min_context = max(1, int(os.environ.get("MARKOV_MIN_CONTEXT", 2)))
    output_columns = train_real.columns.tolist()
    value_columns, value_frames = trajectory_value_frames(train_real)
    trajectories = [_encode_trajectory(frame) for frame in value_frames]
    tables, unigram = _build_tables(trajectories, order)
    rng = random.Random(seed)
    lengths, length_metadata = sample_trajectory_lengths(train_real, seed, generation_shape)
    generated: list[tuple[str, ...]] = []
    used_backoff = 0
    for length in lengths:
        learner_rows, learner_backoff = _generate_trajectory(
            tables, unigram, order, min_context, length, rng
        )
        generated.extend(learner_rows)
        used_backoff += learner_backoff

    values = pd.DataFrame(generated, columns=value_columns)
    synthetic = attach_synthetic_identifiers(values, lengths, output_columns)

    return write_generation_output(
        model_name=MODEL_NAME,
        train_real=train_real,
        synthetic=synthetic,
        output_dir=output_dir,
        seed=seed,
        extra_metadata={
            "generator_script": __name__,
            "method": "order_k_markov_ngram_stupid_backoff",
            "order": order,
            "min_context_count": min_context,
            "distinct_rows": len(unigram),
            "unigram_backoff_rows": used_backoff,
            "trajectory_boundaries": "learner_isolated_contexts",
            "length_model": length_metadata,
        },
    )
