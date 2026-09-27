# Synthetic Educational Trajectory Benchmark

This repository contains a reproducible benchmark of three synthetic sequence
generators on three educational trajectory datasets: ASSISTments, EdNet KT1,
and OULAD weekly engagement. KDD Cup 2010 is intentionally outside the locked
paper benchmark.

Start here:

- [`experiments/README.md`](experiments/README.md): environment, preprocessing,
  one-cell-at-a-time reproduction commands, outputs, and strict audit behavior.
- [`experiments/reports/RESULTS_GUIDE.md`](experiments/reports/RESULTS_GUIDE.md):
  detailed explanation of the committed results and how to interpret every
  report family.
- [`experiments/PUBLICATION_METRICS.md`](experiments/PUBLICATION_METRICS.md):
  machine-readable metric selection rules for paper tables and model ranking.
- [`experiments/EVALUATION_METRICS.md`](experiments/EVALUATION_METRICS.md): all
  diagnostic metrics retained for auditing and debugging.

The committed result matrix is complete for three models (`markov_ngram`,
`sequence_vae_timevae`, and `timegan`), three datasets, and three fixed seeds
(`20260703`, `20260704`, and `20260705`). Standard RQ1 results and the RQ2
standard-versus-tail-targeted comparison are stored separately under
[`experiments/reports/`](experiments/reports/).
