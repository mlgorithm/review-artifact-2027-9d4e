# Experiment Pipeline

This folder contains the experiment pipeline for synthetic educational trajectory generation and evaluation.

## Reviewer Preflight

The locked production matrix is in
[`configs/final_experiments.yaml`](configs/final_experiments.yaml). Run its
fail-fast preflight first:

```text
PYTHONPATH=experiments .venv/bin/python experiments/scripts/run_final_experiments.py --preflight-only
```

The preflight performs no generation. After it passes, run exactly one
dataset/model command at a time from the **Run** section below. Wait for that
cell to finish and inspect its report before starting the next command. The
README intentionally provides no command that launches the full matrix.

Strict mode treats a single-class synthetic-only downstream target as an
explicit `not_estimable_model_collapse` benchmark outcome and continues. It
still fails on malformed artifacts, leakage, missing or skipped tail/outcome
evaluations, empty RQ1/RQ2 publication trees, publication-schema drift,
real-data degeneracy, and evaluation exceptions. Diagnostic reports are saved
before a strict failure is raised.

The final runner is locked to the three configured datasets, three core models,
and seeds `20260703 20260704 20260705`. Its `--datasets` and `--models`
arguments select one cell from that matrix; they do not extend it. KDD or any
other unconfigured entry is rejected. A reduced seed list is accepted only with
`--allow-fewer-seeds` for a non-final smoke test and must not be promoted as a
final benchmark result.

## Generator Files

The three core generators and one auxiliary baseline share one contract:

```text
markov_ngram.py          # core statistical: order-k Markov / n-gram with backoff
sequence_vae_timevae.py  # core encoder-based: sequence VAE / TimeVAE (needs torch)
timegan.py               # core adversarial: SynthCity TimeGAN adapter
block_bootstrap.py       # auxiliary sanity baseline
```

Block bootstrap deliberately resamples real training subsequences. It is a
memorization-sensitive experimental baseline, not a privacy-safe release method.

Each script implements one function:

```python
def generate(train_real, output_dir, seed, generation_shape=None):
    ...
```

`train_real` is the real training data prepared for generation. It retains
`learner_id` and `order` strictly as structural boundaries: generators group and
order by them, exclude them from the learned value representation, and emit fresh
synthetic identifiers. Every generator samples new trajectory lengths from the
shared train-fitted parametric length model; real per-learner lengths are never
copied or empirically resampled. Each `generate` writes
`output_dir / "synth_generation.csv"` (same columns as `train_real`) and returns a
metadata dict.

`generation_shape` is used by the pipeline for the tail-targeted arm: tail
trajectories may be up-weighted for fitting, while the emitted learner and row
budgets stay identical to the standard arm for a size-controlled comparison.

Generator hyperparameters are overridable via environment variables (see each
script's module docstring), e.g. `MARKOV_ORDER`, `VAE_EPOCHS`,
`TIMEGAN_N_ITER`. TimeGAN is supplied by the pinned external SynthCity package;
the local module is only a train-only categorical and trajectory-contract
adapter. The external version, plugin class, and exact adapter configuration are
recorded in generation metadata.

The locked neural benchmark contract applies to both the sequence VAE and
TimeGAN: every fit uses all available training trajectories, window length 20,
a configured batch-size cap of 200, a 100-value per-column categorical
vocabulary, approximately 4,000 full-data batch updates, and a 50-epoch ceiling.
Effective epochs are calculated from each arm's training-window count so large
and tail-oversampled inputs do not receive more optimization merely because they
contain more batches. SynthCity TimeGAN cannot train on a final minibatch of one;
only when `training_windows % 200 == 1`, the pipeline deterministically selects
the largest smaller batch size without a singleton remainder (199 for the
ASSISTments tail arm). No examples are dropped, and the requested/effective
sizes and reason are recorded in generation metadata. Rare categories use a
train-fitted `<other>` bucket and are decoded from their train-only empirical
distribution. Target steps, effective epochs, and projected steps are recorded
in generation metadata. Architectures, losses, and optimizers remain
model-specific because they define the compared methods; wall-clock time is
reported as an efficiency result, not equalized.

TimeGAN uses a one-layer 50-unit generator and discriminator. Override keys are
`TIMEGAN_TARGET_STEPS`,
`TIMEGAN_GENERATOR_LAYERS`, `TIMEGAN_GENERATOR_HIDDEN`,
`TIMEGAN_DISCRIMINATOR_LAYERS`, and `TIMEGAN_DISCRIMINATOR_HIDDEN`; changing
them creates a different experimental condition and must be reported.

## Evaluation

The exact set to use for paper tables and model ranking is documented in
[PUBLICATION_METRICS.md](PUBLICATION_METRICS.md). The complete raw diagnostic
contract is in [EVALUATION_METRICS.md](EVALUATION_METRICS.md). For the exact
meaning of every report directory, JSON block, arm, status, and dataset-specific
result, see [reports/RESULTS_GUIDE.md](reports/RESULTS_GUIDE.md).

In short: use histogram gradient boosting, rank downstream utility primarily by
AUPRC and secondarily by AUROC, and use only values under
`publication.metrics`. The `metric_summary` field in each
`multi_seed_summary.json` already contains only that curated set.
`publication_status_summary` separately counts non-numeric outcomes such as
model collapse and records their seeds. Fixed-threshold classification scores,
shared length-sampler metrics, novelty counts, and postprocessor-confounded
OULAD metrics are not publication metrics.

`run_general_evaluation` produces, per synthetic set:

- **Fidelity** — global, temporal, tail (prevalence error **and** tail-shape
  curve), cross-signal dependence, subgroup (demographic column if configured,
  else prior-performance terciles), diversity/coverage, and a real-vs-synthetic
  distinguishability AUROC.
- **Tail groups** — defined by percentile cut-points (top/bottom 5% by default)
  fixed on the real train split, matching the paper's Table 3. If a discretized
  signal cannot form any non-empty group at no more than 10% prevalence, that
  percentile group is declared unavailable instead of calling a common bin a
  tail.
- **Downstream utility** (`downstream_utility_models`) — next-response
  correctness with four training arms: real, synthetic, real+synthetic, and
  **tail-targeted synthetic**; each sliced per tail group for tail learnability
  (`delta_tail`, `delta_tail_augmented`, `delta_tail_tail_targeted`) with
  learner-clustered percentile **bootstrap CIs**. Tail-targeted and standard
  synthetic arms have the same output row and learner budgets.
- **Learner-level tasks** (`downstream_utility_tasks`) — early-behaviour →
  future failure/recovery for interaction datasets, plus explicit dropout and
  course-failure outcomes for OULAD weekly. Outcome columns never enter the
  generator or downstream feature matrix; OULAD terminal outcomes are derived
  only after trajectory generation.
- **Privacy** (`privacy_memorization`) — full-value-trajectory duplicates /
  nearest-neighbour / membership-inference, with a separately labelled
  skill-correct-hint projection audit and a per-tail-group breakdown. The
  standard and tail-targeted arms, plus risk-oriented differences, are reported.
- **Tail-targeted fidelity** (`tail_targeted_evaluation`) — the complete global,
  temporal, tail, dependence, subgroup, diversity, lightweight utility, and
  detectability evaluation for the size-matched targeted arm.
- **Subgroups and distinguishability** — performance cut-points are fitted on a
  fixed real-train prefix and reused on synthetic learners; the detection
  classifier is held out by learner rather than by interaction row.

The tail-targeted arm is controlled by `--tail-oversample` (default 3; `1`
disables it). A candidate targeting group is used only when its prevalence in
real training learners is non-zero and at most 10%. Broader semantic groups are
retained as raw risk-group diagnostics, but are excluded from oversampling and
publication rare-group results and recorded under `excluded_target_groups`.

Memory-heavy fidelity/privacy evaluation is bounded by a deterministic
whole-learner sample of at most 4,000 learners and 120,000 rows per split.
Rare-group fidelity uses the complete real and synthetic frames and the same
complete-real-train reference used for targeting. Primary downstream models and
explicit outcome tasks also use the full processed frames. The detector uses at
most 100,000 rows per class after a learner-disjoint split.

Author-side report maintenance after an evaluator-only correction is also
reproducible. The following commands recompute only trajectory-shape and
rare-group blocks from the existing hash-recorded standard and targeted CSVs,
rebuild the publication summaries, and then refresh the independent RQ1 view.
They do not import a generator, retrain a model, or modify a synthetic artifact:

```bash
PYTHONPATH=experiments .venv/bin/python experiments/scripts/refresh_rare_group_metrics.py
PYTHONPATH=experiments .venv/bin/python experiments/scripts/promote_standard_reports.py
```

Reviewers reproducing from scratch should use the dataset/model commands below;
those commands apply the current evaluator directly and do not need this
maintenance step.

## Run

Run these commands from the repository root, one at a time. Each command runs
one dataset/model cell with the three fixed seeds, generates both the standard
and tail-targeted arms, applies the locked environment from
`configs/final_experiments.yaml`, and performs the strict final-report audit.
Each cell has its own manifest, so completing one command never starts another
dataset or model.

The two synthetic arms are physically separate. Unmodified RQ1 generations are
written under `experiments/outputs/standard/`; RQ2 tail-targeted generations are
written under `experiments/outputs/tail_targeted/`. Local evaluation reports and
manifests are written under the git-ignored
`experiments/runs/standard_vs_tail_targeted/` workspace. Curated reports live
separately under `experiments/reports/`, so they never block a clean reviewer
reproduction. The pipeline refuses to overwrite an existing local cell. Archive
an obsolete cell before changing the experimental design.

If a command is interrupted, repeat that same command with `--resume`. The
runner validates hashes, generator source, dataset provenance, normalization,
and the strict evaluation audit before skipping a completed seed. It also
reuses an atomically completed standard or tail-generation phase instead of
training that phase again. `--resume` never accepts a mismatched or partially
written artifact.

The clean reviewer commands below train both arms from scratch. On the original
development machine, add `--reuse-standard-if-valid` to reuse the existing
hash- and provenance-validated RQ1 standard arm; the RQ2 tail-targeted arm is
still trained separately under the active targeting contract.

### ASSISTments

```bash
PYTHONPATH=experiments .venv/bin/python experiments/scripts/run_final_experiments.py \
  --datasets assistments --models markov_ngram \
  --manifest experiments/runs/standard_vs_tail_targeted/manifests/assistments_markov_ngram.json
```

```bash
PYTHONPATH=experiments .venv/bin/python experiments/scripts/run_final_experiments.py \
  --datasets assistments --models sequence_vae_timevae \
  --manifest experiments/runs/standard_vs_tail_targeted/manifests/assistments_sequence_vae_timevae.json
```

```bash
PYTHONPATH=experiments .venv/bin/python experiments/scripts/run_final_experiments.py \
  --datasets assistments --models timegan \
  --manifest experiments/runs/standard_vs_tail_targeted/manifests/assistments_timegan.json
```

### EdNet KT1

```bash
PYTHONPATH=experiments .venv/bin/python experiments/scripts/run_final_experiments.py \
  --datasets ednet --models markov_ngram \
  --manifest experiments/runs/standard_vs_tail_targeted/manifests/ednet_markov_ngram.json
```

```bash
PYTHONPATH=experiments .venv/bin/python experiments/scripts/run_final_experiments.py \
  --datasets ednet --models sequence_vae_timevae \
  --manifest experiments/runs/standard_vs_tail_targeted/manifests/ednet_sequence_vae_timevae.json
```

```bash
PYTHONPATH=experiments .venv/bin/python experiments/scripts/run_final_experiments.py \
  --datasets ednet --models timegan \
  --manifest experiments/runs/standard_vs_tail_targeted/manifests/ednet_timegan.json
```

### OULAD Weekly

```bash
PYTHONPATH=experiments .venv/bin/python experiments/scripts/run_final_experiments.py \
  --datasets oulad_weekly --models markov_ngram \
  --manifest experiments/runs/standard_vs_tail_targeted/manifests/oulad_weekly_markov_ngram.json
```

```bash
PYTHONPATH=experiments .venv/bin/python experiments/scripts/run_final_experiments.py \
  --datasets oulad_weekly --models sequence_vae_timevae \
  --manifest experiments/runs/standard_vs_tail_targeted/manifests/oulad_weekly_sequence_vae_timevae.json
```

```bash
PYTHONPATH=experiments .venv/bin/python experiments/scripts/run_final_experiments.py \
  --datasets oulad_weekly --models timegan \
  --manifest experiments/runs/standard_vs_tail_targeted/manifests/oulad_weekly_timegan.json
```

The OULAD outcome-reprocessing and target-only scripts are historical
maintenance utilities. They are not part of the clean reviewer workflow.

The original development machine has pre-existing standard OULAD trajectories
whose generator-produced behavior is valid but whose three derived terminal
outcome columns predate the behavioral-dropout contract. Before reusing those
local artifacts, migrate and validate them once (this does not train a
generator):

```bash
PYTHONPATH=experiments .venv/bin/python experiments/scripts/migrate_oulad_standard_outcomes.py
PYTHONPATH=experiments .venv/bin/python experiments/scripts/promote_standard_reports.py
```

Clean reviewer runs do not need this migration because both arms are generated
and normalized directly under the current contract.

Final datasets:

```text
assistments
ednet
oulad_weekly
```

KDD Cup 2010 is not part of the locked paper benchmark. Its adapter is retained
as optional development code, but it is absent from the final configuration,
reviewer commands, committed result matrix, and manuscript conclusions.

Core models:

```text
markov_ngram
sequence_vae_timevae
timegan
```

`block_bootstrap` remains available only as an optional auxiliary baseline.

Historical report directories may contain the retired model identifier
`rcgan_crnngan`. They are preserved as prior artifacts, but that custom model is
not selectable by the current pipeline and is not part of the final benchmark.

Default runs use three fixed seeds:

```text
20260703 20260704 20260705
```

For a non-final development run with different seeds:

```bash
PYTHONPATH=experiments .venv/bin/python experiments/pipeline.py --dataset assistments --model markov_ngram --seeds 11 22 33
```

## Outputs

The two synthetic datasets have separate top-level roots:

| Arm | Research use | Generator training data | Output root |
|---|---|---|---|
| Standard | RQ1 and the RQ2 baseline | Original real training split, unchanged | `experiments/outputs/standard/` |
| Tail-targeted | RQ2 comparison | Same training split with eligible rare learners up-weighted | `experiments/outputs/tail_targeted/` |

Each seed is saved as:

```text
experiments/outputs/standard/{dataset}/{model}/seed_{seed}/synth_generation.csv
experiments/outputs/tail_targeted/{dataset}/{model}/seed_{seed}/synth_generation.csv
```

The standard folder never contains a nested `tail_targeted/` directory. Old
layouts and superseded targeting runs are retained only under
`experiments/outputs/archive/` and are never loaded as active experiment arms.

Local evaluation reports and manifests are saved under:

```text
experiments/runs/standard_vs_tail_targeted/evaluation/{dataset}/{model}/
experiments/runs/standard_vs_tail_targeted/manifests/{dataset}_{model}.json
```

Curated immutable RQ1-only report snapshots are committed under
`experiments/reports/standard/`. Completed RQ1/RQ2 comparison snapshots are
committed under `experiments/reports/standard_vs_tail_targeted/`. New pipeline
runs never write to either location. See
[`reports/RESULTS_GUIDE.md`](reports/RESULTS_GUIDE.md) before extracting tables.
Historical mixed reports and superseded targeting experiments are preserved
under `experiments/reports/archive/`.

`experiments/outputs/`, `experiments/runs/`, and
`experiments/data/datasets/` are ignored by git.

## Rebuild the Paper Figures and Supplement Tables

The LAK paper's grouped figures and complete supplementary tables are generated
from the nine committed `publication_metrics_v2` multi-seed summaries. They do
not rerun a generator or recompute an evaluation metric. From the repository
root, write the artifacts into a local paper checkout with:

```bash
PYTHONPATH=experiments .venv/bin/python experiments/scripts/build_paper_figures.py \
  --output-dir <paper-dir>/figures

PYTHONPATH=experiments .venv/bin/python experiments/scripts/build_paper_supplement.py \
  --output-dir <paper-dir>/supplement-generated
```

`build_paper_figures.py` emits four TikZ figures plus the exact plotted values
as CSV: the six-facet RQ1 profile grouped under the first three evaluation-lens
questions, overall utility, per-pathway targeting effects with
representation--utility concordance, and disclosure risk. The
utility figure overlays the three fixed seed values rather than showing only a
bar and SD. `build_paper_supplement.py` emits grouped LaTeX longtables, a
complete CSV retaining every publication JSON path and its mean, population SD,
minimum, maximum, estimable-seed count, and missing-seed count; an outcome-model
status table; a compact rare-pathway shape-support summary plus a 126-row
seed-level export with exact support counts; a complete dataset-by-metric
applicability grid; a signed real/standard/targeted
rare-pathway prevalence table; and `appendix-generation-audit.json`. The signed
table is descriptive and is derived from prevalence values already stored in
the committed seed reports; it does not rerun evaluation or add a ranking
metric. The appendix build fails if numeric row counts, required summary fields,
unique metric paths, or rare-shape seed coverage differ from the source reports.
The current locked matrix contains 1,451 numeric entries, 42 signed prevalence
rows, 27 raw outcome-status entries, 126 rare-shape seed-status rows, and 99
dataset--metric applicability rows.

## Data Folder

After downloading the shared data folder, the layout should look like:

```text
experiments/data/datasets/
  assistments/
    train.csv
    test.csv
  ednet/
    train.csv
    test.csv
  oulad/
    train.csv
    test.csv
  oulad_weekly/     # derived from oulad/raw/oulad.zip
    train.csv
    test.csv
```
