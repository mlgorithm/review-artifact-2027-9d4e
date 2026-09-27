# Results Guide

This document is the map from the committed JSON reports to the research
questions. It explains which files are authoritative, what each result block
means, which direction is better, which data were used, and how non-estimable
model-collapse outcomes must be reported.

## 1. Locked result matrix

The paper benchmark contains exactly three datasets, three core generators,
and three fixed random seeds.

| Dataset key | Paper name | Sequence unit | Primary prediction signal |
| --- | --- | --- | --- |
| `assistments` | ASSISTments 2009--2010 Skill Builder | learner interaction | next-response `correct` |
| `ednet` | EdNet KT1 | learner interaction | next-response `correct` |
| `oulad_weekly` | OULAD weekly engagement | enrollment-week | next-week `engaged` |

| Model key | Role | Implementation |
| --- | --- | --- |
| `markov_ngram` | statistical baseline | order-2 Markov/n-gram over complete modeled row tuples, with minimum context count 2 and unigram backoff |
| `sequence_vae_timevae` | encoder-based neural model | repository sequence VAE over boundary-safe categorical windows |
| `timegan` | adversarial neural model | adapter around the external SynthCity TimeGAN implementation |

The seeds are `20260703`, `20260704`, and `20260705`. This produces nine
dataset/model cells and 27 seed runs. Every cell has a three-seed summary in
both the standalone RQ1 snapshot and the completed RQ1/RQ2 comparison. KDD Cup
2010 is not part of this matrix, even though optional adapter code remains in
the repository.

## 2. Which directory answers which question?

There are two committed report roots. They are separate experiment views, not
two batches to average together.

| Directory | Meaning | Use |
| --- | --- | --- |
| [`standard/`](standard/) | Immutable RQ1-only snapshot of the ordinary generator trained on the unchanged real-train split | Standalone audit of RQ1 standard-generator fidelity |
| [`standard_vs_tail_targeted/`](standard_vs_tail_targeted/) | Final schema-v2 comparison containing the same standard arm plus the separately fitted, size-matched tail-targeted arm | Final paper analysis of RQ1 and RQ2 |
| `archive/` | Superseded local development artifacts; git-ignored | Never use for paper results |

For a final table that combines research questions, use only
`standard_vs_tail_targeted/<dataset>/<model>/multi_seed_summary.json`. Select
keys beginning with `rq1.` for RQ1 and keys beginning with `rq2.` for RQ2. The
standalone `standard/` tree exists so RQ1 can be audited independently; do not
count its values as additional runs.

The committed reports include a `rare_group_metric_refresh` provenance block
because the trajectory interpolation and rare-group support contract were
corrected after generation. Its artifact hashes show that this was a report-only
refresh: no generator was retrained and no standard or targeted synthetic CSV
was changed. The reproducible maintenance commands are
`experiments/scripts/refresh_rare_group_metrics.py` followed by
`experiments/scripts/promote_standard_reports.py`. A clean from-scratch run does
not require either command because it uses the corrected evaluator directly.

The old standalone OULAD artifact identifier
`oulad_weekly_engagement_behavioral_dropout_v2` is a provenance-preserving name
for the same active OULAD weekly dataset. The completed schema-v2 result uses
the cleaner identifier `oulad_weekly_engagement`. This suffix is not a fourth
dataset and must not be treated as an additional experimental condition.

## 3. Complete result index

Each cell directory contains one report per seed and one aggregate report.

| Dataset | Markov/n-gram | Sequence VAE | TimeGAN |
| --- | --- | --- | --- |
| ASSISTments | [summary](standard_vs_tail_targeted/assistments_2009_2010_skill_builder/markov_ngram/multi_seed_summary.json) | [summary](standard_vs_tail_targeted/assistments_2009_2010_skill_builder/sequence_vae_timevae/multi_seed_summary.json) | [summary](standard_vs_tail_targeted/assistments_2009_2010_skill_builder/timegan/multi_seed_summary.json) |
| EdNet KT1 | [summary](standard_vs_tail_targeted/ednet_kt1/markov_ngram/multi_seed_summary.json) | [summary](standard_vs_tail_targeted/ednet_kt1/sequence_vae_timevae/multi_seed_summary.json) | [summary](standard_vs_tail_targeted/ednet_kt1/timegan/multi_seed_summary.json) |
| OULAD weekly | [summary](standard_vs_tail_targeted/oulad_weekly_engagement/markov_ngram/multi_seed_summary.json) | [summary](standard_vs_tail_targeted/oulad_weekly_engagement/sequence_vae_timevae/multi_seed_summary.json) | [summary](standard_vs_tail_targeted/oulad_weekly_engagement/timegan/multi_seed_summary.json) |

Within a model directory:

```text
multi_seed_summary.json
seed_20260703/generation_evaluation_report.json
seed_20260704/generation_evaluation_report.json
seed_20260705/generation_evaluation_report.json
```

Use the multi-seed file for tables. Open seed reports to inspect confidence
intervals, sample counts, generator metadata, tail membership, provenance, or a
model-collapse status. A summary intentionally does not average confidence
interval endpoints.

## 4. Data and leakage contract

The stable hash split uses nominal 80/20 allocation with split seed `20260703`.
ASSISTments and EdNet split by learner. OULAD splits by physical student, so
multiple course enrollments from one student cannot appear on both sides. The
held-out real test split is never used to fit preprocessing transformations,
generator encoders, length distributions, tail thresholds, postprocessors, or
tail-targeting membership.

| Dataset | Full train rows | Full test rows | Train learners | Test learners | Generator-modeled columns |
| --- | ---: | ---: | ---: | ---: | --- |
| ASSISTments | 282,118 | 64,742 | 3,429 | 788 | `skill_id`, `correct`, `attempt_bin`, `hint_bin`, `response_time_bin`, `opportunity_bin` |
| EdNet KT1 | 1,021,565 | 264,798 | 7,703 | 1,963 | `question_id`, `correct`, `elapsed_time_bin`, `timestamp_gap_bin`, `attempt_bin`, `hint_bin` |
| OULAD weekly | 743,533 | 187,324 | 26,072 | 6,521 | `course_id`, `dominant_activity`, `registered`, `click_bin`, `active_days_bin`, and the listed demographic/profile fields |

`learner_id` and `order` are present in generation frames only as structural
sequence boundaries. They are excluded from the learned value representation,
and generators emit fresh synthetic identifiers.

EdNet `skill_id` and `part` are deterministic question metadata. They are
withheld from the learned generator representation and rebuilt from
`question_id`. In OULAD, `module_id` and `presentation_id` are rebuilt from
`course_id`; `engaged` and `gap_bin` are derived from generated weekly behavior.
Terminal `dropout`, `failure`, and `final_result` values are excluded from
generator inputs and constructed after generation. OULAD demographic profiles
are independently resampled by course, so demographic fidelity is not used to
rank the three generators.

The generator output has the same full learner and row budget as real train.
Trajectory lengths come from one shared train-fitted, mean-calibrated truncated
lognormal sampler. Individual real learner lengths are never copied or
empirically resampled. Therefore sequence-length metrics and short/long
trajectory tails are diagnostics of the shared sampler, not model-ranking
metrics.

## 5. Fair generator exposure

The two neural generators receive all available train trajectories through
boundary-safe windows of length 20. Both use a batch-size cap of 200, a maximum
100-category vocabulary per modeled column, approximately 4,000 full-data
optimization steps, and a 50-epoch ceiling. Effective epochs are reduced on
large or oversampled inputs to keep exposure comparable. Exact effective
epochs, training-window counts, batch sizes, and projected steps are stored in
each run's generation metadata.

The sequence VAE uses hidden width 256, latent width 16, learning rate 0.001,
KL weight 1.0, and sampling temperature 1.0. TimeGAN uses the external
SynthCity plugin on CPU with one 50-unit generator layer, one 50-unit
discriminator layer, RNN mode, and a maximum 20 encoder clusters. Architectures
and objectives remain generator-specific because they define the compared
methods. Runtime is an efficiency outcome; it is not artificially equalized.

## 6. RQ1: ordinary generator fidelity

RQ1 asks what aspects of educational trajectories an unmodified, normally
trained generator preserves. It uses only the standard arm. Every RQ1 fidelity
comparison uses real train as its reference because that is the distribution
the generator was fitted to. The duplicate `*_vs_test` raw blocks are useful
generalization diagnostics but are excluded from publication ranking.

The publication subtree is `publication.metrics.rq1` in a seed report and the
flattened `rq1.*` keys in a multi-seed summary.

### Average fidelity

- `primary_signal_rate_error`: absolute difference in overall correctness
  (ASSISTments/EdNet) or engagement (OULAD). Lower is better; zero is exact.
- `skill_frequency_js`: Jensen--Shannon divergence between the real and
  synthetic skill/activity frequency distributions. Lower is better; zero is
  exact. For OULAD, the label refers to activity rather than a tutoring skill.
- `modeled_value_marginal_js_mean`: mean JS divergence over the prespecified
  generator-modeled value columns. Lower is better.

### Temporal fidelity

- `correctness_transition_js`: divergence between adjacent binary-state
  transitions. In OULAD this is engagement transition fidelity.
- `skill_trigram_js`: divergence between ordered three-event skill or activity
  patterns.
- `correctness_curve_mae`: mean absolute error between real and synthetic
  primary-signal curves over normalized sequence progress.
- `correctness_autocorrelation_mae`: absolute error in within-learner serial
  dependence.
- `correct_hint_cross_correlation_mae`: error in prespecified primary-signal and
  hint dependence when hints exist.
- `inter_event_time_js`: divergence in timing behavior when the dataset exposes
  a meaningful event-time signal.

All divergences and errors above are lower-is-better.

### Cross-signal dependence

`prespecified_signal_nmi_mae` is the mean absolute error in normalized mutual
information over prespecified, scientifically relevant signal pairs. It tests
whether joint relationships are retained rather than only one-column
marginals. Lower is better. For OULAD the prespecified pairs are dominant
activity--click intensity, dominant activity--active days, and click
intensity--active days. These are all generator-modeled weekly signals;
reconstructed engagement/gaps and independently postprocessed profiles or
outcomes are excluded.

### Subgroup fidelity

Subgroups are defined once from a fixed prefix of real-train learner behavior,
then the same cut-points are applied to synthetic learners. The report compares
future primary-signal rate, future hint rate where available, and learner share
across those prefix-performance groups. The paper calls the future-hint metric
**subgroup hint-use error**. It is available only for ASSISTments; EdNet's hint
field is constant zero and OULAD has no hint-use signal. These are not
demographic groups.
Lower MAE is better. OULAD demographic subgroup fidelity is deliberately
excluded because the common postprocessor, not the generator, samples profiles.

### Tail fidelity

For each available tail `<tail>`:

- `prevalence_error` is the absolute real-train versus synthetic difference in
  learner prevalence.
- `primary_signal_curve_mae` measures whether the within-tail temporal
  correctness or engagement shape is preserved after every learner is
  interpolated onto the same ten relative positions.

Both are lower-is-better. Publication reports prevalence error whenever the
group rule is available, including when the generator produces zero or fewer
than 10 qualifying learners. Shape alone becomes `not_estimable` unless both
sides have at least 10 qualifying learners. Publication rare groups must also
have complete-real-train prevalence at most 10%; broader risk groups remain raw
diagnostics. Tail thresholds are fitted on complete real train and the same
reference is used for targeting and evaluation. A coarse binned variable is
marked unavailable when no strict or inclusive boundary produces a nonempty
group at no more than 10% prevalence; the pipeline never relabels a common bin
as a rare tail.

### Detectability

`detectability_auc` is the learner-disjoint real-versus-synthetic classifier
AUROC after orienting it so 0.5 is ideal indistinguishability and 1.0 is perfect
detectability. Lower is better, with 0.5 as the target. Detector accuracy is a
raw diagnostic and is not a paper ranking metric.

## 7. RQ2: utility, tail targeting, and privacy

RQ2 compares two separately fitted synthetic arms:

| Arm key | Training data for downstream predictor | Meaning |
| --- | --- | --- |
| `real_train` | real train | TRTR reference |
| `synthetic_train` | standard synthetic | standard TSTR |
| `real_plus_synthetic` | real + standard synthetic | standard augmentation |
| `tail_targeted_synthetic` | tail-targeted synthetic | targeted TSTR |
| `real_plus_tail_targeted_synthetic` | real + tail-targeted synthetic | targeted augmentation |

Every predictor is evaluated on the same held-out real-test learners. The
primary downstream model is histogram gradient boosting. AUPRC is the primary
score because rare outcomes and tail slices are imbalanced; AUROC is secondary.
Accuracy, precision, recall, and F1 at an untuned 0.5 threshold are retained as
raw diagnostics but excluded from paper conclusions.

### Utility values and gains

- `arms.<arm>.{auprc,auroc}`: held-out-real performance; higher is better.
- `synthetic_utility_gap`: `real_train - synthetic_train`; lower is better. A
  positive value is a synthetic-only loss and a negative value means the
  synthetic-trained model scored higher on that metric.
- `real_plus_synthetic_gain_over_real`: standard augmentation minus real-only;
  higher is better.
- `tail_targeted_tstr_gain`: targeted TSTR minus standard TSTR; higher is
  better.
- `targeted_augmentation_gain_over_real`: targeted augmentation minus
  real-only; higher is better.

`tail_learnability.<tail>` repeats these comparisons on held-out real learners
belonging to a fixed tail group. `tail_targeting_fidelity_gain.<tail>` reports
the reduction in prevalence or temporal-shape error from standard to targeted;
positive is an improvement.

### Tail-targeted generator construction

The tail-targeted arm uses the same generator code and hyperparameters as its
standard counterpart. Eligible real-train tail learners are repeated with
oversampling factor 3 only in the generator fitting frame. The emitted targeted
dataset is forced to the same row and learner budget as the standard output.
`tail_targeted_size_match.matched` must therefore be `true`.

An individual targeting group must be nonempty and at most 10% of real-train
learners. The union can exceed 10% because learners from different rare groups
need not be the same. This is why ASSISTments and EdNet show union fractions of
15.9% and 23.2% even though each targeted group satisfies the 10% rule.

| Dataset | Targeted groups | Unique selected learners | Fitting rows after up-weighting | Important exclusions |
| --- | --- | ---: | ---: | --- |
| ASSISTments | high hint use, late failure, persistent failure, rapid guessing, rare skill path, recovery | 545 (15.89%) | 311,268 | persistent misconception is 33.07%, above ceiling; length groups use shared sampler |
| EdNet | late failure, persistent failure, rapid guessing, rare skill path, recovery | 1,790 (23.24%) | 1,076,733 | persistent misconception is 30.65%; long inactivity has an inseparable 33.08% coarse tie; length groups use shared sampler |
| OULAD weekly | late disengagement, rare activity path, reengagement | 2,968 (11.38%) | 911,197 | dropout is postprocessed; misconception is inapplicable; several binned/length groups cannot satisfy the 10% boundary rule |

The machine-readable source of these counts is `tail_targeting_contract` in
every schema-v2 seed report. Tail membership is deterministic and recorded as
`tail_learner_set_sha256`.

### Learner-level outcome tasks

ASSISTments and EdNet include early-history prediction of persistent failure
and recovery. OULAD includes dropout and course-failure prediction from the
first four observed weeks. OULAD outcomes have `scope=pipeline_construct`:
terminal labels are derived by the common leakage-safe post-generation pipeline,
not learned as inputs by the generator. They evaluate the complete synthetic
data pipeline and must be described that way.

### Privacy and memorization

For ASSISTments and EdNet, publication privacy metrics operate on full modeled
value trajectories:

- `exact_duplicate_rate`: fraction of synthetic learners whose complete value
  trajectory exactly matches a real-train trajectory; lower is safer.
- `mean_nearest_neighbor_distance`: mean distance from synthetic trajectories
  to their nearest real-train trajectory; larger implies less direct
  resemblance, but must be interpreted with fidelity rather than maximized in
  isolation.
- `near_duplicate_rate`: fraction below the predefined near-duplicate distance;
  lower is safer.
- `membership_inference_attack_advantage`: attacker advantage derived from a
  prespecified distance-score orientation, `max(0, 2(AUC-0.5))`; lower is safer
  and zero is chance-or-worse evidence under that fixed orientation.

The report contains standard, tail-targeted, and risk-oriented comparison
blocks. Do not interpret a privacy metric as a formal differential-privacy
guarantee. It is an empirical memorization audit.

For OULAD, full records include independently sampled profiles and constructed
terminal outcomes, so their full-record distances are not comparable across
generators. Only behavior-projection exact-duplicate rate is a publication
privacy metric. The excluded full-record diagnostics remain visible for audit.

## 8. What each top-level seed-report block means

| JSON block | Interpretation |
| --- | --- |
| `dataset` / `model` | Explicit report identity, checked against the selected cell during promotion |
| `artifact_provenance` | Portable paths and complete generation metadata for both standard and tail-targeted synthetic artifacts, including output, source, configuration, and data hashes |
| `dataset_sizes` | Row/learner counts in the bounded diagnostic sample, not necessarily the full dataset |
| `evaluation_sampling` | Exact deterministic whole-learner sampling policy and realized sample counts |
| `global_fidelity` | Standard synthetic versus real-train marginal/aggregate diagnostics |
| `global_fidelity_vs_test` | Standard synthetic versus real-test diagnostic; not used for generator-fit fidelity ranking |
| `temporal_fidelity` / `temporal_fidelity_vs_test` | Sequential dynamics against train/test references |
| `cross_signal_dependence` | Joint signal-dependence preservation against real train |
| `subgroup_fidelity` | Future behavior within fixed-prefix performance groups |
| `tail_fidelity` / `tail_fidelity_vs_test` | Tail prevalence and shape against train/test references, plus unavailable/unsupported groups |
| `distinguishability` | Learner-disjoint real-versus-synthetic detector |
| `diversity_coverage` | Pattern coverage, entropy, overlap, and uniqueness diagnostics |
| `downstream_utility` | Lightweight diagnostic predictor; not the publication model |
| `downstream_utility_models` | Full five-arm histogram-gradient-boosting utility evaluation and tail slices |
| `downstream_utility_tasks` | Learner-level failure/recovery or OULAD outcome tasks |
| `privacy_memorization` | Standard/targeted privacy audits and their comparison |
| `tail_targeted_evaluation` | Full diagnostic evaluation repeated for the targeted synthetic arm |
| `tail_targeted_size_match` | Proof that standard and targeted emitted sizes match |
| `tail_targeting_contract` | Tail eligibility, selected counts, exclusions, oversampling, and membership digest |
| `publication` | Curated machine-readable results, directions, exclusions, and schema version |
| `final_audit` | Strict validation status; must be `passed` with no issues for promotion |
| `curated_snapshot` | Source report hashes, metadata-only corrections, and promotion provenance; `metric_values_changed` must be `false` |

Memory-heavy fidelity and privacy blocks use a deterministic whole-learner
sample capped at 4,000 learners and 120,000 rows per split. Rare-group fidelity,
primary downstream, and explicit outcome tasks use the full processed frames;
the rare-group reference is fitted from complete real train for both targeting
and evaluation. The detector is separately capped at 100,000 rows per class.
Therefore a sampled `dataset_sizes.real_train_rows` value near 120,000 does not
mean the generator, rare-group evaluator, or downstream task was trained or
scored on a shrunken dataset. Full source sizes and hashes are in
`data_provenance` in the multi-seed summary and in run metadata.

## 9. How to read a multi-seed summary

`metric_summary` flattens the curated publication tree. Each numeric metric has:

- `mean`: arithmetic mean over estimable seeds;
- `std`: population standard deviation over estimable seeds (`ddof=0`);
- `min` and `max`: observed estimable-seed range;
- `n`: number of estimable seeds included;
- `missing_runs`: expected seeds without a numeric value for that metric.

Always report `n`. Do not silently present a two-seed mean as a three-seed
result. Seed-level 95% percentile bootstrap confidence intervals are clustered
by learner for downstream metrics and tail comparisons; their endpoints are
kept in seed reports and are not averaged in the summary.

`publication_status_summary` is equally important. It aggregates categorical
outcomes and lists the exact seeds for each status. Valid statuses include:

- `completed`: the task and all required arms were estimable;
- `completed_with_model_collapse`: the evaluation completed, but at least one
  synthetic-only arm had a single-class target and could not fit a classifier;
- `not_estimable_model_collapse`: the named arm has no valid AUPRC/AUROC because
  the generator produced only one target class;
- `insufficient_sample`: the declared minimum sample requirement was not met;
- `failed`: an actual evaluation error; promoted final reports must not contain
  strict-audit failures.

Never replace a non-estimable status with zero and never drop it from a model
comparison. Collapse is itself a benchmark result about mode coverage.

### Observed model-collapse results

| Dataset/model | Outcome task 1 | Outcome task 2 |
| --- | --- | --- |
| ASSISTments Markov | 3/3 completed | 3/3 completed |
| ASSISTments sequence VAE | 2/3 completed; standard synthetic collapse in seed 20260704 | 3/3 completed |
| ASSISTments TimeGAN | standard and targeted synthetic collapse in 3/3 | 1/3 completed; both synthetic arms collapse in seeds 20260703 and 20260705 |
| EdNet Markov | 3/3 completed | 3/3 completed |
| EdNet sequence VAE | 3/3 completed | 3/3 completed |
| EdNet TimeGAN | standard and targeted synthetic collapse in 3/3 | standard and targeted synthetic collapse in 3/3 |
| OULAD, all three models | dropout 3/3 completed | course failure 3/3 completed |

These are healthy, explicitly represented experimental outcomes, not missing
files. They show that a generator failed to preserve enough positive and
negative learner-level outcomes for a synthetic-only classifier in those seeds.

## 10. Example interpretation

Suppose a summary contains:

```json
"rq2.downstream_utility.synthetic_utility_gap.auprc": {
  "mean": 0.04,
  "std": 0.01,
  "min": 0.03,
  "max": 0.05,
  "n": 3,
  "missing_runs": 0
}
```

This means that, averaged over three seeds, the real-trained classifier's AUPRC
was 0.04 higher than the standard-synthetic-trained classifier on the same held-
out real test learners. It does not mean synthetic AUPRC was 0.04. To report
the absolute scores, also read `arms.real_train.auprc` and
`arms.synthetic_train.auprc`. Because `synthetic_utility_gap` is a loss, smaller
is better.

## 11. Metrics deliberately excluded from conclusions

The raw report is intentionally larger than the paper metric set. Do not rank
models using fixed-threshold accuracy/F1, the lightweight per-skill predictor,
shared length-sampler fidelity, raw novelty/cardinality counts, detector
accuracy, all-column unfiltered averages, test-referenced duplicates of
train-referenced fidelity, OULAD demographics, OULAD postprocessed dropout
fidelity, OULAD misconception labels, or underpowered tails/tasks.

The exhaustive inclusion/exclusion contract and direction table is in
[`../PUBLICATION_METRICS.md`](../PUBLICATION_METRICS.md). The definitions of all
retained raw diagnostics are in
[`../EVALUATION_METRICS.md`](../EVALUATION_METRICS.md).

## 12. Reproduction and provenance

Run the preflight and then one dataset/model command at a time exactly as shown
in [`../README.md`](../README.md). The locked config records seeds, dataset
matrix, tail factor, and neural environment. Each run records input hashes,
preprocessing config/script hashes, generator source-bundle hashes, effective
training settings, both standard and tail-targeted output hashes, and evaluator
seed. Promotion revalidates both synthetic artifacts against their complete
generation metadata before creating a reference report.

New executions write only to git-ignored `experiments/outputs/` and
`experiments/runs/`; they never overwrite committed reference reports. The
runner refuses to overwrite an existing local cell. `--resume` reuses only
artifacts whose hashes, generator source, data provenance, normalization, and
strict audit match. Promotion to this directory is a separate validated step.

The block-bootstrap implementation is an optional memorization-sensitive sanity
baseline. It is not one of the three ranked core models and its copied local
blocks must not be described as privacy-safe synthetic release data.
