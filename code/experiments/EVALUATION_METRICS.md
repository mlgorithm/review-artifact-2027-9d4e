# Evaluation Metric Contract

This document defines the complete raw diagnostic report. It is the
interpretation contract for what is computed, which direction is better, which
split supplies thresholds, and where uncertainty is available.

For paper tables, statistical comparisons, and model ranking, use only the
curated `publication.metrics` subtree documented in
[`PUBLICATION_METRICS.md`](PUBLICATION_METRICS.md). Raw blocks intentionally
contain diagnostics that are valid for troubleshooting but are confounded,
redundant, or unsuitable as benchmark objectives.

## Evaluation design

- `real_train` is the only data used to fit generators, preprocessing state,
  tail thresholds, subgroup cutoffs, and downstream training baselines.
- `real_test` contains held-out learners. It is used only for generalization,
  downstream testing, and non-member examples in the privacy audit.
- The standard and tail-targeted generators emit the same number of learners
  and rows. This makes their utility, fidelity, and privacy results size-matched.
- The two neural generators use the same train-only categorical codec, window
  length (20), vocabulary cap (100), batch size (200), target exposure
  (approximately 4,000 full-data batches), and epoch ceiling (50). Their
  effective epochs are derived with the same shared function and recorded in
  generation metadata. Model architectures, objectives, and optimizers remain
  method-specific. Markov is a non-iterative full-data estimator and therefore
  has no artificial gradient-step budget.
- Every fidelity family is reported against `real_train`. Global, temporal, and
  tail fidelity are also reported against `real_test` under `*_vs_test`, while
  still using tail cutoffs fitted on `real_train`.
- The full standard-arm evaluation is at the report root. The same fidelity,
  dependence, subgroup, diversity, lightweight utility, and distinguishability
  blocks for the targeted arm are nested under `tail_targeted_evaluation`.
  `tail_targeted_size_match.matched` must be true; evaluation raises an error
  instead of publishing a confounded comparison when row or learner budgets differ.
- The pipeline runs three fixed seeds by default. `metric_summary` aggregates
  only numeric values selected under `publication.metrics` and reports the
  valid-run count, missing-run count, mean, standard deviation, minimum,
  and maximum. Confidence-interval metadata is deliberately not averaged across
  seeds. `publication_status_summary` separately reports counts and exact seeds
  for explicit non-numeric statuses such as model collapse.
- Memory-heavy average-fidelity, temporal, dependence, subgroup, privacy,
  diversity, and lightweight-utility blocks use a deterministic sample of whole
  learners (at most 4,000 learners and 120,000 rows per split). Sampling never
  splits a trajectory, and original/evaluated counts are stored under
  `evaluation_sampling`. Rare-group fidelity, primary downstream models, and
  explicit outcome models use the complete processed frames. Rare-group
  targeting and evaluation therefore reuse the same cutoffs fitted from the
  complete real-training split.

Unless stated otherwise, a distance or absolute error is better when lower and
equals zero for an exact match. Rates and classification metrics lie in `[0, 1]`.
JSON `null` means the metric is mathematically undefined or could not be scored;
it must not be interpreted as zero.

`status=not_estimable_model_collapse` is a benchmark result, not an evaluation
exception. It means a synthetic-only training arm contained no rows or only one
target class, so fitting a classifier would be mathematically invalid. Reports
retain the train/test class counts and reason. Strict mode accepts this explicit
status and continues the remaining seeds; equivalent degeneracy in required
real train/test arms remains a strict failure. Evaluation reports are written
before the final strict audit so diagnostics survive any genuine audit failure.

Percentile tails default to 5% of real-train learners. For a discretized signal,
the evaluator tries strict and inclusive application of the fixed real-train
cutoff. If neither produces a non-empty group at no more than twice the declared
prevalence (10% under the default), the group is omitted and its reason is stored
under `tail_fidelity.unavailable_percentile_tail_groups`.

The targeting union is further restricted to proposal-aligned groups whose
membership can be attributed to generated behavior. Shared short/long trajectory
groups are excluded because every model uses the same length sampler. OULAD also
excludes postprocessed dropout and the pedagogically inapplicable
`persistent_misconception` group. Every individual candidate targeting group,
including rule-defined groups, must also contain no more than 10% of real-train
learners. Broader groups remain available as raw diagnostics but are neither
oversampled nor promoted as publication rare groups. The exact eligible,
implemented, unavailable, excluded, and selected-learner counts are recorded under
`tail_targeting_contract` and in targeted generation metadata.

`primary_binary_signal` names the evaluated signal. It is response correctness
for tutoring datasets and weekly engagement for OULAD weekly. Legacy
JSON keys containing `correctness_*` therefore mean the configured primary
binary signal when `primary_binary_signal_semantics=weekly_engagement`.

## Distribution primitives

| Primitive | Definition | Range and direction |
| --- | --- | --- |
| Jensen-Shannon divergence (`js`) | Symmetric average of the two base-2 KL divergences to the mixture distribution. | `[0, 1]`; lower is better. |
| Total variation (`tv`) | One half of the L1 distance between two discrete probability vectors. | `[0, 1]`; lower is better. |
| Wasserstein-1 (`wasserstein`) | Mean absolute difference between corresponding empirical quantiles. | `>= 0`, in the original variable's units; lower is better. `null` if only one sample is empty. |
| Histogram TV | TV after both numeric samples are placed in the same 10 bins over their combined range. | `[0, 1]`; lower is better. |
| Mean absolute error (`mae`) | Mean absolute difference over the union of comparable keys. | `>= 0`; lower is better. |
| Normalized mutual information (`nmi`) | Mutual information divided by the geometric mean of the two marginal entropies. | `[0, 1]`; it describes dependence, so fidelity uses absolute real-synthetic error. |

The preprocessed value signals are categorical or binned. Attempt-count
Wasserstein converts bin labels to representative values (`3-4` to `3.5`, `5+`
to `5`). It therefore measures distance between attempt-bin representatives,
not unbinned raw attempt counts.

## Global fidelity

The following appear in `global_fidelity` and `global_fidelity_vs_test`.

| Report key | Exact interpretation |
| --- | --- |
| `correctness_rate_error` | Absolute difference in interaction-level correctness rates. |
| `hint_rate_error` | Absolute difference in interaction-level hint-use rates; omitted when the dataset has no hint signal. |
| `correctness_by_skill_mae` | MAE between real and synthetic correctness rates for each skill in the union of skills. |
| `skill_frequency_js` | JS divergence between interaction-level skill frequencies. |
| `skill_frequency_total_variation` | TV between interaction-level skill frequencies. |
| `sequence_length_wasserstein` | Wasserstein-1 between learner trajectory lengths, in interactions. |
| `sequence_length_histogram_tv` | Histogram TV between learner trajectory lengths. |
| `attempt_count_wasserstein` | Wasserstein-1 between attempt-bin representatives; omitted if absent. |
| `attempt_count_histogram_tv` | Histogram TV between attempt-bin representatives; omitted if absent. |
| `all_value_marginals.by_column.*.jensen_shannon_divergence` | Categorical JS divergence for every generated, non-structural value column. |
| `all_value_marginals.by_column.*.total_variation` | Categorical TV for every generated, non-structural value column. |
| `all_value_marginals.*_mean` | Unweighted mean of the per-column JS or TV values. |
| `real_unique_count`, `synthetic_unique_count` | Diagnostic cardinalities, not quality scores. |

Learners and sequence order are structural columns and are never treated as
generated value marginals.

## Temporal fidelity

The following appear in `temporal_fidelity` and
`temporal_fidelity_vs_test`.

| Report key | Exact interpretation |
| --- | --- |
| `correctness_transition_js`, `correctness_transition_tv` | Distance between distributions of adjacent correctness transitions (`0->0`, `0->1`, `1->0`, `1->1`). |
| `hint_transition_js`, `hint_transition_tv` | The same calculation for adjacent hint-use states; omitted if hints are absent. |
| `correct_streak_wasserstein`, `failure_streak_wasserstein` | Wasserstein distance between lengths of maximal consecutive correct or incorrect runs. |
| `skill_trigram_js` | JS divergence between ordered length-three skill subsequences. |
| `correctness_curve_mae` | MAE between ten-position normalized correctness curves. Each learner is linearly interpolated onto all ten relative positions and learners are then averaged equally, so neither long trajectories nor missing bins dominate. |
| `correctness_autocorrelation_real`, `..._synthetic` | Pooled within-learner Pearson autocorrelation of correctness at lags 1, 2, and 3. These are diagnostic values in `[-1, 1]`. |
| `correctness_autocorrelation_mae` | MAE between the three real and synthetic autocorrelations. |
| `correct_hint_cross_correlation_*` | Pooled within-learner Pearson correlation between correctness and hint use at lag 0 and lag 1. |
| `correct_hint_cross_correlation_mae` | Mean absolute real-synthetic error over lag 0 and lag 1. |
| `inter_event_time_js`, `inter_event_time_total_variation` | Distances between the preprocessed inter-event-gap bins; omitted when gaps are absent. |

A correlation is returned as zero when one side has zero variance or fewer than
two aligned observations. This is the conventional neutral value for an
unidentifiable correlation, so the corresponding sample sizes should also be
checked on very small datasets.

## Tail definitions

All percentile cutoffs use learner-level summaries from the complete
`real_train` split and default to the bottom or top 5%. The same frozen cutoffs
label `real_test`, standard synthetic, and tail-targeted synthetic learners. If
a cutoff falls on a majority point mass, strict comparison is used so a small
genuine extreme is retained without assigning the entire population to the
tail. A completely constant signal does not create a percentile tail. Scalar
cutoffs, strict/inclusive boundary choices, and the rare-path transition-model
hash are serialized under `tail_fidelity.tail_reference`.

| Tail group | Operational definition |
| --- | --- |
| `persistent_failure` | Bottom-5% whole-trajectory correctness and at least three interactions. |
| `persistent_misconception` | At least three interactions on one skill with mean correctness `<= 0.25`. This remains a raw risk-group diagnostic but is excluded from rare-group publication results whenever its real-train prevalence exceeds 10%. |
| `recovery` | First-half correctness `<= 0.35`, second-half correctness `>= 0.65`, and length at least four. |
| `late_failure` | First-half correctness `>= 0.65`, second-half correctness `<= 0.35`, and length at least four. |
| `short_trajectory`, `long_trajectory` | Bottom-5% or top-5% learner trajectory length. |
| `rare_skill_path` | Bottom-5% mean log-probability of the learner's skill transitions under an add-one-smoothed `real_train` transition model. |
| `high_hint_use` | Top-5% hints per attempt and at least three interactions; omitted without hints. |
| `rapid_guessing` | Bottom-5% mean response time, whole-trajectory correctness `<= 0.50`, and at least three interactions; omitted without response time. |
| `long_inactivity` | Top-5% maximum inter-event gap and at least three interactions; omitted without gaps. |

`dropout` is emitted only for datasets with an explicit completion process. Real
OULAD labels come from `final_result=Withdrawn`. Synthetic dropout is assigned
after generation by a frozen logistic risk ranker fitted on the real training
split. Its only behavioral inputs are mean and standard deviation of generated
engagement after the four-week downstream prediction prefix, plus course ID;
terminal outcomes, prefix behavior, and trajectory length are excluded. Within
each course, the highest-risk generated learners are labeled as dropout until
the real-training course prevalence is matched (subject to integer rounding).
Interaction datasets without a completion outcome continue to omit this group.

## Tail fidelity

| Report key | Exact interpretation |
| --- | --- |
| `tail_prevalence_real`, `tail_prevalence_synthetic` | Learner fraction assigned to each fixed tail group. These are descriptive rates, not errors. |
| `tail_prevalence_error` | Absolute real-synthetic prevalence difference for each group. |
| `tail_prevalence_mae` | Unweighted mean prevalence error across implemented groups. |
| `tail_prevalence_error_ci` | 95% percentile interval from independently resampling real and synthetic learner memberships 200 times. |
| `tail_shape.*.correctness_curve_mae` | Distance between learner-equal ten-position primary-signal curves within the same tail group. Each learner is linearly interpolated onto the complete relative-position grid before averaging, so missing bins are not converted to zero. |
| `tail_shape.*.correctness_curve_mae_ci` | 95% learner-bootstrap interval with 200 resamples; at most 300 learners per side are used for cost control. |
| `real_learners`, `synthetic_learners` | Required denominator diagnostics. |
| `exploratory` | `true` if either side has fewer than five tail learners; that group is excluded from the mean tail-shape error. |

Publication applies a 10% real-train prevalence ceiling to the term *rare
group*. Prevalence error remains reportable even if the real or synthetic rare
group has fewer than ten qualifying learners, because failure to generate the
group is itself the result. Primary-signal trajectory-shape error requires at
least ten qualifying learners on both sides; otherwise only the shape value is
`not_estimable`. The curve describes correctness or engagement conditional on
group membership. It does not, by itself, compare the temporal evolution of the
signal defining high hint use, rapid guessing, inactivity, or path rarity.

## Cross-signal dependence

`cross_signal_dependence.pairwise_nmi` compares every pair of shared generated
value columns. For each pair it reports real NMI, synthetic NMI, and absolute
error; `nmi_mae` is the unweighted mean error. Equal-size deterministic samples
of at most 100,000 interaction rows are used on each side, with sample sizes
recorded in the report. The publication metric retains only prespecified,
scientifically relevant pairs. OULAD uses the three relationships among
dominant activity, click-intensity bin, and active-days bin; all three variables
are generator-modeled rather than reconstructed or independently postprocessed.
Lower `abs_error` and `nmi_mae` are better.

## Subgroup fidelity

Learners are assigned to low/mid/high performance groups from their first two
interactions using tertile cutoffs fitted on `real_train` and reused unchanged
for synthetic learners. Correctness/engagement and hint-use rates are measured
strictly after those two prefix events, so the observations that define a group
do not also determine its measured outcome.

For each subgroup, the report contains learner share, future interaction-level
correctness/engagement rate, and future hint-use rate when available, plus
absolute real-synthetic errors. Aggregate errors are unweighted means over
comparable subgroups. Empty subgroup values are `null`, not zero. Independently
postprocessed demographic attributes are excluded from the publication set.
The published `future_hint_rate_mae` is named **subgroup hint-use error**. It is
reported only for ASSISTments: EdNet's schema-compatible hint field is constant
zero, and OULAD has no hint-use signal.

## Downstream next-response utility

All training arms are tested on the same held-out real learners:

- TRTR: train on `real_train`, test on `real_test`;
- TSTR: train on standard synthetic, test on `real_test`;
- augmentation: train on real plus standard synthetic, test on `real_test`;
- tail-targeted TSTR and real-plus-tail-targeted augmentation.

The raw report fits a group-mean baseline, logistic regression, and histogram
gradient boosting classifier. Publication comparisons use only histogram
gradient boosting, with AUPRC primary and AUROC secondary. The lightweight
`downstream_utility` block is only a per-skill mean-probability smoke test.

| Metric | Definition and direction |
| --- | --- |
| `auroc` | Probability that a random positive receives a higher score than a random negative; higher is better. `null` for a single-class test slice. |
| `auprc` | Area under the precision-recall curve; higher is better and must be read against outcome prevalence. `null` for a single-class test slice. |
| `accuracy` | Fraction correctly classified at threshold 0.5; higher is better. |
| `precision` | `TP / (TP + FP)` at threshold 0.5; higher is better. |
| `recall` | `TP / (TP + FN)` at threshold 0.5; higher is better. |
| `f1` | Harmonic mean of precision and recall; higher is better. |
| `log_loss` | Mean binary cross-entropy of predicted probabilities; lower is better. |
| `brier` | Mean squared probability error; lower is better. |

Tail learnability applies the fixed `real_train` tail labels to each held-out
real learner and scores the common real test rows belonging to that group.
`delta_tail = M(TRTR) - M(arm)` is calculated only for AUPRC and AUROC;
positive values mean the compared synthetic arm lost predictive signal. Metric
intervals and paired delta intervals are 95% percentile bootstraps with 200
resamples clustered by learner. A tail result is flagged exploratory below 10
held-out learners; row and learner counts are always reported.

## Learner-level rare-outcome tasks

Interaction-only tasks predict future outcomes using only the first two interactions:

- `persistent_failure_prediction`: future mean correctness is at or below the
  `real_train` low-correctness cutoff;
- `recovery_prediction`: among learners whose two-event prefix correctness is
  at most 0.35, future mean correctness is at least 0.65.

The feature window and outcome window never overlap. These labels are related to,
but deliberately not claimed to be identical to, whole-trajectory tail labels.
A task is exploratory and no model is fitted unless both train and test contain
at least 10 positive and 10 negative learners. Completed tasks report the same
classification metrics and real-minus-arm deltas as above, with 95% learner
bootstrap intervals and paired delta intervals.

OULAD weekly additionally uses fixed-prefix prediction of explicit terminal
outcomes. The first four weeks provide features; dropout, failure, and
final-result columns are excluded from `X` and used only as targets or
eligibility criteria. OULAD failure/demographic profiles are generated
only after trajectory generation: dropout is assigned from post-prefix generated
engagement by the frozen train-only risk ranker described above,
failure/distinction are assigned from generated engagement ranks and course-level
real-train rates, and demographic fields are sampled independently from
course-conditioned real-train marginals. Complete real learner profiles are not
copied. Because demographics are sampled independently of behavior, subgroup
outcome analyses measure this pipeline design and not demographic-behavior
relationships learned by a generator.

## Distinguishability

`distinguishability.detectability_auc` comes from a classifier trained to
separate real from synthetic interaction rows and equals
`max(raw_auroc, 1 - raw_auroc)`. Train/test splitting is by learner on both sides, so
rows from one learner cannot leak across the detector split. A detectability AUC
near 0.5 is good (indistinguishable); a value near 1 is poor. Raw AUROC and
accuracy at threshold 0.5 are diagnostics only. After the learner split, at most 100,000 rows per
class are selected for model fitting/scoring, with evaluated counts reported.
This metric measures detectability, not privacy.

## Diversity and coverage

| Report key | Exact interpretation and direction |
| --- | --- |
| `unique_synthetic_full_trajectory_rate` | Fraction of synthetic learner trajectories unique over all generated value columns. Higher indicates less duplication, but is not automatically better if implausible paths are novel. |
| `unique_synthetic_projected_trajectory_rate` | The same fraction using only skill/correct/hint projection. |
| `unique_synthetic_skill_path_count` | Number of distinct ordered skill paths; descriptive and sample-size dependent. |
| `skill_trigram_coverage` | Fraction of real skill trigrams also observed synthetically; higher is better. |
| `skill_trigram_precision` | Fraction of synthetic skill trigrams also observed in real data; higher is better. |
| `skill_trigram_jaccard` | Intersection-over-union of real and synthetic skill-trigram sets; higher is better. |
| `real_skill_trigram_entropy`, `synthetic_skill_trigram_entropy` | Base-2 Shannon entropy of trigram distributions; descriptive and unbounded above. |
| `skill_trigram_entropy_abs_error` | Absolute difference between those entropies; lower is better. |

## Privacy and memorization

These are empirical attacks and similarity audits, not a formal differential
privacy guarantee. They use the complete generated value tuple at every event;
the separately labelled `projected_pattern_audit` uses only skill/correct/hint
and must not be presented as the primary duplicate estimate.

Trajectory distance is normalized positional mismatch: length difference plus
unequal aligned event tuples, divided by the longer length. It lies in `[0, 1]`.

| Report key | Exact interpretation and risk direction |
| --- | --- |
| `exact_duplicate_rate` | Fraction of all synthetic learner trajectories exactly equal to a `real_train` trajectory. Higher means more risk. |
| `mean_nearest_neighbor_distance` | Mean synthetic-to-nearest-`real_train` distance over a seeded sample of at most 1,000 synthetic learners, searched exactly against every real trajectory in the declared whole-learner evaluation sample. Lower means more exposure. |
| `min_nearest_neighbor_distance` | Minimum of those distances; lower means more exposure and zero is an exact match. |
| `near_duplicate_rate` | Fraction of sampled synthetic trajectories with nearest distance `<= 0.1` by default. Higher means more risk. |
| `membership_inference_auc` | Raw AUROC of a nearest-synthetic-neighbour attack: members are `real_train`, non-members are `real_test`, and score is negative distance to synthetic. Values above 0.5 mean members are closer. |
| `membership_inference_attack_auc` | `max(0.5, raw AUC)` for the predefined closer-means-member attack. It lies in `[0.5, 1]`; higher means more membership distinguishability. A below-chance attack is not inverted after observing its result. |
| `membership_inference_attack_advantage` | `max(0, 2 * (raw AUC - 0.5))`. It lies in `[0, 1]`; zero is chance and higher means more risk. |

Duplicate, mean-distance, near-duplicate, and membership-advantage results carry
95% percentile bootstrap intervals with 200 resamples. The membership bootstrap
resamples members and non-members separately. The audit may still be confounded
by real train/test distribution shift, so it should be interpreted jointly with
the train-vs-test fidelity gap and across generation seeds.

For each tail group, privacy is measured from each real tail learner to the
nearest trajectory in a seeded synthetic reference sample of at most 400. The
report includes distance, near-duplicate rate, and membership attack metrics for
tail learners, corresponding non-tail baselines, tail counts, and bootstrap
intervals. Risk-coded `privacy_tail_gap` values are positive when tail learners
are more exposed: non-tail distance minus tail distance, tail minus non-tail
near-duplicate rate, and tail minus non-tail membership advantage. A group is
exploratory if either its train or held-out test side has fewer than 10 learners.

`privacy_memorization.comparison` contrasts the size-matched tail-targeted arm
with the standard arm. Every `*_risk_delta` is oriented so a positive value means
tail targeting increased empirical privacy risk. These per-seed comparison
deltas are summarized across the experiment's generation seeds.

## Availability rules

- Hint, attempts, response-time, gap, and demographic metrics are omitted when
  the necessary source signal does not exist; fabricated zero-valued metrics are
  not emitted.
- Unsupported or unscorable analyses return `status: skipped`, `status:
  exploratory`, or JSON `null` with a reason/count where applicable.
- Always report tail learner counts and consult exploratory flags before drawing
  conclusions. Do not average `null` metrics or treat missing metrics as zero.
