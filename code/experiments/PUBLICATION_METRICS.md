# Metrics to Use in the Paper

Use only the metrics under `publication.metrics` in each seed report and the
corresponding `metric_summary` field in each model's `multi_seed_summary.json`.
The raw evaluation blocks are retained for debugging and auditability; they are
not an additional menu of metrics for model ranking.

The machine-readable selection is implemented in
[`evaluation/publication.py`](evaluation/publication.py) and has schema version
`publication_metrics_v2`.

## Primary comparison rule

- Use `hist_gradient_boosting` for every downstream comparison.
- Use **AUPRC as the primary utility metric**, because rare outcomes and tail
  slices are imbalanced. Interpret it together with the reported held-out-real
  positive rate.
- Use **AUROC as the secondary utility metric**.
- Rank fidelity errors and divergences lower-is-better.
- Rank utility gains higher-is-better. A `synthetic_utility_gap` is
  `real-trained - synthetic-trained`, so lower is better and a negative value
  means synthetic training scored higher.
- Never replace an undefined value with zero. Report a model-collapse or
  insufficient-sample status as such.

Each `multi_seed_summary.json` stores numeric results under `metric_summary` and
explicit non-numeric outcomes under `publication_status_summary`. A collapsed
synthetic arm therefore remains a benchmark result even when AUPRC/AUROC and a
synthetic-utility gap cannot be computed.

## RQ1: what behavior does a generator preserve?

| Construct | Publication JSON path | Direction |
| --- | --- | --- |
| Average behavior | `rq1.average_fidelity.primary_signal_rate_error` | Lower |
| Activity/skill mix | `rq1.average_fidelity.skill_frequency_js` | Lower |
| Other modeled marginals | `rq1.average_fidelity.modeled_value_marginal_js_mean` | Lower |
| Adjacent response/engagement dynamics | `rq1.temporal_fidelity.correctness_transition_js` | Lower |
| Ordered activity/skill dynamics | `rq1.temporal_fidelity.skill_trigram_js` | Lower |
| Progress over normalized time | `rq1.temporal_fidelity.correctness_curve_mae` | Lower |
| Within-learner serial dependence | `rq1.temporal_fidelity.correctness_autocorrelation_mae` | Lower |
| Correctness-hint dependence | `rq1.temporal_fidelity.correct_hint_cross_correlation_mae` | Lower |
| Timing dependence, when available | `rq1.temporal_fidelity.inter_event_time_js` | Lower |
| Prespecified cross-signal dependence | `rq1.cross_signal_dependence.prespecified_signal_nmi_mae` | Lower |
| Rare pathway prevalence | `rq1.tail_fidelity.<tail>.prevalence_error` | Lower |
| Rare pathway temporal shape | `rq1.tail_fidelity.<tail>.primary_signal_curve_mae` | Lower |
| Future behavior by fixed-prefix group | `rq1.subgroup_fidelity.*` | Lower |
| Real/synthetic detectability | `rq1.detectability.detectability_auc` | 0.5 best; 1 worst |

Publication rare groups must have real-train learner prevalence at most 10%.
Rare-group prevalence error is reported whenever the group rule is available,
including when the generator produces fewer than 10 qualifying learners.
For interpretation, the paper supplement also reports real prevalence,
synthetic prevalence, and the signed difference `synthetic - real` in
percentage points. Positive signed values mean overrepresentation and negative
values mean underrepresentation. These descriptive values are read from the
saved seed reports' `tail_prevalence_real` and `tail_prevalence_synthetic`
fields; absolute prevalence error remains the publication ranking metric.
Trajectory-shape error is reported only when both the real and synthetic groups
contain at least 10 learners; otherwise its status is `not_estimable` while the
prevalence result remains present. For OULAD, behavioral names are used:
`persistent_disengagement`, `reengagement`, `late_disengagement`,
`rare_activity_path`, and `long_inactivity`.

A binned percentile signal is omitted when neither strict nor inclusive
boundary selection can create a non-empty group at no more than twice the
declared 5% prevalence. This prevents a coarse maximum bin such as `7d+` from
turning `long_inactivity` into a 33--75% group.

Tail-targeted generator fitting uses only proposal-aligned, generator-attributable
groups. It never targets shared trajectory-length groups, and OULAD targeting
does not use postprocessed dropout or pedagogically inapplicable misconception
labels. The report records the exact selected learner set by a deterministic
SHA-256 identifier-set digest.

## RQ2: does synthetic data help utility, tails, and privacy?

| Construct | Publication JSON path | Direction |
| --- | --- | --- |
| TRTR/TSTR/augmentation performance | `rq2.downstream_utility.arms.*.{auprc,auroc}` | Higher |
| Synthetic utility loss | `rq2.downstream_utility.synthetic_utility_gap.*` | Lower |
| Standard augmentation gain | `rq2.downstream_utility.real_plus_synthetic_gain_over_real.*` | Higher |
| Tail-targeted augmentation gain | `rq2.downstream_utility.targeted_augmentation_gain_over_real.*` | Higher |
| Targeting gain over standard TSTR | `rq2.downstream_utility.tail_targeted_tstr_gain.*` | Higher |
| Same comparisons on held-out real tails | `rq2.tail_learnability.<tail>.*` | As above |
| Fidelity improvement from targeting | `rq2.tail_targeting_fidelity_gain.<tail>.*_reduction` | Higher |
| Early-history outcome tasks | `rq2.learner_level_outcome_tasks.<task>.*` | As above |
| Exact train-trajectory reproduction | `rq2.privacy_risk.*.exact_duplicate_rate` | Lower risk |
| Mean nearest-train distance | `rq2.privacy_risk.*.mean_nearest_neighbor_distance` | Higher distance, lower risk |
| Near-duplicate rate | `rq2.privacy_risk.*.near_duplicate_rate` | Lower risk |
| Membership attack advantage | `rq2.privacy_risk.*.membership_inference_attack_advantage` | Lower risk |

The membership attack has a predefined score orientation: a real learner that
is closer to the synthetic data is more likely to be a training member. Its
publication advantage is `max(0, 2 * (AUC - 0.5))`. An AUC below 0.5 is not
inverted into apparent leakage.

For OULAD, independently sampled profiles and constructed terminal outcomes
make full-record nearest-neighbor and membership distances incomparable. The
publication privacy audit therefore uses only
`behavior_projection_exact_duplicate_rate`. OULAD outcome prediction is marked
`scope=pipeline_construct`; it evaluates the complete synthetic-data pipeline,
not an outcome learned directly by the trajectory generator.

## Do not use for benchmark conclusions

The following remain only in the raw report, or are omitted entirely from the
publication subtree:

- accuracy, precision, recall, and F1 at the arbitrary untuned threshold 0.5;
- the lightweight per-skill mean downstream baseline;
- sequence-length fidelity and short/long trajectory tails, because every
  generator uses the same shared length sampler;
- uniqueness rates, raw cardinalities, detector accuracy, and minimum-neighbor
  extremes as model-quality rankings;
- unfiltered all-column marginal and all-pair NMI averages;
- fidelity duplicated against held-out `real_test` rather than generator-fit
  `real_train`;
- OULAD demographic subgroup fidelity, because profiles are sampled
  independently of generated behavior;
- OULAD synthetic dropout prevalence/shape, because dropout is assigned by the
  common postprocessor at a train-fitted course prevalence;
- OULAD persistent-misconception tails, because an activity category is not a
  pedagogical skill;
- any tail or outcome comparison below its declared minimum sample size.

These exclusions are also recorded in every report under
`publication.excluded_from_publication`.
