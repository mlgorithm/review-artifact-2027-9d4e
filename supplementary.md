# Supplementary Material: Synthetic Data for Whom? Evaluating Synthetic Data for Uncommon Learning Pathways

This Markdown companion contains the supplement's text, numerical tables, and references. The [PDF supplement](supplementary.pdf) is authoritative for the supporting figure and final layout. Machine-readable aggregate results are in [supplement-generated](supplement-generated/) and additional CSVs are in [tables](tables/).

## How to Read This Supplement

In this supplement, a trajectory is one learner’s event sequence (one
course enrollment in OULAD), while a pathway is a behavioral pattern
identified by a rule that multiple trajectories can share. The main
paper organizes evaluation around five learning-analytics questions:
what occurs in the data, how learning unfolds, who is represented, what
analysts can learn from the synthetic data, and what information might
be exposed. This supplement provides the complete operational detail
behind those questions.
Sections <a href="#sec:symbols" data-reference-type="ref"
data-reference="sec:symbols">3</a>–<a href="#sec:dictionary" data-reference-type="ref"
data-reference="sec:dictionary">4</a> define every publication metric,
including its intuition, direction, scope, and conditions under which it
is not estimable.
Section <a href="#sec:complete-results" data-reference-type="ref"
data-reference="sec:complete-results">6</a> then lists every numeric
entry in the nine final `publication_metrics_v2` multi-seed summaries.
Section <a href="#sec:statuses" data-reference-type="ref"
data-reference="sec:statuses">8</a> retains outcome-model collapse, a
compact rare-pathway shape-support summary, and the complete
dataset-by-metric applicability contract. The accompanying
`rare-shape-estimability.csv` retains every seed-level support decision
and its real/synthetic learner counts. Thus an absent numeric row is
never left for the reader to interpret as either zero or an undocumented
omission.

The final benchmark contains three datasets (ASSISTments, EdNet, and
OULAD), three generators (Markov, sequence VAE, and TimeGAN), and the
fixed seeds 20260703, 20260704, and 20260705. All reported standard
deviations are population standard deviations across those generator
seeds. Single-run entries are labeled “one available run” without an SD;
the missing-run count remains explicit. Undefined values are never
replaced by zero.

# Implementation and Reproducibility Details

#### Trajectory frames and postprocessing.

ASSISTments models skill, correctness, attempt, hint, response-time, and
opportunity bins. EdNet models question, correctness, elapsed-time,
timestamp-gap, attempt, and hint bins; deterministic question metadata
are rebuilt from the generated question identifier. OULAD is an explicit
weekly panel. Its generator-fitting frame contains course, dominant
activity, registration, click, active-day, and profile attributes.
Generated profile values, derived fields, and terminal outcomes are
handled by the common train-fitted postprocessor detailed below. Learner
identifiers and event order mark boundaries only and are excluded from
the learned value representation.
Table <a href="#tab:supp-data-contract" data-reference-type="ref"
data-reference="tab:supp-data-contract">1</a> states the complete
implemented contract rather than only the fields retained in the final
normalized output.

| Dataset | Split unit | Train/test trajectories | Train/test rows | Generator-fitted value columns | Rebuilt, replaced, or constructed after generation |
|:---|:---|:---|:---|:---|:---|
| Dataset | Split unit | Train/test trajectories | Train/test rows | Generator-fitted value columns | Rebuilt, replaced, or constructed after generation |
| ASSISTments | Learner ID | 3,429 / 788 | 282,118 / 64,742 | 6: skill ID, correctness, attempt bin, hint bin, response-time bin, opportunity bin | Learner ID and zero-based event order are newly assigned structural fields. |
| EdNet | Learner ID | 7,703 / 1,963 | 1,021,565 / 264,798 | 6: question ID, correctness, elapsed-time bin, timestamp-gap bin, attempt bin, hint bin | Skill ID and part are deterministically rebuilt from question ID; learner ID and order are newly assigned. |
| OULAD | Physical student ID | 26,072 / 6,521 | 743,533 / 187,324 | 13: course ID, dominant activity, registration, click bin, active-days bin, and eight profile attributes | Module and presentation from course; engagement and gap from generated behavior; profiles independently resampled; dropout, failure, and final result constructed; learner ID and week order newly assigned. |

Processed-data and generator column contract. Learner counts refer to
trajectory identifiers; for OULAD these are course enrollments, while
the split itself is by physical student. Structural fields establish
sequence boundaries and are not learned values.
{#tab:supp-data-contract}

#### OULAD shared postprocessor.

Normalization uses only the real-training split as reference and runs in
a fixed order. First, course identifier is made learner-static and every
retained enrollment-week is marked registered. The eight generated
profile fields are then discarded. For each synthetic learner and each
field independently, one value is sampled with replacement from
real-training learners in the same course; the full real-training
profile pool is the fallback for an unseen course. This targets, and
preserves in expectation, course-conditional univariate marginals
without copying one complete real profile as a unit. Module and
presentation are rebuilt from course; engagement is one exactly when
click intensity is nonzero and the learner is registered. Non-engaged
weeks receive activity `none` and zero active days, an engaged week
decoded with zero active days is repaired to the `1--2` bin, and gap
categories are recomputed from weeks since the most recent engaged week.

Dropout, failure, and final result are never generator inputs. A
class-balanced logistic dropout ranker is fitted on real-training
learners using course plus the mean and standard deviation of engagement
after the first four observed weeks; missing post-prefix summaries are
set to zero. The predictor features exclude terminal labels, total
trajectory length, and behavior from the four-week prediction prefix;
real dropout is used only as the fitting target. The frozen model ranks
synthetic learners within course, with seeded random tie-breaking, and
assigns the rounded product of the real-training course dropout rate and
the number of synthetic learners in that course (using the global
training rate if a course rate is unavailable). Dropouts receive
`Withdrawn` and failure zero. Among completers, the lowest mean
generated engagement values are assigned `Fail` to match the
real-training course failure rate; among the remaining learners, the
highest engagement values receive `Distinction` to match the
real-training course distinction rate, and the remainder receive `Pass`.
These labels are pipeline constructs, so OULAD outcome tasks evaluate
the complete generation-and-postprocessing pipeline rather than direct
outcome generation.

#### Generator settings.

The Markov baseline is an order-2 tuple n-gram with minimum context
count two and unigram backoff. The sequence VAE uses categorical windows
of length 20, two 256-unit hidden layers, latent dimension 16, learning
rate 0.001, KL weight 1.0, and sampling temperature 1.0. The external
SynthCity TimeGAN uses boundary-safe length-20 categorical windows, one
50-unit generator layer, one 50-unit discriminator layer, RNN mode, and
CPU execution. Both neural models receive all available training
trajectories, use a batch-size cap of 200 and at most 100 observed
categories per modeled column plus an `<other>` category when needed,
and have a 50-epoch ceiling. The effective epoch count is chosen so
fitting does not exceed approximately 4,000 minibatch updates. The
effective settings, library version, source hashes, preprocessing and
input hashes, and synthetic-output hash are recorded in each run report.
A contiguous block-bootstrap generator remains in the software only as a
copying-sensitive diagnostic; it is not one of the three ranked
generators and is not proposed for privacy-safe release.

#### Targeting and output budgets.

The targeted fit forms the union of eligible real-training pathway
learners and assigns each learner three times the sampling weight of
other learners; membership in multiple pathways does not compound the
weight. This factor is fixed across datasets, generators, pathways, and
seeds and is not tuned against evaluation outcomes. ASSISTments targets
high hint use, late correctness decline, persistent low correctness,
rapid low-accuracy responding, rare skill path, and recovery; EdNet uses
the same set except high hint use; OULAD targets late disengagement,
rare activity path, and reengagement. The selected unions contain 545
ASSISTments learners (15.89%), 1,790 EdNet learners (23.24%), and 2,968
OULAD enrollments (11.38%). Standard and targeted outputs have identical
learner and row budgets. All generators use the same train-fitted,
mean-calibrated truncated-lognormal length sampler; no individual real
learner length is copied or empirically resampled. For neural outputs,
the implementation allocates enough independently generated
20-observation windows to each learner, concatenates them in order, and
truncates the final window to the sampled trajectory length.
Cross-window continuity is not modeled by this assembly.

#### Evaluation sampling and uncertainty.

Memory-heavy fidelity and privacy diagnostics use a deterministic
whole-learner sample capped at 4,000 learners and 120,000 rows per
split; source detection uses at most 100,000 rows per class. Rare-group
prevalence and shape and the primary downstream tasks use the complete
processed frames. Nearest-neighbor queries use at most 1,000 synthetic
trajectories against the complete evaluated real-train reference;
membership inference uses at most 1,000 members, nonmembers, and
synthetic reference trajectories. The pipeline uses 200
learner-clustered percentile-bootstrap resamples for threshold-free
downstream scores and paired differences. For OULAD, clusters are
course-enrollment trajectory identifiers (module, presentation, and
student ID), not physical students. All prediction rows within a sampled
enrollment are retained together; different enrollments of the same
student are not resampled jointly. The additional outcome tasks have one
prediction row per enrollment. Rare-group prevalence uses an independent
learner-membership bootstrap; conditional shape resamples qualifying
learners, capped at 300 per side. Quantities below declared learner or
class-count minima are reported as not estimable rather than replaced by
zero.

#### Authoritative reports and schema.

The final paired reports are under
`experiments/reports/standard_vs_tail_targeted/<dataset>/<model>/`.
Dataset keys are `assistments_2009_2010_skill_builder`, `ednet_kt1`, and
`oulad_weekly_engagement`; model keys are `markov_ngram`,
`sequence_vae_timevae`, and `timegan`. Each
`seed_<seed>/generation_evaluation_report.json` contains the seed-level
result, and `multi_seed_summary.json` contains the aggregate. The
immutable `experiments/reports/standard/` directory is the RQ1 audit
snapshot; it is neither averaged into the paired results nor counted as
additional runs. Runtime manifests are under the corresponding
`experiments/runs/` tree and are git-ignored.

Seed reports declare
`publication.schema_version=publication_metrics_v2`, organize promoted
results under `publication.metrics.rq1` and `publication.metrics.rq2`,
retain exact exclusions and reasons under
`publication.excluded_from_publication`; every `final_audit.status` must
equal `passed` with an empty issues list. Aggregates declare
`publication_schema_version=publication_metrics_v2` and
`metric_summary_scope=publication_only`. Their flattened numeric paths
retain mean, population SD, minimum, maximum, estimable-seed count, and
missing runs; `publication_status_summary` retains nonnumeric collapse
and support states with their exact seeds. The deterministic learner
split uses seed 20260703, and generator seeds are 20260703, 20260704,
and 20260705. Seed-level bootstrap intervals remain in the seed reports
and are not averaged.

# Notation and Shared Comparison Measures

Let $`\mathcal{R}`$ denote the real reference trajectories and
$`\mathcal{S}`$ the synthetic trajectories. Learner $`i`$’s ordered
trajectory is
$`x_i=((z_{it},c_{it},\mathbf{v}_{it},\theta_{it}))_{t=1}^{T_i}`$. The
primary binary behavior is $`z`$: correctness in ASSISTments and EdNet
and weekly engagement in OULAD. The categorical event type $`c`$ is a
skill, question, or activity. The vector $`\mathbf{v}`$ contains the
other modeled signals, and $`\theta`$ is event time or week when a
meaningful time signal is available. Jensen–Shannon divergence follows
Lin (Lin 1991); the geometric-mean normalization of mutual information
follows Strehl and Ghosh (Strehl and Ghosh 2002); source detectability
follows classifier two-sample testing (Lopez-Paz and Oquab 2017);
precision–recall and ROC interpretation follows Davis and
Goadrich (Davis and Goadrich 2006), Saito and Rehmsmeier (Saito and
Rehmsmeier 2015), and Fawcett (Fawcett 2006); train-on-synthetic,
test-on-real evaluation follows prior time-series work (Esteban et al.
2017; Yoon et al. 2019); and the disclosure audit follows the
membership-inference threat model of Shokri et al. (Shokri et al. 2017)
and synthetic-data privacy cautions of Stadler et al. (Stadler et al.
2022).

| Quantity | Definition | Intuition |
|:---|:---|:---|
| Quantity | Definition | Intuition |
| Mean absolute error | $`\operatorname{MAE}(\mathbf a,\mathbf b)=K^{-1}\sum_{k=1}^{K}|a_k-b_k|`$. | Average separation between matched summaries; zero means exact agreement. |
| Jensen–Shannon divergence | $`\operatorname{JS}(P,Q)=\tfrac12\mathrm{KL}_2(P\Vert M)+\tfrac12\mathrm{KL}_2(Q\Vert M)`$, where $`M=(P+Q)/2`$. | Symmetric difference between two discrete distributions. With base-2 logs it lies in $`[0,1]`$, and zero means identical distributions. |
| Normalized mutual information | $`\operatorname{NMI}(X,Y)=I(X;Y)/\sqrt{H(X)H(Y)}`$, with zero when either entropy is zero. | Strength of a categorical relationship after normalizing for the variables’ entropies. |
| Normalized trajectory position | Each learner’s primary-behavior sequence is linearly interpolated onto ten positions from beginning to end before learner averaging. | Compares early-to-late development without allowing longer trajectories to contribute more learners’ worth of evidence. |
| Population SD | $`\sqrt{n^{-1}\sum_{s=1}^{n}(m_s-\bar m)^2}`$ over the estimable fixed generator seeds. | Describes seed-to-seed generator variability; it is not a standard error over learners. |

Shared notation and comparison functions. {#tab:supp-notation}

# Complete Metric Dictionary

The dictionary follows the five questions used in the main paper. Each
subsection gives the metric definitions, learning-analytics intuition,
preferred direction, and applicability conditions for one evaluation
construct.

## Population Composition: What Occurs?

The population-composition metrics ask whether synthetic records
reproduce the amount and mixture of observed behavior.
Aggregate-behavior checks are necessary but insufficient because they
ignore educational order and learner pathways.

| Metric | Formal or operational definition | Learning-analytics intuition | Better |
|:---|:---|:---|:---|
| Metric | Formal or operational definition | Learning-analytics intuition | Better |
| Behavior-rate error | $`|\bar z_{\mathcal{R}}-\bar z_{\mathcal{S}}|`$, using event-weighted means. Because $`z`$ is binary, the negative-state error is identical and is not duplicated. | Does the synthetic dataset contain the same overall correctness or engagement rate? | Lower |
| Event-type frequency divergence | $`\operatorname{JS}(P_{\mathcal{R}}(c),P_{\mathcal{S}}(c))`$. | Does it contain the same mixture of skills, questions, or weekly activity types? | Lower |
| Average feature-distribution divergence | $`J^{-1}\sum_{j=1}^{J}\operatorname{JS}(P_{\mathcal{R}}(v_j),P_{\mathcal{S}}(v_j))`$ over prespecified generator-modeled attributes. | Are modeled attributes such as attempts, hints, timing bins, or activity intensities present in realistic proportions? | Lower |
| Real-versus-synthetic detectability AUROC | $`\max(A,1-A)`$, where $`A`$ is AUROC of a classifier distinguishing real from synthetic event records. Detector partitions are learner-disjoint. | Can a classifier easily tell which modeled records are artificial without seeing records from the same learner on both sides? A value of 0.5 is chance-level indistinguishability; 1 is perfect separation. | 0.5 |

Population-composition metrics. {#tab:supp-dictionary-composition}

## Learning Process: How Does Learning Unfold?

This group tests temporal order, progression, persistence, spacing, and
relationships among educational signals. It distinguishes a realistic
bag of events from a realistic learning trajectory.

| Metric | Formal or operational definition | Learning-analytics intuition | Better |
|:---|:---|:---|:---|
| Metric | Formal or operational definition | Learning-analytics intuition | Better |
| Behavior-transition divergence | $`\operatorname{JS}`$ between pooled within-learner adjacent pairs $`(z_t,z_{t+1})`$. | Are persistence, recovery, and decline transitions produced at realistic rates? | Lower |
| Three-event sequence divergence | $`\operatorname{JS}`$ between pooled within-learner event-type triples $`(c_t,c_{t+1},c_{t+2})`$. Windows never cross learner boundaries. | Are short ordered skill or activity motifs realistic? | Lower |
| Trajectory-shape error | $`10^{-1}\sum_{b=1}^{10}|q_{\mathcal{R}}(b)-q_{\mathcal{S}}(b)|`$, where $`q_D(b)`$ is the learner-averaged primary behavior at normalized position $`b`$. | Does the average trajectory rise, fall, or recover in the right way from beginning to end? | Lower |
| Behavior-persistence error | $`3^{-1}\sum_{\ell\in\{1,2,3\}}|\rho_{\mathcal{R}}(z_t,z_{t+\ell})-\rho_{\mathcal{S}}(z_t,z_{t+\ell})|`$. | Do correct/incorrect or engaged/disengaged states form similarly persistent streaks? | Lower |
| Response/inactivity-gap distribution divergence | $`\operatorname{JS}(P_{\mathcal{R}}(g),P_{\mathcal{S}}(g))`$. In EdNet, $`g`$ is binned elapsed time since the prior response; in OULAD, it is binned weeks since the most recent engaged week. It is not applicable in ASSISTments. | Does the generator preserve response spacing or inactivity history, rather than response duration within one event? | Lower |
| Behavior–hint lag error | Mean absolute difference in $`\rho(z_t,h_{t+\ell})`$ for $`\ell\in\{0,1\}`$. It is not estimated if hint use is absent or constant. | Is help-seeking related to current and next-step performance in the same way? | Lower |
| Feature-dependence error | Mean absolute real–synthetic NMI difference over prespecified scientifically relevant signal pairs. | Are relationships among modeled signals retained even if individual columns look realistic? | Lower |

Learning-process metrics. {#tab:supp-dictionary-process}

## Learner-Pathway Representation: Who Is Represented?

This group asks whether common learner strata and uncommon but
intervention-relevant pathways appear in realistic proportions, and
whether learners in those groups show realistic later behavior.
Prefix-defined subgroups use a rule fitted on real training data and
evaluate only later observations. Rare-group rules and thresholds are
also fitted only on real training data and then applied unchanged to
synthetic and held-out real learners.

| Metric | Formal or operational definition | Learning-analytics intuition | Better |
|:---|:---|:---|:---|
| Metric | Formal or operational definition | Learning-analytics intuition | Better |
| Subgroup behavior error | MAE across fixed prefix-defined groups in their later primary-behavior rates. | Do learners with different early histories show realistic later correctness or engagement? | Lower |
| Subgroup-size error | MAE across the same groups in learner share. | Does the generator produce the same mixture of early-history learner types? | Lower |
| Subgroup hint-use error | MAE across eligible groups in later hint-request rates; not estimated without a varying hint signal. | Do early-history learner groups retain their later help-seeking behavior? | Lower |
| Rare-group prevalence error | For group $`g`$, $`|\pi_g(\mathcal{R})-\pi_g(\mathcal{S})|`$, where $`\pi_g(D)`$ is learner prevalence. | Does the generator include the right fraction of learners following a rare pathway? | Lower |
| Signed rare-group prevalence difference | For group $`g`$, $`\pi_g(\mathcal{S})-\pi_g(\mathcal{R})`$. It is reported in percentage points alongside the absolute error. | Is the pathway underrepresented (negative), accurately represented (near zero), or overrepresented (positive)? | Zero |
| Rare-group trajectory-shape error | Trajectory-shape error recomputed among learners satisfying $`g`$. It requires at least ten qualifying real and ten qualifying synthetic learners. | Conditional on producing the group, does correctness or engagement among qualifying learners develop realistically? | Lower |

Learner-pathway representation metrics. {#tab:supp-dictionary-pathways}

Distribution-based rarity rules target the outer 5% of the real-training
distribution. The 10% ceiling allows for ties in discretized variables
without treating a common behavior as rare. Fixed change thresholds and
minimum lengths are benchmark defaults for interpretable behavioral
contrasts, not universal educational cut points. Prevalence remains
reportable when no or too few synthetic learners qualify, because
absence is itself a representation result. Shape is marked not estimable
when either arm has fewer than ten qualifying learners. Publication rare
groups have complete-real-training learner prevalence at most 10%. The
published groups are high hint use, late correctness decline, persistent
low correctness, rapid low-accuracy responding, rare skill path, and
recovery where applicable; OULAD uses late disengagement, rare activity
path, and reengagement. A coarse binned signal is excluded if no strict
or inclusive boundary yields a nonempty group within the rarity ceiling.

Rare skill/activity paths are scored by the mean log$`_2`$ probability
of successive skill/activity transitions under a real-training
transition model with add-one smoothing. The bottom 5% defines the
candidate rare-pathway group, subject to the rarity ceiling and at least
one observed transition.

## Analytic Usefulness: What Can Analysts Learn?

The main paper’s prediction-specification table lists the actual
predictor inputs, fixed early-history lengths, and numerical eligibility
requirements. Every downstream predictor is evaluated on the same
held-out real learners. The primary task is next-response correctness
for ASSISTments and EdNet and next-week engagement for OULAD. The five
training arms are real only (TRTR), standard synthetic only (TSTR), real
plus standard synthetic, pathway-targeted synthetic only, and real plus
pathway-targeted synthetic. Learner-level outcome tasks use four arms
because real plus targeted synthetic is not defined there. In
synthetic-only arms, no real outcome rows fit the predictor, although
all arms use the same real-training-fitted schema and transformations.
AUPRC and AUROC are computed over eligible prediction rows, so longer
trajectories contribute more instances; learner-clustered bootstrapping
preserves within- learner dependence without changing that
event-weighted point estimate.

| Metric | Formal or operational definition | Learning-analytics intuition | Better |
|:---|:---|:---|:---|
| Metric | Formal or operational definition | Learning-analytics intuition | Better |
| AUPRC | Area under the precision–recall curve on held-out real data. The observed positive rate is the no-skill reference. | Primary score for imbalanced educational outcomes and rare learner slices; rewards identifying positives without many false alarms. | Higher |
| AUROC | Probability that a randomly chosen positive receives a higher score than a randomly chosen negative. | Secondary threshold-free discrimination score; less sensitive than AUPRC to positive-class prevalence. | Higher |
| Standard synthetic utility loss | $`M_{\rm TRTR}-M_{\rm standard}`$, separately for AUPRC and AUROC. | How much held-out-real predictive performance is lost when analysts train only on ordinary synthetic data? | Lower |
| Standard augmentation gain | $`M_{\rm real+standard}-M_{\rm TRTR}`$. | Does adding ordinary synthetic data improve a predictor already trained on real data? | Higher |
| Targeted-vs-standard TSTR gain | $`M_{\rm targeted}-M_{\rm standard}`$. | Does emphasizing rare training learners make synthetic-only data more useful? | Higher |
| Targeted augmentation gain | $`M_{\rm real+targeted}-M_{\rm TRTR}`$. | Does pathway-targeted synthetic augmentation improve over real-only training? | Higher |
| Pathway-level predictive usefulness | The same five arms and four contrasts, evaluated only on held-out real learners satisfying a fixed rare-group rule. | Can a model trained on synthetic data learn the signal needed for uncommon, educationally consequential learner pathways? | Varies |
| Rare-group prevalence-match change | $`E^{\rm prev}_{\rm standard}-E^{\rm prev}_{\rm targeted}`$. | Did targeting move the learner proportion closer to its real-training reference, without treating overproduction as success? | Higher |
| Rare-group shape-error reduction | $`E^{\rm shape}_{\rm standard}-E^{\rm shape}_{\rm targeted}`$, only when both errors are estimable. | Did targeting improve the group’s within-trajectory primary-behavior pattern? | Higher |
| Learner-level outcome-task AUPRC/AUROC and contrasts | Persistent-low-correctness and recovery prediction in ASSISTments/EdNet; dropout and course-failure prediction in OULAD, using only declared early history. | Does synthetic data support intervention-oriented outcomes beyond the primary next-step prediction task? | Varies |

Analytic-usefulness metrics and contrasts.
{#tab:supp-dictionary-utility}

OULAD terminal labels are excluded from generator inputs and constructed
by the common leakage-safe post-generation pipeline. These
pipeline-level outcome tasks evaluate the complete synthetic-data
pipeline, not direct generation of future outcomes. The machine-readable
reports retain the corresponding internal scope code for auditability.

## Disclosure Risk: What Might Be Exposed?

These measures are empirical memorization audits, not
differential-privacy guarantees. A large nearest-neighbor distance
should not be maximized without regard to fidelity: unrelated noise can
be far from training data while being useless.

#### Trajectory representation and distance.

Each event is represented by the tuple of all nonidentifier, non-order
audit columns shared by the normalized real and synthetic frames. For
trajectories $`x=(u_1,\ldots,u_{T_x})`$ and $`y=(v_1,\ldots,v_{T_y})`$,
the implemented positional distance is
``` math
d(x,y)=\frac{|T_x-T_y|+
\sum_{t=1}^{\min(T_x,T_y)}\mathbf{1}[u_t\ne v_t]}
{\max(T_x,T_y)}.
```
Distance is zero when both trajectories are empty. An aligned position
counts as one mismatch when its complete event tuple differs, regardless
of how many fields differ, and unmatched suffix positions count through
the length term. Thus $`d\in[0,1]`$, with no temporal realignment and
$`d=0`$ indicating an exact normalized audit-trajectory match.

#### Near-copy and membership rules.

The threshold $`\tau=0.1`$ is fixed for every dataset, generator, arm,
and seed. Under this distance, $`d\leq0.1`$ means that at least 90% of
positions in the longer trajectory match exactly after counting length
mismatch. This is a prespecified benchmark flag, not a
literature-derived or validated privacy boundary. The membership attack
gives each real learner the fixed score
$`-\min_{z\in\mathcal{S}}d(x,z)`$, so greater proximity always means
“more likely a training member.” A below-chance AUROC is not inverted
after seeing the labels; reported advantage is $`\max\{0,2(A-0.5)\}`$.
For copying and membership, targeting risk change is targeted minus
standard; for mean distance it is standard minus targeted, so positive
values consistently indicate greater empirical exposure after targeting.
Sampling limits and bootstrap counts are given in
Section <a href="#sec:implementation" data-reference-type="ref"
data-reference="sec:implementation">2</a>.

| Metric | Formal or operational definition | Learning-analytics intuition | Safer |
|:---|:---|:---|:---|
| Metric | Formal or operational definition | Learning-analytics intuition | Safer |
| Exact-duplicate rate | Fraction of synthetic learners whose full normalized audit trajectory exactly matches a real-training trajectory. | How often does the generator literally reproduce a learner’s normalized sequence over the audit columns? | Lower |
| Near-duplicate rate | Fraction of synthetic trajectories with $`\min_{r\in\mathcal{R}}d(x,r)\leq\tau`$, where $`\tau=0.1`$. | How often does the generator produce a trajectory unusually close to one seen in training? | Lower |
| Mean nearest-training distance | Mean $`\min_{r\in\mathcal{R}}d(x,r)`$ over evaluated synthetic trajectories. | On average, how closely does each synthetic learner resemble its nearest training learner? | Higher |
| Membership-inference advantage | $`\max(0,2(A-0.5))`$ for the fixed proximity-oriented attack AUROC $`A`$. | Can an attacker distinguish training learners from held-out learners using proximity to the synthetic data? | Lower |
| Behavior-projection exact-duplicate rate | Exact-copy rate after projecting OULAD to ordered dominant-activity/engagement pairs. | How often are weekly behavioral patterns repeated when independently assigned profiles and constructed outcomes are removed? | Lower |
| Targeted-minus-standard privacy-risk change | Targeted minus standard for copy and membership metrics; for distance, standard distance minus targeted distance so positive always means greater exposure. | Did pathway targeting make the empirical memorization signal stronger? | Lower |

Disclosure-risk metrics. {#tab:supp-dictionary-privacy}

OULAD full-record nearest-neighbor and membership comparisons are
excluded because demographic profiles are independently sampled and
terminal outcomes are constructed after generation. Only exact copying
on the ordered dominant-activity/engagement projection is promoted for
comparison. Exploratory pathway-specific privacy blocks remain in raw
reports, but the publication audit is deliberately dataset-level and
does not estimate risk specifically for uncommon pathways.

# Applicability, Uncertainty, and Exclusions

Metrics use the real training split as the fidelity reference because
generators are trained on the real-training distribution. Preprocessing
transforms, category maps, trajectory-length models, rare-group
thresholds, subgroup cut points, postprocessors, and targeting
membership are fit without held-out real test learners. Downstream
evaluation uses held-out real learners only. OULAD is split by physical
student, preventing different course enrollments from the same student
from crossing the train/test boundary.

All numeric tables report means and population SD over fixed generator
seeds. Seed-level bootstrap intervals over held-out learners remain in
the JSON seed reports and are not averaged across seeds. Bootstrap
intervals quantify learner-sampling uncertainty conditional on one fixed
generated dataset and seed, whereas cross-seed SD describes generator
stochasticity. With three generator seeds and one fixed learner-disjoint
split, the paper treats cross-seed differences descriptively; neither
summary captures split-to-split variability. A metric based on a
constant or absent signal is not applicable. A rare-group conditional
metric is not estimable below its declared support. Model-collapse
outcomes are preserved as statuses.
Table <a href="#tab:supp-exclusions" data-reference-type="ref"
data-reference="tab:supp-exclusions">8</a> lists every exclusion
category present in the final seed reports. The seed-level source of
truth is `publication.excluded_from_publication`; exclusion from
comparison is distinct from non-estimability, whose support or class
status remains a publication result.

| Report exclusion key | Excluded item | Reason |
|:---|:---|:---|
| Report exclusion key | Excluded item | Reason |
| `fixed_threshold_classification_scores` | Accuracy, precision, recall, and F1 at an untuned 0.5 threshold | Threshold-free AUPRC and AUROC are the prespecified predictive comparisons. |
| `threshold_metric_deltas` | Targeting and augmentation contrasts for fixed-threshold scores | Only AUPRC/AUROC gaps are promoted. |
| `lightweight_downstream_utility` | Skill-mean probability baseline | It is a smoke-test diagnostic superseded by the declared histogram-gradient-boosting model. |
| `sequence_length_model_ranking` | Length-distribution metrics | All generators use the same train-fitted length sampler, so the result is not architecture-attributable. |
| `short_and_long_trajectory_tails` | Length-based uncommon pathways | The shared length sampler prevents generator attribution. |
| `diversity_uniqueness_rates` | Raw diversity/coverage, uniqueness, entropy, and cardinality diagnostics | Novelty is not monotonic quality and remains a raw diagnostic. |
| `raw_membership_auc_inversion` | Flipping a below-chance membership AUROC | Attack orientation is fixed before labels are inspected; the published advantage is not post-hoc inverted. |
| `raw_test_fidelity_duplicates` | Fidelity recomputed against held-out real test | Real train is the generator’s fidelity reference; held-out real learners are reserved for generalization and nonmembership evaluation. |
| `raw_diagnostic_aggregates` | Unfiltered all-column marginals/dependence, detector accuracy, and minimum-distance extremes | These broad exploratory outputs do not belong to the prespecified construct metrics. |
| `omitted_standard_tail_groups` | Inapplicable or unsupported standard-arm pathway blocks | Reasons include absent signals, postprocessed outcomes, and fewer than ten qualifying learners; prevalence is retained when defined. |
| `omitted_targeted_tail_groups` | The corresponding targeted-arm pathway blocks | The same frozen applicability and support rules apply to both generation arms. |
| `omitted_tail_learnability_groups` | Ineligible held-out-real pathway slices | Groups above the rarity ceiling, outside the dataset semantics, or below predictive support are not included in pathway-level prediction comparisons. |
| `unavailable_percentile_tail_groups` | Percentile pathways for which ties cannot yield a nonempty group at or below 10% | A common discretized behavior is not relabeled as rare. |
| `distinguishability` | OULAD full-record source detection | Independently resampled profiles and constructed outcomes confound generator attribution. |

Complete publication-exclusion dictionary. Dataset-specific
omitted-group reasons and fitted prevalences remain verbatim in each
seed report. {#tab:supp-exclusions}

Dataset-specific scope rules add the following consequences. OULAD
demographic subgroup fidelity, postprocessed dropout prevalence/shape,
full-record source detection, and full-record
nearest-neighbor/membership privacy are not used for generator
comparison. The diagnostic persistent-misconception rule is omitted
where the dataset does not support that pedagogical interpretation. Only
the prespecified signal pairs, generator-attributable fields, and
real-train fidelity reference enter publication comparisons.

The full applicability grid in
Table <a href="#tab:supp-applicability" data-reference-type="ref"
data-reference="tab:supp-applicability">17</a> distinguishes three cases
that would otherwise all appear as missing cells: a metric can be
inapplicable because a required signal is absent or constant,
deliberately excluded because postprocessing prevents a
generator-comparable interpretation, or applicable but not estimable in
a particular seed because learner or class support is insufficient.

# Complete Numerical Publication Results

The following tables contain all 1,451 numeric entries in the nine final
multi-seed publication summaries. The generation audit verifies equality
between source and emitted row counts. Printed tables show mean,
population SD, estimable-seed count, and missing-seed count. The
accompanying `supplement-generated/complete-publication-results.csv`
also retains the source minimum, maximum, exact JSON path, and all six
summary fields for every row.

<table id="tab:supp-population-composition">
<caption>Complete publication results for population composition. Every
numeric metric-summary entry is shown; SD is the population SD across
the fixed seeds and
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>n</mi><annotation encoding="application/x-tex">n</annotation></semantics></math>
is the number of estimable seeds.</caption>
<thead>
<tr>
<th style="text-align: left;">Dataset</th>
<th style="text-align: left;">Generator</th>
<th style="text-align: left;">Scope / comparison</th>
<th style="text-align: left;">Metric</th>
<th style="text-align: left;">Mean
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>±</mi><annotation encoding="application/x-tex">\pm</annotation></semantics></math>
SD
(<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>n</mi><annotation encoding="application/x-tex">n</annotation></semantics></math>)</th>
<th style="text-align: left;">Direction</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="6" style="text-align: left;"><em>Continued from previous
page</em></td>
</tr>
<tr>
<td style="text-align: left;">Dataset</td>
<td style="text-align: left;">Generator</td>
<td style="text-align: left;">Scope / comparison</td>
<td style="text-align: left;">Metric</td>
<td style="text-align: left;">Mean
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>±</mi><annotation encoding="application/x-tex">\pm</annotation></semantics></math>
SD
(<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>n</mi><annotation encoding="application/x-tex">n</annotation></semantics></math>)</td>
<td style="text-align: left;">Direction</td>
</tr>
<tr>
<td colspan="6" style="text-align: right;"><em>Continued on next
page</em></td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Average feature-distribution
divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0014</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.0014 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-rate error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0120</mn><mo>±</mo><mn>0.0029</mn></mrow><annotation encoding="application/x-tex">0.0120 \mathbin{\pm} 0.0029</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Event-type frequency divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0053</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.0053 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Real-versus-synthetic detectability
AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5334</mn><mo>±</mo><mn>0.0037</mn></mrow><annotation encoding="application/x-tex">0.5334 \mathbin{\pm} 0.0037</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">0.5 best</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Average feature-distribution
divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0003</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0003 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-rate error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0012</mn><mo>±</mo><mn>0.0008</mn></mrow><annotation encoding="application/x-tex">0.0012 \mathbin{\pm} 0.0008</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Event-type frequency divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0013</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.0013 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Real-versus-synthetic detectability
AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8500</mn><mo>±</mo><mn>0.0008</mn></mrow><annotation encoding="application/x-tex">0.8500 \mathbin{\pm} 0.0008</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">0.5 best</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Average feature-distribution
divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1907</mn><mo>±</mo><mn>0.0923</mn></mrow><annotation encoding="application/x-tex">0.1907 \mathbin{\pm} 0.0923</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-rate error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3036</mn><mo>±</mo><mn>0.0745</mn></mrow><annotation encoding="application/x-tex">0.3036 \mathbin{\pm} 0.0745</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Event-type frequency divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4101</mn><mo>±</mo><mn>0.1875</mn></mrow><annotation encoding="application/x-tex">0.4101 \mathbin{\pm} 0.1875</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Real-versus-synthetic detectability
AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9672</mn><mo>±</mo><mn>0.0214</mn></mrow><annotation encoding="application/x-tex">0.9672 \mathbin{\pm} 0.0214</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">0.5 best</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Average feature-distribution
divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0152</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0152 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-rate error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0163</mn><mo>±</mo><mn>0.0013</mn></mrow><annotation encoding="application/x-tex">0.0163 \mathbin{\pm} 0.0013</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Event-type frequency divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0131</mn><mo>±</mo><mn>0.0008</mn></mrow><annotation encoding="application/x-tex">0.0131 \mathbin{\pm} 0.0008</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Real-versus-synthetic detectability
AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5794</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">0.5794 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">0.5 best</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Average feature-distribution
divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0070</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">0.0070 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-rate error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0192</mn><mo>±</mo><mn>0.0128</mn></mrow><annotation encoding="application/x-tex">0.0192 \mathbin{\pm} 0.0128</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Event-type frequency divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0020</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.0020 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Real-versus-synthetic detectability
AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6894</mn><mo>±</mo><mn>0.0015</mn></mrow><annotation encoding="application/x-tex">0.6894 \mathbin{\pm} 0.0015</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">0.5 best</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Average feature-distribution
divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1197</mn><mo>±</mo><mn>0.0013</mn></mrow><annotation encoding="application/x-tex">0.1197 \mathbin{\pm} 0.0013</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-rate error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3345</mn><mo>±</mo><mn>0.0004</mn></mrow><annotation encoding="application/x-tex">0.3345 \mathbin{\pm} 0.0004</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Event-type frequency divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0101</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.0101 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Real-versus-synthetic detectability
AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9092</mn><mo>±</mo><mn>0.0008</mn></mrow><annotation encoding="application/x-tex">0.9092 \mathbin{\pm} 0.0008</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">0.5 best</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Average feature-distribution
divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0174</mn><mo>±</mo><mn>0.0003</mn></mrow><annotation encoding="application/x-tex">0.0174 \mathbin{\pm} 0.0003</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-rate error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1644</mn><mo>±</mo><mn>0.0038</mn></mrow><annotation encoding="application/x-tex">0.1644 \mathbin{\pm} 0.0038</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Event-type frequency divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0203</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">0.0203 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Average feature-distribution
divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0255</mn><mo>±</mo><mn>0.0031</mn></mrow><annotation encoding="application/x-tex">0.0255 \mathbin{\pm} 0.0031</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-rate error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0100</mn><mo>±</mo><mn>0.0051</mn></mrow><annotation encoding="application/x-tex">0.0100 \mathbin{\pm} 0.0051</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Event-type frequency divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0231</mn><mo>±</mo><mn>0.0032</mn></mrow><annotation encoding="application/x-tex">0.0231 \mathbin{\pm} 0.0032</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Average feature-distribution
divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1605</mn><mo>±</mo><mn>0.1231</mn></mrow><annotation encoding="application/x-tex">0.1605 \mathbin{\pm} 0.1231</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-rate error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3494</mn><mo>±</mo><mn>0.1877</mn></mrow><annotation encoding="application/x-tex">0.3494 \mathbin{\pm} 0.1877</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Event-type frequency divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2222</mn><mo>±</mo><mn>0.1331</mn></mrow><annotation encoding="application/x-tex">0.2222 \mathbin{\pm} 0.1331</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
</tbody>
</table>

<table id="tab:supp-learning-process">
<caption>Complete publication results for learning process. Every
numeric metric-summary entry is shown; SD is the population SD across
the fixed seeds and
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>n</mi><annotation encoding="application/x-tex">n</annotation></semantics></math>
is the number of estimable seeds.</caption>
<thead>
<tr>
<th style="text-align: left;">Dataset</th>
<th style="text-align: left;">Generator</th>
<th style="text-align: left;">Scope / comparison</th>
<th style="text-align: left;">Metric</th>
<th style="text-align: left;">Mean
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>±</mi><annotation encoding="application/x-tex">\pm</annotation></semantics></math>
SD
(<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>n</mi><annotation encoding="application/x-tex">n</annotation></semantics></math>)</th>
<th style="text-align: left;">Direction</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="6" style="text-align: left;"><em>Continued from previous
page</em></td>
</tr>
<tr>
<td style="text-align: left;">Dataset</td>
<td style="text-align: left;">Generator</td>
<td style="text-align: left;">Scope / comparison</td>
<td style="text-align: left;">Metric</td>
<td style="text-align: left;">Mean
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>±</mi><annotation encoding="application/x-tex">\pm</annotation></semantics></math>
SD
(<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>n</mi><annotation encoding="application/x-tex">n</annotation></semantics></math>)</td>
<td style="text-align: left;">Direction</td>
</tr>
<tr>
<td colspan="6" style="text-align: right;"><em>Continued on next
page</em></td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Feature-dependence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0017</mn><mo>±</mo><mn>0.0004</mn></mrow><annotation encoding="application/x-tex">0.0017 \mathbin{\pm} 0.0004</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior–hint lag error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0071</mn><mo>±</mo><mn>0.0021</mn></mrow><annotation encoding="application/x-tex">0.0071 \mathbin{\pm} 0.0021</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-persistence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0293</mn><mo>±</mo><mn>0.0042</mn></mrow><annotation encoding="application/x-tex">0.0293 \mathbin{\pm} 0.0042</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0409</mn><mo>±</mo><mn>0.0041</mn></mrow><annotation encoding="application/x-tex">0.0409 \mathbin{\pm} 0.0041</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-transition divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0002</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.0002 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Three-event sequence divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0828</mn><mo>±</mo><mn>0.0006</mn></mrow><annotation encoding="application/x-tex">0.0828 \mathbin{\pm} 0.0006</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Feature-dependence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0993</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0993 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior–hint lag error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4437</mn><mo>±</mo><mn>0.0026</mn></mrow><annotation encoding="application/x-tex">0.4437 \mathbin{\pm} 0.0026</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-persistence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2017</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">0.2017 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0272</mn><mo>±</mo><mn>0.0029</mn></mrow><annotation encoding="application/x-tex">0.0272 \mathbin{\pm} 0.0029</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-transition divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0100</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">0.0100 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Three-event sequence divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8635</mn><mo>±</mo><mn>0.0032</mn></mrow><annotation encoding="application/x-tex">0.8635 \mathbin{\pm} 0.0032</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Feature-dependence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1029</mn><mo>±</mo><mn>0.0123</mn></mrow><annotation encoding="application/x-tex">0.1029 \mathbin{\pm} 0.0123</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior–hint lag error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3507</mn><mo>±</mo><mn>0.1340</mn></mrow><annotation encoding="application/x-tex">0.3507 \mathbin{\pm} 0.1340</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-persistence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1881</mn><mo>±</mo><mn>0.0192</mn></mrow><annotation encoding="application/x-tex">0.1881 \mathbin{\pm} 0.0192</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3307</mn><mo>±</mo><mn>0.0760</mn></mrow><annotation encoding="application/x-tex">0.3307 \mathbin{\pm} 0.0760</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-transition divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2366</mn><mo>±</mo><mn>0.1211</mn></mrow><annotation encoding="application/x-tex">0.2366 \mathbin{\pm} 0.1211</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Three-event sequence divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7653</mn><mo>±</mo><mn>0.0025</mn></mrow><annotation encoding="application/x-tex">0.7653 \mathbin{\pm} 0.0025</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Feature-dependence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0029</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.0029 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-persistence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0230</mn><mo>±</mo><mn>0.0014</mn></mrow><annotation encoding="application/x-tex">0.0230 \mathbin{\pm} 0.0014</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1550</mn><mo>±</mo><mn>0.0049</mn></mrow><annotation encoding="application/x-tex">0.1550 \mathbin{\pm} 0.0049</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-transition divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0005</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.0005 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Response/inactivity-gap distribution
divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0036</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.0036 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Three-event sequence divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4438</mn><mo>±</mo><mn>0.0012</mn></mrow><annotation encoding="application/x-tex">0.4438 \mathbin{\pm} 0.0012</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Feature-dependence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0088</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0088 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-persistence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0941</mn><mo>±</mo><mn>0.0006</mn></mrow><annotation encoding="application/x-tex">0.0941 \mathbin{\pm} 0.0006</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1460</mn><mo>±</mo><mn>0.0126</mn></mrow><annotation encoding="application/x-tex">0.1460 \mathbin{\pm} 0.0126</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-transition divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0028</mn><mo>±</mo><mn>0.0008</mn></mrow><annotation encoding="application/x-tex">0.0028 \mathbin{\pm} 0.0008</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Response/inactivity-gap distribution
divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0008</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">0.0008 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Three-event sequence divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8978</mn><mo>±</mo><mn>0.0003</mn></mrow><annotation encoding="application/x-tex">0.8978 \mathbin{\pm} 0.0003</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Feature-dependence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0087</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.0087 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-persistence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0932</mn><mo>±</mo><mn>0.0015</mn></mrow><annotation encoding="application/x-tex">0.0932 \mathbin{\pm} 0.0015</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5047</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">0.5047 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-transition divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3248</mn><mo>±</mo><mn>0.0027</mn></mrow><annotation encoding="application/x-tex">0.3248 \mathbin{\pm} 0.0027</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Response/inactivity-gap distribution
divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1745</mn><mo>±</mo><mn>0.0008</mn></mrow><annotation encoding="application/x-tex">0.1745 \mathbin{\pm} 0.0008</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Three-event sequence divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8904</mn><mo>±</mo><mn>0.0016</mn></mrow><annotation encoding="application/x-tex">0.8904 \mathbin{\pm} 0.0016</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Feature-dependence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0842</mn><mo>±</mo><mn>0.0018</mn></mrow><annotation encoding="application/x-tex">0.0842 \mathbin{\pm} 0.0018</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-persistence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0269</mn><mo>±</mo><mn>0.0017</mn></mrow><annotation encoding="application/x-tex">0.0269 \mathbin{\pm} 0.0017</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1043</mn><mo>±</mo><mn>0.0033</mn></mrow><annotation encoding="application/x-tex">0.1043 \mathbin{\pm} 0.0033</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-transition divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0282</mn><mo>±</mo><mn>0.0012</mn></mrow><annotation encoding="application/x-tex">0.0282 \mathbin{\pm} 0.0012</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Response/inactivity-gap distribution
divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0294</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">0.0294 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Three-event sequence divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0420</mn><mo>±</mo><mn>0.0014</mn></mrow><annotation encoding="application/x-tex">0.0420 \mathbin{\pm} 0.0014</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Feature-dependence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1971</mn><mo>±</mo><mn>0.0036</mn></mrow><annotation encoding="application/x-tex">0.1971 \mathbin{\pm} 0.0036</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-persistence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2506</mn><mo>±</mo><mn>0.0033</mn></mrow><annotation encoding="application/x-tex">0.2506 \mathbin{\pm} 0.0033</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0841</mn><mo>±</mo><mn>0.0090</mn></mrow><annotation encoding="application/x-tex">0.0841 \mathbin{\pm} 0.0090</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-transition divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0161</mn><mo>±</mo><mn>0.0003</mn></mrow><annotation encoding="application/x-tex">0.0161 \mathbin{\pm} 0.0003</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Response/inactivity-gap distribution
divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0091</mn><mo>±</mo><mn>0.0006</mn></mrow><annotation encoding="application/x-tex">0.0091 \mathbin{\pm} 0.0006</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Three-event sequence divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1053</mn><mo>±</mo><mn>0.0056</mn></mrow><annotation encoding="application/x-tex">0.1053 \mathbin{\pm} 0.0056</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Feature-dependence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2012</mn><mo>±</mo><mn>0.1637</mn></mrow><annotation encoding="application/x-tex">0.2012 \mathbin{\pm} 0.1637</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-persistence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3582</mn><mo>±</mo><mn>0.0930</mn></mrow><annotation encoding="application/x-tex">0.3582 \mathbin{\pm} 0.0930</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2853</mn><mo>±</mo><mn>0.1825</mn></mrow><annotation encoding="application/x-tex">0.2853 \mathbin{\pm} 0.1825</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Behavior-transition divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2215</mn><mo>±</mo><mn>0.2116</mn></mrow><annotation encoding="application/x-tex">0.2215 \mathbin{\pm} 0.2116</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Response/inactivity-gap distribution
divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3152</mn><mo>±</mo><mn>0.3144</mn></mrow><annotation encoding="application/x-tex">0.3152 \mathbin{\pm} 0.3144</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Three-event sequence divergence</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4071</mn><mo>±</mo><mn>0.1321</mn></mrow><annotation encoding="application/x-tex">0.4071 \mathbin{\pm} 0.1321</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
</tbody>
</table>

<table id="tab:supp-learner-pathway-representation">
<caption>Complete publication results for learner-pathway
representation. Every numeric metric-summary entry is shown; SD is the
population SD across the fixed seeds and
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>n</mi><annotation encoding="application/x-tex">n</annotation></semantics></math>
is the number of estimable seeds.</caption>
<thead>
<tr>
<th style="text-align: left;">Dataset</th>
<th style="text-align: left;">Generator</th>
<th style="text-align: left;">Scope / comparison</th>
<th style="text-align: left;">Metric</th>
<th style="text-align: left;">Mean
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>±</mi><annotation encoding="application/x-tex">\pm</annotation></semantics></math>
SD
(<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>n</mi><annotation encoding="application/x-tex">n</annotation></semantics></math>)</th>
<th style="text-align: left;">Direction</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="6" style="text-align: left;"><em>Continued from previous
page</em></td>
</tr>
<tr>
<td style="text-align: left;">Dataset</td>
<td style="text-align: left;">Generator</td>
<td style="text-align: left;">Scope / comparison</td>
<td style="text-align: left;">Metric</td>
<td style="text-align: left;">Mean
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>±</mi><annotation encoding="application/x-tex">\pm</annotation></semantics></math>
SD
(<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>n</mi><annotation encoding="application/x-tex">n</annotation></semantics></math>)</td>
<td style="text-align: left;">Direction</td>
</tr>
<tr>
<td colspan="6" style="text-align: right;"><em>Continued on next
page</em></td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Subgroup hint-use error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0509</mn><mo>±</mo><mn>0.0007</mn></mrow><annotation encoding="application/x-tex">0.0509 \mathbin{\pm} 0.0007</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Subgroup behavior error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0646</mn><mo>±</mo><mn>0.0011</mn></mrow><annotation encoding="application/x-tex">0.0646 \mathbin{\pm} 0.0011</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Subgroup-size error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0546</mn><mo>±</mo><mn>0.0030</mn></mrow><annotation encoding="application/x-tex">0.0546 \mathbin{\pm} 0.0030</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">High Hint Use</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0293</mn><mo>±</mo><mn>0.0015</mn></mrow><annotation encoding="application/x-tex">0.0293 \mathbin{\pm} 0.0015</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">High Hint Use</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0559</mn><mo>±</mo><mn>0.0099</mn></mrow><annotation encoding="application/x-tex">0.0559 \mathbin{\pm} 0.0099</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0050</mn><mo>±</mo><mn>0.0013</mn></mrow><annotation encoding="application/x-tex">0.0050 \mathbin{\pm} 0.0013</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0439</mn><mo>±</mo><mn>0.0153</mn></mrow><annotation encoding="application/x-tex">0.0439 \mathbin{\pm} 0.0153</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0197</mn><mo>±</mo><mn>0.0017</mn></mrow><annotation encoding="application/x-tex">0.0197 \mathbin{\pm} 0.0017</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0096</mn><mo>±</mo><mn>0.0029</mn></mrow><annotation encoding="application/x-tex">0.0096 \mathbin{\pm} 0.0029</annotation></semantics></math>
(2; 1 missing)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0140</mn><mo>±</mo><mn>0.0014</mn></mrow><annotation encoding="application/x-tex">0.0140 \mathbin{\pm} 0.0014</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1054</mn><mo>±</mo><mn>0.0229</mn></mrow><annotation encoding="application/x-tex">0.1054 \mathbin{\pm} 0.0229</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0276</mn><mo>±</mo><mn>0.0076</mn></mrow><annotation encoding="application/x-tex">0.0276 \mathbin{\pm} 0.0076</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0540</mn><mo>±</mo><mn>0.0215</mn></mrow><annotation encoding="application/x-tex">0.0540 \mathbin{\pm} 0.0215</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0031</mn><mo>±</mo><mn>0.0011</mn></mrow><annotation encoding="application/x-tex">0.0031 \mathbin{\pm} 0.0011</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0659</mn><mo>±</mo><mn>0.0112</mn></mrow><annotation encoding="application/x-tex">0.0659 \mathbin{\pm} 0.0112</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Subgroup hint-use error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0515</mn><mo>±</mo><mn>0.0015</mn></mrow><annotation encoding="application/x-tex">0.0515 \mathbin{\pm} 0.0015</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Subgroup behavior error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0661</mn><mo>±</mo><mn>0.0014</mn></mrow><annotation encoding="application/x-tex">0.0661 \mathbin{\pm} 0.0014</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Subgroup-size error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0907</mn><mo>±</mo><mn>0.0060</mn></mrow><annotation encoding="application/x-tex">0.0907 \mathbin{\pm} 0.0060</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">High Hint Use</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0390</mn><mo>±</mo><mn>0.0027</mn></mrow><annotation encoding="application/x-tex">0.0390 \mathbin{\pm} 0.0027</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">High Hint Use</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>0.5031</mn><annotation encoding="application/x-tex">0.5031</annotation></semantics></math>
(one available run; 2 missing)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0085</mn><mo>±</mo><mn>0.0070</mn></mrow><annotation encoding="application/x-tex">0.0085 \mathbin{\pm} 0.0070</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0730</mn><mo>±</mo><mn>0.0163</mn></mrow><annotation encoding="application/x-tex">0.0730 \mathbin{\pm} 0.0163</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0237</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">0.0237 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0246</mn><mo>±</mo><mn>0.0006</mn></mrow><annotation encoding="application/x-tex">0.0246 \mathbin{\pm} 0.0006</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9365</mn><mo>±</mo><mn>0.0073</mn></mrow><annotation encoding="application/x-tex">0.9365 \mathbin{\pm} 0.0073</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0532</mn><mo>±</mo><mn>0.0042</mn></mrow><annotation encoding="application/x-tex">0.0532 \mathbin{\pm} 0.0042</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0103</mn><mo>±</mo><mn>0.0093</mn></mrow><annotation encoding="application/x-tex">0.0103 \mathbin{\pm} 0.0093</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0693</mn><mo>±</mo><mn>0.0160</mn></mrow><annotation encoding="application/x-tex">0.0693 \mathbin{\pm} 0.0160</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Subgroup hint-use error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1348</mn><mo>±</mo><mn>0.0357</mn></mrow><annotation encoding="application/x-tex">0.1348 \mathbin{\pm} 0.0357</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Subgroup behavior error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2930</mn><mo>±</mo><mn>0.0653</mn></mrow><annotation encoding="application/x-tex">0.2930 \mathbin{\pm} 0.0653</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Subgroup-size error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3320</mn><mo>±</mo><mn>0.0705</mn></mrow><annotation encoding="application/x-tex">0.3320 \mathbin{\pm} 0.0705</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">High Hint Use</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0418</mn><mo>±</mo><mn>0.0003</mn></mrow><annotation encoding="application/x-tex">0.0418 \mathbin{\pm} 0.0003</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0237</mn><mo>±</mo><mn>0.0040</mn></mrow><annotation encoding="application/x-tex">0.0237 \mathbin{\pm} 0.0040</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>0.0899</mn><annotation encoding="application/x-tex">0.0899</annotation></semantics></math>
(one available run; 2 missing)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0245</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0245 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0257</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0257 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1579</mn><mo>±</mo><mn>0.1623</mn></mrow><annotation encoding="application/x-tex">0.1579 \mathbin{\pm} 0.1623</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1743</mn><mo>±</mo><mn>0.1345</mn></mrow><annotation encoding="application/x-tex">0.1743 \mathbin{\pm} 0.1345</annotation></semantics></math>
(2; 1 missing)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0296</mn><mo>±</mo><mn>0.0051</mn></mrow><annotation encoding="application/x-tex">0.0296 \mathbin{\pm} 0.0051</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>0.0804</mn><annotation encoding="application/x-tex">0.0804</annotation></semantics></math>
(one available run; 2 missing)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Subgroup behavior error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0194</mn><mo>±</mo><mn>0.0015</mn></mrow><annotation encoding="application/x-tex">0.0194 \mathbin{\pm} 0.0015</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Subgroup-size error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0755</mn><mo>±</mo><mn>0.0092</mn></mrow><annotation encoding="application/x-tex">0.0755 \mathbin{\pm} 0.0092</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0554</mn><mo>±</mo><mn>0.0014</mn></mrow><annotation encoding="application/x-tex">0.0554 \mathbin{\pm} 0.0014</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0705</mn><mo>±</mo><mn>0.0113</mn></mrow><annotation encoding="application/x-tex">0.0705 \mathbin{\pm} 0.0113</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0441</mn><mo>±</mo><mn>0.0010</mn></mrow><annotation encoding="application/x-tex">0.0441 \mathbin{\pm} 0.0010</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0343</mn><mo>±</mo><mn>0.0127</mn></mrow><annotation encoding="application/x-tex">0.0343 \mathbin{\pm} 0.0127</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0353</mn><mo>±</mo><mn>0.0010</mn></mrow><annotation encoding="application/x-tex">0.0353 \mathbin{\pm} 0.0010</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0719</mn><mo>±</mo><mn>0.0159</mn></mrow><annotation encoding="application/x-tex">0.0719 \mathbin{\pm} 0.0159</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0325</mn><mo>±</mo><mn>0.0060</mn></mrow><annotation encoding="application/x-tex">0.0325 \mathbin{\pm} 0.0060</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2191</mn><mo>±</mo><mn>0.0133</mn></mrow><annotation encoding="application/x-tex">0.2191 \mathbin{\pm} 0.0133</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0222</mn><mo>±</mo><mn>0.0007</mn></mrow><annotation encoding="application/x-tex">0.0222 \mathbin{\pm} 0.0007</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0422</mn><mo>±</mo><mn>0.0026</mn></mrow><annotation encoding="application/x-tex">0.0422 \mathbin{\pm} 0.0026</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Subgroup behavior error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0234</mn><mo>±</mo><mn>0.0049</mn></mrow><annotation encoding="application/x-tex">0.0234 \mathbin{\pm} 0.0049</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Subgroup-size error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0461</mn><mo>±</mo><mn>0.0116</mn></mrow><annotation encoding="application/x-tex">0.0461 \mathbin{\pm} 0.0116</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0620</mn><mo>±</mo><mn>0.0035</mn></mrow><annotation encoding="application/x-tex">0.0620 \mathbin{\pm} 0.0035</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0675</mn><mo>±</mo><mn>0.0098</mn></mrow><annotation encoding="application/x-tex">0.0675 \mathbin{\pm} 0.0098</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0459</mn><mo>±</mo><mn>0.0008</mn></mrow><annotation encoding="application/x-tex">0.0459 \mathbin{\pm} 0.0008</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0469</mn><mo>±</mo><mn>0.0091</mn></mrow><annotation encoding="application/x-tex">0.0469 \mathbin{\pm} 0.0091</annotation></semantics></math>
(2; 1 missing)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0390</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">0.0390 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8528</mn><mo>±</mo><mn>0.0099</mn></mrow><annotation encoding="application/x-tex">0.8528 \mathbin{\pm} 0.0099</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2256</mn><mo>±</mo><mn>0.0140</mn></mrow><annotation encoding="application/x-tex">0.2256 \mathbin{\pm} 0.0140</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0238</mn><mo>±</mo><mn>0.0040</mn></mrow><annotation encoding="application/x-tex">0.0238 \mathbin{\pm} 0.0040</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0519</mn><mo>±</mo><mn>0.0054</mn></mrow><annotation encoding="application/x-tex">0.0519 \mathbin{\pm} 0.0054</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Subgroup behavior error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3258</mn><mo>±</mo><mn>0.0052</mn></mrow><annotation encoding="application/x-tex">0.3258 \mathbin{\pm} 0.0052</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Subgroup-size error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4389</mn><mo>±</mo><mn>0.0008</mn></mrow><annotation encoding="application/x-tex">0.4389 \mathbin{\pm} 0.0008</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0761</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0761 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0476</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0476 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0392</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0392 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8305</mn><mo>±</mo><mn>0.0119</mn></mrow><annotation encoding="application/x-tex">0.8305 \mathbin{\pm} 0.0119</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5816</mn><mo>±</mo><mn>0.0003</mn></mrow><annotation encoding="application/x-tex">0.5816 \mathbin{\pm} 0.0003</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0435</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0435 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Subgroup behavior error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0645</mn><mo>±</mo><mn>0.0054</mn></mrow><annotation encoding="application/x-tex">0.0645 \mathbin{\pm} 0.0054</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Subgroup-size error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0917</mn><mo>±</mo><mn>0.0017</mn></mrow><annotation encoding="application/x-tex">0.0917 \mathbin{\pm} 0.0017</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Disengagement</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0256</mn><mo>±</mo><mn>0.0014</mn></mrow><annotation encoding="application/x-tex">0.0256 \mathbin{\pm} 0.0014</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Disengagement</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0363</mn><mo>±</mo><mn>0.0044</mn></mrow><annotation encoding="application/x-tex">0.0363 \mathbin{\pm} 0.0044</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Activity Path</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0273</mn><mo>±</mo><mn>0.0007</mn></mrow><annotation encoding="application/x-tex">0.0273 \mathbin{\pm} 0.0007</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Activity Path</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0790</mn><mo>±</mo><mn>0.0057</mn></mrow><annotation encoding="application/x-tex">0.0790 \mathbin{\pm} 0.0057</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Reengagement</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0321</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">0.0321 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Reengagement</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0993</mn><mo>±</mo><mn>0.0053</mn></mrow><annotation encoding="application/x-tex">0.0993 \mathbin{\pm} 0.0053</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Subgroup behavior error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1187</mn><mo>±</mo><mn>0.0069</mn></mrow><annotation encoding="application/x-tex">0.1187 \mathbin{\pm} 0.0069</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Subgroup-size error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1072</mn><mo>±</mo><mn>0.0032</mn></mrow><annotation encoding="application/x-tex">0.1072 \mathbin{\pm} 0.0032</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Disengagement</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0188</mn><mo>±</mo><mn>0.0033</mn></mrow><annotation encoding="application/x-tex">0.0188 \mathbin{\pm} 0.0033</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Disengagement</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0768</mn><mo>±</mo><mn>0.0021</mn></mrow><annotation encoding="application/x-tex">0.0768 \mathbin{\pm} 0.0021</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Activity Path</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0209</mn><mo>±</mo><mn>0.0010</mn></mrow><annotation encoding="application/x-tex">0.0209 \mathbin{\pm} 0.0010</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Activity Path</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0652</mn><mo>±</mo><mn>0.0038</mn></mrow><annotation encoding="application/x-tex">0.0652 \mathbin{\pm} 0.0038</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Reengagement</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0219</mn><mo>±</mo><mn>0.0012</mn></mrow><annotation encoding="application/x-tex">0.0219 \mathbin{\pm} 0.0012</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Reengagement</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1045</mn><mo>±</mo><mn>0.0064</mn></mrow><annotation encoding="application/x-tex">0.1045 \mathbin{\pm} 0.0064</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Subgroup behavior error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2205</mn><mo>±</mo><mn>0.0882</mn></mrow><annotation encoding="application/x-tex">0.2205 \mathbin{\pm} 0.0882</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">All eligible learners/events</td>
<td style="text-align: left;">Subgroup-size error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2601</mn><mo>±</mo><mn>0.1935</mn></mrow><annotation encoding="application/x-tex">0.2601 \mathbin{\pm} 0.1935</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Disengagement</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0558</mn><mo>±</mo><mn>0.0100</mn></mrow><annotation encoding="application/x-tex">0.0558 \mathbin{\pm} 0.0100</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Disengagement</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0511</mn><mo>±</mo><mn>0.0030</mn></mrow><annotation encoding="application/x-tex">0.0511 \mathbin{\pm} 0.0030</annotation></semantics></math>
(2; 1 missing)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Activity Path</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0426</mn><mo>±</mo><mn>0.0015</mn></mrow><annotation encoding="application/x-tex">0.0426 \mathbin{\pm} 0.0015</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Activity Path</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1263</mn><mo>±</mo><mn>0.0021</mn></mrow><annotation encoding="application/x-tex">0.1263 \mathbin{\pm} 0.0021</annotation></semantics></math>
(2; 1 missing)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Reengagement</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0285</mn><mo>±</mo><mn>0.0186</mn></mrow><annotation encoding="application/x-tex">0.0285 \mathbin{\pm} 0.0186</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Reengagement</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1264</mn><mo>±</mo><mn>0.0013</mn></mrow><annotation encoding="application/x-tex">0.1264 \mathbin{\pm} 0.0013</annotation></semantics></math>
(2; 1 missing)</td>
<td style="text-align: left;">Lower</td>
</tr>
</tbody>
</table>

<table id="tab:supp-analytic-usefulness">
<caption>Complete publication results for analytic usefulness. Every
numeric metric-summary entry is shown; SD is the population SD across
the fixed seeds and
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>n</mi><annotation encoding="application/x-tex">n</annotation></semantics></math>
is the number of estimable seeds.</caption>
<thead>
<tr>
<th style="text-align: left;">Dataset</th>
<th style="text-align: left;">Generator</th>
<th style="text-align: left;">Scope / comparison</th>
<th style="text-align: left;">Metric</th>
<th style="text-align: left;">Mean
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>±</mi><annotation encoding="application/x-tex">\pm</annotation></semantics></math>
SD
(<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>n</mi><annotation encoding="application/x-tex">n</annotation></semantics></math>)</th>
<th style="text-align: left;">Direction</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="6" style="text-align: left;"><em>Continued from previous
page</em></td>
</tr>
<tr>
<td style="text-align: left;">Dataset</td>
<td style="text-align: left;">Generator</td>
<td style="text-align: left;">Scope / comparison</td>
<td style="text-align: left;">Metric</td>
<td style="text-align: left;">Mean
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>±</mi><annotation encoding="application/x-tex">\pm</annotation></semantics></math>
SD
(<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>n</mi><annotation encoding="application/x-tex">n</annotation></semantics></math>)</td>
<td style="text-align: left;">Direction</td>
</tr>
<tr>
<td colspan="6" style="text-align: right;"><em>Continued on next
page</em></td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Real + standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8198</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">0.8198 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Real + standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7340</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.7340 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Real + targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8189</mn><mo>±</mo><mn>0.0004</mn></mrow><annotation encoding="application/x-tex">0.8189 \mathbin{\pm} 0.0004</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Real + targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7322</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7322 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8232</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.8232 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7384</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7384 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7956</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">0.7956 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7053</mn><mo>±</mo><mn>0.0007</mn></mrow><annotation encoding="application/x-tex">0.7053 \mathbin{\pm} 0.0007</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7954</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">0.7954 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7048</mn><mo>±</mo><mn>0.0010</mn></mrow><annotation encoding="application/x-tex">0.7048 \mathbin{\pm} 0.0010</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0034</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">-0.0034 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0043</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">-0.0043 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0276</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">0.0276 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0331</mn><mo>±</mo><mn>0.0007</mn></mrow><annotation encoding="application/x-tex">0.0331 \mathbin{\pm} 0.0007</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0002</mn><mo>±</mo><mn>0.0004</mn></mrow><annotation encoding="application/x-tex">-0.0002 \mathbin{\pm} 0.0004</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0005</mn><mo>±</mo><mn>0.0004</mn></mrow><annotation encoding="application/x-tex">-0.0005 \mathbin{\pm} 0.0004</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0043</mn><mo>±</mo><mn>0.0004</mn></mrow><annotation encoding="application/x-tex">-0.0043 \mathbin{\pm} 0.0004</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0062</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">-0.0062 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Real + standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1593</mn><mo>±</mo><mn>0.0135</mn></mrow><annotation encoding="application/x-tex">0.1593 \mathbin{\pm} 0.0135</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Real + standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7486</mn><mo>±</mo><mn>0.0176</mn></mrow><annotation encoding="application/x-tex">0.7486 \mathbin{\pm} 0.0176</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1345</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.1345 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7011</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7011 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0991</mn><mo>±</mo><mn>0.0287</mn></mrow><annotation encoding="application/x-tex">0.0991 \mathbin{\pm} 0.0287</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5516</mn><mo>±</mo><mn>0.0840</mn></mrow><annotation encoding="application/x-tex">0.5516 \mathbin{\pm} 0.0840</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Targeted synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1192</mn><mo>±</mo><mn>0.0164</mn></mrow><annotation encoding="application/x-tex">0.1192 \mathbin{\pm} 0.0164</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Targeted synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5745</mn><mo>±</mo><mn>0.0103</mn></mrow><annotation encoding="application/x-tex">0.5745 \mathbin{\pm} 0.0103</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0248</mn><mo>±</mo><mn>0.0135</mn></mrow><annotation encoding="application/x-tex">0.0248 \mathbin{\pm} 0.0135</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0475</mn><mo>±</mo><mn>0.0176</mn></mrow><annotation encoding="application/x-tex">0.0475 \mathbin{\pm} 0.0176</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0354</mn><mo>±</mo><mn>0.0287</mn></mrow><annotation encoding="application/x-tex">0.0354 \mathbin{\pm} 0.0287</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1495</mn><mo>±</mo><mn>0.0840</mn></mrow><annotation encoding="application/x-tex">0.1495 \mathbin{\pm} 0.0840</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery Prediction; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4226</mn><mo>±</mo><mn>0.0427</mn></mrow><annotation encoding="application/x-tex">0.4226 \mathbin{\pm} 0.0427</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery Prediction; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6441</mn><mo>±</mo><mn>0.0229</mn></mrow><annotation encoding="application/x-tex">0.6441 \mathbin{\pm} 0.0229</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery Prediction; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4387</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.4387 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery Prediction; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6897</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.6897 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery Prediction; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3185</mn><mo>±</mo><mn>0.0339</mn></mrow><annotation encoding="application/x-tex">0.3185 \mathbin{\pm} 0.0339</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery Prediction; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5396</mn><mo>±</mo><mn>0.0316</mn></mrow><annotation encoding="application/x-tex">0.5396 \mathbin{\pm} 0.0316</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery Prediction; Targeted synthetic
only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3254</mn><mo>±</mo><mn>0.0197</mn></mrow><annotation encoding="application/x-tex">0.3254 \mathbin{\pm} 0.0197</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery Prediction; Targeted synthetic
only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5303</mn><mo>±</mo><mn>0.0383</mn></mrow><annotation encoding="application/x-tex">0.5303 \mathbin{\pm} 0.0383</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery Prediction; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0160</mn><mo>±</mo><mn>0.0427</mn></mrow><annotation encoding="application/x-tex">-0.0160 \mathbin{\pm} 0.0427</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery Prediction; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0456</mn><mo>±</mo><mn>0.0229</mn></mrow><annotation encoding="application/x-tex">-0.0456 \mathbin{\pm} 0.0229</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery Prediction; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1201</mn><mo>±</mo><mn>0.0339</mn></mrow><annotation encoding="application/x-tex">0.1201 \mathbin{\pm} 0.0339</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery Prediction; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1502</mn><mo>±</mo><mn>0.0316</mn></mrow><annotation encoding="application/x-tex">0.1502 \mathbin{\pm} 0.0316</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">High Hint Use; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4179</mn><mo>±</mo><mn>0.0047</mn></mrow><annotation encoding="application/x-tex">0.4179 \mathbin{\pm} 0.0047</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">High Hint Use; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8513</mn><mo>±</mo><mn>0.0075</mn></mrow><annotation encoding="application/x-tex">0.8513 \mathbin{\pm} 0.0075</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">High Hint Use; Real + targeted
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4100</mn><mo>±</mo><mn>0.0058</mn></mrow><annotation encoding="application/x-tex">0.4100 \mathbin{\pm} 0.0058</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">High Hint Use; Real + targeted
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8518</mn><mo>±</mo><mn>0.0063</mn></mrow><annotation encoding="application/x-tex">0.8518 \mathbin{\pm} 0.0063</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">High Hint Use; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4399</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.4399 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">High Hint Use; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8697</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.8697 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">High Hint Use; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3552</mn><mo>±</mo><mn>0.0169</mn></mrow><annotation encoding="application/x-tex">0.3552 \mathbin{\pm} 0.0169</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">High Hint Use; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8225</mn><mo>±</mo><mn>0.0073</mn></mrow><annotation encoding="application/x-tex">0.8225 \mathbin{\pm} 0.0073</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">High Hint Use; Targeted synthetic
only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3774</mn><mo>±</mo><mn>0.0077</mn></mrow><annotation encoding="application/x-tex">0.3774 \mathbin{\pm} 0.0077</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">High Hint Use; Targeted synthetic
only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8254</mn><mo>±</mo><mn>0.0011</mn></mrow><annotation encoding="application/x-tex">0.8254 \mathbin{\pm} 0.0011</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">High Hint Use; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0221</mn><mo>±</mo><mn>0.0047</mn></mrow><annotation encoding="application/x-tex">-0.0221 \mathbin{\pm} 0.0047</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">High Hint Use; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0184</mn><mo>±</mo><mn>0.0075</mn></mrow><annotation encoding="application/x-tex">-0.0184 \mathbin{\pm} 0.0075</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">High Hint Use; Standard synthetic utility
loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0847</mn><mo>±</mo><mn>0.0169</mn></mrow><annotation encoding="application/x-tex">0.0847 \mathbin{\pm} 0.0169</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">High Hint Use; Standard synthetic utility
loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0472</mn><mo>±</mo><mn>0.0073</mn></mrow><annotation encoding="application/x-tex">0.0472 \mathbin{\pm} 0.0073</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">High Hint Use; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0222</mn><mo>±</mo><mn>0.0130</mn></mrow><annotation encoding="application/x-tex">0.0222 \mathbin{\pm} 0.0130</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">High Hint Use; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0030</mn><mo>±</mo><mn>0.0074</mn></mrow><annotation encoding="application/x-tex">0.0030 \mathbin{\pm} 0.0074</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">High Hint Use; Targeted augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0299</mn><mo>±</mo><mn>0.0058</mn></mrow><annotation encoding="application/x-tex">-0.0299 \mathbin{\pm} 0.0058</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">High Hint Use; Targeted augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0179</mn><mo>±</mo><mn>0.0063</mn></mrow><annotation encoding="application/x-tex">-0.0179 \mathbin{\pm} 0.0063</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7505</mn><mo>±</mo><mn>0.0006</mn></mrow><annotation encoding="application/x-tex">0.7505 \mathbin{\pm} 0.0006</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7374</mn><mo>±</mo><mn>0.0013</mn></mrow><annotation encoding="application/x-tex">0.7374 \mathbin{\pm} 0.0013</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Real + targeted
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7568</mn><mo>±</mo><mn>0.0041</mn></mrow><annotation encoding="application/x-tex">0.7568 \mathbin{\pm} 0.0041</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Real + targeted
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7389</mn><mo>±</mo><mn>0.0028</mn></mrow><annotation encoding="application/x-tex">0.7389 \mathbin{\pm} 0.0028</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Real only
(TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7467</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7467 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Real only
(TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7451</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7451 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7475</mn><mo>±</mo><mn>0.0032</mn></mrow><annotation encoding="application/x-tex">0.7475 \mathbin{\pm} 0.0032</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7227</mn><mo>±</mo><mn>0.0008</mn></mrow><annotation encoding="application/x-tex">0.7227 \mathbin{\pm} 0.0008</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Targeted
synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7465</mn><mo>±</mo><mn>0.0056</mn></mrow><annotation encoding="application/x-tex">0.7465 \mathbin{\pm} 0.0056</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Targeted
synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7200</mn><mo>±</mo><mn>0.0086</mn></mrow><annotation encoding="application/x-tex">0.7200 \mathbin{\pm} 0.0086</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Standard
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0037</mn><mo>±</mo><mn>0.0006</mn></mrow><annotation encoding="application/x-tex">0.0037 \mathbin{\pm} 0.0006</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Standard
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0077</mn><mo>±</mo><mn>0.0013</mn></mrow><annotation encoding="application/x-tex">-0.0077 \mathbin{\pm} 0.0013</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0008</mn><mo>±</mo><mn>0.0032</mn></mrow><annotation encoding="application/x-tex">-0.0008 \mathbin{\pm} 0.0032</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0224</mn><mo>±</mo><mn>0.0008</mn></mrow><annotation encoding="application/x-tex">0.0224 \mathbin{\pm} 0.0008</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0010</mn><mo>±</mo><mn>0.0023</mn></mrow><annotation encoding="application/x-tex">-0.0010 \mathbin{\pm} 0.0023</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0026</mn><mo>±</mo><mn>0.0079</mn></mrow><annotation encoding="application/x-tex">-0.0026 \mathbin{\pm} 0.0079</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Targeted
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0100</mn><mo>±</mo><mn>0.0041</mn></mrow><annotation encoding="application/x-tex">0.0100 \mathbin{\pm} 0.0041</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Targeted
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0062</mn><mo>±</mo><mn>0.0028</mn></mrow><annotation encoding="application/x-tex">-0.0062 \mathbin{\pm} 0.0028</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Real +
standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1252</mn><mo>±</mo><mn>0.0036</mn></mrow><annotation encoding="application/x-tex">0.1252 \mathbin{\pm} 0.0036</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Real +
standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7428</mn><mo>±</mo><mn>0.0216</mn></mrow><annotation encoding="application/x-tex">0.7428 \mathbin{\pm} 0.0216</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Real +
targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1164</mn><mo>±</mo><mn>0.0062</mn></mrow><annotation encoding="application/x-tex">0.1164 \mathbin{\pm} 0.0062</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Real +
targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7761</mn><mo>±</mo><mn>0.0253</mn></mrow><annotation encoding="application/x-tex">0.7761 \mathbin{\pm} 0.0253</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Real only
(TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1436</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.1436 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Real only
(TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8222</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.8222 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0815</mn><mo>±</mo><mn>0.0070</mn></mrow><annotation encoding="application/x-tex">0.0815 \mathbin{\pm} 0.0070</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7503</mn><mo>±</mo><mn>0.0138</mn></mrow><annotation encoding="application/x-tex">0.7503 \mathbin{\pm} 0.0138</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Targeted
synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0782</mn><mo>±</mo><mn>0.0054</mn></mrow><annotation encoding="application/x-tex">0.0782 \mathbin{\pm} 0.0054</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Targeted
synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7584</mn><mo>±</mo><mn>0.0164</mn></mrow><annotation encoding="application/x-tex">0.7584 \mathbin{\pm} 0.0164</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0184</mn><mo>±</mo><mn>0.0036</mn></mrow><annotation encoding="application/x-tex">-0.0184 \mathbin{\pm} 0.0036</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0794</mn><mo>±</mo><mn>0.0216</mn></mrow><annotation encoding="application/x-tex">-0.0794 \mathbin{\pm} 0.0216</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0621</mn><mo>±</mo><mn>0.0070</mn></mrow><annotation encoding="application/x-tex">0.0621 \mathbin{\pm} 0.0070</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0719</mn><mo>±</mo><mn>0.0138</mn></mrow><annotation encoding="application/x-tex">0.0719 \mathbin{\pm} 0.0138</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0033</mn><mo>±</mo><mn>0.0058</mn></mrow><annotation encoding="application/x-tex">-0.0033 \mathbin{\pm} 0.0058</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0081</mn><mo>±</mo><mn>0.0302</mn></mrow><annotation encoding="application/x-tex">0.0081 \mathbin{\pm} 0.0302</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Targeted
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0273</mn><mo>±</mo><mn>0.0062</mn></mrow><annotation encoding="application/x-tex">-0.0273 \mathbin{\pm} 0.0062</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Targeted
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0461</mn><mo>±</mo><mn>0.0253</mn></mrow><annotation encoding="application/x-tex">-0.0461 \mathbin{\pm} 0.0253</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real +
standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2812</mn><mo>±</mo><mn>0.0013</mn></mrow><annotation encoding="application/x-tex">0.2812 \mathbin{\pm} 0.0013</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real +
standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8515</mn><mo>±</mo><mn>0.0086</mn></mrow><annotation encoding="application/x-tex">0.8515 \mathbin{\pm} 0.0086</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real +
targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2591</mn><mo>±</mo><mn>0.0101</mn></mrow><annotation encoding="application/x-tex">0.2591 \mathbin{\pm} 0.0101</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real +
targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8549</mn><mo>±</mo><mn>0.0111</mn></mrow><annotation encoding="application/x-tex">0.8549 \mathbin{\pm} 0.0111</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real only
(TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3409</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.3409 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real only
(TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8810</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.8810 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2165</mn><mo>±</mo><mn>0.0181</mn></mrow><annotation encoding="application/x-tex">0.2165 \mathbin{\pm} 0.0181</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8139</mn><mo>±</mo><mn>0.0179</mn></mrow><annotation encoding="application/x-tex">0.8139 \mathbin{\pm} 0.0179</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Targeted
synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2197</mn><mo>±</mo><mn>0.0137</mn></mrow><annotation encoding="application/x-tex">0.2197 \mathbin{\pm} 0.0137</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Targeted
synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8159</mn><mo>±</mo><mn>0.0062</mn></mrow><annotation encoding="application/x-tex">0.8159 \mathbin{\pm} 0.0062</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0596</mn><mo>±</mo><mn>0.0013</mn></mrow><annotation encoding="application/x-tex">-0.0596 \mathbin{\pm} 0.0013</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0294</mn><mo>±</mo><mn>0.0086</mn></mrow><annotation encoding="application/x-tex">-0.0294 \mathbin{\pm} 0.0086</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1244</mn><mo>±</mo><mn>0.0181</mn></mrow><annotation encoding="application/x-tex">0.1244 \mathbin{\pm} 0.0181</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0671</mn><mo>±</mo><mn>0.0179</mn></mrow><annotation encoding="application/x-tex">0.0671 \mathbin{\pm} 0.0179</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0033</mn><mo>±</mo><mn>0.0177</mn></mrow><annotation encoding="application/x-tex">0.0033 \mathbin{\pm} 0.0177</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0020</mn><mo>±</mo><mn>0.0205</mn></mrow><annotation encoding="application/x-tex">0.0020 \mathbin{\pm} 0.0205</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Targeted
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0818</mn><mo>±</mo><mn>0.0101</mn></mrow><annotation encoding="application/x-tex">-0.0818 \mathbin{\pm} 0.0101</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Targeted
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0261</mn><mo>±</mo><mn>0.0111</mn></mrow><annotation encoding="application/x-tex">-0.0261 \mathbin{\pm} 0.0111</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8973</mn><mo>±</mo><mn>0.0072</mn></mrow><annotation encoding="application/x-tex">0.8973 \mathbin{\pm} 0.0072</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7142</mn><mo>±</mo><mn>0.0149</mn></mrow><annotation encoding="application/x-tex">0.7142 \mathbin{\pm} 0.0149</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Real + targeted
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8881</mn><mo>±</mo><mn>0.0036</mn></mrow><annotation encoding="application/x-tex">0.8881 \mathbin{\pm} 0.0036</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Real + targeted
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7093</mn><mo>±</mo><mn>0.0045</mn></mrow><annotation encoding="application/x-tex">0.7093 \mathbin{\pm} 0.0045</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9009</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.9009 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7239</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7239 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8786</mn><mo>±</mo><mn>0.0078</mn></mrow><annotation encoding="application/x-tex">0.8786 \mathbin{\pm} 0.0078</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6843</mn><mo>±</mo><mn>0.0163</mn></mrow><annotation encoding="application/x-tex">0.6843 \mathbin{\pm} 0.0163</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Targeted synthetic
only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8784</mn><mo>±</mo><mn>0.0063</mn></mrow><annotation encoding="application/x-tex">0.8784 \mathbin{\pm} 0.0063</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Targeted synthetic
only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7069</mn><mo>±</mo><mn>0.0091</mn></mrow><annotation encoding="application/x-tex">0.7069 \mathbin{\pm} 0.0091</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0036</mn><mo>±</mo><mn>0.0072</mn></mrow><annotation encoding="application/x-tex">-0.0036 \mathbin{\pm} 0.0072</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0097</mn><mo>±</mo><mn>0.0149</mn></mrow><annotation encoding="application/x-tex">-0.0097 \mathbin{\pm} 0.0149</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0224</mn><mo>±</mo><mn>0.0078</mn></mrow><annotation encoding="application/x-tex">0.0224 \mathbin{\pm} 0.0078</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0396</mn><mo>±</mo><mn>0.0163</mn></mrow><annotation encoding="application/x-tex">0.0396 \mathbin{\pm} 0.0163</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0002</mn><mo>±</mo><mn>0.0125</mn></mrow><annotation encoding="application/x-tex">-0.0002 \mathbin{\pm} 0.0125</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0226</mn><mo>±</mo><mn>0.0193</mn></mrow><annotation encoding="application/x-tex">0.0226 \mathbin{\pm} 0.0193</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Targeted augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0129</mn><mo>±</mo><mn>0.0036</mn></mrow><annotation encoding="application/x-tex">-0.0129 \mathbin{\pm} 0.0036</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Targeted augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0146</mn><mo>±</mo><mn>0.0045</mn></mrow><annotation encoding="application/x-tex">-0.0146 \mathbin{\pm} 0.0045</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Real + standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7534</mn><mo>±</mo><mn>0.0032</mn></mrow><annotation encoding="application/x-tex">0.7534 \mathbin{\pm} 0.0032</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Real + standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7322</mn><mo>±</mo><mn>0.0044</mn></mrow><annotation encoding="application/x-tex">0.7322 \mathbin{\pm} 0.0044</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Real + targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7437</mn><mo>±</mo><mn>0.0044</mn></mrow><annotation encoding="application/x-tex">0.7437 \mathbin{\pm} 0.0044</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Real + targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7229</mn><mo>±</mo><mn>0.0065</mn></mrow><annotation encoding="application/x-tex">0.7229 \mathbin{\pm} 0.0065</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7445</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7445 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7329</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7329 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7534</mn><mo>±</mo><mn>0.0096</mn></mrow><annotation encoding="application/x-tex">0.7534 \mathbin{\pm} 0.0096</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7269</mn><mo>±</mo><mn>0.0086</mn></mrow><annotation encoding="application/x-tex">0.7269 \mathbin{\pm} 0.0086</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Targeted synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7310</mn><mo>±</mo><mn>0.0072</mn></mrow><annotation encoding="application/x-tex">0.7310 \mathbin{\pm} 0.0072</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Targeted synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7060</mn><mo>±</mo><mn>0.0091</mn></mrow><annotation encoding="application/x-tex">0.7060 \mathbin{\pm} 0.0091</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Standard augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0089</mn><mo>±</mo><mn>0.0032</mn></mrow><annotation encoding="application/x-tex">0.0089 \mathbin{\pm} 0.0032</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Standard augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0006</mn><mo>±</mo><mn>0.0044</mn></mrow><annotation encoding="application/x-tex">-0.0006 \mathbin{\pm} 0.0044</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Standard synthetic utility
loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0089</mn><mo>±</mo><mn>0.0096</mn></mrow><annotation encoding="application/x-tex">-0.0089 \mathbin{\pm} 0.0096</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Standard synthetic utility
loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0060</mn><mo>±</mo><mn>0.0086</mn></mrow><annotation encoding="application/x-tex">0.0060 \mathbin{\pm} 0.0086</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0224</mn><mo>±</mo><mn>0.0025</mn></mrow><annotation encoding="application/x-tex">-0.0224 \mathbin{\pm} 0.0025</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0209</mn><mo>±</mo><mn>0.0047</mn></mrow><annotation encoding="application/x-tex">-0.0209 \mathbin{\pm} 0.0047</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Targeted augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0008</mn><mo>±</mo><mn>0.0044</mn></mrow><annotation encoding="application/x-tex">-0.0008 \mathbin{\pm} 0.0044</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Targeted augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0099</mn><mo>±</mo><mn>0.0065</mn></mrow><annotation encoding="application/x-tex">-0.0099 \mathbin{\pm} 0.0065</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">High Hint Use</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0133</mn><mo>±</mo><mn>0.0043</mn></mrow><annotation encoding="application/x-tex">0.0133 \mathbin{\pm} 0.0043</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">High Hint Use</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0055</mn><mo>±</mo><mn>0.0034</mn></mrow><annotation encoding="application/x-tex">-0.0055 \mathbin{\pm} 0.0034</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0151</mn><mo>±</mo><mn>0.0037</mn></mrow><annotation encoding="application/x-tex">-0.0151 \mathbin{\pm} 0.0037</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0048</mn><mo>±</mo><mn>0.0103</mn></mrow><annotation encoding="application/x-tex">-0.0048 \mathbin{\pm} 0.0103</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0146</mn><mo>±</mo><mn>0.0013</mn></mrow><annotation encoding="application/x-tex">0.0146 \mathbin{\pm} 0.0013</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0044</mn><mo>±</mo><mn>0.0056</mn></mrow><annotation encoding="application/x-tex">-0.0044 \mathbin{\pm} 0.0056</annotation></semantics></math>
(2; 1 missing)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0016</mn><mo>±</mo><mn>0.0048</mn></mrow><annotation encoding="application/x-tex">-0.0016 \mathbin{\pm} 0.0048</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0430</mn><mo>±</mo><mn>0.0110</mn></mrow><annotation encoding="application/x-tex">0.0430 \mathbin{\pm} 0.0110</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0066</mn><mo>±</mo><mn>0.0023</mn></mrow><annotation encoding="application/x-tex">0.0066 \mathbin{\pm} 0.0023</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0080</mn><mo>±</mo><mn>0.0282</mn></mrow><annotation encoding="application/x-tex">-0.0080 \mathbin{\pm} 0.0282</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0067</mn><mo>±</mo><mn>0.0057</mn></mrow><annotation encoding="application/x-tex">-0.0067 \mathbin{\pm} 0.0057</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0079</mn><mo>±</mo><mn>0.0136</mn></mrow><annotation encoding="application/x-tex">-0.0079 \mathbin{\pm} 0.0136</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Real + standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8187</mn><mo>±</mo><mn>0.0003</mn></mrow><annotation encoding="application/x-tex">0.8187 \mathbin{\pm} 0.0003</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Real + standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7327</mn><mo>±</mo><mn>0.0006</mn></mrow><annotation encoding="application/x-tex">0.7327 \mathbin{\pm} 0.0006</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Real + targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8186</mn><mo>±</mo><mn>0.0004</mn></mrow><annotation encoding="application/x-tex">0.8186 \mathbin{\pm} 0.0004</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Real + targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7326</mn><mo>±</mo><mn>0.0005</mn></mrow><annotation encoding="application/x-tex">0.7326 \mathbin{\pm} 0.0005</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8232</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.8232 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7384</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7384 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6453</mn><mo>±</mo><mn>0.0056</mn></mrow><annotation encoding="application/x-tex">0.6453 \mathbin{\pm} 0.0056</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5056</mn><mo>±</mo><mn>0.0148</mn></mrow><annotation encoding="application/x-tex">0.5056 \mathbin{\pm} 0.0148</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6594</mn><mo>±</mo><mn>0.0137</mn></mrow><annotation encoding="application/x-tex">0.6594 \mathbin{\pm} 0.0137</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5176</mn><mo>±</mo><mn>0.0220</mn></mrow><annotation encoding="application/x-tex">0.5176 \mathbin{\pm} 0.0220</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0045</mn><mo>±</mo><mn>0.0003</mn></mrow><annotation encoding="application/x-tex">-0.0045 \mathbin{\pm} 0.0003</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0056</mn><mo>±</mo><mn>0.0006</mn></mrow><annotation encoding="application/x-tex">-0.0056 \mathbin{\pm} 0.0006</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1779</mn><mo>±</mo><mn>0.0056</mn></mrow><annotation encoding="application/x-tex">0.1779 \mathbin{\pm} 0.0056</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2328</mn><mo>±</mo><mn>0.0148</mn></mrow><annotation encoding="application/x-tex">0.2328 \mathbin{\pm} 0.0148</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0141</mn><mo>±</mo><mn>0.0081</mn></mrow><annotation encoding="application/x-tex">0.0141 \mathbin{\pm} 0.0081</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0120</mn><mo>±</mo><mn>0.0088</mn></mrow><annotation encoding="application/x-tex">0.0120 \mathbin{\pm} 0.0088</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0046</mn><mo>±</mo><mn>0.0004</mn></mrow><annotation encoding="application/x-tex">-0.0046 \mathbin{\pm} 0.0004</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0058</mn><mo>±</mo><mn>0.0005</mn></mrow><annotation encoding="application/x-tex">-0.0058 \mathbin{\pm} 0.0005</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Real + standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1964</mn><mo>±</mo><mn>0.0091</mn></mrow><annotation encoding="application/x-tex">0.1964 \mathbin{\pm} 0.0091</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Real + standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7731</mn><mo>±</mo><mn>0.0142</mn></mrow><annotation encoding="application/x-tex">0.7731 \mathbin{\pm} 0.0142</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1345</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.1345 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7011</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7011 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0681</mn><mo>±</mo><mn>0.0135</mn></mrow><annotation encoding="application/x-tex">0.0681 \mathbin{\pm} 0.0135</annotation></semantics></math>
(2; 1 missing)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5538</mn><mo>±</mo><mn>0.1434</mn></mrow><annotation encoding="application/x-tex">0.5538 \mathbin{\pm} 0.1434</annotation></semantics></math>
(2; 1 missing)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Targeted synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0630</mn><mo>±</mo><mn>0.0225</mn></mrow><annotation encoding="application/x-tex">0.0630 \mathbin{\pm} 0.0225</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Targeted synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4811</mn><mo>±</mo><mn>0.0780</mn></mrow><annotation encoding="application/x-tex">0.4811 \mathbin{\pm} 0.0780</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0618</mn><mo>±</mo><mn>0.0091</mn></mrow><annotation encoding="application/x-tex">0.0618 \mathbin{\pm} 0.0091</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0720</mn><mo>±</mo><mn>0.0142</mn></mrow><annotation encoding="application/x-tex">0.0720 \mathbin{\pm} 0.0142</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0664</mn><mo>±</mo><mn>0.0135</mn></mrow><annotation encoding="application/x-tex">0.0664 \mathbin{\pm} 0.0135</annotation></semantics></math>
(2; 1 missing)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1473</mn><mo>±</mo><mn>0.1434</mn></mrow><annotation encoding="application/x-tex">0.1473 \mathbin{\pm} 0.1434</annotation></semantics></math>
(2; 1 missing)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery Prediction; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3816</mn><mo>±</mo><mn>0.0306</mn></mrow><annotation encoding="application/x-tex">0.3816 \mathbin{\pm} 0.0306</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery Prediction; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6364</mn><mo>±</mo><mn>0.0184</mn></mrow><annotation encoding="application/x-tex">0.6364 \mathbin{\pm} 0.0184</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery Prediction; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4387</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.4387 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery Prediction; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6897</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.6897 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery Prediction; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3080</mn><mo>±</mo><mn>0.0356</mn></mrow><annotation encoding="application/x-tex">0.3080 \mathbin{\pm} 0.0356</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery Prediction; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5099</mn><mo>±</mo><mn>0.0476</mn></mrow><annotation encoding="application/x-tex">0.5099 \mathbin{\pm} 0.0476</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery Prediction; Targeted synthetic
only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2984</mn><mo>±</mo><mn>0.0227</mn></mrow><annotation encoding="application/x-tex">0.2984 \mathbin{\pm} 0.0227</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery Prediction; Targeted synthetic
only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5012</mn><mo>±</mo><mn>0.0487</mn></mrow><annotation encoding="application/x-tex">0.5012 \mathbin{\pm} 0.0487</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery Prediction; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0570</mn><mo>±</mo><mn>0.0306</mn></mrow><annotation encoding="application/x-tex">-0.0570 \mathbin{\pm} 0.0306</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery Prediction; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0534</mn><mo>±</mo><mn>0.0184</mn></mrow><annotation encoding="application/x-tex">-0.0534 \mathbin{\pm} 0.0184</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery Prediction; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1307</mn><mo>±</mo><mn>0.0356</mn></mrow><annotation encoding="application/x-tex">0.1307 \mathbin{\pm} 0.0356</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery Prediction; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1798</mn><mo>±</mo><mn>0.0476</mn></mrow><annotation encoding="application/x-tex">0.1798 \mathbin{\pm} 0.0476</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">High Hint Use; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4170</mn><mo>±</mo><mn>0.0083</mn></mrow><annotation encoding="application/x-tex">0.4170 \mathbin{\pm} 0.0083</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">High Hint Use; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8591</mn><mo>±</mo><mn>0.0010</mn></mrow><annotation encoding="application/x-tex">0.8591 \mathbin{\pm} 0.0010</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">High Hint Use; Real + targeted
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3914</mn><mo>±</mo><mn>0.0054</mn></mrow><annotation encoding="application/x-tex">0.3914 \mathbin{\pm} 0.0054</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">High Hint Use; Real + targeted
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8494</mn><mo>±</mo><mn>0.0052</mn></mrow><annotation encoding="application/x-tex">0.8494 \mathbin{\pm} 0.0052</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">High Hint Use; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4399</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.4399 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">High Hint Use; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8697</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.8697 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">High Hint Use; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1086</mn><mo>±</mo><mn>0.0087</mn></mrow><annotation encoding="application/x-tex">0.1086 \mathbin{\pm} 0.0087</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">High Hint Use; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5344</mn><mo>±</mo><mn>0.0237</mn></mrow><annotation encoding="application/x-tex">0.5344 \mathbin{\pm} 0.0237</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">High Hint Use; Targeted synthetic
only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1042</mn><mo>±</mo><mn>0.0184</mn></mrow><annotation encoding="application/x-tex">0.1042 \mathbin{\pm} 0.0184</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">High Hint Use; Targeted synthetic
only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4872</mn><mo>±</mo><mn>0.0601</mn></mrow><annotation encoding="application/x-tex">0.4872 \mathbin{\pm} 0.0601</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">High Hint Use; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0229</mn><mo>±</mo><mn>0.0083</mn></mrow><annotation encoding="application/x-tex">-0.0229 \mathbin{\pm} 0.0083</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">High Hint Use; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0106</mn><mo>±</mo><mn>0.0010</mn></mrow><annotation encoding="application/x-tex">-0.0106 \mathbin{\pm} 0.0010</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">High Hint Use; Standard synthetic utility
loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3313</mn><mo>±</mo><mn>0.0087</mn></mrow><annotation encoding="application/x-tex">0.3313 \mathbin{\pm} 0.0087</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">High Hint Use; Standard synthetic utility
loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3353</mn><mo>±</mo><mn>0.0237</mn></mrow><annotation encoding="application/x-tex">0.3353 \mathbin{\pm} 0.0237</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">High Hint Use; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0044</mn><mo>±</mo><mn>0.0255</mn></mrow><annotation encoding="application/x-tex">-0.0044 \mathbin{\pm} 0.0255</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">High Hint Use; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0472</mn><mo>±</mo><mn>0.0765</mn></mrow><annotation encoding="application/x-tex">-0.0472 \mathbin{\pm} 0.0765</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">High Hint Use; Targeted augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0485</mn><mo>±</mo><mn>0.0054</mn></mrow><annotation encoding="application/x-tex">-0.0485 \mathbin{\pm} 0.0054</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">High Hint Use; Targeted augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0203</mn><mo>±</mo><mn>0.0052</mn></mrow><annotation encoding="application/x-tex">-0.0203 \mathbin{\pm} 0.0052</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7488</mn><mo>±</mo><mn>0.0027</mn></mrow><annotation encoding="application/x-tex">0.7488 \mathbin{\pm} 0.0027</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7420</mn><mo>±</mo><mn>0.0024</mn></mrow><annotation encoding="application/x-tex">0.7420 \mathbin{\pm} 0.0024</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Real + targeted
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7548</mn><mo>±</mo><mn>0.0012</mn></mrow><annotation encoding="application/x-tex">0.7548 \mathbin{\pm} 0.0012</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Real + targeted
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7449</mn><mo>±</mo><mn>0.0004</mn></mrow><annotation encoding="application/x-tex">0.7449 \mathbin{\pm} 0.0004</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Real only
(TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7467</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7467 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Real only
(TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7451</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7451 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5608</mn><mo>±</mo><mn>0.0143</mn></mrow><annotation encoding="application/x-tex">0.5608 \mathbin{\pm} 0.0143</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5327</mn><mo>±</mo><mn>0.0011</mn></mrow><annotation encoding="application/x-tex">0.5327 \mathbin{\pm} 0.0011</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Targeted
synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5402</mn><mo>±</mo><mn>0.0446</mn></mrow><annotation encoding="application/x-tex">0.5402 \mathbin{\pm} 0.0446</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Targeted
synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5150</mn><mo>±</mo><mn>0.0468</mn></mrow><annotation encoding="application/x-tex">0.5150 \mathbin{\pm} 0.0468</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Standard
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0021</mn><mo>±</mo><mn>0.0027</mn></mrow><annotation encoding="application/x-tex">0.0021 \mathbin{\pm} 0.0027</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Standard
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0031</mn><mo>±</mo><mn>0.0024</mn></mrow><annotation encoding="application/x-tex">-0.0031 \mathbin{\pm} 0.0024</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1860</mn><mo>±</mo><mn>0.0143</mn></mrow><annotation encoding="application/x-tex">0.1860 \mathbin{\pm} 0.0143</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2124</mn><mo>±</mo><mn>0.0011</mn></mrow><annotation encoding="application/x-tex">0.2124 \mathbin{\pm} 0.0011</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0206</mn><mo>±</mo><mn>0.0419</mn></mrow><annotation encoding="application/x-tex">-0.0206 \mathbin{\pm} 0.0419</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0177</mn><mo>±</mo><mn>0.0459</mn></mrow><annotation encoding="application/x-tex">-0.0177 \mathbin{\pm} 0.0459</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Targeted
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0081</mn><mo>±</mo><mn>0.0012</mn></mrow><annotation encoding="application/x-tex">0.0081 \mathbin{\pm} 0.0012</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Targeted
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0002</mn><mo>±</mo><mn>0.0004</mn></mrow><annotation encoding="application/x-tex">-0.0002 \mathbin{\pm} 0.0004</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Real +
standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1212</mn><mo>±</mo><mn>0.0073</mn></mrow><annotation encoding="application/x-tex">0.1212 \mathbin{\pm} 0.0073</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Real +
standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7884</mn><mo>±</mo><mn>0.0049</mn></mrow><annotation encoding="application/x-tex">0.7884 \mathbin{\pm} 0.0049</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Real +
targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1084</mn><mo>±</mo><mn>0.0056</mn></mrow><annotation encoding="application/x-tex">0.1084 \mathbin{\pm} 0.0056</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Real +
targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7558</mn><mo>±</mo><mn>0.0178</mn></mrow><annotation encoding="application/x-tex">0.7558 \mathbin{\pm} 0.0178</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Real only
(TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1436</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.1436 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Real only
(TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8222</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.8222 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0361</mn><mo>±</mo><mn>0.0071</mn></mrow><annotation encoding="application/x-tex">0.0361 \mathbin{\pm} 0.0071</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5410</mn><mo>±</mo><mn>0.0544</mn></mrow><annotation encoding="application/x-tex">0.5410 \mathbin{\pm} 0.0544</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Targeted
synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0302</mn><mo>±</mo><mn>0.0059</mn></mrow><annotation encoding="application/x-tex">0.0302 \mathbin{\pm} 0.0059</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Targeted
synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5015</mn><mo>±</mo><mn>0.0544</mn></mrow><annotation encoding="application/x-tex">0.5015 \mathbin{\pm} 0.0544</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0224</mn><mo>±</mo><mn>0.0073</mn></mrow><annotation encoding="application/x-tex">-0.0224 \mathbin{\pm} 0.0073</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0338</mn><mo>±</mo><mn>0.0049</mn></mrow><annotation encoding="application/x-tex">-0.0338 \mathbin{\pm} 0.0049</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1076</mn><mo>±</mo><mn>0.0071</mn></mrow><annotation encoding="application/x-tex">0.1076 \mathbin{\pm} 0.0071</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2812</mn><mo>±</mo><mn>0.0544</mn></mrow><annotation encoding="application/x-tex">0.2812 \mathbin{\pm} 0.0544</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0058</mn><mo>±</mo><mn>0.0130</mn></mrow><annotation encoding="application/x-tex">-0.0058 \mathbin{\pm} 0.0130</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0395</mn><mo>±</mo><mn>0.0964</mn></mrow><annotation encoding="application/x-tex">-0.0395 \mathbin{\pm} 0.0964</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Targeted
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0352</mn><mo>±</mo><mn>0.0056</mn></mrow><annotation encoding="application/x-tex">-0.0352 \mathbin{\pm} 0.0056</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Targeted
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0664</mn><mo>±</mo><mn>0.0178</mn></mrow><annotation encoding="application/x-tex">-0.0664 \mathbin{\pm} 0.0178</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real +
standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2959</mn><mo>±</mo><mn>0.0053</mn></mrow><annotation encoding="application/x-tex">0.2959 \mathbin{\pm} 0.0053</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real +
standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8593</mn><mo>±</mo><mn>0.0013</mn></mrow><annotation encoding="application/x-tex">0.8593 \mathbin{\pm} 0.0013</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real +
targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2828</mn><mo>±</mo><mn>0.0080</mn></mrow><annotation encoding="application/x-tex">0.2828 \mathbin{\pm} 0.0080</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real +
targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8448</mn><mo>±</mo><mn>0.0088</mn></mrow><annotation encoding="application/x-tex">0.8448 \mathbin{\pm} 0.0088</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real only
(TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3409</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.3409 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real only
(TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8810</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.8810 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0870</mn><mo>±</mo><mn>0.0060</mn></mrow><annotation encoding="application/x-tex">0.0870 \mathbin{\pm} 0.0060</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5216</mn><mo>±</mo><mn>0.0335</mn></mrow><annotation encoding="application/x-tex">0.5216 \mathbin{\pm} 0.0335</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Targeted
synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0743</mn><mo>±</mo><mn>0.0129</mn></mrow><annotation encoding="application/x-tex">0.0743 \mathbin{\pm} 0.0129</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Targeted
synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4541</mn><mo>±</mo><mn>0.0738</mn></mrow><annotation encoding="application/x-tex">0.4541 \mathbin{\pm} 0.0738</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0450</mn><mo>±</mo><mn>0.0053</mn></mrow><annotation encoding="application/x-tex">-0.0450 \mathbin{\pm} 0.0053</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0216</mn><mo>±</mo><mn>0.0013</mn></mrow><annotation encoding="application/x-tex">-0.0216 \mathbin{\pm} 0.0013</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2539</mn><mo>±</mo><mn>0.0060</mn></mrow><annotation encoding="application/x-tex">0.2539 \mathbin{\pm} 0.0060</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3594</mn><mo>±</mo><mn>0.0335</mn></mrow><annotation encoding="application/x-tex">0.3594 \mathbin{\pm} 0.0335</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0127</mn><mo>±</mo><mn>0.0189</mn></mrow><annotation encoding="application/x-tex">-0.0127 \mathbin{\pm} 0.0189</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0675</mn><mo>±</mo><mn>0.1069</mn></mrow><annotation encoding="application/x-tex">-0.0675 \mathbin{\pm} 0.1069</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Targeted
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0580</mn><mo>±</mo><mn>0.0080</mn></mrow><annotation encoding="application/x-tex">-0.0580 \mathbin{\pm} 0.0080</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Targeted
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0362</mn><mo>±</mo><mn>0.0088</mn></mrow><annotation encoding="application/x-tex">-0.0362 \mathbin{\pm} 0.0088</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8805</mn><mo>±</mo><mn>0.0061</mn></mrow><annotation encoding="application/x-tex">0.8805 \mathbin{\pm} 0.0061</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7039</mn><mo>±</mo><mn>0.0032</mn></mrow><annotation encoding="application/x-tex">0.7039 \mathbin{\pm} 0.0032</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Real + targeted
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8953</mn><mo>±</mo><mn>0.0005</mn></mrow><annotation encoding="application/x-tex">0.8953 \mathbin{\pm} 0.0005</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Real + targeted
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7309</mn><mo>±</mo><mn>0.0064</mn></mrow><annotation encoding="application/x-tex">0.7309 \mathbin{\pm} 0.0064</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9009</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.9009 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7239</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7239 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7866</mn><mo>±</mo><mn>0.0348</mn></mrow><annotation encoding="application/x-tex">0.7866 \mathbin{\pm} 0.0348</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4846</mn><mo>±</mo><mn>0.0572</mn></mrow><annotation encoding="application/x-tex">0.4846 \mathbin{\pm} 0.0572</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Targeted synthetic
only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8085</mn><mo>±</mo><mn>0.0168</mn></mrow><annotation encoding="application/x-tex">0.8085 \mathbin{\pm} 0.0168</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Targeted synthetic
only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5413</mn><mo>±</mo><mn>0.0480</mn></mrow><annotation encoding="application/x-tex">0.5413 \mathbin{\pm} 0.0480</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0205</mn><mo>±</mo><mn>0.0061</mn></mrow><annotation encoding="application/x-tex">-0.0205 \mathbin{\pm} 0.0061</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0201</mn><mo>±</mo><mn>0.0032</mn></mrow><annotation encoding="application/x-tex">-0.0201 \mathbin{\pm} 0.0032</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1143</mn><mo>±</mo><mn>0.0348</mn></mrow><annotation encoding="application/x-tex">0.1143 \mathbin{\pm} 0.0348</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2393</mn><mo>±</mo><mn>0.0572</mn></mrow><annotation encoding="application/x-tex">0.2393 \mathbin{\pm} 0.0572</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0219</mn><mo>±</mo><mn>0.0489</mn></mrow><annotation encoding="application/x-tex">0.0219 \mathbin{\pm} 0.0489</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0567</mn><mo>±</mo><mn>0.1009</mn></mrow><annotation encoding="application/x-tex">0.0567 \mathbin{\pm} 0.1009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Targeted augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0056</mn><mo>±</mo><mn>0.0005</mn></mrow><annotation encoding="application/x-tex">-0.0056 \mathbin{\pm} 0.0005</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Targeted augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0070</mn><mo>±</mo><mn>0.0064</mn></mrow><annotation encoding="application/x-tex">0.0070 \mathbin{\pm} 0.0064</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Real + standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7538</mn><mo>±</mo><mn>0.0056</mn></mrow><annotation encoding="application/x-tex">0.7538 \mathbin{\pm} 0.0056</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Real + standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7353</mn><mo>±</mo><mn>0.0039</mn></mrow><annotation encoding="application/x-tex">0.7353 \mathbin{\pm} 0.0039</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Real + targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7379</mn><mo>±</mo><mn>0.0088</mn></mrow><annotation encoding="application/x-tex">0.7379 \mathbin{\pm} 0.0088</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Real + targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7246</mn><mo>±</mo><mn>0.0064</mn></mrow><annotation encoding="application/x-tex">0.7246 \mathbin{\pm} 0.0064</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7445</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7445 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7329</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7329 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5886</mn><mo>±</mo><mn>0.0867</mn></mrow><annotation encoding="application/x-tex">0.5886 \mathbin{\pm} 0.0867</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5290</mn><mo>±</mo><mn>0.0933</mn></mrow><annotation encoding="application/x-tex">0.5290 \mathbin{\pm} 0.0933</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Targeted synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5215</mn><mo>±</mo><mn>0.0252</mn></mrow><annotation encoding="application/x-tex">0.5215 \mathbin{\pm} 0.0252</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Targeted synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4260</mn><mo>±</mo><mn>0.0431</mn></mrow><annotation encoding="application/x-tex">0.4260 \mathbin{\pm} 0.0431</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Standard augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0093</mn><mo>±</mo><mn>0.0056</mn></mrow><annotation encoding="application/x-tex">0.0093 \mathbin{\pm} 0.0056</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Standard augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0024</mn><mo>±</mo><mn>0.0039</mn></mrow><annotation encoding="application/x-tex">0.0024 \mathbin{\pm} 0.0039</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Standard synthetic utility
loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1559</mn><mo>±</mo><mn>0.0867</mn></mrow><annotation encoding="application/x-tex">0.1559 \mathbin{\pm} 0.0867</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Standard synthetic utility
loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2038</mn><mo>±</mo><mn>0.0933</mn></mrow><annotation encoding="application/x-tex">0.2038 \mathbin{\pm} 0.0933</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0671</mn><mo>±</mo><mn>0.0764</mn></mrow><annotation encoding="application/x-tex">-0.0671 \mathbin{\pm} 0.0764</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.1030</mn><mo>±</mo><mn>0.0921</mn></mrow><annotation encoding="application/x-tex">-0.1030 \mathbin{\pm} 0.0921</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Targeted augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0065</mn><mo>±</mo><mn>0.0088</mn></mrow><annotation encoding="application/x-tex">-0.0065 \mathbin{\pm} 0.0088</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Targeted augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0083</mn><mo>±</mo><mn>0.0064</mn></mrow><annotation encoding="application/x-tex">-0.0083 \mathbin{\pm} 0.0064</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">High Hint Use</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0020</mn><mo>±</mo><mn>0.0017</mn></mrow><annotation encoding="application/x-tex">0.0020 \mathbin{\pm} 0.0017</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">High Hint Use</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>0.0411</mn><annotation encoding="application/x-tex">0.0411</annotation></semantics></math>
(one available run; 2 missing)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0039</mn><mo>±</mo><mn>0.0048</mn></mrow><annotation encoding="application/x-tex">0.0039 \mathbin{\pm} 0.0048</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0143</mn><mo>±</mo><mn>0.0197</mn></mrow><annotation encoding="application/x-tex">0.0143 \mathbin{\pm} 0.0197</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0001</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.0001 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0001</mn><mo>±</mo><mn>0.0010</mn></mrow><annotation encoding="application/x-tex">0.0001 \mathbin{\pm} 0.0010</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0046</mn><mo>±</mo><mn>0.0014</mn></mrow><annotation encoding="application/x-tex">-0.0046 \mathbin{\pm} 0.0014</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0282</mn><mo>±</mo><mn>0.0038</mn></mrow><annotation encoding="application/x-tex">-0.0282 \mathbin{\pm} 0.0038</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0022</mn><mo>±</mo><mn>0.0037</mn></mrow><annotation encoding="application/x-tex">0.0022 \mathbin{\pm} 0.0037</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0126</mn><mo>±</mo><mn>0.0142</mn></mrow><annotation encoding="application/x-tex">-0.0126 \mathbin{\pm} 0.0142</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Real + standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8146</mn><mo>±</mo><mn>0.0094</mn></mrow><annotation encoding="application/x-tex">0.8146 \mathbin{\pm} 0.0094</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Real + standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7335</mn><mo>±</mo><mn>0.0062</mn></mrow><annotation encoding="application/x-tex">0.7335 \mathbin{\pm} 0.0062</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Real + targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8176</mn><mo>±</mo><mn>0.0048</mn></mrow><annotation encoding="application/x-tex">0.8176 \mathbin{\pm} 0.0048</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Real + targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7358</mn><mo>±</mo><mn>0.0026</mn></mrow><annotation encoding="application/x-tex">0.7358 \mathbin{\pm} 0.0026</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8232</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.8232 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7384</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7384 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6630</mn><mo>±</mo><mn>0.0381</mn></mrow><annotation encoding="application/x-tex">0.6630 \mathbin{\pm} 0.0381</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5259</mn><mo>±</mo><mn>0.0671</mn></mrow><annotation encoding="application/x-tex">0.5259 \mathbin{\pm} 0.0671</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6612</mn><mo>±</mo><mn>0.0293</mn></mrow><annotation encoding="application/x-tex">0.6612 \mathbin{\pm} 0.0293</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5210</mn><mo>±</mo><mn>0.0567</mn></mrow><annotation encoding="application/x-tex">0.5210 \mathbin{\pm} 0.0567</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0086</mn><mo>±</mo><mn>0.0094</mn></mrow><annotation encoding="application/x-tex">-0.0086 \mathbin{\pm} 0.0094</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0049</mn><mo>±</mo><mn>0.0062</mn></mrow><annotation encoding="application/x-tex">-0.0049 \mathbin{\pm} 0.0062</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1602</mn><mo>±</mo><mn>0.0381</mn></mrow><annotation encoding="application/x-tex">0.1602 \mathbin{\pm} 0.0381</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2125</mn><mo>±</mo><mn>0.0671</mn></mrow><annotation encoding="application/x-tex">0.2125 \mathbin{\pm} 0.0671</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0017</mn><mo>±</mo><mn>0.0187</mn></mrow><annotation encoding="application/x-tex">-0.0017 \mathbin{\pm} 0.0187</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0050</mn><mo>±</mo><mn>0.0336</mn></mrow><annotation encoding="application/x-tex">-0.0050 \mathbin{\pm} 0.0336</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0056</mn><mo>±</mo><mn>0.0048</mn></mrow><annotation encoding="application/x-tex">-0.0056 \mathbin{\pm} 0.0048</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0025</mn><mo>±</mo><mn>0.0026</mn></mrow><annotation encoding="application/x-tex">-0.0025 \mathbin{\pm} 0.0026</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Real + standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1349</mn><mo>±</mo><mn>0.0154</mn></mrow><annotation encoding="application/x-tex">0.1349 \mathbin{\pm} 0.0154</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Real + standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7280</mn><mo>±</mo><mn>0.0011</mn></mrow><annotation encoding="application/x-tex">0.7280 \mathbin{\pm} 0.0011</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1345</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.1345 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7011</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7011 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0004</mn><mo>±</mo><mn>0.0154</mn></mrow><annotation encoding="application/x-tex">0.0004 \mathbin{\pm} 0.0154</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0269</mn><mo>±</mo><mn>0.0011</mn></mrow><annotation encoding="application/x-tex">0.0269 \mathbin{\pm} 0.0011</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery Prediction; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4304</mn><mo>±</mo><mn>0.0116</mn></mrow><annotation encoding="application/x-tex">0.4304 \mathbin{\pm} 0.0116</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery Prediction; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6759</mn><mo>±</mo><mn>0.0196</mn></mrow><annotation encoding="application/x-tex">0.6759 \mathbin{\pm} 0.0196</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery Prediction; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4387</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.4387 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery Prediction; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6897</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.6897 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery Prediction; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>0.3738</mn><annotation encoding="application/x-tex">0.3738</annotation></semantics></math>
(one available run; 2 missing)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery Prediction; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>0.5787</mn><annotation encoding="application/x-tex">0.5787</annotation></semantics></math>
(one available run; 2 missing)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery Prediction; Targeted synthetic
only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>0.2593</mn><annotation encoding="application/x-tex">0.2593</annotation></semantics></math>
(one available run; 2 missing)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery Prediction; Targeted synthetic
only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>0.4649</mn><annotation encoding="application/x-tex">0.4649</annotation></semantics></math>
(one available run; 2 missing)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery Prediction; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0082</mn><mo>±</mo><mn>0.0116</mn></mrow><annotation encoding="application/x-tex">-0.0082 \mathbin{\pm} 0.0116</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery Prediction; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0138</mn><mo>±</mo><mn>0.0196</mn></mrow><annotation encoding="application/x-tex">-0.0138 \mathbin{\pm} 0.0196</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery Prediction; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>0.0649</mn><annotation encoding="application/x-tex">0.0649</annotation></semantics></math>
(one available run; 2 missing)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery Prediction; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>0.1111</mn><annotation encoding="application/x-tex">0.1111</annotation></semantics></math>
(one available run; 2 missing)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">High Hint Use; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4302</mn><mo>±</mo><mn>0.0068</mn></mrow><annotation encoding="application/x-tex">0.4302 \mathbin{\pm} 0.0068</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">High Hint Use; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8583</mn><mo>±</mo><mn>0.0064</mn></mrow><annotation encoding="application/x-tex">0.8583 \mathbin{\pm} 0.0064</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">High Hint Use; Real + targeted
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4327</mn><mo>±</mo><mn>0.0063</mn></mrow><annotation encoding="application/x-tex">0.4327 \mathbin{\pm} 0.0063</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">High Hint Use; Real + targeted
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8578</mn><mo>±</mo><mn>0.0073</mn></mrow><annotation encoding="application/x-tex">0.8578 \mathbin{\pm} 0.0073</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">High Hint Use; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4399</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.4399 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">High Hint Use; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8697</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.8697 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">High Hint Use; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1550</mn><mo>±</mo><mn>0.0671</mn></mrow><annotation encoding="application/x-tex">0.1550 \mathbin{\pm} 0.0671</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">High Hint Use; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5438</mn><mo>±</mo><mn>0.1026</mn></mrow><annotation encoding="application/x-tex">0.5438 \mathbin{\pm} 0.1026</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">High Hint Use; Targeted synthetic
only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1250</mn><mo>±</mo><mn>0.0487</mn></mrow><annotation encoding="application/x-tex">0.1250 \mathbin{\pm} 0.0487</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">High Hint Use; Targeted synthetic
only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4969</mn><mo>±</mo><mn>0.0691</mn></mrow><annotation encoding="application/x-tex">0.4969 \mathbin{\pm} 0.0691</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">High Hint Use; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0098</mn><mo>±</mo><mn>0.0068</mn></mrow><annotation encoding="application/x-tex">-0.0098 \mathbin{\pm} 0.0068</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">High Hint Use; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0113</mn><mo>±</mo><mn>0.0064</mn></mrow><annotation encoding="application/x-tex">-0.0113 \mathbin{\pm} 0.0064</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">High Hint Use; Standard synthetic utility
loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2850</mn><mo>±</mo><mn>0.0671</mn></mrow><annotation encoding="application/x-tex">0.2850 \mathbin{\pm} 0.0671</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">High Hint Use; Standard synthetic utility
loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3258</mn><mo>±</mo><mn>0.1026</mn></mrow><annotation encoding="application/x-tex">0.3258 \mathbin{\pm} 0.1026</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">High Hint Use; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0300</mn><mo>±</mo><mn>0.0219</mn></mrow><annotation encoding="application/x-tex">-0.0300 \mathbin{\pm} 0.0219</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">High Hint Use; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0469</mn><mo>±</mo><mn>0.0421</mn></mrow><annotation encoding="application/x-tex">-0.0469 \mathbin{\pm} 0.0421</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">High Hint Use; Targeted augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0072</mn><mo>±</mo><mn>0.0063</mn></mrow><annotation encoding="application/x-tex">-0.0072 \mathbin{\pm} 0.0063</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">High Hint Use; Targeted augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0119</mn><mo>±</mo><mn>0.0073</mn></mrow><annotation encoding="application/x-tex">-0.0119 \mathbin{\pm} 0.0073</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7269</mn><mo>±</mo><mn>0.0145</mn></mrow><annotation encoding="application/x-tex">0.7269 \mathbin{\pm} 0.0145</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7344</mn><mo>±</mo><mn>0.0101</mn></mrow><annotation encoding="application/x-tex">0.7344 \mathbin{\pm} 0.0101</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Real + targeted
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7239</mn><mo>±</mo><mn>0.0118</mn></mrow><annotation encoding="application/x-tex">0.7239 \mathbin{\pm} 0.0118</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Real + targeted
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7355</mn><mo>±</mo><mn>0.0064</mn></mrow><annotation encoding="application/x-tex">0.7355 \mathbin{\pm} 0.0064</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Real only
(TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7467</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7467 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Real only
(TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7451</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7451 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5569</mn><mo>±</mo><mn>0.0491</mn></mrow><annotation encoding="application/x-tex">0.5569 \mathbin{\pm} 0.0491</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5803</mn><mo>±</mo><mn>0.0548</mn></mrow><annotation encoding="application/x-tex">0.5803 \mathbin{\pm} 0.0548</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Targeted
synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5160</mn><mo>±</mo><mn>0.0229</mn></mrow><annotation encoding="application/x-tex">0.5160 \mathbin{\pm} 0.0229</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Targeted
synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5377</mn><mo>±</mo><mn>0.0273</mn></mrow><annotation encoding="application/x-tex">0.5377 \mathbin{\pm} 0.0273</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Standard
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0198</mn><mo>±</mo><mn>0.0145</mn></mrow><annotation encoding="application/x-tex">-0.0198 \mathbin{\pm} 0.0145</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Standard
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0107</mn><mo>±</mo><mn>0.0101</mn></mrow><annotation encoding="application/x-tex">-0.0107 \mathbin{\pm} 0.0101</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1898</mn><mo>±</mo><mn>0.0491</mn></mrow><annotation encoding="application/x-tex">0.1898 \mathbin{\pm} 0.0491</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1648</mn><mo>±</mo><mn>0.0548</mn></mrow><annotation encoding="application/x-tex">0.1648 \mathbin{\pm} 0.0548</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0409</mn><mo>±</mo><mn>0.0718</mn></mrow><annotation encoding="application/x-tex">-0.0409 \mathbin{\pm} 0.0718</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0426</mn><mo>±</mo><mn>0.0791</mn></mrow><annotation encoding="application/x-tex">-0.0426 \mathbin{\pm} 0.0791</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Targeted
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0228</mn><mo>±</mo><mn>0.0118</mn></mrow><annotation encoding="application/x-tex">-0.0228 \mathbin{\pm} 0.0118</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Targeted
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0096</mn><mo>±</mo><mn>0.0064</mn></mrow><annotation encoding="application/x-tex">-0.0096 \mathbin{\pm} 0.0064</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Real +
standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1541</mn><mo>±</mo><mn>0.0212</mn></mrow><annotation encoding="application/x-tex">0.1541 \mathbin{\pm} 0.0212</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Real +
standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7800</mn><mo>±</mo><mn>0.0231</mn></mrow><annotation encoding="application/x-tex">0.7800 \mathbin{\pm} 0.0231</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Real +
targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1439</mn><mo>±</mo><mn>0.0032</mn></mrow><annotation encoding="application/x-tex">0.1439 \mathbin{\pm} 0.0032</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Real +
targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7862</mn><mo>±</mo><mn>0.0269</mn></mrow><annotation encoding="application/x-tex">0.7862 \mathbin{\pm} 0.0269</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Real only
(TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1436</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.1436 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Real only
(TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8222</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.8222 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0833</mn><mo>±</mo><mn>0.0670</mn></mrow><annotation encoding="application/x-tex">0.0833 \mathbin{\pm} 0.0670</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5595</mn><mo>±</mo><mn>0.1198</mn></mrow><annotation encoding="application/x-tex">0.5595 \mathbin{\pm} 0.1198</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Targeted
synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0631</mn><mo>±</mo><mn>0.0576</mn></mrow><annotation encoding="application/x-tex">0.0631 \mathbin{\pm} 0.0576</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Targeted
synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4593</mn><mo>±</mo><mn>0.1080</mn></mrow><annotation encoding="application/x-tex">0.4593 \mathbin{\pm} 0.1080</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0105</mn><mo>±</mo><mn>0.0212</mn></mrow><annotation encoding="application/x-tex">0.0105 \mathbin{\pm} 0.0212</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0422</mn><mo>±</mo><mn>0.0231</mn></mrow><annotation encoding="application/x-tex">-0.0422 \mathbin{\pm} 0.0231</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0603</mn><mo>±</mo><mn>0.0670</mn></mrow><annotation encoding="application/x-tex">0.0603 \mathbin{\pm} 0.0670</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2627</mn><mo>±</mo><mn>0.1198</mn></mrow><annotation encoding="application/x-tex">0.2627 \mathbin{\pm} 0.1198</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0203</mn><mo>±</mo><mn>0.0106</mn></mrow><annotation encoding="application/x-tex">-0.0203 \mathbin{\pm} 0.0106</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.1001</mn><mo>±</mo><mn>0.0382</mn></mrow><annotation encoding="application/x-tex">-0.1001 \mathbin{\pm} 0.0382</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Targeted
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0003</mn><mo>±</mo><mn>0.0032</mn></mrow><annotation encoding="application/x-tex">0.0003 \mathbin{\pm} 0.0032</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Targeted
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0360</mn><mo>±</mo><mn>0.0269</mn></mrow><annotation encoding="application/x-tex">-0.0360 \mathbin{\pm} 0.0269</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real +
standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3171</mn><mo>±</mo><mn>0.0103</mn></mrow><annotation encoding="application/x-tex">0.3171 \mathbin{\pm} 0.0103</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real +
standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8622</mn><mo>±</mo><mn>0.0152</mn></mrow><annotation encoding="application/x-tex">0.8622 \mathbin{\pm} 0.0152</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real +
targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3161</mn><mo>±</mo><mn>0.0073</mn></mrow><annotation encoding="application/x-tex">0.3161 \mathbin{\pm} 0.0073</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real +
targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8652</mn><mo>±</mo><mn>0.0099</mn></mrow><annotation encoding="application/x-tex">0.8652 \mathbin{\pm} 0.0099</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real only
(TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3409</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.3409 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real only
(TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8810</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.8810 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1296</mn><mo>±</mo><mn>0.0627</mn></mrow><annotation encoding="application/x-tex">0.1296 \mathbin{\pm} 0.0627</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5301</mn><mo>±</mo><mn>0.1321</mn></mrow><annotation encoding="application/x-tex">0.5301 \mathbin{\pm} 0.1321</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Targeted
synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1017</mn><mo>±</mo><mn>0.0397</mn></mrow><annotation encoding="application/x-tex">0.1017 \mathbin{\pm} 0.0397</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Targeted
synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4677</mn><mo>±</mo><mn>0.1231</mn></mrow><annotation encoding="application/x-tex">0.4677 \mathbin{\pm} 0.1231</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0238</mn><mo>±</mo><mn>0.0103</mn></mrow><annotation encoding="application/x-tex">-0.0238 \mathbin{\pm} 0.0103</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0188</mn><mo>±</mo><mn>0.0152</mn></mrow><annotation encoding="application/x-tex">-0.0188 \mathbin{\pm} 0.0152</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2112</mn><mo>±</mo><mn>0.0627</mn></mrow><annotation encoding="application/x-tex">0.2112 \mathbin{\pm} 0.0627</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3509</mn><mo>±</mo><mn>0.1321</mn></mrow><annotation encoding="application/x-tex">0.3509 \mathbin{\pm} 0.1321</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0279</mn><mo>±</mo><mn>0.0307</mn></mrow><annotation encoding="application/x-tex">-0.0279 \mathbin{\pm} 0.0307</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0624</mn><mo>±</mo><mn>0.1196</mn></mrow><annotation encoding="application/x-tex">-0.0624 \mathbin{\pm} 0.1196</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Targeted
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0248</mn><mo>±</mo><mn>0.0073</mn></mrow><annotation encoding="application/x-tex">-0.0248 \mathbin{\pm} 0.0073</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Targeted
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0157</mn><mo>±</mo><mn>0.0099</mn></mrow><annotation encoding="application/x-tex">-0.0157 \mathbin{\pm} 0.0099</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8878</mn><mo>±</mo><mn>0.0169</mn></mrow><annotation encoding="application/x-tex">0.8878 \mathbin{\pm} 0.0169</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7094</mn><mo>±</mo><mn>0.0032</mn></mrow><annotation encoding="application/x-tex">0.7094 \mathbin{\pm} 0.0032</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Real + targeted
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8950</mn><mo>±</mo><mn>0.0136</mn></mrow><annotation encoding="application/x-tex">0.8950 \mathbin{\pm} 0.0136</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Real + targeted
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7159</mn><mo>±</mo><mn>0.0101</mn></mrow><annotation encoding="application/x-tex">0.7159 \mathbin{\pm} 0.0101</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9009</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.9009 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7239</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7239 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8103</mn><mo>±</mo><mn>0.0480</mn></mrow><annotation encoding="application/x-tex">0.8103 \mathbin{\pm} 0.0480</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5168</mn><mo>±</mo><mn>0.1208</mn></mrow><annotation encoding="application/x-tex">0.5168 \mathbin{\pm} 0.1208</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Targeted synthetic
only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8049</mn><mo>±</mo><mn>0.0345</mn></mrow><annotation encoding="application/x-tex">0.8049 \mathbin{\pm} 0.0345</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Targeted synthetic
only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5161</mn><mo>±</mo><mn>0.0898</mn></mrow><annotation encoding="application/x-tex">0.5161 \mathbin{\pm} 0.0898</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0131</mn><mo>±</mo><mn>0.0169</mn></mrow><annotation encoding="application/x-tex">-0.0131 \mathbin{\pm} 0.0169</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0145</mn><mo>±</mo><mn>0.0032</mn></mrow><annotation encoding="application/x-tex">-0.0145 \mathbin{\pm} 0.0032</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0907</mn><mo>±</mo><mn>0.0480</mn></mrow><annotation encoding="application/x-tex">0.0907 \mathbin{\pm} 0.0480</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2072</mn><mo>±</mo><mn>0.1208</mn></mrow><annotation encoding="application/x-tex">0.2072 \mathbin{\pm} 0.1208</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0054</mn><mo>±</mo><mn>0.0286</mn></mrow><annotation encoding="application/x-tex">-0.0054 \mathbin{\pm} 0.0286</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0006</mn><mo>±</mo><mn>0.0762</mn></mrow><annotation encoding="application/x-tex">-0.0006 \mathbin{\pm} 0.0762</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Targeted augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0059</mn><mo>±</mo><mn>0.0136</mn></mrow><annotation encoding="application/x-tex">-0.0059 \mathbin{\pm} 0.0136</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Targeted augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0080</mn><mo>±</mo><mn>0.0101</mn></mrow><annotation encoding="application/x-tex">-0.0080 \mathbin{\pm} 0.0101</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Real + standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7340</mn><mo>±</mo><mn>0.0033</mn></mrow><annotation encoding="application/x-tex">0.7340 \mathbin{\pm} 0.0033</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Real + standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7231</mn><mo>±</mo><mn>0.0021</mn></mrow><annotation encoding="application/x-tex">0.7231 \mathbin{\pm} 0.0021</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Real + targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7330</mn><mo>±</mo><mn>0.0067</mn></mrow><annotation encoding="application/x-tex">0.7330 \mathbin{\pm} 0.0067</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Real + targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7243</mn><mo>±</mo><mn>0.0046</mn></mrow><annotation encoding="application/x-tex">0.7243 \mathbin{\pm} 0.0046</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7445</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7445 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7329</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7329 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5913</mn><mo>±</mo><mn>0.0318</mn></mrow><annotation encoding="application/x-tex">0.5913 \mathbin{\pm} 0.0318</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5224</mn><mo>±</mo><mn>0.0568</mn></mrow><annotation encoding="application/x-tex">0.5224 \mathbin{\pm} 0.0568</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Targeted synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6155</mn><mo>±</mo><mn>0.0619</mn></mrow><annotation encoding="application/x-tex">0.6155 \mathbin{\pm} 0.0619</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Targeted synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5607</mn><mo>±</mo><mn>0.0721</mn></mrow><annotation encoding="application/x-tex">0.5607 \mathbin{\pm} 0.0721</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Standard augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0105</mn><mo>±</mo><mn>0.0033</mn></mrow><annotation encoding="application/x-tex">-0.0105 \mathbin{\pm} 0.0033</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Standard augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0098</mn><mo>±</mo><mn>0.0021</mn></mrow><annotation encoding="application/x-tex">-0.0098 \mathbin{\pm} 0.0021</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Standard synthetic utility
loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1532</mn><mo>±</mo><mn>0.0318</mn></mrow><annotation encoding="application/x-tex">0.1532 \mathbin{\pm} 0.0318</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Standard synthetic utility
loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2104</mn><mo>±</mo><mn>0.0568</mn></mrow><annotation encoding="application/x-tex">0.2104 \mathbin{\pm} 0.0568</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0242</mn><mo>±</mo><mn>0.0479</mn></mrow><annotation encoding="application/x-tex">0.0242 \mathbin{\pm} 0.0479</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0383</mn><mo>±</mo><mn>0.0707</mn></mrow><annotation encoding="application/x-tex">0.0383 \mathbin{\pm} 0.0707</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Targeted augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0115</mn><mo>±</mo><mn>0.0067</mn></mrow><annotation encoding="application/x-tex">-0.0115 \mathbin{\pm} 0.0067</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Targeted augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0086</mn><mo>±</mo><mn>0.0046</mn></mrow><annotation encoding="application/x-tex">-0.0086 \mathbin{\pm} 0.0046</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">High Hint Use</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0001</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">-0.0001 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0023</mn><mo>±</mo><mn>0.0033</mn></mrow><annotation encoding="application/x-tex">-0.0023 \mathbin{\pm} 0.0033</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0617</mn><mo>±</mo><mn>0.0904</mn></mrow><annotation encoding="application/x-tex">0.0617 \mathbin{\pm} 0.0904</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0380</mn><mo>±</mo><mn>0.0380</mn></mrow><annotation encoding="application/x-tex">-0.0380 \mathbin{\pm} 0.0380</annotation></semantics></math>
(2; 1 missing)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0028</mn><mo>±</mo><mn>0.0040</mn></mrow><annotation encoding="application/x-tex">-0.0028 \mathbin{\pm} 0.0040</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Real + standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7664</mn><mo>±</mo><mn>0.0003</mn></mrow><annotation encoding="application/x-tex">0.7664 \mathbin{\pm} 0.0003</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Real + standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6684</mn><mo>±</mo><mn>0.0003</mn></mrow><annotation encoding="application/x-tex">0.6684 \mathbin{\pm} 0.0003</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Real + targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7607</mn><mo>±</mo><mn>0.0005</mn></mrow><annotation encoding="application/x-tex">0.7607 \mathbin{\pm} 0.0005</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Real + targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6609</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">0.6609 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7687</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7687 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6722</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.6722 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7380</mn><mo>±</mo><mn>0.0012</mn></mrow><annotation encoding="application/x-tex">0.7380 \mathbin{\pm} 0.0012</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6359</mn><mo>±</mo><mn>0.0017</mn></mrow><annotation encoding="application/x-tex">0.6359 \mathbin{\pm} 0.0017</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7313</mn><mo>±</mo><mn>0.0016</mn></mrow><annotation encoding="application/x-tex">0.7313 \mathbin{\pm} 0.0016</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6245</mn><mo>±</mo><mn>0.0011</mn></mrow><annotation encoding="application/x-tex">0.6245 \mathbin{\pm} 0.0011</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0024</mn><mo>±</mo><mn>0.0003</mn></mrow><annotation encoding="application/x-tex">-0.0024 \mathbin{\pm} 0.0003</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0037</mn><mo>±</mo><mn>0.0003</mn></mrow><annotation encoding="application/x-tex">-0.0037 \mathbin{\pm} 0.0003</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0308</mn><mo>±</mo><mn>0.0012</mn></mrow><annotation encoding="application/x-tex">0.0308 \mathbin{\pm} 0.0012</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0362</mn><mo>±</mo><mn>0.0017</mn></mrow><annotation encoding="application/x-tex">0.0362 \mathbin{\pm} 0.0017</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0066</mn><mo>±</mo><mn>0.0020</mn></mrow><annotation encoding="application/x-tex">-0.0066 \mathbin{\pm} 0.0020</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0114</mn><mo>±</mo><mn>0.0024</mn></mrow><annotation encoding="application/x-tex">-0.0114 \mathbin{\pm} 0.0024</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0080</mn><mo>±</mo><mn>0.0005</mn></mrow><annotation encoding="application/x-tex">-0.0080 \mathbin{\pm} 0.0005</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0112</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">-0.0112 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Real + standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1130</mn><mo>±</mo><mn>0.0014</mn></mrow><annotation encoding="application/x-tex">0.1130 \mathbin{\pm} 0.0014</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Real + standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5596</mn><mo>±</mo><mn>0.0057</mn></mrow><annotation encoding="application/x-tex">0.5596 \mathbin{\pm} 0.0057</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1310</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.1310 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5859</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.5859 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0877</mn><mo>±</mo><mn>0.0044</mn></mrow><annotation encoding="application/x-tex">0.0877 \mathbin{\pm} 0.0044</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4889</mn><mo>±</mo><mn>0.0139</mn></mrow><annotation encoding="application/x-tex">0.4889 \mathbin{\pm} 0.0139</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Targeted synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0943</mn><mo>±</mo><mn>0.0055</mn></mrow><annotation encoding="application/x-tex">0.0943 \mathbin{\pm} 0.0055</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Targeted synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4945</mn><mo>±</mo><mn>0.0166</mn></mrow><annotation encoding="application/x-tex">0.4945 \mathbin{\pm} 0.0166</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0179</mn><mo>±</mo><mn>0.0014</mn></mrow><annotation encoding="application/x-tex">-0.0179 \mathbin{\pm} 0.0014</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0263</mn><mo>±</mo><mn>0.0057</mn></mrow><annotation encoding="application/x-tex">-0.0263 \mathbin{\pm} 0.0057</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0432</mn><mo>±</mo><mn>0.0044</mn></mrow><annotation encoding="application/x-tex">0.0432 \mathbin{\pm} 0.0044</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0969</mn><mo>±</mo><mn>0.0139</mn></mrow><annotation encoding="application/x-tex">0.0969 \mathbin{\pm} 0.0139</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery Prediction; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2130</mn><mo>±</mo><mn>0.0116</mn></mrow><annotation encoding="application/x-tex">0.2130 \mathbin{\pm} 0.0116</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery Prediction; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4764</mn><mo>±</mo><mn>0.0151</mn></mrow><annotation encoding="application/x-tex">0.4764 \mathbin{\pm} 0.0151</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery Prediction; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2194</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.2194 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery Prediction; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4910</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.4910 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery Prediction; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2239</mn><mo>±</mo><mn>0.0052</mn></mrow><annotation encoding="application/x-tex">0.2239 \mathbin{\pm} 0.0052</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery Prediction; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5052</mn><mo>±</mo><mn>0.0194</mn></mrow><annotation encoding="application/x-tex">0.5052 \mathbin{\pm} 0.0194</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery Prediction; Targeted synthetic
only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2142</mn><mo>±</mo><mn>0.0125</mn></mrow><annotation encoding="application/x-tex">0.2142 \mathbin{\pm} 0.0125</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery Prediction; Targeted synthetic
only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4924</mn><mo>±</mo><mn>0.0199</mn></mrow><annotation encoding="application/x-tex">0.4924 \mathbin{\pm} 0.0199</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery Prediction; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0064</mn><mo>±</mo><mn>0.0116</mn></mrow><annotation encoding="application/x-tex">-0.0064 \mathbin{\pm} 0.0116</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery Prediction; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0146</mn><mo>±</mo><mn>0.0151</mn></mrow><annotation encoding="application/x-tex">-0.0146 \mathbin{\pm} 0.0151</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery Prediction; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0045</mn><mo>±</mo><mn>0.0052</mn></mrow><annotation encoding="application/x-tex">-0.0045 \mathbin{\pm} 0.0052</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery Prediction; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0142</mn><mo>±</mo><mn>0.0194</mn></mrow><annotation encoding="application/x-tex">-0.0142 \mathbin{\pm} 0.0194</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6295</mn><mo>±</mo><mn>0.0015</mn></mrow><annotation encoding="application/x-tex">0.6295 \mathbin{\pm} 0.0015</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6826</mn><mo>±</mo><mn>0.0019</mn></mrow><annotation encoding="application/x-tex">0.6826 \mathbin{\pm} 0.0019</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Real + targeted
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6280</mn><mo>±</mo><mn>0.0049</mn></mrow><annotation encoding="application/x-tex">0.6280 \mathbin{\pm} 0.0049</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Real + targeted
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6813</mn><mo>±</mo><mn>0.0021</mn></mrow><annotation encoding="application/x-tex">0.6813 \mathbin{\pm} 0.0021</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Real only
(TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6426</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.6426 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Real only
(TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6902</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.6902 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6286</mn><mo>±</mo><mn>0.0103</mn></mrow><annotation encoding="application/x-tex">0.6286 \mathbin{\pm} 0.0103</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6819</mn><mo>±</mo><mn>0.0092</mn></mrow><annotation encoding="application/x-tex">0.6819 \mathbin{\pm} 0.0092</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Targeted
synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6197</mn><mo>±</mo><mn>0.0051</mn></mrow><annotation encoding="application/x-tex">0.6197 \mathbin{\pm} 0.0051</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Targeted
synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6619</mn><mo>±</mo><mn>0.0043</mn></mrow><annotation encoding="application/x-tex">0.6619 \mathbin{\pm} 0.0043</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Standard
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0131</mn><mo>±</mo><mn>0.0015</mn></mrow><annotation encoding="application/x-tex">-0.0131 \mathbin{\pm} 0.0015</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Standard
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0076</mn><mo>±</mo><mn>0.0019</mn></mrow><annotation encoding="application/x-tex">-0.0076 \mathbin{\pm} 0.0019</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0140</mn><mo>±</mo><mn>0.0103</mn></mrow><annotation encoding="application/x-tex">0.0140 \mathbin{\pm} 0.0103</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0083</mn><mo>±</mo><mn>0.0092</mn></mrow><annotation encoding="application/x-tex">0.0083 \mathbin{\pm} 0.0092</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0088</mn><mo>±</mo><mn>0.0057</mn></mrow><annotation encoding="application/x-tex">-0.0088 \mathbin{\pm} 0.0057</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0200</mn><mo>±</mo><mn>0.0071</mn></mrow><annotation encoding="application/x-tex">-0.0200 \mathbin{\pm} 0.0071</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Targeted
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0146</mn><mo>±</mo><mn>0.0049</mn></mrow><annotation encoding="application/x-tex">-0.0146 \mathbin{\pm} 0.0049</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline; Targeted
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0089</mn><mo>±</mo><mn>0.0021</mn></mrow><annotation encoding="application/x-tex">-0.0089 \mathbin{\pm} 0.0021</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Real +
standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1065</mn><mo>±</mo><mn>0.0004</mn></mrow><annotation encoding="application/x-tex">0.1065 \mathbin{\pm} 0.0004</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Real +
standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5436</mn><mo>±</mo><mn>0.0006</mn></mrow><annotation encoding="application/x-tex">0.5436 \mathbin{\pm} 0.0006</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Real +
targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0947</mn><mo>±</mo><mn>0.0005</mn></mrow><annotation encoding="application/x-tex">0.0947 \mathbin{\pm} 0.0005</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Real +
targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5238</mn><mo>±</mo><mn>0.0027</mn></mrow><annotation encoding="application/x-tex">0.5238 \mathbin{\pm} 0.0027</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Real only
(TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1132</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.1132 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Real only
(TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5416</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.5416 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1009</mn><mo>±</mo><mn>0.0023</mn></mrow><annotation encoding="application/x-tex">0.1009 \mathbin{\pm} 0.0023</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5186</mn><mo>±</mo><mn>0.0048</mn></mrow><annotation encoding="application/x-tex">0.5186 \mathbin{\pm} 0.0048</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Targeted
synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0887</mn><mo>±</mo><mn>0.0042</mn></mrow><annotation encoding="application/x-tex">0.0887 \mathbin{\pm} 0.0042</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Targeted
synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4886</mn><mo>±</mo><mn>0.0130</mn></mrow><annotation encoding="application/x-tex">0.4886 \mathbin{\pm} 0.0130</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0067</mn><mo>±</mo><mn>0.0004</mn></mrow><annotation encoding="application/x-tex">-0.0067 \mathbin{\pm} 0.0004</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0020</mn><mo>±</mo><mn>0.0006</mn></mrow><annotation encoding="application/x-tex">0.0020 \mathbin{\pm} 0.0006</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0123</mn><mo>±</mo><mn>0.0023</mn></mrow><annotation encoding="application/x-tex">0.0123 \mathbin{\pm} 0.0023</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0230</mn><mo>±</mo><mn>0.0048</mn></mrow><annotation encoding="application/x-tex">0.0230 \mathbin{\pm} 0.0048</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0123</mn><mo>±</mo><mn>0.0063</mn></mrow><annotation encoding="application/x-tex">-0.0123 \mathbin{\pm} 0.0063</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0301</mn><mo>±</mo><mn>0.0178</mn></mrow><annotation encoding="application/x-tex">-0.0301 \mathbin{\pm} 0.0178</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Targeted
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0186</mn><mo>±</mo><mn>0.0005</mn></mrow><annotation encoding="application/x-tex">-0.0186 \mathbin{\pm} 0.0005</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness; Targeted
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0178</mn><mo>±</mo><mn>0.0027</mn></mrow><annotation encoding="application/x-tex">-0.0178 \mathbin{\pm} 0.0027</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real +
standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3019</mn><mo>±</mo><mn>0.0036</mn></mrow><annotation encoding="application/x-tex">0.3019 \mathbin{\pm} 0.0036</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real +
standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5622</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">0.5622 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real +
targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3096</mn><mo>±</mo><mn>0.0011</mn></mrow><annotation encoding="application/x-tex">0.3096 \mathbin{\pm} 0.0011</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real +
targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5849</mn><mo>±</mo><mn>0.0038</mn></mrow><annotation encoding="application/x-tex">0.5849 \mathbin{\pm} 0.0038</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real only
(TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3092</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.3092 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real only
(TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5619</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.5619 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2954</mn><mo>±</mo><mn>0.0035</mn></mrow><annotation encoding="application/x-tex">0.2954 \mathbin{\pm} 0.0035</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5631</mn><mo>±</mo><mn>0.0030</mn></mrow><annotation encoding="application/x-tex">0.5631 \mathbin{\pm} 0.0030</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Targeted
synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3110</mn><mo>±</mo><mn>0.0037</mn></mrow><annotation encoding="application/x-tex">0.3110 \mathbin{\pm} 0.0037</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Targeted
synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5904</mn><mo>±</mo><mn>0.0043</mn></mrow><annotation encoding="application/x-tex">0.5904 \mathbin{\pm} 0.0043</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0073</mn><mo>±</mo><mn>0.0036</mn></mrow><annotation encoding="application/x-tex">-0.0073 \mathbin{\pm} 0.0036</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0003</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">0.0003 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0137</mn><mo>±</mo><mn>0.0035</mn></mrow><annotation encoding="application/x-tex">0.0137 \mathbin{\pm} 0.0035</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0012</mn><mo>±</mo><mn>0.0030</mn></mrow><annotation encoding="application/x-tex">-0.0012 \mathbin{\pm} 0.0030</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0155</mn><mo>±</mo><mn>0.0023</mn></mrow><annotation encoding="application/x-tex">0.0155 \mathbin{\pm} 0.0023</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0273</mn><mo>±</mo><mn>0.0020</mn></mrow><annotation encoding="application/x-tex">0.0273 \mathbin{\pm} 0.0020</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Targeted
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0004</mn><mo>±</mo><mn>0.0011</mn></mrow><annotation encoding="application/x-tex">0.0004 \mathbin{\pm} 0.0011</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Targeted
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0230</mn><mo>±</mo><mn>0.0038</mn></mrow><annotation encoding="application/x-tex">0.0230 \mathbin{\pm} 0.0038</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5333</mn><mo>±</mo><mn>0.0052</mn></mrow><annotation encoding="application/x-tex">0.5333 \mathbin{\pm} 0.0052</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5799</mn><mo>±</mo><mn>0.0056</mn></mrow><annotation encoding="application/x-tex">0.5799 \mathbin{\pm} 0.0056</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Real + targeted
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5212</mn><mo>±</mo><mn>0.0100</mn></mrow><annotation encoding="application/x-tex">0.5212 \mathbin{\pm} 0.0100</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Real + targeted
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5504</mn><mo>±</mo><mn>0.0025</mn></mrow><annotation encoding="application/x-tex">0.5504 \mathbin{\pm} 0.0025</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5203</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.5203 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5892</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.5892 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5230</mn><mo>±</mo><mn>0.0030</mn></mrow><annotation encoding="application/x-tex">0.5230 \mathbin{\pm} 0.0030</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5556</mn><mo>±</mo><mn>0.0059</mn></mrow><annotation encoding="application/x-tex">0.5556 \mathbin{\pm} 0.0059</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Targeted synthetic
only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5315</mn><mo>±</mo><mn>0.0170</mn></mrow><annotation encoding="application/x-tex">0.5315 \mathbin{\pm} 0.0170</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Targeted synthetic
only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5611</mn><mo>±</mo><mn>0.0196</mn></mrow><annotation encoding="application/x-tex">0.5611 \mathbin{\pm} 0.0196</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0130</mn><mo>±</mo><mn>0.0052</mn></mrow><annotation encoding="application/x-tex">0.0130 \mathbin{\pm} 0.0052</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0093</mn><mo>±</mo><mn>0.0056</mn></mrow><annotation encoding="application/x-tex">-0.0093 \mathbin{\pm} 0.0056</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0027</mn><mo>±</mo><mn>0.0030</mn></mrow><annotation encoding="application/x-tex">-0.0027 \mathbin{\pm} 0.0030</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0337</mn><mo>±</mo><mn>0.0059</mn></mrow><annotation encoding="application/x-tex">0.0337 \mathbin{\pm} 0.0059</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0085</mn><mo>±</mo><mn>0.0150</mn></mrow><annotation encoding="application/x-tex">0.0085 \mathbin{\pm} 0.0150</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0056</mn><mo>±</mo><mn>0.0199</mn></mrow><annotation encoding="application/x-tex">0.0056 \mathbin{\pm} 0.0199</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Targeted augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0009</mn><mo>±</mo><mn>0.0100</mn></mrow><annotation encoding="application/x-tex">0.0009 \mathbin{\pm} 0.0100</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path; Targeted augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0388</mn><mo>±</mo><mn>0.0025</mn></mrow><annotation encoding="application/x-tex">-0.0388 \mathbin{\pm} 0.0025</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Real + standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7186</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">0.7186 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Real + standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7010</mn><mo>±</mo><mn>0.0014</mn></mrow><annotation encoding="application/x-tex">0.7010 \mathbin{\pm} 0.0014</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Real + targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6883</mn><mo>±</mo><mn>0.0015</mn></mrow><annotation encoding="application/x-tex">0.6883 \mathbin{\pm} 0.0015</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Real + targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6689</mn><mo>±</mo><mn>0.0037</mn></mrow><annotation encoding="application/x-tex">0.6689 \mathbin{\pm} 0.0037</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7117</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7117 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6912</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.6912 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7122</mn><mo>±</mo><mn>0.0043</mn></mrow><annotation encoding="application/x-tex">0.7122 \mathbin{\pm} 0.0043</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6953</mn><mo>±</mo><mn>0.0019</mn></mrow><annotation encoding="application/x-tex">0.6953 \mathbin{\pm} 0.0019</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Targeted synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6670</mn><mo>±</mo><mn>0.0089</mn></mrow><annotation encoding="application/x-tex">0.6670 \mathbin{\pm} 0.0089</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Targeted synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6456</mn><mo>±</mo><mn>0.0160</mn></mrow><annotation encoding="application/x-tex">0.6456 \mathbin{\pm} 0.0160</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Standard augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0069</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">0.0069 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Standard augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0098</mn><mo>±</mo><mn>0.0014</mn></mrow><annotation encoding="application/x-tex">0.0098 \mathbin{\pm} 0.0014</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Standard synthetic utility
loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0005</mn><mo>±</mo><mn>0.0043</mn></mrow><annotation encoding="application/x-tex">-0.0005 \mathbin{\pm} 0.0043</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Standard synthetic utility
loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0041</mn><mo>±</mo><mn>0.0019</mn></mrow><annotation encoding="application/x-tex">-0.0041 \mathbin{\pm} 0.0019</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0452</mn><mo>±</mo><mn>0.0125</mn></mrow><annotation encoding="application/x-tex">-0.0452 \mathbin{\pm} 0.0125</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0497</mn><mo>±</mo><mn>0.0179</mn></mrow><annotation encoding="application/x-tex">-0.0497 \mathbin{\pm} 0.0179</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Targeted augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0234</mn><mo>±</mo><mn>0.0015</mn></mrow><annotation encoding="application/x-tex">-0.0234 \mathbin{\pm} 0.0015</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery; Targeted augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0224</mn><mo>±</mo><mn>0.0037</mn></mrow><annotation encoding="application/x-tex">-0.0224 \mathbin{\pm} 0.0037</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0117</mn><mo>±</mo><mn>0.0020</mn></mrow><annotation encoding="application/x-tex">0.0117 \mathbin{\pm} 0.0020</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0078</mn><mo>±</mo><mn>0.0036</mn></mrow><annotation encoding="application/x-tex">-0.0078 \mathbin{\pm} 0.0036</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0043</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">0.0043 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0002</mn><mo>±</mo><mn>0.0033</mn></mrow><annotation encoding="application/x-tex">-0.0002 \mathbin{\pm} 0.0033</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0018</mn><mo>±</mo><mn>0.0048</mn></mrow><annotation encoding="application/x-tex">-0.0018 \mathbin{\pm} 0.0048</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0035</mn><mo>±</mo><mn>0.0170</mn></mrow><annotation encoding="application/x-tex">0.0035 \mathbin{\pm} 0.0170</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0031</mn><mo>±</mo><mn>0.0043</mn></mrow><annotation encoding="application/x-tex">0.0031 \mathbin{\pm} 0.0043</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1097</mn><mo>±</mo><mn>0.0027</mn></mrow><annotation encoding="application/x-tex">0.1097 \mathbin{\pm} 0.0027</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0009</mn><mo>±</mo><mn>0.0012</mn></mrow><annotation encoding="application/x-tex">-0.0009 \mathbin{\pm} 0.0012</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0009</mn><mo>±</mo><mn>0.0049</mn></mrow><annotation encoding="application/x-tex">-0.0009 \mathbin{\pm} 0.0049</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Real + standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7642</mn><mo>±</mo><mn>0.0005</mn></mrow><annotation encoding="application/x-tex">0.7642 \mathbin{\pm} 0.0005</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Real + standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6670</mn><mo>±</mo><mn>0.0005</mn></mrow><annotation encoding="application/x-tex">0.6670 \mathbin{\pm} 0.0005</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Real + targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7635</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.7635 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Real + targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6663</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.6663 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7687</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7687 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6722</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.6722 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6257</mn><mo>±</mo><mn>0.0159</mn></mrow><annotation encoding="application/x-tex">0.6257 \mathbin{\pm} 0.0159</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4906</mn><mo>±</mo><mn>0.0212</mn></mrow><annotation encoding="application/x-tex">0.4906 \mathbin{\pm} 0.0212</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6326</mn><mo>±</mo><mn>0.0130</mn></mrow><annotation encoding="application/x-tex">0.6326 \mathbin{\pm} 0.0130</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5114</mn><mo>±</mo><mn>0.0264</mn></mrow><annotation encoding="application/x-tex">0.5114 \mathbin{\pm} 0.0264</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0046</mn><mo>±</mo><mn>0.0005</mn></mrow><annotation encoding="application/x-tex">-0.0046 \mathbin{\pm} 0.0005</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0051</mn><mo>±</mo><mn>0.0005</mn></mrow><annotation encoding="application/x-tex">-0.0051 \mathbin{\pm} 0.0005</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1430</mn><mo>±</mo><mn>0.0159</mn></mrow><annotation encoding="application/x-tex">0.1430 \mathbin{\pm} 0.0159</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1816</mn><mo>±</mo><mn>0.0212</mn></mrow><annotation encoding="application/x-tex">0.1816 \mathbin{\pm} 0.0212</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0069</mn><mo>±</mo><mn>0.0056</mn></mrow><annotation encoding="application/x-tex">0.0069 \mathbin{\pm} 0.0056</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0208</mn><mo>±</mo><mn>0.0073</mn></mrow><annotation encoding="application/x-tex">0.0208 \mathbin{\pm} 0.0073</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0052</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">-0.0052 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0059</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">-0.0059 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Real + standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1227</mn><mo>±</mo><mn>0.0049</mn></mrow><annotation encoding="application/x-tex">0.1227 \mathbin{\pm} 0.0049</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Real + standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5725</mn><mo>±</mo><mn>0.0124</mn></mrow><annotation encoding="application/x-tex">0.5725 \mathbin{\pm} 0.0124</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1310</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.1310 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5859</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.5859 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0810</mn><mo>±</mo><mn>0.0007</mn></mrow><annotation encoding="application/x-tex">0.0810 \mathbin{\pm} 0.0007</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4758</mn><mo>±</mo><mn>0.0071</mn></mrow><annotation encoding="application/x-tex">0.4758 \mathbin{\pm} 0.0071</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Targeted synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0799</mn><mo>±</mo><mn>0.0034</mn></mrow><annotation encoding="application/x-tex">0.0799 \mathbin{\pm} 0.0034</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Targeted synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4500</mn><mo>±</mo><mn>0.0185</mn></mrow><annotation encoding="application/x-tex">0.4500 \mathbin{\pm} 0.0185</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0082</mn><mo>±</mo><mn>0.0049</mn></mrow><annotation encoding="application/x-tex">-0.0082 \mathbin{\pm} 0.0049</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0133</mn><mo>±</mo><mn>0.0124</mn></mrow><annotation encoding="application/x-tex">-0.0133 \mathbin{\pm} 0.0124</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0500</mn><mo>±</mo><mn>0.0007</mn></mrow><annotation encoding="application/x-tex">0.0500 \mathbin{\pm} 0.0007</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1101</mn><mo>±</mo><mn>0.0071</mn></mrow><annotation encoding="application/x-tex">0.1101 \mathbin{\pm} 0.0071</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery Prediction; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2222</mn><mo>±</mo><mn>0.0175</mn></mrow><annotation encoding="application/x-tex">0.2222 \mathbin{\pm} 0.0175</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery Prediction; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4739</mn><mo>±</mo><mn>0.0140</mn></mrow><annotation encoding="application/x-tex">0.4739 \mathbin{\pm} 0.0140</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery Prediction; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2194</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.2194 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery Prediction; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4910</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.4910 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery Prediction; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2026</mn><mo>±</mo><mn>0.0040</mn></mrow><annotation encoding="application/x-tex">0.2026 \mathbin{\pm} 0.0040</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery Prediction; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4699</mn><mo>±</mo><mn>0.0190</mn></mrow><annotation encoding="application/x-tex">0.4699 \mathbin{\pm} 0.0190</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery Prediction; Targeted synthetic
only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2206</mn><mo>±</mo><mn>0.0101</mn></mrow><annotation encoding="application/x-tex">0.2206 \mathbin{\pm} 0.0101</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery Prediction; Targeted synthetic
only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5181</mn><mo>±</mo><mn>0.0140</mn></mrow><annotation encoding="application/x-tex">0.5181 \mathbin{\pm} 0.0140</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery Prediction; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0027</mn><mo>±</mo><mn>0.0175</mn></mrow><annotation encoding="application/x-tex">0.0027 \mathbin{\pm} 0.0175</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery Prediction; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0171</mn><mo>±</mo><mn>0.0140</mn></mrow><annotation encoding="application/x-tex">-0.0171 \mathbin{\pm} 0.0140</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery Prediction; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0168</mn><mo>±</mo><mn>0.0040</mn></mrow><annotation encoding="application/x-tex">0.0168 \mathbin{\pm} 0.0040</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery Prediction; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0211</mn><mo>±</mo><mn>0.0190</mn></mrow><annotation encoding="application/x-tex">0.0211 \mathbin{\pm} 0.0190</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6061</mn><mo>±</mo><mn>0.0043</mn></mrow><annotation encoding="application/x-tex">0.6061 \mathbin{\pm} 0.0043</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6556</mn><mo>±</mo><mn>0.0030</mn></mrow><annotation encoding="application/x-tex">0.6556 \mathbin{\pm} 0.0030</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Real + targeted
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6107</mn><mo>±</mo><mn>0.0052</mn></mrow><annotation encoding="application/x-tex">0.6107 \mathbin{\pm} 0.0052</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Real + targeted
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6621</mn><mo>±</mo><mn>0.0044</mn></mrow><annotation encoding="application/x-tex">0.6621 \mathbin{\pm} 0.0044</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Real only
(TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6426</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.6426 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Real only
(TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6902</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.6902 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3975</mn><mo>±</mo><mn>0.0139</mn></mrow><annotation encoding="application/x-tex">0.3975 \mathbin{\pm} 0.0139</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4341</mn><mo>±</mo><mn>0.0209</mn></mrow><annotation encoding="application/x-tex">0.4341 \mathbin{\pm} 0.0209</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Targeted
synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4080</mn><mo>±</mo><mn>0.0160</mn></mrow><annotation encoding="application/x-tex">0.4080 \mathbin{\pm} 0.0160</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Targeted
synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4711</mn><mo>±</mo><mn>0.0428</mn></mrow><annotation encoding="application/x-tex">0.4711 \mathbin{\pm} 0.0428</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Standard
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0365</mn><mo>±</mo><mn>0.0043</mn></mrow><annotation encoding="application/x-tex">-0.0365 \mathbin{\pm} 0.0043</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Standard
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0346</mn><mo>±</mo><mn>0.0030</mn></mrow><annotation encoding="application/x-tex">-0.0346 \mathbin{\pm} 0.0030</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2451</mn><mo>±</mo><mn>0.0139</mn></mrow><annotation encoding="application/x-tex">0.2451 \mathbin{\pm} 0.0139</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2561</mn><mo>±</mo><mn>0.0209</mn></mrow><annotation encoding="application/x-tex">0.2561 \mathbin{\pm} 0.0209</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0105</mn><mo>±</mo><mn>0.0152</mn></mrow><annotation encoding="application/x-tex">0.0105 \mathbin{\pm} 0.0152</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0369</mn><mo>±</mo><mn>0.0278</mn></mrow><annotation encoding="application/x-tex">0.0369 \mathbin{\pm} 0.0278</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Targeted
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0319</mn><mo>±</mo><mn>0.0052</mn></mrow><annotation encoding="application/x-tex">-0.0319 \mathbin{\pm} 0.0052</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline; Targeted
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0281</mn><mo>±</mo><mn>0.0044</mn></mrow><annotation encoding="application/x-tex">-0.0281 \mathbin{\pm} 0.0044</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Real +
standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1083</mn><mo>±</mo><mn>0.0090</mn></mrow><annotation encoding="application/x-tex">0.1083 \mathbin{\pm} 0.0090</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Real +
standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5288</mn><mo>±</mo><mn>0.0017</mn></mrow><annotation encoding="application/x-tex">0.5288 \mathbin{\pm} 0.0017</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Real +
targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1068</mn><mo>±</mo><mn>0.0030</mn></mrow><annotation encoding="application/x-tex">0.1068 \mathbin{\pm} 0.0030</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Real +
targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5390</mn><mo>±</mo><mn>0.0055</mn></mrow><annotation encoding="application/x-tex">0.5390 \mathbin{\pm} 0.0055</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Real only
(TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1132</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.1132 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Real only
(TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5416</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.5416 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0992</mn><mo>±</mo><mn>0.0021</mn></mrow><annotation encoding="application/x-tex">0.0992 \mathbin{\pm} 0.0021</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4726</mn><mo>±</mo><mn>0.0111</mn></mrow><annotation encoding="application/x-tex">0.4726 \mathbin{\pm} 0.0111</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Targeted
synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0918</mn><mo>±</mo><mn>0.0049</mn></mrow><annotation encoding="application/x-tex">0.0918 \mathbin{\pm} 0.0049</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Targeted
synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4718</mn><mo>±</mo><mn>0.0242</mn></mrow><annotation encoding="application/x-tex">0.4718 \mathbin{\pm} 0.0242</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0049</mn><mo>±</mo><mn>0.0090</mn></mrow><annotation encoding="application/x-tex">-0.0049 \mathbin{\pm} 0.0090</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0129</mn><mo>±</mo><mn>0.0017</mn></mrow><annotation encoding="application/x-tex">-0.0129 \mathbin{\pm} 0.0017</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0141</mn><mo>±</mo><mn>0.0021</mn></mrow><annotation encoding="application/x-tex">0.0141 \mathbin{\pm} 0.0021</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0691</mn><mo>±</mo><mn>0.0111</mn></mrow><annotation encoding="application/x-tex">0.0691 \mathbin{\pm} 0.0111</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0073</mn><mo>±</mo><mn>0.0063</mn></mrow><annotation encoding="application/x-tex">-0.0073 \mathbin{\pm} 0.0063</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0008</mn><mo>±</mo><mn>0.0214</mn></mrow><annotation encoding="application/x-tex">-0.0008 \mathbin{\pm} 0.0214</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Targeted
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0064</mn><mo>±</mo><mn>0.0030</mn></mrow><annotation encoding="application/x-tex">-0.0064 \mathbin{\pm} 0.0030</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness; Targeted
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0027</mn><mo>±</mo><mn>0.0055</mn></mrow><annotation encoding="application/x-tex">-0.0027 \mathbin{\pm} 0.0055</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real +
standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3009</mn><mo>±</mo><mn>0.0020</mn></mrow><annotation encoding="application/x-tex">0.3009 \mathbin{\pm} 0.0020</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real +
standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5487</mn><mo>±</mo><mn>0.0005</mn></mrow><annotation encoding="application/x-tex">0.5487 \mathbin{\pm} 0.0005</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real +
targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3010</mn><mo>±</mo><mn>0.0021</mn></mrow><annotation encoding="application/x-tex">0.3010 \mathbin{\pm} 0.0021</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real +
targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5487</mn><mo>±</mo><mn>0.0024</mn></mrow><annotation encoding="application/x-tex">0.5487 \mathbin{\pm} 0.0024</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real only
(TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3092</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.3092 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real only
(TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5619</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.5619 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2962</mn><mo>±</mo><mn>0.0156</mn></mrow><annotation encoding="application/x-tex">0.2962 \mathbin{\pm} 0.0156</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5348</mn><mo>±</mo><mn>0.0233</mn></mrow><annotation encoding="application/x-tex">0.5348 \mathbin{\pm} 0.0233</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Targeted
synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2575</mn><mo>±</mo><mn>0.0058</mn></mrow><annotation encoding="application/x-tex">0.2575 \mathbin{\pm} 0.0058</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Targeted
synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4839</mn><mo>±</mo><mn>0.0125</mn></mrow><annotation encoding="application/x-tex">0.4839 \mathbin{\pm} 0.0125</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0083</mn><mo>±</mo><mn>0.0020</mn></mrow><annotation encoding="application/x-tex">-0.0083 \mathbin{\pm} 0.0020</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0132</mn><mo>±</mo><mn>0.0005</mn></mrow><annotation encoding="application/x-tex">-0.0132 \mathbin{\pm} 0.0005</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0130</mn><mo>±</mo><mn>0.0156</mn></mrow><annotation encoding="application/x-tex">0.0130 \mathbin{\pm} 0.0156</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0271</mn><mo>±</mo><mn>0.0233</mn></mrow><annotation encoding="application/x-tex">0.0271 \mathbin{\pm} 0.0233</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0387</mn><mo>±</mo><mn>0.0098</mn></mrow><annotation encoding="application/x-tex">-0.0387 \mathbin{\pm} 0.0098</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0508</mn><mo>±</mo><mn>0.0120</mn></mrow><annotation encoding="application/x-tex">-0.0508 \mathbin{\pm} 0.0120</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Targeted
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0082</mn><mo>±</mo><mn>0.0021</mn></mrow><annotation encoding="application/x-tex">-0.0082 \mathbin{\pm} 0.0021</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Targeted
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0132</mn><mo>±</mo><mn>0.0024</mn></mrow><annotation encoding="application/x-tex">-0.0132 \mathbin{\pm} 0.0024</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5408</mn><mo>±</mo><mn>0.0099</mn></mrow><annotation encoding="application/x-tex">0.5408 \mathbin{\pm} 0.0099</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5871</mn><mo>±</mo><mn>0.0025</mn></mrow><annotation encoding="application/x-tex">0.5871 \mathbin{\pm} 0.0025</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Real + targeted
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5349</mn><mo>±</mo><mn>0.0082</mn></mrow><annotation encoding="application/x-tex">0.5349 \mathbin{\pm} 0.0082</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Real + targeted
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5921</mn><mo>±</mo><mn>0.0032</mn></mrow><annotation encoding="application/x-tex">0.5921 \mathbin{\pm} 0.0032</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5203</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.5203 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5892</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.5892 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4843</mn><mo>±</mo><mn>0.0154</mn></mrow><annotation encoding="application/x-tex">0.4843 \mathbin{\pm} 0.0154</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5309</mn><mo>±</mo><mn>0.0197</mn></mrow><annotation encoding="application/x-tex">0.5309 \mathbin{\pm} 0.0197</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Targeted synthetic
only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4782</mn><mo>±</mo><mn>0.0078</mn></mrow><annotation encoding="application/x-tex">0.4782 \mathbin{\pm} 0.0078</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Targeted synthetic
only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5251</mn><mo>±</mo><mn>0.0124</mn></mrow><annotation encoding="application/x-tex">0.5251 \mathbin{\pm} 0.0124</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0205</mn><mo>±</mo><mn>0.0099</mn></mrow><annotation encoding="application/x-tex">0.0205 \mathbin{\pm} 0.0099</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0021</mn><mo>±</mo><mn>0.0025</mn></mrow><annotation encoding="application/x-tex">-0.0021 \mathbin{\pm} 0.0025</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0360</mn><mo>±</mo><mn>0.0154</mn></mrow><annotation encoding="application/x-tex">0.0360 \mathbin{\pm} 0.0154</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0583</mn><mo>±</mo><mn>0.0197</mn></mrow><annotation encoding="application/x-tex">0.0583 \mathbin{\pm} 0.0197</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0061</mn><mo>±</mo><mn>0.0186</mn></mrow><annotation encoding="application/x-tex">-0.0061 \mathbin{\pm} 0.0186</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0059</mn><mo>±</mo><mn>0.0185</mn></mrow><annotation encoding="application/x-tex">-0.0059 \mathbin{\pm} 0.0185</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Targeted augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0146</mn><mo>±</mo><mn>0.0082</mn></mrow><annotation encoding="application/x-tex">0.0146 \mathbin{\pm} 0.0082</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path; Targeted augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0028</mn><mo>±</mo><mn>0.0032</mn></mrow><annotation encoding="application/x-tex">0.0028 \mathbin{\pm} 0.0032</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Real + standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7254</mn><mo>±</mo><mn>0.0033</mn></mrow><annotation encoding="application/x-tex">0.7254 \mathbin{\pm} 0.0033</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Real + standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7016</mn><mo>±</mo><mn>0.0034</mn></mrow><annotation encoding="application/x-tex">0.7016 \mathbin{\pm} 0.0034</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Real + targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7296</mn><mo>±</mo><mn>0.0053</mn></mrow><annotation encoding="application/x-tex">0.7296 \mathbin{\pm} 0.0053</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Real + targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7058</mn><mo>±</mo><mn>0.0045</mn></mrow><annotation encoding="application/x-tex">0.7058 \mathbin{\pm} 0.0045</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7117</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7117 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6912</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.6912 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5820</mn><mo>±</mo><mn>0.0102</mn></mrow><annotation encoding="application/x-tex">0.5820 \mathbin{\pm} 0.0102</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5853</mn><mo>±</mo><mn>0.0097</mn></mrow><annotation encoding="application/x-tex">0.5853 \mathbin{\pm} 0.0097</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Targeted synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5731</mn><mo>±</mo><mn>0.0206</mn></mrow><annotation encoding="application/x-tex">0.5731 \mathbin{\pm} 0.0206</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Targeted synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5594</mn><mo>±</mo><mn>0.0360</mn></mrow><annotation encoding="application/x-tex">0.5594 \mathbin{\pm} 0.0360</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Standard augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0137</mn><mo>±</mo><mn>0.0033</mn></mrow><annotation encoding="application/x-tex">0.0137 \mathbin{\pm} 0.0033</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Standard augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0103</mn><mo>±</mo><mn>0.0034</mn></mrow><annotation encoding="application/x-tex">0.0103 \mathbin{\pm} 0.0034</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Standard synthetic utility
loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1297</mn><mo>±</mo><mn>0.0102</mn></mrow><annotation encoding="application/x-tex">0.1297 \mathbin{\pm} 0.0102</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Standard synthetic utility
loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1059</mn><mo>±</mo><mn>0.0097</mn></mrow><annotation encoding="application/x-tex">0.1059 \mathbin{\pm} 0.0097</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0089</mn><mo>±</mo><mn>0.0167</mn></mrow><annotation encoding="application/x-tex">-0.0089 \mathbin{\pm} 0.0167</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0259</mn><mo>±</mo><mn>0.0290</mn></mrow><annotation encoding="application/x-tex">-0.0259 \mathbin{\pm} 0.0290</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Targeted augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0179</mn><mo>±</mo><mn>0.0053</mn></mrow><annotation encoding="application/x-tex">0.0179 \mathbin{\pm} 0.0053</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery; Targeted augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0146</mn><mo>±</mo><mn>0.0045</mn></mrow><annotation encoding="application/x-tex">0.0146 \mathbin{\pm} 0.0045</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0029</mn><mo>±</mo><mn>0.0024</mn></mrow><annotation encoding="application/x-tex">0.0029 \mathbin{\pm} 0.0024</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0017</mn><mo>±</mo><mn>0.0129</mn></mrow><annotation encoding="application/x-tex">0.0017 \mathbin{\pm} 0.0129</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0006</mn><mo>±</mo><mn>0.0005</mn></mrow><annotation encoding="application/x-tex">-0.0006 \mathbin{\pm} 0.0005</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0051</mn><mo>±</mo><mn>0.0036</mn></mrow><annotation encoding="application/x-tex">-0.0051 \mathbin{\pm} 0.0036</annotation></semantics></math>
(2; 1 missing)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0122</mn><mo>±</mo><mn>0.0054</mn></mrow><annotation encoding="application/x-tex">-0.0122 \mathbin{\pm} 0.0054</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0123</mn><mo>±</mo><mn>0.0165</mn></mrow><annotation encoding="application/x-tex">0.0123 \mathbin{\pm} 0.0165</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0007</mn><mo>±</mo><mn>0.0056</mn></mrow><annotation encoding="application/x-tex">-0.0007 \mathbin{\pm} 0.0056</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0117</mn><mo>±</mo><mn>0.0084</mn></mrow><annotation encoding="application/x-tex">0.0117 \mathbin{\pm} 0.0084</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Real + standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7631</mn><mo>±</mo><mn>0.0003</mn></mrow><annotation encoding="application/x-tex">0.7631 \mathbin{\pm} 0.0003</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Real + standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6675</mn><mo>±</mo><mn>0.0003</mn></mrow><annotation encoding="application/x-tex">0.6675 \mathbin{\pm} 0.0003</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Real + targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7634</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.7634 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Real + targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6682</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">0.6682 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7687</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7687 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6722</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.6722 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6045</mn><mo>±</mo><mn>0.0084</mn></mrow><annotation encoding="application/x-tex">0.6045 \mathbin{\pm} 0.0084</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4593</mn><mo>±</mo><mn>0.0184</mn></mrow><annotation encoding="application/x-tex">0.4593 \mathbin{\pm} 0.0184</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6127</mn><mo>±</mo><mn>0.0034</mn></mrow><annotation encoding="application/x-tex">0.6127 \mathbin{\pm} 0.0034</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4633</mn><mo>±</mo><mn>0.0113</mn></mrow><annotation encoding="application/x-tex">0.4633 \mathbin{\pm} 0.0113</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0056</mn><mo>±</mo><mn>0.0003</mn></mrow><annotation encoding="application/x-tex">-0.0056 \mathbin{\pm} 0.0003</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0046</mn><mo>±</mo><mn>0.0003</mn></mrow><annotation encoding="application/x-tex">-0.0046 \mathbin{\pm} 0.0003</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1642</mn><mo>±</mo><mn>0.0084</mn></mrow><annotation encoding="application/x-tex">0.1642 \mathbin{\pm} 0.0084</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2128</mn><mo>±</mo><mn>0.0184</mn></mrow><annotation encoding="application/x-tex">0.2128 \mathbin{\pm} 0.0184</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0082</mn><mo>±</mo><mn>0.0092</mn></mrow><annotation encoding="application/x-tex">0.0082 \mathbin{\pm} 0.0092</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0040</mn><mo>±</mo><mn>0.0200</mn></mrow><annotation encoding="application/x-tex">0.0040 \mathbin{\pm} 0.0200</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0053</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">-0.0053 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0040</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">-0.0040 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Real + standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1334</mn><mo>±</mo><mn>0.0084</mn></mrow><annotation encoding="application/x-tex">0.1334 \mathbin{\pm} 0.0084</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Real + standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6099</mn><mo>±</mo><mn>0.0023</mn></mrow><annotation encoding="application/x-tex">0.6099 \mathbin{\pm} 0.0023</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1310</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.1310 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5859</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.5859 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0025</mn><mo>±</mo><mn>0.0084</mn></mrow><annotation encoding="application/x-tex">0.0025 \mathbin{\pm} 0.0084</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low-Correctness Prediction;
Standard augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0241</mn><mo>±</mo><mn>0.0023</mn></mrow><annotation encoding="application/x-tex">0.0241 \mathbin{\pm} 0.0023</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery Prediction; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2194</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.2194 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery Prediction; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4910</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.4910 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery Prediction; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2194</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.2194 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery Prediction; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4910</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.4910 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery Prediction; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery Prediction; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6089</mn><mo>±</mo><mn>0.0016</mn></mrow><annotation encoding="application/x-tex">0.6089 \mathbin{\pm} 0.0016</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6742</mn><mo>±</mo><mn>0.0003</mn></mrow><annotation encoding="application/x-tex">0.6742 \mathbin{\pm} 0.0003</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Real + targeted
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6095</mn><mo>±</mo><mn>0.0036</mn></mrow><annotation encoding="application/x-tex">0.6095 \mathbin{\pm} 0.0036</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Real + targeted
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6751</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">0.6751 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Real only
(TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6426</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.6426 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Real only
(TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6902</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.6902 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4824</mn><mo>±</mo><mn>0.0294</mn></mrow><annotation encoding="application/x-tex">0.4824 \mathbin{\pm} 0.0294</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5451</mn><mo>±</mo><mn>0.0086</mn></mrow><annotation encoding="application/x-tex">0.5451 \mathbin{\pm} 0.0086</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Targeted
synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3833</mn><mo>±</mo><mn>0.0107</mn></mrow><annotation encoding="application/x-tex">0.3833 \mathbin{\pm} 0.0107</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Targeted
synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4067</mn><mo>±</mo><mn>0.0175</mn></mrow><annotation encoding="application/x-tex">0.4067 \mathbin{\pm} 0.0175</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Standard
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0337</mn><mo>±</mo><mn>0.0016</mn></mrow><annotation encoding="application/x-tex">-0.0337 \mathbin{\pm} 0.0016</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Standard
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0160</mn><mo>±</mo><mn>0.0003</mn></mrow><annotation encoding="application/x-tex">-0.0160 \mathbin{\pm} 0.0003</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1602</mn><mo>±</mo><mn>0.0294</mn></mrow><annotation encoding="application/x-tex">0.1602 \mathbin{\pm} 0.0294</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1451</mn><mo>±</mo><mn>0.0086</mn></mrow><annotation encoding="application/x-tex">0.1451 \mathbin{\pm} 0.0086</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0991</mn><mo>±</mo><mn>0.0216</mn></mrow><annotation encoding="application/x-tex">-0.0991 \mathbin{\pm} 0.0216</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.1385</mn><mo>±</mo><mn>0.0139</mn></mrow><annotation encoding="application/x-tex">-0.1385 \mathbin{\pm} 0.0139</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Targeted
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0331</mn><mo>±</mo><mn>0.0036</mn></mrow><annotation encoding="application/x-tex">-0.0331 \mathbin{\pm} 0.0036</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline; Targeted
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0151</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">-0.0151 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Real +
standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1146</mn><mo>±</mo><mn>0.0014</mn></mrow><annotation encoding="application/x-tex">0.1146 \mathbin{\pm} 0.0014</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Real +
standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5399</mn><mo>±</mo><mn>0.0020</mn></mrow><annotation encoding="application/x-tex">0.5399 \mathbin{\pm} 0.0020</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Real +
targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1162</mn><mo>±</mo><mn>0.0031</mn></mrow><annotation encoding="application/x-tex">0.1162 \mathbin{\pm} 0.0031</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Real +
targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5522</mn><mo>±</mo><mn>0.0046</mn></mrow><annotation encoding="application/x-tex">0.5522 \mathbin{\pm} 0.0046</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Real only
(TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1132</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.1132 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Real only
(TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5416</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.5416 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1048</mn><mo>±</mo><mn>0.0134</mn></mrow><annotation encoding="application/x-tex">0.1048 \mathbin{\pm} 0.0134</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5324</mn><mo>±</mo><mn>0.0355</mn></mrow><annotation encoding="application/x-tex">0.5324 \mathbin{\pm} 0.0355</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Targeted
synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0898</mn><mo>±</mo><mn>0.0053</mn></mrow><annotation encoding="application/x-tex">0.0898 \mathbin{\pm} 0.0053</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Targeted
synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4774</mn><mo>±</mo><mn>0.0172</mn></mrow><annotation encoding="application/x-tex">0.4774 \mathbin{\pm} 0.0172</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0014</mn><mo>±</mo><mn>0.0014</mn></mrow><annotation encoding="application/x-tex">0.0014 \mathbin{\pm} 0.0014</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0018</mn><mo>±</mo><mn>0.0020</mn></mrow><annotation encoding="application/x-tex">-0.0018 \mathbin{\pm} 0.0020</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0084</mn><mo>±</mo><mn>0.0134</mn></mrow><annotation encoding="application/x-tex">0.0084 \mathbin{\pm} 0.0134</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0092</mn><mo>±</mo><mn>0.0355</mn></mrow><annotation encoding="application/x-tex">0.0092 \mathbin{\pm} 0.0355</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0150</mn><mo>±</mo><mn>0.0181</mn></mrow><annotation encoding="application/x-tex">-0.0150 \mathbin{\pm} 0.0181</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0551</mn><mo>±</mo><mn>0.0492</mn></mrow><annotation encoding="application/x-tex">-0.0551 \mathbin{\pm} 0.0492</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Targeted
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0030</mn><mo>±</mo><mn>0.0031</mn></mrow><annotation encoding="application/x-tex">0.0030 \mathbin{\pm} 0.0031</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness; Targeted
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0105</mn><mo>±</mo><mn>0.0046</mn></mrow><annotation encoding="application/x-tex">0.0105 \mathbin{\pm} 0.0046</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real +
standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3042</mn><mo>±</mo><mn>0.0016</mn></mrow><annotation encoding="application/x-tex">0.3042 \mathbin{\pm} 0.0016</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real +
standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5571</mn><mo>±</mo><mn>0.0025</mn></mrow><annotation encoding="application/x-tex">0.5571 \mathbin{\pm} 0.0025</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real +
targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3053</mn><mo>±</mo><mn>0.0014</mn></mrow><annotation encoding="application/x-tex">0.3053 \mathbin{\pm} 0.0014</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real +
targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5602</mn><mo>±</mo><mn>0.0016</mn></mrow><annotation encoding="application/x-tex">0.5602 \mathbin{\pm} 0.0016</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real only
(TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3092</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.3092 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Real only
(TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5619</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.5619 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2823</mn><mo>±</mo><mn>0.0059</mn></mrow><annotation encoding="application/x-tex">0.2823 \mathbin{\pm} 0.0059</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5352</mn><mo>±</mo><mn>0.0102</mn></mrow><annotation encoding="application/x-tex">0.5352 \mathbin{\pm} 0.0102</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Targeted
synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2627</mn><mo>±</mo><mn>0.0043</mn></mrow><annotation encoding="application/x-tex">0.2627 \mathbin{\pm} 0.0043</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Targeted
synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4904</mn><mo>±</mo><mn>0.0090</mn></mrow><annotation encoding="application/x-tex">0.4904 \mathbin{\pm} 0.0090</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0050</mn><mo>±</mo><mn>0.0016</mn></mrow><annotation encoding="application/x-tex">-0.0050 \mathbin{\pm} 0.0016</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0048</mn><mo>±</mo><mn>0.0025</mn></mrow><annotation encoding="application/x-tex">-0.0048 \mathbin{\pm} 0.0025</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0269</mn><mo>±</mo><mn>0.0059</mn></mrow><annotation encoding="application/x-tex">0.0269 \mathbin{\pm} 0.0059</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0267</mn><mo>±</mo><mn>0.0102</mn></mrow><annotation encoding="application/x-tex">0.0267 \mathbin{\pm} 0.0102</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0196</mn><mo>±</mo><mn>0.0101</mn></mrow><annotation encoding="application/x-tex">-0.0196 \mathbin{\pm} 0.0101</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding;
Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0448</mn><mo>±</mo><mn>0.0190</mn></mrow><annotation encoding="application/x-tex">-0.0448 \mathbin{\pm} 0.0190</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Targeted
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0039</mn><mo>±</mo><mn>0.0014</mn></mrow><annotation encoding="application/x-tex">-0.0039 \mathbin{\pm} 0.0014</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding; Targeted
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0017</mn><mo>±</mo><mn>0.0016</mn></mrow><annotation encoding="application/x-tex">-0.0017 \mathbin{\pm} 0.0016</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5320</mn><mo>±</mo><mn>0.0048</mn></mrow><annotation encoding="application/x-tex">0.5320 \mathbin{\pm} 0.0048</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5826</mn><mo>±</mo><mn>0.0029</mn></mrow><annotation encoding="application/x-tex">0.5826 \mathbin{\pm} 0.0029</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Real + targeted
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5323</mn><mo>±</mo><mn>0.0043</mn></mrow><annotation encoding="application/x-tex">0.5323 \mathbin{\pm} 0.0043</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Real + targeted
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5868</mn><mo>±</mo><mn>0.0054</mn></mrow><annotation encoding="application/x-tex">0.5868 \mathbin{\pm} 0.0054</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5203</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.5203 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5892</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.5892 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4560</mn><mo>±</mo><mn>0.0178</mn></mrow><annotation encoding="application/x-tex">0.4560 \mathbin{\pm} 0.0178</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4766</mn><mo>±</mo><mn>0.0214</mn></mrow><annotation encoding="application/x-tex">0.4766 \mathbin{\pm} 0.0214</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Targeted synthetic
only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4759</mn><mo>±</mo><mn>0.0100</mn></mrow><annotation encoding="application/x-tex">0.4759 \mathbin{\pm} 0.0100</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Targeted synthetic
only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5215</mn><mo>±</mo><mn>0.0142</mn></mrow><annotation encoding="application/x-tex">0.5215 \mathbin{\pm} 0.0142</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0117</mn><mo>±</mo><mn>0.0048</mn></mrow><annotation encoding="application/x-tex">0.0117 \mathbin{\pm} 0.0048</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0066</mn><mo>±</mo><mn>0.0029</mn></mrow><annotation encoding="application/x-tex">-0.0066 \mathbin{\pm} 0.0029</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0643</mn><mo>±</mo><mn>0.0178</mn></mrow><annotation encoding="application/x-tex">0.0643 \mathbin{\pm} 0.0178</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1126</mn><mo>±</mo><mn>0.0214</mn></mrow><annotation encoding="application/x-tex">0.1126 \mathbin{\pm} 0.0214</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0198</mn><mo>±</mo><mn>0.0156</mn></mrow><annotation encoding="application/x-tex">0.0198 \mathbin{\pm} 0.0156</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0449</mn><mo>±</mo><mn>0.0310</mn></mrow><annotation encoding="application/x-tex">0.0449 \mathbin{\pm} 0.0310</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Targeted augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0120</mn><mo>±</mo><mn>0.0043</mn></mrow><annotation encoding="application/x-tex">0.0120 \mathbin{\pm} 0.0043</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path; Targeted augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0025</mn><mo>±</mo><mn>0.0054</mn></mrow><annotation encoding="application/x-tex">-0.0025 \mathbin{\pm} 0.0054</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Real + standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7074</mn><mo>±</mo><mn>0.0012</mn></mrow><annotation encoding="application/x-tex">0.7074 \mathbin{\pm} 0.0012</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Real + standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6890</mn><mo>±</mo><mn>0.0028</mn></mrow><annotation encoding="application/x-tex">0.6890 \mathbin{\pm} 0.0028</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Real + targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7146</mn><mo>±</mo><mn>0.0007</mn></mrow><annotation encoding="application/x-tex">0.7146 \mathbin{\pm} 0.0007</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Real + targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6964</mn><mo>±</mo><mn>0.0004</mn></mrow><annotation encoding="application/x-tex">0.6964 \mathbin{\pm} 0.0004</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7117</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7117 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6912</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.6912 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5143</mn><mo>±</mo><mn>0.0159</mn></mrow><annotation encoding="application/x-tex">0.5143 \mathbin{\pm} 0.0159</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4371</mn><mo>±</mo><mn>0.0233</mn></mrow><annotation encoding="application/x-tex">0.4371 \mathbin{\pm} 0.0233</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Targeted synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5762</mn><mo>±</mo><mn>0.0282</mn></mrow><annotation encoding="application/x-tex">0.5762 \mathbin{\pm} 0.0282</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Targeted synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5583</mn><mo>±</mo><mn>0.0407</mn></mrow><annotation encoding="application/x-tex">0.5583 \mathbin{\pm} 0.0407</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Standard augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0043</mn><mo>±</mo><mn>0.0012</mn></mrow><annotation encoding="application/x-tex">-0.0043 \mathbin{\pm} 0.0012</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Standard augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0022</mn><mo>±</mo><mn>0.0028</mn></mrow><annotation encoding="application/x-tex">-0.0022 \mathbin{\pm} 0.0028</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Standard synthetic utility
loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1974</mn><mo>±</mo><mn>0.0159</mn></mrow><annotation encoding="application/x-tex">0.1974 \mathbin{\pm} 0.0159</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Standard synthetic utility
loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2542</mn><mo>±</mo><mn>0.0233</mn></mrow><annotation encoding="application/x-tex">0.2542 \mathbin{\pm} 0.0233</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0619</mn><mo>±</mo><mn>0.0232</mn></mrow><annotation encoding="application/x-tex">0.0619 \mathbin{\pm} 0.0232</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1212</mn><mo>±</mo><mn>0.0313</mn></mrow><annotation encoding="application/x-tex">0.1212 \mathbin{\pm} 0.0313</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Targeted augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0029</mn><mo>±</mo><mn>0.0007</mn></mrow><annotation encoding="application/x-tex">0.0029 \mathbin{\pm} 0.0007</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery; Targeted augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0051</mn><mo>±</mo><mn>0.0004</mn></mrow><annotation encoding="application/x-tex">0.0051 \mathbin{\pm} 0.0004</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0172</mn><mo>±</mo><mn>0.0070</mn></mrow><annotation encoding="application/x-tex">-0.0172 \mathbin{\pm} 0.0070</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0004</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">-0.0004 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Real + standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9179</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.9179 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Real + standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8802</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.8802 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Real + targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9181</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.9181 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Real + targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8805</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">0.8805 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9225</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.9225 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8872</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.8872 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8991</mn><mo>±</mo><mn>0.0006</mn></mrow><annotation encoding="application/x-tex">0.8991 \mathbin{\pm} 0.0006</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8572</mn><mo>±</mo><mn>0.0004</mn></mrow><annotation encoding="application/x-tex">0.8572 \mathbin{\pm} 0.0004</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9009</mn><mo>±</mo><mn>0.0003</mn></mrow><annotation encoding="application/x-tex">0.9009 \mathbin{\pm} 0.0003</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8589</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">0.8589 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0046</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">-0.0046 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0070</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">-0.0070 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0234</mn><mo>±</mo><mn>0.0006</mn></mrow><annotation encoding="application/x-tex">0.0234 \mathbin{\pm} 0.0006</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0301</mn><mo>±</mo><mn>0.0004</mn></mrow><annotation encoding="application/x-tex">0.0301 \mathbin{\pm} 0.0004</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0018</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">0.0018 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0017</mn><mo>±</mo><mn>0.0005</mn></mrow><annotation encoding="application/x-tex">0.0017 \mathbin{\pm} 0.0005</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0044</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">-0.0044 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0068</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">-0.0068 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Course Failure Prediction; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5450</mn><mo>±</mo><mn>0.0027</mn></mrow><annotation encoding="application/x-tex">0.5450 \mathbin{\pm} 0.0027</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Course Failure Prediction; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7000</mn><mo>±</mo><mn>0.0016</mn></mrow><annotation encoding="application/x-tex">0.7000 \mathbin{\pm} 0.0016</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Course Failure Prediction; Real only
(TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5467</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.5467 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Course Failure Prediction; Real only
(TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7034</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7034 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Course Failure Prediction; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5261</mn><mo>±</mo><mn>0.0044</mn></mrow><annotation encoding="application/x-tex">0.5261 \mathbin{\pm} 0.0044</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Course Failure Prediction; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6826</mn><mo>±</mo><mn>0.0028</mn></mrow><annotation encoding="application/x-tex">0.6826 \mathbin{\pm} 0.0028</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Course Failure Prediction; Targeted
synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5234</mn><mo>±</mo><mn>0.0040</mn></mrow><annotation encoding="application/x-tex">0.5234 \mathbin{\pm} 0.0040</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Course Failure Prediction; Targeted
synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6815</mn><mo>±</mo><mn>0.0043</mn></mrow><annotation encoding="application/x-tex">0.6815 \mathbin{\pm} 0.0043</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Course Failure Prediction; Standard
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0017</mn><mo>±</mo><mn>0.0027</mn></mrow><annotation encoding="application/x-tex">-0.0017 \mathbin{\pm} 0.0027</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Course Failure Prediction; Standard
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0034</mn><mo>±</mo><mn>0.0016</mn></mrow><annotation encoding="application/x-tex">-0.0034 \mathbin{\pm} 0.0016</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Course Failure Prediction; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0206</mn><mo>±</mo><mn>0.0044</mn></mrow><annotation encoding="application/x-tex">0.0206 \mathbin{\pm} 0.0044</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Course Failure Prediction; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0208</mn><mo>±</mo><mn>0.0028</mn></mrow><annotation encoding="application/x-tex">0.0208 \mathbin{\pm} 0.0028</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Dropout Prediction; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2963</mn><mo>±</mo><mn>0.0010</mn></mrow><annotation encoding="application/x-tex">0.2963 \mathbin{\pm} 0.0010</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Dropout Prediction; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6444</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">0.6444 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Dropout Prediction; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3074</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.3074 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Dropout Prediction; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6526</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.6526 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Dropout Prediction; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2753</mn><mo>±</mo><mn>0.0019</mn></mrow><annotation encoding="application/x-tex">0.2753 \mathbin{\pm} 0.0019</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Dropout Prediction; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6156</mn><mo>±</mo><mn>0.0013</mn></mrow><annotation encoding="application/x-tex">0.6156 \mathbin{\pm} 0.0013</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Dropout Prediction; Targeted synthetic
only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2813</mn><mo>±</mo><mn>0.0018</mn></mrow><annotation encoding="application/x-tex">0.2813 \mathbin{\pm} 0.0018</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Dropout Prediction; Targeted synthetic
only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6225</mn><mo>±</mo><mn>0.0006</mn></mrow><annotation encoding="application/x-tex">0.6225 \mathbin{\pm} 0.0006</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Dropout Prediction; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0111</mn><mo>±</mo><mn>0.0010</mn></mrow><annotation encoding="application/x-tex">-0.0111 \mathbin{\pm} 0.0010</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Dropout Prediction; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0082</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">-0.0082 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Dropout Prediction; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0321</mn><mo>±</mo><mn>0.0019</mn></mrow><annotation encoding="application/x-tex">0.0321 \mathbin{\pm} 0.0019</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Dropout Prediction; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0370</mn><mo>±</mo><mn>0.0013</mn></mrow><annotation encoding="application/x-tex">0.0370 \mathbin{\pm} 0.0013</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Disengagement; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8200</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.8200 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Disengagement; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8477</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.8477 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Disengagement; Real + targeted
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8189</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.8189 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Disengagement; Real + targeted
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8467</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">0.8467 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Disengagement; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8294</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.8294 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Disengagement; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8542</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.8542 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Disengagement; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7986</mn><mo>±</mo><mn>0.0008</mn></mrow><annotation encoding="application/x-tex">0.7986 \mathbin{\pm} 0.0008</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Disengagement; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8322</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">0.8322 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Disengagement; Targeted synthetic
only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8004</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">0.8004 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Disengagement; Targeted synthetic
only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8319</mn><mo>±</mo><mn>0.0006</mn></mrow><annotation encoding="application/x-tex">0.8319 \mathbin{\pm} 0.0006</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Disengagement; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0094</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">-0.0094 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Disengagement; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0066</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">-0.0066 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Disengagement; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0307</mn><mo>±</mo><mn>0.0008</mn></mrow><annotation encoding="application/x-tex">0.0307 \mathbin{\pm} 0.0008</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Disengagement; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0220</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">0.0220 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Disengagement; Targeted-vs-standard
TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0018</mn><mo>±</mo><mn>0.0017</mn></mrow><annotation encoding="application/x-tex">0.0018 \mathbin{\pm} 0.0017</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Disengagement; Targeted-vs-standard
TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0003</mn><mo>±</mo><mn>0.0007</mn></mrow><annotation encoding="application/x-tex">-0.0003 \mathbin{\pm} 0.0007</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Disengagement; Targeted augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0105</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">-0.0105 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Disengagement; Targeted augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0075</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">-0.0075 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Activity Path; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9534</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">0.9534 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Activity Path; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7968</mn><mo>±</mo><mn>0.0007</mn></mrow><annotation encoding="application/x-tex">0.7968 \mathbin{\pm} 0.0007</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Activity Path; Real + targeted
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9524</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.9524 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Activity Path; Real + targeted
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7918</mn><mo>±</mo><mn>0.0007</mn></mrow><annotation encoding="application/x-tex">0.7918 \mathbin{\pm} 0.0007</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Activity Path; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9562</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.9562 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Activity Path; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8097</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.8097 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Activity Path; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9349</mn><mo>±</mo><mn>0.0014</mn></mrow><annotation encoding="application/x-tex">0.9349 \mathbin{\pm} 0.0014</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Activity Path; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7281</mn><mo>±</mo><mn>0.0044</mn></mrow><annotation encoding="application/x-tex">0.7281 \mathbin{\pm} 0.0044</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Activity Path; Targeted synthetic
only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9350</mn><mo>±</mo><mn>0.0008</mn></mrow><annotation encoding="application/x-tex">0.9350 \mathbin{\pm} 0.0008</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Activity Path; Targeted synthetic
only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7279</mn><mo>±</mo><mn>0.0015</mn></mrow><annotation encoding="application/x-tex">0.7279 \mathbin{\pm} 0.0015</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Activity Path; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0028</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">-0.0028 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Activity Path; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0129</mn><mo>±</mo><mn>0.0007</mn></mrow><annotation encoding="application/x-tex">-0.0129 \mathbin{\pm} 0.0007</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Activity Path; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0213</mn><mo>±</mo><mn>0.0014</mn></mrow><annotation encoding="application/x-tex">0.0213 \mathbin{\pm} 0.0014</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Activity Path; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0816</mn><mo>±</mo><mn>0.0044</mn></mrow><annotation encoding="application/x-tex">0.0816 \mathbin{\pm} 0.0044</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Activity Path; Targeted-vs-standard
TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0002</mn><mo>±</mo><mn>0.0020</mn></mrow><annotation encoding="application/x-tex">0.0002 \mathbin{\pm} 0.0020</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Activity Path; Targeted-vs-standard
TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0002</mn><mo>±</mo><mn>0.0052</mn></mrow><annotation encoding="application/x-tex">-0.0002 \mathbin{\pm} 0.0052</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Activity Path; Targeted augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0037</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">-0.0037 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Activity Path; Targeted augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0179</mn><mo>±</mo><mn>0.0007</mn></mrow><annotation encoding="application/x-tex">-0.0179 \mathbin{\pm} 0.0007</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Reengagement; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7599</mn><mo>±</mo><mn>0.0021</mn></mrow><annotation encoding="application/x-tex">0.7599 \mathbin{\pm} 0.0021</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Reengagement; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7462</mn><mo>±</mo><mn>0.0017</mn></mrow><annotation encoding="application/x-tex">0.7462 \mathbin{\pm} 0.0017</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Reengagement; Real + targeted
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7669</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">0.7669 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Reengagement; Real + targeted
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7489</mn><mo>±</mo><mn>0.0004</mn></mrow><annotation encoding="application/x-tex">0.7489 \mathbin{\pm} 0.0004</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Reengagement; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7060</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7060 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Reengagement; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7062</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7062 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Reengagement; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7503</mn><mo>±</mo><mn>0.0034</mn></mrow><annotation encoding="application/x-tex">0.7503 \mathbin{\pm} 0.0034</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Reengagement; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7459</mn><mo>±</mo><mn>0.0025</mn></mrow><annotation encoding="application/x-tex">0.7459 \mathbin{\pm} 0.0025</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Reengagement; Targeted synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7553</mn><mo>±</mo><mn>0.0015</mn></mrow><annotation encoding="application/x-tex">0.7553 \mathbin{\pm} 0.0015</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Reengagement; Targeted synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7463</mn><mo>±</mo><mn>0.0014</mn></mrow><annotation encoding="application/x-tex">0.7463 \mathbin{\pm} 0.0014</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Reengagement; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0539</mn><mo>±</mo><mn>0.0021</mn></mrow><annotation encoding="application/x-tex">0.0539 \mathbin{\pm} 0.0021</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Reengagement; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0399</mn><mo>±</mo><mn>0.0017</mn></mrow><annotation encoding="application/x-tex">0.0399 \mathbin{\pm} 0.0017</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Reengagement; Standard synthetic utility
loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0443</mn><mo>±</mo><mn>0.0034</mn></mrow><annotation encoding="application/x-tex">-0.0443 \mathbin{\pm} 0.0034</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Reengagement; Standard synthetic utility
loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0397</mn><mo>±</mo><mn>0.0025</mn></mrow><annotation encoding="application/x-tex">-0.0397 \mathbin{\pm} 0.0025</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Reengagement; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0050</mn><mo>±</mo><mn>0.0022</mn></mrow><annotation encoding="application/x-tex">0.0050 \mathbin{\pm} 0.0022</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Reengagement; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0004</mn><mo>±</mo><mn>0.0016</mn></mrow><annotation encoding="application/x-tex">0.0004 \mathbin{\pm} 0.0016</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Reengagement; Targeted augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0610</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">0.0610 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Reengagement; Targeted augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0426</mn><mo>±</mo><mn>0.0004</mn></mrow><annotation encoding="application/x-tex">0.0426 \mathbin{\pm} 0.0004</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Disengagement</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0119</mn><mo>±</mo><mn>0.0011</mn></mrow><annotation encoding="application/x-tex">0.0119 \mathbin{\pm} 0.0011</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Disengagement</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0014</mn><mo>±</mo><mn>0.0056</mn></mrow><annotation encoding="application/x-tex">-0.0014 \mathbin{\pm} 0.0056</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Activity Path</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0347</mn><mo>±</mo><mn>0.0030</mn></mrow><annotation encoding="application/x-tex">-0.0347 \mathbin{\pm} 0.0030</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Activity Path</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0006</mn><mo>±</mo><mn>0.0050</mn></mrow><annotation encoding="application/x-tex">-0.0006 \mathbin{\pm} 0.0050</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Reengagement</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0052</mn><mo>±</mo><mn>0.0007</mn></mrow><annotation encoding="application/x-tex">0.0052 \mathbin{\pm} 0.0007</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Reengagement</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0015</mn><mo>±</mo><mn>0.0061</mn></mrow><annotation encoding="application/x-tex">-0.0015 \mathbin{\pm} 0.0061</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Real + standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9179</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">0.9179 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Real + standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8809</mn><mo>±</mo><mn>0.0003</mn></mrow><annotation encoding="application/x-tex">0.8809 \mathbin{\pm} 0.0003</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Real + targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9181</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.9181 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Real + targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8810</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.8810 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9225</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.9225 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8872</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.8872 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8952</mn><mo>±</mo><mn>0.0006</mn></mrow><annotation encoding="application/x-tex">0.8952 \mathbin{\pm} 0.0006</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8529</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">0.8529 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8966</mn><mo>±</mo><mn>0.0008</mn></mrow><annotation encoding="application/x-tex">0.8966 \mathbin{\pm} 0.0008</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8543</mn><mo>±</mo><mn>0.0008</mn></mrow><annotation encoding="application/x-tex">0.8543 \mathbin{\pm} 0.0008</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0046</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">-0.0046 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0063</mn><mo>±</mo><mn>0.0003</mn></mrow><annotation encoding="application/x-tex">-0.0063 \mathbin{\pm} 0.0003</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0273</mn><mo>±</mo><mn>0.0006</mn></mrow><annotation encoding="application/x-tex">0.0273 \mathbin{\pm} 0.0006</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0344</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">0.0344 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0014</mn><mo>±</mo><mn>0.0012</mn></mrow><annotation encoding="application/x-tex">0.0014 \mathbin{\pm} 0.0012</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0014</mn><mo>±</mo><mn>0.0008</mn></mrow><annotation encoding="application/x-tex">0.0014 \mathbin{\pm} 0.0008</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0044</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">-0.0044 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0062</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">-0.0062 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Course Failure Prediction; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5451</mn><mo>±</mo><mn>0.0020</mn></mrow><annotation encoding="application/x-tex">0.5451 \mathbin{\pm} 0.0020</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Course Failure Prediction; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7019</mn><mo>±</mo><mn>0.0016</mn></mrow><annotation encoding="application/x-tex">0.7019 \mathbin{\pm} 0.0016</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Course Failure Prediction; Real only
(TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5467</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.5467 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Course Failure Prediction; Real only
(TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7034</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7034 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Course Failure Prediction; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5277</mn><mo>±</mo><mn>0.0029</mn></mrow><annotation encoding="application/x-tex">0.5277 \mathbin{\pm} 0.0029</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Course Failure Prediction; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6876</mn><mo>±</mo><mn>0.0026</mn></mrow><annotation encoding="application/x-tex">0.6876 \mathbin{\pm} 0.0026</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Course Failure Prediction; Targeted
synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5302</mn><mo>±</mo><mn>0.0024</mn></mrow><annotation encoding="application/x-tex">0.5302 \mathbin{\pm} 0.0024</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Course Failure Prediction; Targeted
synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6873</mn><mo>±</mo><mn>0.0051</mn></mrow><annotation encoding="application/x-tex">0.6873 \mathbin{\pm} 0.0051</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Course Failure Prediction; Standard
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0015</mn><mo>±</mo><mn>0.0020</mn></mrow><annotation encoding="application/x-tex">-0.0015 \mathbin{\pm} 0.0020</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Course Failure Prediction; Standard
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0015</mn><mo>±</mo><mn>0.0016</mn></mrow><annotation encoding="application/x-tex">-0.0015 \mathbin{\pm} 0.0016</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Course Failure Prediction; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0190</mn><mo>±</mo><mn>0.0029</mn></mrow><annotation encoding="application/x-tex">0.0190 \mathbin{\pm} 0.0029</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Course Failure Prediction; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0158</mn><mo>±</mo><mn>0.0026</mn></mrow><annotation encoding="application/x-tex">0.0158 \mathbin{\pm} 0.0026</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Dropout Prediction; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3013</mn><mo>±</mo><mn>0.0015</mn></mrow><annotation encoding="application/x-tex">0.3013 \mathbin{\pm} 0.0015</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Dropout Prediction; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6512</mn><mo>±</mo><mn>0.0025</mn></mrow><annotation encoding="application/x-tex">0.6512 \mathbin{\pm} 0.0025</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Dropout Prediction; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3074</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.3074 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Dropout Prediction; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6526</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.6526 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Dropout Prediction; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2802</mn><mo>±</mo><mn>0.0034</mn></mrow><annotation encoding="application/x-tex">0.2802 \mathbin{\pm} 0.0034</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Dropout Prediction; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6271</mn><mo>±</mo><mn>0.0068</mn></mrow><annotation encoding="application/x-tex">0.6271 \mathbin{\pm} 0.0068</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Dropout Prediction; Targeted synthetic
only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2802</mn><mo>±</mo><mn>0.0037</mn></mrow><annotation encoding="application/x-tex">0.2802 \mathbin{\pm} 0.0037</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Dropout Prediction; Targeted synthetic
only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6276</mn><mo>±</mo><mn>0.0011</mn></mrow><annotation encoding="application/x-tex">0.6276 \mathbin{\pm} 0.0011</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Dropout Prediction; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0061</mn><mo>±</mo><mn>0.0015</mn></mrow><annotation encoding="application/x-tex">-0.0061 \mathbin{\pm} 0.0015</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Dropout Prediction; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0014</mn><mo>±</mo><mn>0.0025</mn></mrow><annotation encoding="application/x-tex">-0.0014 \mathbin{\pm} 0.0025</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Dropout Prediction; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0272</mn><mo>±</mo><mn>0.0034</mn></mrow><annotation encoding="application/x-tex">0.0272 \mathbin{\pm} 0.0034</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Dropout Prediction; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0254</mn><mo>±</mo><mn>0.0068</mn></mrow><annotation encoding="application/x-tex">0.0254 \mathbin{\pm} 0.0068</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Disengagement; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8233</mn><mo>±</mo><mn>0.0006</mn></mrow><annotation encoding="application/x-tex">0.8233 \mathbin{\pm} 0.0006</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Disengagement; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8491</mn><mo>±</mo><mn>0.0004</mn></mrow><annotation encoding="application/x-tex">0.8491 \mathbin{\pm} 0.0004</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Disengagement; Real + targeted
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8228</mn><mo>±</mo><mn>0.0006</mn></mrow><annotation encoding="application/x-tex">0.8228 \mathbin{\pm} 0.0006</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Disengagement; Real + targeted
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8494</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">0.8494 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Disengagement; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8294</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.8294 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Disengagement; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8542</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.8542 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Disengagement; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8037</mn><mo>±</mo><mn>0.0017</mn></mrow><annotation encoding="application/x-tex">0.8037 \mathbin{\pm} 0.0017</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Disengagement; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8289</mn><mo>±</mo><mn>0.0014</mn></mrow><annotation encoding="application/x-tex">0.8289 \mathbin{\pm} 0.0014</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Disengagement; Targeted synthetic
only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8034</mn><mo>±</mo><mn>0.0011</mn></mrow><annotation encoding="application/x-tex">0.8034 \mathbin{\pm} 0.0011</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Disengagement; Targeted synthetic
only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8292</mn><mo>±</mo><mn>0.0005</mn></mrow><annotation encoding="application/x-tex">0.8292 \mathbin{\pm} 0.0005</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Disengagement; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0061</mn><mo>±</mo><mn>0.0006</mn></mrow><annotation encoding="application/x-tex">-0.0061 \mathbin{\pm} 0.0006</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Disengagement; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0051</mn><mo>±</mo><mn>0.0004</mn></mrow><annotation encoding="application/x-tex">-0.0051 \mathbin{\pm} 0.0004</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Disengagement; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0256</mn><mo>±</mo><mn>0.0017</mn></mrow><annotation encoding="application/x-tex">0.0256 \mathbin{\pm} 0.0017</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Disengagement; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0254</mn><mo>±</mo><mn>0.0014</mn></mrow><annotation encoding="application/x-tex">0.0254 \mathbin{\pm} 0.0014</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Disengagement; Targeted-vs-standard
TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0004</mn><mo>±</mo><mn>0.0025</mn></mrow><annotation encoding="application/x-tex">-0.0004 \mathbin{\pm} 0.0025</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Disengagement; Targeted-vs-standard
TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0003</mn><mo>±</mo><mn>0.0008</mn></mrow><annotation encoding="application/x-tex">0.0003 \mathbin{\pm} 0.0008</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Disengagement; Targeted augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0066</mn><mo>±</mo><mn>0.0006</mn></mrow><annotation encoding="application/x-tex">-0.0066 \mathbin{\pm} 0.0006</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Disengagement; Targeted augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0048</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">-0.0048 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Activity Path; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9533</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">0.9533 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Activity Path; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7998</mn><mo>±</mo><mn>0.0008</mn></mrow><annotation encoding="application/x-tex">0.7998 \mathbin{\pm} 0.0008</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Activity Path; Real + targeted
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9531</mn><mo>±</mo><mn>0.0003</mn></mrow><annotation encoding="application/x-tex">0.9531 \mathbin{\pm} 0.0003</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Activity Path; Real + targeted
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7990</mn><mo>±</mo><mn>0.0010</mn></mrow><annotation encoding="application/x-tex">0.7990 \mathbin{\pm} 0.0010</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Activity Path; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9562</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.9562 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Activity Path; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8097</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.8097 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Activity Path; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9397</mn><mo>±</mo><mn>0.0004</mn></mrow><annotation encoding="application/x-tex">0.9397 \mathbin{\pm} 0.0004</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Activity Path; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7454</mn><mo>±</mo><mn>0.0008</mn></mrow><annotation encoding="application/x-tex">0.7454 \mathbin{\pm} 0.0008</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Activity Path; Targeted synthetic
only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9403</mn><mo>±</mo><mn>0.0014</mn></mrow><annotation encoding="application/x-tex">0.9403 \mathbin{\pm} 0.0014</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Activity Path; Targeted synthetic
only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7460</mn><mo>±</mo><mn>0.0048</mn></mrow><annotation encoding="application/x-tex">0.7460 \mathbin{\pm} 0.0048</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Activity Path; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0029</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">-0.0029 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Activity Path; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0099</mn><mo>±</mo><mn>0.0008</mn></mrow><annotation encoding="application/x-tex">-0.0099 \mathbin{\pm} 0.0008</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Activity Path; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0165</mn><mo>±</mo><mn>0.0004</mn></mrow><annotation encoding="application/x-tex">0.0165 \mathbin{\pm} 0.0004</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Activity Path; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0643</mn><mo>±</mo><mn>0.0008</mn></mrow><annotation encoding="application/x-tex">0.0643 \mathbin{\pm} 0.0008</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Activity Path; Targeted-vs-standard
TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0006</mn><mo>±</mo><mn>0.0012</mn></mrow><annotation encoding="application/x-tex">0.0006 \mathbin{\pm} 0.0012</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Activity Path; Targeted-vs-standard
TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0005</mn><mo>±</mo><mn>0.0040</mn></mrow><annotation encoding="application/x-tex">0.0005 \mathbin{\pm} 0.0040</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Activity Path; Targeted augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0030</mn><mo>±</mo><mn>0.0003</mn></mrow><annotation encoding="application/x-tex">-0.0030 \mathbin{\pm} 0.0003</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Activity Path; Targeted augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0107</mn><mo>±</mo><mn>0.0010</mn></mrow><annotation encoding="application/x-tex">-0.0107 \mathbin{\pm} 0.0010</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Reengagement; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7246</mn><mo>±</mo><mn>0.0040</mn></mrow><annotation encoding="application/x-tex">0.7246 \mathbin{\pm} 0.0040</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Reengagement; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7154</mn><mo>±</mo><mn>0.0028</mn></mrow><annotation encoding="application/x-tex">0.7154 \mathbin{\pm} 0.0028</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Reengagement; Real + targeted
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7327</mn><mo>±</mo><mn>0.0055</mn></mrow><annotation encoding="application/x-tex">0.7327 \mathbin{\pm} 0.0055</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Reengagement; Real + targeted
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7208</mn><mo>±</mo><mn>0.0030</mn></mrow><annotation encoding="application/x-tex">0.7208 \mathbin{\pm} 0.0030</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Reengagement; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7060</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7060 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Reengagement; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7062</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7062 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Reengagement; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6964</mn><mo>±</mo><mn>0.0025</mn></mrow><annotation encoding="application/x-tex">0.6964 \mathbin{\pm} 0.0025</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Reengagement; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6982</mn><mo>±</mo><mn>0.0031</mn></mrow><annotation encoding="application/x-tex">0.6982 \mathbin{\pm} 0.0031</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Reengagement; Targeted synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6997</mn><mo>±</mo><mn>0.0053</mn></mrow><annotation encoding="application/x-tex">0.6997 \mathbin{\pm} 0.0053</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Reengagement; Targeted synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7008</mn><mo>±</mo><mn>0.0068</mn></mrow><annotation encoding="application/x-tex">0.7008 \mathbin{\pm} 0.0068</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Reengagement; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0186</mn><mo>±</mo><mn>0.0040</mn></mrow><annotation encoding="application/x-tex">0.0186 \mathbin{\pm} 0.0040</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Reengagement; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0091</mn><mo>±</mo><mn>0.0028</mn></mrow><annotation encoding="application/x-tex">0.0091 \mathbin{\pm} 0.0028</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Reengagement; Standard synthetic utility
loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0096</mn><mo>±</mo><mn>0.0025</mn></mrow><annotation encoding="application/x-tex">0.0096 \mathbin{\pm} 0.0025</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Reengagement; Standard synthetic utility
loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0080</mn><mo>±</mo><mn>0.0031</mn></mrow><annotation encoding="application/x-tex">0.0080 \mathbin{\pm} 0.0031</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Reengagement; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0033</mn><mo>±</mo><mn>0.0033</mn></mrow><annotation encoding="application/x-tex">0.0033 \mathbin{\pm} 0.0033</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Reengagement; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0026</mn><mo>±</mo><mn>0.0038</mn></mrow><annotation encoding="application/x-tex">0.0026 \mathbin{\pm} 0.0038</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Reengagement; Targeted augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0267</mn><mo>±</mo><mn>0.0055</mn></mrow><annotation encoding="application/x-tex">0.0267 \mathbin{\pm} 0.0055</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Reengagement; Targeted augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0146</mn><mo>±</mo><mn>0.0030</mn></mrow><annotation encoding="application/x-tex">0.0146 \mathbin{\pm} 0.0030</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Disengagement</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0030</mn><mo>±</mo><mn>0.0046</mn></mrow><annotation encoding="application/x-tex">-0.0030 \mathbin{\pm} 0.0046</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Disengagement</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0018</mn><mo>±</mo><mn>0.0029</mn></mrow><annotation encoding="application/x-tex">-0.0018 \mathbin{\pm} 0.0029</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Activity Path</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0166</mn><mo>±</mo><mn>0.0034</mn></mrow><annotation encoding="application/x-tex">0.0166 \mathbin{\pm} 0.0034</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Activity Path</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0013</mn><mo>±</mo><mn>0.0068</mn></mrow><annotation encoding="application/x-tex">0.0013 \mathbin{\pm} 0.0068</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Reengagement</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0001</mn><mo>±</mo><mn>0.0017</mn></mrow><annotation encoding="application/x-tex">-0.0001 \mathbin{\pm} 0.0017</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Reengagement</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0002</mn><mo>±</mo><mn>0.0102</mn></mrow><annotation encoding="application/x-tex">-0.0002 \mathbin{\pm} 0.0102</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Real + standard synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9166</mn><mo>±</mo><mn>0.0034</mn></mrow><annotation encoding="application/x-tex">0.9166 \mathbin{\pm} 0.0034</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Real + standard synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8790</mn><mo>±</mo><mn>0.0041</mn></mrow><annotation encoding="application/x-tex">0.8790 \mathbin{\pm} 0.0041</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Real + targeted synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9168</mn><mo>±</mo><mn>0.0035</mn></mrow><annotation encoding="application/x-tex">0.9168 \mathbin{\pm} 0.0035</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Real + targeted synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8793</mn><mo>±</mo><mn>0.0043</mn></mrow><annotation encoding="application/x-tex">0.8793 \mathbin{\pm} 0.0043</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9225</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.9225 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8872</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.8872 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8201</mn><mo>±</mo><mn>0.0924</mn></mrow><annotation encoding="application/x-tex">0.8201 \mathbin{\pm} 0.0924</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7503</mn><mo>±</mo><mn>0.1291</mn></mrow><annotation encoding="application/x-tex">0.7503 \mathbin{\pm} 0.1291</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7857</mn><mo>±</mo><mn>0.1353</mn></mrow><annotation encoding="application/x-tex">0.7857 \mathbin{\pm} 0.1353</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7182</mn><mo>±</mo><mn>0.1712</mn></mrow><annotation encoding="application/x-tex">0.7182 \mathbin{\pm} 0.1712</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0059</mn><mo>±</mo><mn>0.0034</mn></mrow><annotation encoding="application/x-tex">-0.0059 \mathbin{\pm} 0.0034</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0082</mn><mo>±</mo><mn>0.0041</mn></mrow><annotation encoding="application/x-tex">-0.0082 \mathbin{\pm} 0.0041</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1023</mn><mo>±</mo><mn>0.0924</mn></mrow><annotation encoding="application/x-tex">0.1023 \mathbin{\pm} 0.0924</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1369</mn><mo>±</mo><mn>0.1291</mn></mrow><annotation encoding="application/x-tex">0.1369 \mathbin{\pm} 0.1291</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0344</mn><mo>±</mo><mn>0.0429</mn></mrow><annotation encoding="application/x-tex">-0.0344 \mathbin{\pm} 0.0429</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0321</mn><mo>±</mo><mn>0.0421</mn></mrow><annotation encoding="application/x-tex">-0.0321 \mathbin{\pm} 0.0421</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0057</mn><mo>±</mo><mn>0.0035</mn></mrow><annotation encoding="application/x-tex">-0.0057 \mathbin{\pm} 0.0035</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0080</mn><mo>±</mo><mn>0.0043</mn></mrow><annotation encoding="application/x-tex">-0.0080 \mathbin{\pm} 0.0043</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Course Failure Prediction; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5287</mn><mo>±</mo><mn>0.0211</mn></mrow><annotation encoding="application/x-tex">0.5287 \mathbin{\pm} 0.0211</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Course Failure Prediction; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6968</mn><mo>±</mo><mn>0.0067</mn></mrow><annotation encoding="application/x-tex">0.6968 \mathbin{\pm} 0.0067</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Course Failure Prediction; Real only
(TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5467</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.5467 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Course Failure Prediction; Real only
(TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7034</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7034 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Course Failure Prediction; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4411</mn><mo>±</mo><mn>0.0581</mn></mrow><annotation encoding="application/x-tex">0.4411 \mathbin{\pm} 0.0581</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Course Failure Prediction; Standard
synthetic only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6276</mn><mo>±</mo><mn>0.0645</mn></mrow><annotation encoding="application/x-tex">0.6276 \mathbin{\pm} 0.0645</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Course Failure Prediction; Targeted
synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4503</mn><mo>±</mo><mn>0.0558</mn></mrow><annotation encoding="application/x-tex">0.4503 \mathbin{\pm} 0.0558</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Course Failure Prediction; Targeted
synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6210</mn><mo>±</mo><mn>0.0800</mn></mrow><annotation encoding="application/x-tex">0.6210 \mathbin{\pm} 0.0800</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Course Failure Prediction; Standard
augmentation gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0179</mn><mo>±</mo><mn>0.0211</mn></mrow><annotation encoding="application/x-tex">-0.0179 \mathbin{\pm} 0.0211</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Course Failure Prediction; Standard
augmentation gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0066</mn><mo>±</mo><mn>0.0067</mn></mrow><annotation encoding="application/x-tex">-0.0066 \mathbin{\pm} 0.0067</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Course Failure Prediction; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1055</mn><mo>±</mo><mn>0.0581</mn></mrow><annotation encoding="application/x-tex">0.1055 \mathbin{\pm} 0.0581</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Course Failure Prediction; Standard
synthetic utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0758</mn><mo>±</mo><mn>0.0645</mn></mrow><annotation encoding="application/x-tex">0.0758 \mathbin{\pm} 0.0645</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Dropout Prediction; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2983</mn><mo>±</mo><mn>0.0042</mn></mrow><annotation encoding="application/x-tex">0.2983 \mathbin{\pm} 0.0042</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Dropout Prediction; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6404</mn><mo>±</mo><mn>0.0090</mn></mrow><annotation encoding="application/x-tex">0.6404 \mathbin{\pm} 0.0090</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Dropout Prediction; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.3074</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.3074 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Dropout Prediction; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6526</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.6526 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Dropout Prediction; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2758</mn><mo>±</mo><mn>0.0109</mn></mrow><annotation encoding="application/x-tex">0.2758 \mathbin{\pm} 0.0109</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Dropout Prediction; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6092</mn><mo>±</mo><mn>0.0023</mn></mrow><annotation encoding="application/x-tex">0.6092 \mathbin{\pm} 0.0023</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Dropout Prediction; Targeted synthetic
only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.2687</mn><mo>±</mo><mn>0.0100</mn></mrow><annotation encoding="application/x-tex">0.2687 \mathbin{\pm} 0.0100</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Dropout Prediction; Targeted synthetic
only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5934</mn><mo>±</mo><mn>0.0051</mn></mrow><annotation encoding="application/x-tex">0.5934 \mathbin{\pm} 0.0051</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Dropout Prediction; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0091</mn><mo>±</mo><mn>0.0042</mn></mrow><annotation encoding="application/x-tex">-0.0091 \mathbin{\pm} 0.0042</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Dropout Prediction; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0121</mn><mo>±</mo><mn>0.0090</mn></mrow><annotation encoding="application/x-tex">-0.0121 \mathbin{\pm} 0.0090</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Dropout Prediction; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0316</mn><mo>±</mo><mn>0.0109</mn></mrow><annotation encoding="application/x-tex">0.0316 \mathbin{\pm} 0.0109</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Dropout Prediction; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0433</mn><mo>±</mo><mn>0.0023</mn></mrow><annotation encoding="application/x-tex">0.0433 \mathbin{\pm} 0.0023</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Disengagement; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8254</mn><mo>±</mo><mn>0.0010</mn></mrow><annotation encoding="application/x-tex">0.8254 \mathbin{\pm} 0.0010</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Disengagement; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8492</mn><mo>±</mo><mn>0.0014</mn></mrow><annotation encoding="application/x-tex">0.8492 \mathbin{\pm} 0.0014</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Disengagement; Real + targeted
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8258</mn><mo>±</mo><mn>0.0013</mn></mrow><annotation encoding="application/x-tex">0.8258 \mathbin{\pm} 0.0013</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Disengagement; Real + targeted
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8499</mn><mo>±</mo><mn>0.0013</mn></mrow><annotation encoding="application/x-tex">0.8499 \mathbin{\pm} 0.0013</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Disengagement; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8294</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.8294 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Disengagement; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8542</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.8542 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Disengagement; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7247</mn><mo>±</mo><mn>0.1029</mn></mrow><annotation encoding="application/x-tex">0.7247 \mathbin{\pm} 0.1029</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Disengagement; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7491</mn><mo>±</mo><mn>0.1010</mn></mrow><annotation encoding="application/x-tex">0.7491 \mathbin{\pm} 0.1010</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Disengagement; Targeted synthetic
only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6644</mn><mo>±</mo><mn>0.1858</mn></mrow><annotation encoding="application/x-tex">0.6644 \mathbin{\pm} 0.1858</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Disengagement; Targeted synthetic
only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6869</mn><mo>±</mo><mn>0.1912</mn></mrow><annotation encoding="application/x-tex">0.6869 \mathbin{\pm} 0.1912</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Disengagement; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0040</mn><mo>±</mo><mn>0.0010</mn></mrow><annotation encoding="application/x-tex">-0.0040 \mathbin{\pm} 0.0010</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Disengagement; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0051</mn><mo>±</mo><mn>0.0014</mn></mrow><annotation encoding="application/x-tex">-0.0051 \mathbin{\pm} 0.0014</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Disengagement; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1046</mn><mo>±</mo><mn>0.1029</mn></mrow><annotation encoding="application/x-tex">0.1046 \mathbin{\pm} 0.1029</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Disengagement; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1051</mn><mo>±</mo><mn>0.1010</mn></mrow><annotation encoding="application/x-tex">0.1051 \mathbin{\pm} 0.1010</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Disengagement; Targeted-vs-standard
TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0603</mn><mo>±</mo><mn>0.0829</mn></mrow><annotation encoding="application/x-tex">-0.0603 \mathbin{\pm} 0.0829</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Disengagement; Targeted-vs-standard
TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0623</mn><mo>±</mo><mn>0.0903</mn></mrow><annotation encoding="application/x-tex">-0.0623 \mathbin{\pm} 0.0903</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Disengagement; Targeted augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0035</mn><mo>±</mo><mn>0.0013</mn></mrow><annotation encoding="application/x-tex">-0.0035 \mathbin{\pm} 0.0013</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Disengagement; Targeted augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0044</mn><mo>±</mo><mn>0.0013</mn></mrow><annotation encoding="application/x-tex">-0.0044 \mathbin{\pm} 0.0013</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Activity Path; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9533</mn><mo>±</mo><mn>0.0015</mn></mrow><annotation encoding="application/x-tex">0.9533 \mathbin{\pm} 0.0015</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Activity Path; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7968</mn><mo>±</mo><mn>0.0074</mn></mrow><annotation encoding="application/x-tex">0.7968 \mathbin{\pm} 0.0074</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Activity Path; Real + targeted
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9531</mn><mo>±</mo><mn>0.0017</mn></mrow><annotation encoding="application/x-tex">0.9531 \mathbin{\pm} 0.0017</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Activity Path; Real + targeted
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7959</mn><mo>±</mo><mn>0.0083</mn></mrow><annotation encoding="application/x-tex">0.7959 \mathbin{\pm} 0.0083</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Activity Path; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9562</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.9562 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Activity Path; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8097</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.8097 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Activity Path; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9169</mn><mo>±</mo><mn>0.0222</mn></mrow><annotation encoding="application/x-tex">0.9169 \mathbin{\pm} 0.0222</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Activity Path; Standard synthetic
only (TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6805</mn><mo>±</mo><mn>0.0635</mn></mrow><annotation encoding="application/x-tex">0.6805 \mathbin{\pm} 0.0635</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Activity Path; Targeted synthetic
only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8905</mn><mo>±</mo><mn>0.0570</mn></mrow><annotation encoding="application/x-tex">0.8905 \mathbin{\pm} 0.0570</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Activity Path; Targeted synthetic
only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6100</mn><mo>±</mo><mn>0.1554</mn></mrow><annotation encoding="application/x-tex">0.6100 \mathbin{\pm} 0.1554</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Activity Path; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0029</mn><mo>±</mo><mn>0.0015</mn></mrow><annotation encoding="application/x-tex">-0.0029 \mathbin{\pm} 0.0015</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Activity Path; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0129</mn><mo>±</mo><mn>0.0074</mn></mrow><annotation encoding="application/x-tex">-0.0129 \mathbin{\pm} 0.0074</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Activity Path; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0393</mn><mo>±</mo><mn>0.0222</mn></mrow><annotation encoding="application/x-tex">0.0393 \mathbin{\pm} 0.0222</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Activity Path; Standard synthetic
utility loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1292</mn><mo>±</mo><mn>0.0635</mn></mrow><annotation encoding="application/x-tex">0.1292 \mathbin{\pm} 0.0635</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Activity Path; Targeted-vs-standard
TSTR gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0264</mn><mo>±</mo><mn>0.0349</mn></mrow><annotation encoding="application/x-tex">-0.0264 \mathbin{\pm} 0.0349</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Activity Path; Targeted-vs-standard
TSTR gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0706</mn><mo>±</mo><mn>0.0922</mn></mrow><annotation encoding="application/x-tex">-0.0706 \mathbin{\pm} 0.0922</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Activity Path; Targeted augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0031</mn><mo>±</mo><mn>0.0017</mn></mrow><annotation encoding="application/x-tex">-0.0031 \mathbin{\pm} 0.0017</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Activity Path; Targeted augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0138</mn><mo>±</mo><mn>0.0083</mn></mrow><annotation encoding="application/x-tex">-0.0138 \mathbin{\pm} 0.0083</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Reengagement; Real + standard
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7524</mn><mo>±</mo><mn>0.0077</mn></mrow><annotation encoding="application/x-tex">0.7524 \mathbin{\pm} 0.0077</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Reengagement; Real + standard
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7544</mn><mo>±</mo><mn>0.0034</mn></mrow><annotation encoding="application/x-tex">0.7544 \mathbin{\pm} 0.0034</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Reengagement; Real + targeted
synthetic</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7536</mn><mo>±</mo><mn>0.0099</mn></mrow><annotation encoding="application/x-tex">0.7536 \mathbin{\pm} 0.0099</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Reengagement; Real + targeted
synthetic</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7580</mn><mo>±</mo><mn>0.0025</mn></mrow><annotation encoding="application/x-tex">0.7580 \mathbin{\pm} 0.0025</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Reengagement; Real only (TRTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7060</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7060 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Reengagement; Real only (TRTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.7062</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.7062 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Reengagement; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6463</mn><mo>±</mo><mn>0.1331</mn></mrow><annotation encoding="application/x-tex">0.6463 \mathbin{\pm} 0.1331</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Reengagement; Standard synthetic only
(TSTR)</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6268</mn><mo>±</mo><mn>0.1687</mn></mrow><annotation encoding="application/x-tex">0.6268 \mathbin{\pm} 0.1687</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Reengagement; Targeted synthetic only</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6866</mn><mo>±</mo><mn>0.0826</mn></mrow><annotation encoding="application/x-tex">0.6866 \mathbin{\pm} 0.0826</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Reengagement; Targeted synthetic only</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6953</mn><mo>±</mo><mn>0.0869</mn></mrow><annotation encoding="application/x-tex">0.6953 \mathbin{\pm} 0.0869</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Reengagement; Standard augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0464</mn><mo>±</mo><mn>0.0077</mn></mrow><annotation encoding="application/x-tex">0.0464 \mathbin{\pm} 0.0077</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Reengagement; Standard augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0482</mn><mo>±</mo><mn>0.0034</mn></mrow><annotation encoding="application/x-tex">0.0482 \mathbin{\pm} 0.0034</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Reengagement; Standard synthetic utility
loss</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0597</mn><mo>±</mo><mn>0.1331</mn></mrow><annotation encoding="application/x-tex">0.0597 \mathbin{\pm} 0.1331</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Reengagement; Standard synthetic utility
loss</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0795</mn><mo>±</mo><mn>0.1687</mn></mrow><annotation encoding="application/x-tex">0.0795 \mathbin{\pm} 0.1687</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Reengagement; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0403</mn><mo>±</mo><mn>0.0513</mn></mrow><annotation encoding="application/x-tex">0.0403 \mathbin{\pm} 0.0513</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Reengagement; Targeted-vs-standard TSTR
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0685</mn><mo>±</mo><mn>0.0826</mn></mrow><annotation encoding="application/x-tex">0.0685 \mathbin{\pm} 0.0826</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Reengagement; Targeted augmentation
gain</td>
<td style="text-align: left;">AUPRC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0477</mn><mo>±</mo><mn>0.0099</mn></mrow><annotation encoding="application/x-tex">0.0477 \mathbin{\pm} 0.0099</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Reengagement; Targeted augmentation
gain</td>
<td style="text-align: left;">AUROC</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0517</mn><mo>±</mo><mn>0.0025</mn></mrow><annotation encoding="application/x-tex">0.0517 \mathbin{\pm} 0.0025</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Disengagement</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0086</mn><mo>±</mo><mn>0.0076</mn></mrow><annotation encoding="application/x-tex">0.0086 \mathbin{\pm} 0.0076</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Disengagement</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0064</mn><mo>±</mo><mn>0.0054</mn></mrow><annotation encoding="application/x-tex">-0.0064 \mathbin{\pm} 0.0054</annotation></semantics></math>
(2; 1 missing)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Activity Path</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0000</mn><mo>±</mo><mn>0.0002</mn></mrow><annotation encoding="application/x-tex">-0.0000 \mathbin{\pm} 0.0002</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Activity Path</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0053</mn><mo>±</mo><mn>0.0155</mn></mrow><annotation encoding="application/x-tex">0.0053 \mathbin{\pm} 0.0155</annotation></semantics></math>
(2; 1 missing)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Reengagement</td>
<td style="text-align: left;">Rare-group prevalence-error reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0016</mn><mo>±</mo><mn>0.0022</mn></mrow><annotation encoding="application/x-tex">-0.0016 \mathbin{\pm} 0.0022</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Reengagement</td>
<td style="text-align: left;">Rare-group trajectory-shape-error
reduction</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0046</mn><mo>±</mo><mn>0.0052</mn></mrow><annotation encoding="application/x-tex">-0.0046 \mathbin{\pm} 0.0052</annotation></semantics></math>
(2; 1 missing)</td>
<td style="text-align: left;">Higher</td>
</tr>
</tbody>
</table>

<table id="tab:supp-disclosure-risk">
<caption>Complete publication results for disclosure risk. Every numeric
metric-summary entry is shown; SD is the population SD across the fixed
seeds and
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>n</mi><annotation encoding="application/x-tex">n</annotation></semantics></math>
is the number of estimable seeds.</caption>
<thead>
<tr>
<th style="text-align: left;">Dataset</th>
<th style="text-align: left;">Generator</th>
<th style="text-align: left;">Scope / comparison</th>
<th style="text-align: left;">Metric</th>
<th style="text-align: left;">Mean
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>±</mi><annotation encoding="application/x-tex">\pm</annotation></semantics></math>
SD
(<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>n</mi><annotation encoding="application/x-tex">n</annotation></semantics></math>)</th>
<th style="text-align: left;">Direction</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="6" style="text-align: left;"><em>Continued from previous
page</em></td>
</tr>
<tr>
<td style="text-align: left;">Dataset</td>
<td style="text-align: left;">Generator</td>
<td style="text-align: left;">Scope / comparison</td>
<td style="text-align: left;">Metric</td>
<td style="text-align: left;">Mean
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>±</mi><annotation encoding="application/x-tex">\pm</annotation></semantics></math>
SD
(<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>n</mi><annotation encoding="application/x-tex">n</annotation></semantics></math>)</td>
<td style="text-align: left;">Direction</td>
</tr>
<tr>
<td colspan="6" style="text-align: right;"><em>Continued on next
page</em></td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Exact-duplicate rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Mean nearest-training distance</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9532</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">0.9532 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Membership-inference advantage</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0119</mn><mo>±</mo><mn>0.0088</mn></mrow><annotation encoding="application/x-tex">0.0119 \mathbin{\pm} 0.0088</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Near-duplicate rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Exact-duplicate rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Mean nearest-training distance</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9517</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">0.9517 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Membership-inference advantage</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0528</mn><mo>±</mo><mn>0.0194</mn></mrow><annotation encoding="application/x-tex">0.0528 \mathbin{\pm} 0.0194</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Near-duplicate rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Exact-duplicate risk change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Nearest-neighbor exposure risk change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0015</mn><mo>±</mo><mn>0.0016</mn></mrow><annotation encoding="application/x-tex">0.0015 \mathbin{\pm} 0.0016</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Membership-inference risk change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0409</mn><mo>±</mo><mn>0.0142</mn></mrow><annotation encoding="application/x-tex">0.0409 \mathbin{\pm} 0.0142</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Near-duplicate risk change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Exact-duplicate rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Mean nearest-training distance</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9657</mn><mo>±</mo><mn>0.0015</mn></mrow><annotation encoding="application/x-tex">0.9657 \mathbin{\pm} 0.0015</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Membership-inference advantage</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0235</mn><mo>±</mo><mn>0.0095</mn></mrow><annotation encoding="application/x-tex">0.0235 \mathbin{\pm} 0.0095</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Near-duplicate rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Exact-duplicate rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Mean nearest-training distance</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9677</mn><mo>±</mo><mn>0.0020</mn></mrow><annotation encoding="application/x-tex">0.9677 \mathbin{\pm} 0.0020</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Membership-inference advantage</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0412</mn><mo>±</mo><mn>0.0253</mn></mrow><annotation encoding="application/x-tex">0.0412 \mathbin{\pm} 0.0253</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Near-duplicate rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Exact-duplicate risk change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Nearest-neighbor exposure risk change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0020</mn><mo>±</mo><mn>0.0005</mn></mrow><annotation encoding="application/x-tex">-0.0020 \mathbin{\pm} 0.0005</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Membership-inference risk change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0178</mn><mo>±</mo><mn>0.0276</mn></mrow><annotation encoding="application/x-tex">0.0178 \mathbin{\pm} 0.0276</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Near-duplicate risk change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Exact-duplicate rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Mean nearest-training distance</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8614</mn><mo>±</mo><mn>0.0412</mn></mrow><annotation encoding="application/x-tex">0.8614 \mathbin{\pm} 0.0412</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Membership-inference advantage</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0011</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">0.0011 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Near-duplicate rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Exact-duplicate rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Mean nearest-training distance</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.8676</mn><mo>±</mo><mn>0.0294</mn></mrow><annotation encoding="application/x-tex">0.8676 \mathbin{\pm} 0.0294</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Membership-inference advantage</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0221</mn><mo>±</mo><mn>0.0173</mn></mrow><annotation encoding="application/x-tex">0.0221 \mathbin{\pm} 0.0173</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Near-duplicate rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Exact-duplicate risk change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Nearest-neighbor exposure risk change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0062</mn><mo>±</mo><mn>0.0124</mn></mrow><annotation encoding="application/x-tex">-0.0062 \mathbin{\pm} 0.0124</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Membership-inference risk change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0210</mn><mo>±</mo><mn>0.0169</mn></mrow><annotation encoding="application/x-tex">0.0210 \mathbin{\pm} 0.0169</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Near-duplicate risk change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Exact-duplicate rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Mean nearest-training distance</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9937</mn><mo>±</mo><mn>0.0008</mn></mrow><annotation encoding="application/x-tex">0.9937 \mathbin{\pm} 0.0008</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Membership-inference advantage</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Near-duplicate rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Exact-duplicate rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Mean nearest-training distance</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9923</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">0.9923 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Membership-inference advantage</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Near-duplicate rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Exact-duplicate risk change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Nearest-neighbor exposure risk change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0015</mn><mo>±</mo><mn>0.0008</mn></mrow><annotation encoding="application/x-tex">0.0015 \mathbin{\pm} 0.0008</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Membership-inference risk change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Near-duplicate risk change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Exact-duplicate rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Mean nearest-training distance</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9869</mn><mo>±</mo><mn>0.0017</mn></mrow><annotation encoding="application/x-tex">0.9869 \mathbin{\pm} 0.0017</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Membership-inference advantage</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Near-duplicate rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Exact-duplicate rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Mean nearest-training distance</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9852</mn><mo>±</mo><mn>0.0013</mn></mrow><annotation encoding="application/x-tex">0.9852 \mathbin{\pm} 0.0013</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Membership-inference advantage</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Near-duplicate rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Exact-duplicate risk change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Nearest-neighbor exposure risk change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0017</mn><mo>±</mo><mn>0.0008</mn></mrow><annotation encoding="application/x-tex">0.0017 \mathbin{\pm} 0.0008</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Membership-inference risk change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Near-duplicate risk change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Exact-duplicate rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Mean nearest-training distance</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9933</mn><mo>±</mo><mn>0.0007</mn></mrow><annotation encoding="application/x-tex">0.9933 \mathbin{\pm} 0.0007</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Membership-inference advantage</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0049</mn><mo>±</mo><mn>0.0069</mn></mrow><annotation encoding="application/x-tex">0.0049 \mathbin{\pm} 0.0069</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Near-duplicate rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Exact-duplicate rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Mean nearest-training distance</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.9942</mn><mo>±</mo><mn>0.0001</mn></mrow><annotation encoding="application/x-tex">0.9942 \mathbin{\pm} 0.0001</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Higher</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Membership-inference advantage</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Near-duplicate rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Exact-duplicate risk change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Nearest-neighbor exposure risk change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0009</mn><mo>±</mo><mn>0.0007</mn></mrow><annotation encoding="application/x-tex">-0.0009 \mathbin{\pm} 0.0007</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Membership-inference risk change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0049</mn><mo>±</mo><mn>0.0069</mn></mrow><annotation encoding="application/x-tex">-0.0049 \mathbin{\pm} 0.0069</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Near-duplicate risk change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0000</mn><mo>±</mo><mn>0.0000</mn></mrow><annotation encoding="application/x-tex">0.0000 \mathbin{\pm} 0.0000</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Behavior-projection exact-duplicate
rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0290</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">0.0290 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Behavior-projection exact-duplicate
rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0315</mn><mo>±</mo><mn>0.0005</mn></mrow><annotation encoding="application/x-tex">0.0315 \mathbin{\pm} 0.0005</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Behavior-projection exact-copy risk
change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0025</mn><mo>±</mo><mn>0.0011</mn></mrow><annotation encoding="application/x-tex">0.0025 \mathbin{\pm} 0.0011</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Behavior-projection exact-duplicate
rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0031</mn><mo>±</mo><mn>0.0009</mn></mrow><annotation encoding="application/x-tex">0.0031 \mathbin{\pm} 0.0009</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Behavior-projection exact-duplicate
rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0027</mn><mo>±</mo><mn>0.0003</mn></mrow><annotation encoding="application/x-tex">0.0027 \mathbin{\pm} 0.0003</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Behavior-projection exact-copy risk
change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.0004</mn><mo>±</mo><mn>0.0012</mn></mrow><annotation encoding="application/x-tex">-0.0004 \mathbin{\pm} 0.0012</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Standard synthetic</td>
<td style="text-align: left;">Behavior-projection exact-duplicate
rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1557</mn><mo>±</mo><mn>0.0906</mn></mrow><annotation encoding="application/x-tex">0.1557 \mathbin{\pm} 0.0906</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted synthetic</td>
<td style="text-align: left;">Behavior-projection exact-duplicate
rate</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.1587</mn><mo>±</mo><mn>0.0973</mn></mrow><annotation encoding="application/x-tex">0.1587 \mathbin{\pm} 0.0973</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Targeted-minus-standard risk</td>
<td style="text-align: left;">Behavior-projection exact-copy risk
change</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0030</mn><mo>±</mo><mn>0.0071</mn></mrow><annotation encoding="application/x-tex">0.0030 \mathbin{\pm} 0.0071</annotation></semantics></math>
(3)</td>
<td style="text-align: left;">Lower</td>
</tr>
</tbody>
</table>

<div class="landscape">

# Signed Rare-Pathway Prevalence

Absolute prevalence error supports generator comparison but does not
identify whether a pathway was underproduced or overproduced.
Table <a href="#tab:supp-signed-prevalence" data-reference-type="ref"
data-reference="tab:supp-signed-prevalence">14</a> therefore reports the
real, standard-synthetic, and targeted-synthetic learner prevalences and
both signed differences for every publication pathway.

<table id="tab:supp-signed-prevalence">
<caption>Signed rare-pathway prevalence results. Prevalences are learner
percentages and differences are synthetic minus real in percentage
points. Positive differences mean overrepresentation; negative
differences mean underrepresentation. Values are mean
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>±</mi><annotation encoding="application/x-tex">\pm</annotation></semantics></math>
population SD across the three fixed generator seeds.</caption>
<thead>
<tr>
<th style="text-align: left;">Dataset</th>
<th style="text-align: left;">Generator</th>
<th style="text-align: left;">Pathway</th>
<th style="text-align: left;">Real (%)</th>
<th style="text-align: left;">Standard synth. (%)</th>
<th style="text-align: left;">Targeted synth. (%)</th>
<th style="text-align: left;">Standard signed diff. (pp)</th>
<th style="text-align: left;">Targeted signed diff. (pp)</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">Dataset</td>
<td style="text-align: left;">Generator</td>
<td style="text-align: left;">Pathway</td>
<td style="text-align: left;">Real (%)</td>
<td style="text-align: left;">Standard synth. (%)</td>
<td style="text-align: left;">Targeted synth. (%)</td>
<td style="text-align: left;">Standard signed diff. (pp)</td>
<td style="text-align: left;">Targeted signed diff. (pp)</td>
</tr>
<tr>
<td colspan="8" style="text-align: right;"><em>Continued on next
page</em></td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">High Hint Use</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>4.20</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">4.20 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>1.27</mn><mo>±</mo><mn>0.15</mn></mrow><annotation encoding="application/x-tex">1.27 \mathbin{\pm} 0.15</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>5.79</mn><mo>±</mo><mn>0.30</mn></mrow><annotation encoding="application/x-tex">5.79 \mathbin{\pm} 0.30</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>2.93</mn><mo>±</mo><mn>0.15</mn></mrow><annotation encoding="application/x-tex">-2.93 \mathbin{\pm} 0.15</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>1.59</mn><mo>±</mo><mn>0.30</mn></mrow><annotation encoding="application/x-tex">+1.59 \mathbin{\pm} 0.30</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2.65</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">2.65 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>3.15</mn><mo>±</mo><mn>0.13</mn></mrow><annotation encoding="application/x-tex">3.15 \mathbin{\pm} 0.13</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>4.66</mn><mo>±</mo><mn>0.28</mn></mrow><annotation encoding="application/x-tex">4.66 \mathbin{\pm} 0.28</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>0.50</mn><mo>±</mo><mn>0.13</mn></mrow><annotation encoding="application/x-tex">+0.50 \mathbin{\pm} 0.13</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>2.00</mn><mo>±</mo><mn>0.28</mn></mrow><annotation encoding="application/x-tex">+2.00 \mathbin{\pm} 0.28</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2.45</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">2.45 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.48</mn><mo>±</mo><mn>0.17</mn></mrow><annotation encoding="application/x-tex">0.48 \mathbin{\pm} 0.17</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2.23</mn><mo>±</mo><mn>0.55</mn></mrow><annotation encoding="application/x-tex">2.23 \mathbin{\pm} 0.55</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>1.97</mn><mo>±</mo><mn>0.17</mn></mrow><annotation encoding="application/x-tex">-1.97 \mathbin{\pm} 0.17</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.22</mn><mo>±</mo><mn>0.55</mn></mrow><annotation encoding="application/x-tex">-0.22 \mathbin{\pm} 0.55</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2.57</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">2.57 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>1.17</mn><mo>±</mo><mn>0.14</mn></mrow><annotation encoding="application/x-tex">1.17 \mathbin{\pm} 0.14</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>4.12</mn><mo>±</mo><mn>0.34</mn></mrow><annotation encoding="application/x-tex">4.12 \mathbin{\pm} 0.34</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>1.40</mn><mo>±</mo><mn>0.14</mn></mrow><annotation encoding="application/x-tex">-1.40 \mathbin{\pm} 0.14</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>1.56</mn><mo>±</mo><mn>0.34</mn></mrow><annotation encoding="application/x-tex">+1.56 \mathbin{\pm} 0.34</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>4.87</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">4.87 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2.11</mn><mo>±</mo><mn>0.76</mn></mrow><annotation encoding="application/x-tex">2.11 \mathbin{\pm} 0.76</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2.77</mn><mo>±</mo><mn>0.68</mn></mrow><annotation encoding="application/x-tex">2.77 \mathbin{\pm} 0.68</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>2.76</mn><mo>±</mo><mn>0.76</mn></mrow><annotation encoding="application/x-tex">-2.76 \mathbin{\pm} 0.76</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>2.10</mn><mo>±</mo><mn>0.68</mn></mrow><annotation encoding="application/x-tex">-2.10 \mathbin{\pm} 0.68</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>3.32</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">3.32 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>3.42</mn><mo>±</mo><mn>0.31</mn></mrow><annotation encoding="application/x-tex">3.42 \mathbin{\pm} 0.31</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>4.31</mn><mo>±</mo><mn>0.46</mn></mrow><annotation encoding="application/x-tex">4.31 \mathbin{\pm} 0.46</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>0.10</mn><mo>±</mo><mn>0.31</mn></mrow><annotation encoding="application/x-tex">+0.10 \mathbin{\pm} 0.31</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>0.98</mn><mo>±</mo><mn>0.46</mn></mrow><annotation encoding="application/x-tex">+0.98 \mathbin{\pm} 0.46</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">High Hint Use</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>4.20</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">4.20 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.30</mn><mo>±</mo><mn>0.27</mn></mrow><annotation encoding="application/x-tex">0.30 \mathbin{\pm} 0.27</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.51</mn><mo>±</mo><mn>0.24</mn></mrow><annotation encoding="application/x-tex">0.51 \mathbin{\pm} 0.24</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>3.90</mn><mo>±</mo><mn>0.27</mn></mrow><annotation encoding="application/x-tex">-3.90 \mathbin{\pm} 0.27</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>3.69</mn><mo>±</mo><mn>0.24</mn></mrow><annotation encoding="application/x-tex">-3.69 \mathbin{\pm} 0.24</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2.65</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">2.65 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>1.81</mn><mo>±</mo><mn>0.70</mn></mrow><annotation encoding="application/x-tex">1.81 \mathbin{\pm} 0.70</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2.43</mn><mo>±</mo><mn>0.56</mn></mrow><annotation encoding="application/x-tex">2.43 \mathbin{\pm} 0.56</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.85</mn><mo>±</mo><mn>0.70</mn></mrow><annotation encoding="application/x-tex">-0.85 \mathbin{\pm} 0.70</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.22</mn><mo>±</mo><mn>0.56</mn></mrow><annotation encoding="application/x-tex">-0.22 \mathbin{\pm} 0.56</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2.45</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">2.45 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.08</mn><mo>±</mo><mn>0.09</mn></mrow><annotation encoding="application/x-tex">0.08 \mathbin{\pm} 0.09</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.09</mn><mo>±</mo><mn>0.10</mn></mrow><annotation encoding="application/x-tex">0.09 \mathbin{\pm} 0.10</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>2.37</mn><mo>±</mo><mn>0.09</mn></mrow><annotation encoding="application/x-tex">-2.37 \mathbin{\pm} 0.09</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>2.36</mn><mo>±</mo><mn>0.10</mn></mrow><annotation encoding="application/x-tex">-2.36 \mathbin{\pm} 0.10</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2.57</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">2.57 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.11</mn><mo>±</mo><mn>0.06</mn></mrow><annotation encoding="application/x-tex">0.11 \mathbin{\pm} 0.06</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.12</mn><mo>±</mo><mn>0.14</mn></mrow><annotation encoding="application/x-tex">0.12 \mathbin{\pm} 0.14</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>2.46</mn><mo>±</mo><mn>0.06</mn></mrow><annotation encoding="application/x-tex">-2.46 \mathbin{\pm} 0.06</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>2.45</mn><mo>±</mo><mn>0.14</mn></mrow><annotation encoding="application/x-tex">-2.45 \mathbin{\pm} 0.14</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>4.87</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">4.87 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>98.52</mn><mo>±</mo><mn>0.73</mn></mrow><annotation encoding="application/x-tex">98.52 \mathbin{\pm} 0.73</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>98.98</mn><mo>±</mo><mn>0.62</mn></mrow><annotation encoding="application/x-tex">98.98 \mathbin{\pm} 0.62</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>93.65</mn><mo>±</mo><mn>0.73</mn></mrow><annotation encoding="application/x-tex">+93.65 \mathbin{\pm} 0.73</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>94.11</mn><mo>±</mo><mn>0.62</mn></mrow><annotation encoding="application/x-tex">+94.11 \mathbin{\pm} 0.62</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>3.32</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">3.32 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2.33</mn><mo>±</mo><mn>0.97</mn></mrow><annotation encoding="application/x-tex">2.33 \mathbin{\pm} 0.97</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2.52</mn><mo>±</mo><mn>0.63</mn></mrow><annotation encoding="application/x-tex">2.52 \mathbin{\pm} 0.63</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.99</mn><mo>±</mo><mn>0.97</mn></mrow><annotation encoding="application/x-tex">-0.99 \mathbin{\pm} 0.97</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.81</mn><mo>±</mo><mn>0.63</mn></mrow><annotation encoding="application/x-tex">-0.81 \mathbin{\pm} 0.63</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">High Hint Use</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>4.20</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">4.20 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.02</mn><mo>±</mo><mn>0.03</mn></mrow><annotation encoding="application/x-tex">0.02 \mathbin{\pm} 0.03</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.01</mn><mo>±</mo><mn>0.01</mn></mrow><annotation encoding="application/x-tex">0.01 \mathbin{\pm} 0.01</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>4.18</mn><mo>±</mo><mn>0.03</mn></mrow><annotation encoding="application/x-tex">-4.18 \mathbin{\pm} 0.03</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>4.19</mn><mo>±</mo><mn>0.01</mn></mrow><annotation encoding="application/x-tex">-4.19 \mathbin{\pm} 0.01</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2.65</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">2.65 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.28</mn><mo>±</mo><mn>0.40</mn></mrow><annotation encoding="application/x-tex">0.28 \mathbin{\pm} 0.40</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.05</mn><mo>±</mo><mn>0.07</mn></mrow><annotation encoding="application/x-tex">0.05 \mathbin{\pm} 0.07</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>2.37</mn><mo>±</mo><mn>0.40</mn></mrow><annotation encoding="application/x-tex">-2.37 \mathbin{\pm} 0.40</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>2.61</mn><mo>±</mo><mn>0.07</mn></mrow><annotation encoding="application/x-tex">-2.61 \mathbin{\pm} 0.07</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2.45</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">2.45 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.00</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">0.00 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.00</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">0.00 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>2.45</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">-2.45 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>2.45</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">-2.45 \mathbin{\pm} 0.00</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2.57</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">2.57 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.00</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">0.00 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.00</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">0.00 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>2.57</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">-2.57 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>2.57</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">-2.57 \mathbin{\pm} 0.00</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>4.87</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">4.87 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>14.90</mn><mo>±</mo><mn>20.30</mn></mrow><annotation encoding="application/x-tex">14.90 \mathbin{\pm} 20.30</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>8.44</mn><mo>±</mo><mn>11.46</mn></mrow><annotation encoding="application/x-tex">8.44 \mathbin{\pm} 11.46</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>10.03</mn><mo>±</mo><mn>20.30</mn></mrow><annotation encoding="application/x-tex">+10.03 \mathbin{\pm} 20.30</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>3.57</mn><mo>±</mo><mn>11.46</mn></mrow><annotation encoding="application/x-tex">+3.57 \mathbin{\pm} 11.46</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>3.32</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">3.32 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.36</mn><mo>±</mo><mn>0.51</mn></mrow><annotation encoding="application/x-tex">0.36 \mathbin{\pm} 0.51</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.08</mn><mo>±</mo><mn>0.11</mn></mrow><annotation encoding="application/x-tex">0.08 \mathbin{\pm} 0.11</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>2.96</mn><mo>±</mo><mn>0.51</mn></mrow><annotation encoding="application/x-tex">-2.96 \mathbin{\pm} 0.51</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>3.25</mn><mo>±</mo><mn>0.11</mn></mrow><annotation encoding="application/x-tex">-3.25 \mathbin{\pm} 0.11</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>7.61</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">7.61 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2.07</mn><mo>±</mo><mn>0.14</mn></mrow><annotation encoding="application/x-tex">2.07 \mathbin{\pm} 0.14</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>3.24</mn><mo>±</mo><mn>0.08</mn></mrow><annotation encoding="application/x-tex">3.24 \mathbin{\pm} 0.08</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>5.54</mn><mo>±</mo><mn>0.14</mn></mrow><annotation encoding="application/x-tex">-5.54 \mathbin{\pm} 0.14</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>4.37</mn><mo>±</mo><mn>0.08</mn></mrow><annotation encoding="application/x-tex">-4.37 \mathbin{\pm} 0.08</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Persistent Low Correctness</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>4.76</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">4.76 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.35</mn><mo>±</mo><mn>0.10</mn></mrow><annotation encoding="application/x-tex">0.35 \mathbin{\pm} 0.10</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.78</mn><mo>±</mo><mn>0.09</mn></mrow><annotation encoding="application/x-tex">0.78 \mathbin{\pm} 0.09</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>4.41</mn><mo>±</mo><mn>0.10</mn></mrow><annotation encoding="application/x-tex">-4.41 \mathbin{\pm} 0.10</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>3.98</mn><mo>±</mo><mn>0.09</mn></mrow><annotation encoding="application/x-tex">-3.98 \mathbin{\pm} 0.09</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>3.92</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">3.92 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.39</mn><mo>±</mo><mn>0.10</mn></mrow><annotation encoding="application/x-tex">0.39 \mathbin{\pm} 0.10</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>7.63</mn><mo>±</mo><mn>0.38</mn></mrow><annotation encoding="application/x-tex">7.63 \mathbin{\pm} 0.38</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>3.53</mn><mo>±</mo><mn>0.10</mn></mrow><annotation encoding="application/x-tex">-3.53 \mathbin{\pm} 0.10</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>3.71</mn><mo>±</mo><mn>0.38</mn></mrow><annotation encoding="application/x-tex">+3.71 \mathbin{\pm} 0.38</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Skill Path</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>5.01</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">5.01 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>1.76</mn><mo>±</mo><mn>0.60</mn></mrow><annotation encoding="application/x-tex">1.76 \mathbin{\pm} 0.60</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2.07</mn><mo>±</mo><mn>0.17</mn></mrow><annotation encoding="application/x-tex">2.07 \mathbin{\pm} 0.17</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>3.25</mn><mo>±</mo><mn>0.60</mn></mrow><annotation encoding="application/x-tex">-3.25 \mathbin{\pm} 0.60</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>2.94</mn><mo>±</mo><mn>0.17</mn></mrow><annotation encoding="application/x-tex">-2.94 \mathbin{\pm} 0.17</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Recovery</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>4.35</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">4.35 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2.13</mn><mo>±</mo><mn>0.07</mn></mrow><annotation encoding="application/x-tex">2.13 \mathbin{\pm} 0.07</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2.04</mn><mo>±</mo><mn>0.20</mn></mrow><annotation encoding="application/x-tex">2.04 \mathbin{\pm} 0.20</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>2.22</mn><mo>±</mo><mn>0.07</mn></mrow><annotation encoding="application/x-tex">-2.22 \mathbin{\pm} 0.07</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>2.31</mn><mo>±</mo><mn>0.20</mn></mrow><annotation encoding="application/x-tex">-2.31 \mathbin{\pm} 0.20</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>7.61</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">7.61 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>1.41</mn><mo>±</mo><mn>0.35</mn></mrow><annotation encoding="application/x-tex">1.41 \mathbin{\pm} 0.35</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>1.70</mn><mo>±</mo><mn>0.17</mn></mrow><annotation encoding="application/x-tex">1.70 \mathbin{\pm} 0.17</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>6.20</mn><mo>±</mo><mn>0.35</mn></mrow><annotation encoding="application/x-tex">-6.20 \mathbin{\pm} 0.35</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>5.91</mn><mo>±</mo><mn>0.17</mn></mrow><annotation encoding="application/x-tex">-5.91 \mathbin{\pm} 0.17</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Persistent Low Correctness</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>4.76</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">4.76 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.17</mn><mo>±</mo><mn>0.08</mn></mrow><annotation encoding="application/x-tex">0.17 \mathbin{\pm} 0.08</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.11</mn><mo>±</mo><mn>0.03</mn></mrow><annotation encoding="application/x-tex">0.11 \mathbin{\pm} 0.03</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>4.59</mn><mo>±</mo><mn>0.08</mn></mrow><annotation encoding="application/x-tex">-4.59 \mathbin{\pm} 0.08</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>4.65</mn><mo>±</mo><mn>0.03</mn></mrow><annotation encoding="application/x-tex">-4.65 \mathbin{\pm} 0.03</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>3.92</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">3.92 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.02</mn><mo>±</mo><mn>0.02</mn></mrow><annotation encoding="application/x-tex">0.02 \mathbin{\pm} 0.02</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.02</mn><mo>±</mo><mn>0.02</mn></mrow><annotation encoding="application/x-tex">0.02 \mathbin{\pm} 0.02</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>3.90</mn><mo>±</mo><mn>0.02</mn></mrow><annotation encoding="application/x-tex">-3.90 \mathbin{\pm} 0.02</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>3.90</mn><mo>±</mo><mn>0.02</mn></mrow><annotation encoding="application/x-tex">-3.90 \mathbin{\pm} 0.02</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Skill Path</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>5.01</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">5.01 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>90.29</mn><mo>±</mo><mn>0.99</mn></mrow><annotation encoding="application/x-tex">90.29 \mathbin{\pm} 0.99</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>91.51</mn><mo>±</mo><mn>0.48</mn></mrow><annotation encoding="application/x-tex">91.51 \mathbin{\pm} 0.48</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>85.28</mn><mo>±</mo><mn>0.99</mn></mrow><annotation encoding="application/x-tex">+85.28 \mathbin{\pm} 0.99</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>86.50</mn><mo>±</mo><mn>0.48</mn></mrow><annotation encoding="application/x-tex">+86.50 \mathbin{\pm} 0.48</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Recovery</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>4.35</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">4.35 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>1.96</mn><mo>±</mo><mn>0.40</mn></mrow><annotation encoding="application/x-tex">1.96 \mathbin{\pm} 0.40</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>1.90</mn><mo>±</mo><mn>0.16</mn></mrow><annotation encoding="application/x-tex">1.90 \mathbin{\pm} 0.16</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>2.38</mn><mo>±</mo><mn>0.40</mn></mrow><annotation encoding="application/x-tex">-2.38 \mathbin{\pm} 0.40</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>2.45</mn><mo>±</mo><mn>0.16</mn></mrow><annotation encoding="application/x-tex">-2.45 \mathbin{\pm} 0.16</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Correctness Decline</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>7.61</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">7.61 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.00</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">0.00 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.00</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">0.00 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>7.61</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">-7.61 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>7.61</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">-7.61 \mathbin{\pm} 0.00</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Persistent Low Correctness</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>4.76</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">4.76 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.00</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">0.00 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.00</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">0.00 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>4.76</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">-4.76 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>4.76</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">-4.76 \mathbin{\pm} 0.00</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rapid Low-Accuracy Responding</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>3.92</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">3.92 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.00</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">0.00 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.00</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">0.00 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>3.92</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">-3.92 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>3.92</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">-3.92 \mathbin{\pm} 0.00</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Skill Path</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>5.01</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">5.01 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>88.06</mn><mo>±</mo><mn>1.19</mn></mrow><annotation encoding="application/x-tex">88.06 \mathbin{\pm} 1.19</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>89.78</mn><mo>±</mo><mn>0.49</mn></mrow><annotation encoding="application/x-tex">89.78 \mathbin{\pm} 0.49</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>83.05</mn><mo>±</mo><mn>1.19</mn></mrow><annotation encoding="application/x-tex">+83.05 \mathbin{\pm} 1.19</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>84.77</mn><mo>±</mo><mn>0.49</mn></mrow><annotation encoding="application/x-tex">+84.77 \mathbin{\pm} 0.49</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Recovery</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>4.35</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">4.35 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.00</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">0.00 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.00</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">0.00 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>4.35</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">-4.35 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>4.35</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">-4.35 \mathbin{\pm} 0.00</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Late Disengagement</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>6.74</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">6.74 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>9.30</mn><mo>±</mo><mn>0.14</mn></mrow><annotation encoding="application/x-tex">9.30 \mathbin{\pm} 0.14</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>8.11</mn><mo>±</mo><mn>0.05</mn></mrow><annotation encoding="application/x-tex">8.11 \mathbin{\pm} 0.05</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>2.56</mn><mo>±</mo><mn>0.14</mn></mrow><annotation encoding="application/x-tex">+2.56 \mathbin{\pm} 0.14</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>1.38</mn><mo>±</mo><mn>0.05</mn></mrow><annotation encoding="application/x-tex">+1.38 \mathbin{\pm} 0.05</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Rare Activity Path</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>4.46</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">4.46 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>1.73</mn><mo>±</mo><mn>0.07</mn></mrow><annotation encoding="application/x-tex">1.73 \mathbin{\pm} 0.07</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>10.67</mn><mo>±</mo><mn>0.24</mn></mrow><annotation encoding="application/x-tex">10.67 \mathbin{\pm} 0.24</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>2.73</mn><mo>±</mo><mn>0.07</mn></mrow><annotation encoding="application/x-tex">-2.73 \mathbin{\pm} 0.07</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>6.20</mn><mo>±</mo><mn>0.24</mn></mrow><annotation encoding="application/x-tex">+6.20 \mathbin{\pm} 0.24</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">Reengagement</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.22</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">0.22 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>3.43</mn><mo>±</mo><mn>0.09</mn></mrow><annotation encoding="application/x-tex">3.43 \mathbin{\pm} 0.09</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2.91</mn><mo>±</mo><mn>0.10</mn></mrow><annotation encoding="application/x-tex">2.91 \mathbin{\pm} 0.10</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>3.21</mn><mo>±</mo><mn>0.09</mn></mrow><annotation encoding="application/x-tex">+3.21 \mathbin{\pm} 0.09</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>2.68</mn><mo>±</mo><mn>0.10</mn></mrow><annotation encoding="application/x-tex">+2.68 \mathbin{\pm} 0.10</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Late Disengagement</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>6.74</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">6.74 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>4.85</mn><mo>±</mo><mn>0.33</mn></mrow><annotation encoding="application/x-tex">4.85 \mathbin{\pm} 0.33</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>4.55</mn><mo>±</mo><mn>0.25</mn></mrow><annotation encoding="application/x-tex">4.55 \mathbin{\pm} 0.25</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>1.88</mn><mo>±</mo><mn>0.33</mn></mrow><annotation encoding="application/x-tex">-1.88 \mathbin{\pm} 0.33</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>2.18</mn><mo>±</mo><mn>0.25</mn></mrow><annotation encoding="application/x-tex">-2.18 \mathbin{\pm} 0.25</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Rare Activity Path</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>4.46</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">4.46 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2.38</mn><mo>±</mo><mn>0.10</mn></mrow><annotation encoding="application/x-tex">2.38 \mathbin{\pm} 0.10</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>4.04</mn><mo>±</mo><mn>0.28</mn></mrow><annotation encoding="application/x-tex">4.04 \mathbin{\pm} 0.28</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>2.09</mn><mo>±</mo><mn>0.10</mn></mrow><annotation encoding="application/x-tex">-2.09 \mathbin{\pm} 0.10</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.43</mn><mo>±</mo><mn>0.28</mn></mrow><annotation encoding="application/x-tex">-0.43 \mathbin{\pm} 0.28</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">Reengagement</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.22</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">0.22 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2.41</mn><mo>±</mo><mn>0.12</mn></mrow><annotation encoding="application/x-tex">2.41 \mathbin{\pm} 0.12</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2.42</mn><mo>±</mo><mn>0.12</mn></mrow><annotation encoding="application/x-tex">2.42 \mathbin{\pm} 0.12</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>2.19</mn><mo>±</mo><mn>0.12</mn></mrow><annotation encoding="application/x-tex">+2.19 \mathbin{\pm} 0.12</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>2.20</mn><mo>±</mo><mn>0.12</mn></mrow><annotation encoding="application/x-tex">+2.20 \mathbin{\pm} 0.12</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Late Disengagement</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>6.74</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">6.74 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>7.82</mn><mo>±</mo><mn>5.56</mn></mrow><annotation encoding="application/x-tex">7.82 \mathbin{\pm} 5.56</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>6.96</mn><mo>±</mo><mn>4.92</mn></mrow><annotation encoding="application/x-tex">6.96 \mathbin{\pm} 4.92</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>1.09</mn><mo>±</mo><mn>5.56</mn></mrow><annotation encoding="application/x-tex">+1.09 \mathbin{\pm} 5.56</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>0.23</mn><mo>±</mo><mn>4.92</mn></mrow><annotation encoding="application/x-tex">+0.23 \mathbin{\pm} 4.92</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Rare Activity Path</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>4.46</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">4.46 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.21</mn><mo>±</mo><mn>0.15</mn></mrow><annotation encoding="application/x-tex">0.21 \mathbin{\pm} 0.15</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.20</mn><mo>±</mo><mn>0.15</mn></mrow><annotation encoding="application/x-tex">0.20 \mathbin{\pm} 0.15</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>4.26</mn><mo>±</mo><mn>0.15</mn></mrow><annotation encoding="application/x-tex">-4.26 \mathbin{\pm} 0.15</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>4.26</mn><mo>±</mo><mn>0.15</mn></mrow><annotation encoding="application/x-tex">-4.26 \mathbin{\pm} 0.15</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">Reengagement</td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.22</mn><mo>±</mo><mn>0.00</mn></mrow><annotation encoding="application/x-tex">0.22 \mathbin{\pm} 0.00</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2.92</mn><mo>±</mo><mn>2.07</mn></mrow><annotation encoding="application/x-tex">2.92 \mathbin{\pm} 2.07</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>3.08</mn><mo>±</mo><mn>2.18</mn></mrow><annotation encoding="application/x-tex">3.08 \mathbin{\pm} 2.18</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>2.70</mn><mo>±</mo><mn>2.07</mn></mrow><annotation encoding="application/x-tex">+2.70 \mathbin{\pm} 2.07</annotation></semantics></math></td>
<td
style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>2.86</mn><mo>±</mo><mn>2.18</mn></mrow><annotation encoding="application/x-tex">+2.86 \mathbin{\pm} 2.18</annotation></semantics></math></td>
</tr>
</tbody>
</table>

</div>

<div class="landscape">

# Complete Publication Status Results

Table <a href="#tab:supp-statuses" data-reference-type="ref"
data-reference="tab:supp-statuses">15</a> reports learner-level
outcome-model collapse.
Table <a href="#tab:supp-shape-statuses" data-reference-type="ref"
data-reference="tab:supp-shape-statuses">16</a> summarizes shape
estimability by dataset and generator. The machine-readable
`supplement-generated/rare-shape-estimability.csv` retains all 126
seed-level pathway rows, including exact support counts and fully
estimable combinations.
Table <a href="#tab:supp-applicability" data-reference-type="ref"
data-reference="tab:supp-applicability">17</a> then records the
dataset-level contract for every metric in the dictionary. The companion
files `supplement-generated/complete-publication-statuses.csv` and
`supplement-generated/metric-applicability.csv` retain the same status
and applicability information in machine-readable form.

| Dataset | Generator | Task | Overall task status and seeds | Non-estimable arms and seeds |
|:---|:---|:---|:---|:---|
| Dataset | Generator | Task | Overall task status and seeds | Non-estimable arms and seeds |
| ASSISTments | Markov | Persistent Low-Correctness Prediction | completed (3): 20260703, 20260704, 20260705 | None |
| ASSISTments | Markov | Recovery Prediction | completed (3): 20260703, 20260704, 20260705 | None |
| ASSISTments | SeqVAE | Persistent Low-Correctness Prediction | completed (2): 20260703, 20260705; completed with model collapse (1): 20260704 | Standard synthetic only (TSTR): not estimable model collapse (1): 20260704 |
| ASSISTments | SeqVAE | Recovery Prediction | completed (3): 20260703, 20260704, 20260705 | None |
| ASSISTments | TimeGAN | Persistent Low-Correctness Prediction | completed with model collapse (3): 20260703, 20260704, 20260705 | Standard synthetic only (TSTR): not estimable model collapse (3): 20260703, 20260704, 20260705 Targeted synthetic only: not estimable model collapse (3): 20260703, 20260704, 20260705 |
| ASSISTments | TimeGAN | Recovery Prediction | completed (1): 20260704; completed with model collapse (2): 20260703, 20260705 | Standard synthetic only (TSTR): not estimable model collapse (2): 20260703, 20260705 Targeted synthetic only: not estimable model collapse (2): 20260703, 20260705 |
| EdNet | Markov | Persistent Low-Correctness Prediction | completed (3): 20260703, 20260704, 20260705 | None |
| EdNet | Markov | Recovery Prediction | completed (3): 20260703, 20260704, 20260705 | None |
| EdNet | SeqVAE | Persistent Low-Correctness Prediction | completed (3): 20260703, 20260704, 20260705 | None |
| EdNet | SeqVAE | Recovery Prediction | completed (3): 20260703, 20260704, 20260705 | None |
| EdNet | TimeGAN | Persistent Low-Correctness Prediction | completed with model collapse (3): 20260703, 20260704, 20260705 | Standard synthetic only (TSTR): not estimable model collapse (3): 20260703, 20260704, 20260705 Targeted synthetic only: not estimable model collapse (3): 20260703, 20260704, 20260705 |
| EdNet | TimeGAN | Recovery Prediction | completed with model collapse (3): 20260703, 20260704, 20260705 | Standard synthetic only (TSTR): not estimable model collapse (3): 20260703, 20260704, 20260705 Targeted synthetic only: not estimable model collapse (3): 20260703, 20260704, 20260705 |
| OULAD | Markov | Course Failure Prediction | completed (3): 20260703, 20260704, 20260705 | None |
| OULAD | Markov | Dropout Prediction | completed (3): 20260703, 20260704, 20260705 | None |
| OULAD | SeqVAE | Course Failure Prediction | completed (3): 20260703, 20260704, 20260705 | None |
| OULAD | SeqVAE | Dropout Prediction | completed (3): 20260703, 20260704, 20260705 | None |
| OULAD | TimeGAN | Course Failure Prediction | completed (3): 20260703, 20260704, 20260705 | None |
| OULAD | TimeGAN | Dropout Prediction | completed (3): 20260703, 20260704, 20260705 | None |

Learner-level outcome-task statuses. Every source status entry is
represented: overall task status and any non-estimable arm are shown
together rather than replacing undefined metrics with zero.
{#tab:supp-statuses}

<table id="tab:supp-shape-statuses">
<caption>Rare-pathway conditional-shape support summarized by dataset
and generator. Each fraction is the number of estimable pathway–seed
cells over all eligible cells; each arm requires at least 10 qualifying
real and synthetic learners. “Affected pathways” counts pathways with at
least one non-estimable standard, targeted, or paired cell. The
accompanying <code>rare-shape-estimability.csv</code> retains all 126
seed-level records, exact statuses, and real/synthetic learner
counts.</caption>
<thead>
<tr>
<th style="text-align: left;">Dataset</th>
<th style="text-align: left;">Generator</th>
<th style="text-align: left;">Eligible cells</th>
<th style="text-align: left;">Standard shape</th>
<th style="text-align: left;">Targeted shape</th>
<th style="text-align: left;">Paired comparison</th>
<th style="text-align: left;">Affected pathways</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="7" style="text-align: left;"><em>Continued from previous
page</em></td>
</tr>
<tr>
<td style="text-align: left;">Dataset</td>
<td style="text-align: left;">Generator</td>
<td style="text-align: left;">Eligible cells</td>
<td style="text-align: left;">Standard shape</td>
<td style="text-align: left;">Targeted shape</td>
<td style="text-align: left;">Paired comparison</td>
<td style="text-align: left;">Affected pathways</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">18</td>
<td style="text-align: left;">17/18</td>
<td style="text-align: left;">18/18</td>
<td style="text-align: left;">17/18</td>
<td style="text-align: left;">1/6</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">18</td>
<td style="text-align: left;">10/18</td>
<td style="text-align: left;">12/18</td>
<td style="text-align: left;">10/18</td>
<td style="text-align: left;">3/6</td>
</tr>
<tr>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">18</td>
<td style="text-align: left;">4/18</td>
<td style="text-align: left;">2/18</td>
<td style="text-align: left;">2/18</td>
<td style="text-align: left;">6/6</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">15</td>
<td style="text-align: left;">15/15</td>
<td style="text-align: left;">15/15</td>
<td style="text-align: left;">15/15</td>
<td style="text-align: left;">0/5</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">15</td>
<td style="text-align: left;">11/15</td>
<td style="text-align: left;">11/15</td>
<td style="text-align: left;">11/15</td>
<td style="text-align: left;">2/5</td>
</tr>
<tr>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">15</td>
<td style="text-align: left;">3/15</td>
<td style="text-align: left;">3/15</td>
<td style="text-align: left;">3/15</td>
<td style="text-align: left;">4/5</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">Markov</td>
<td style="text-align: left;">9</td>
<td style="text-align: left;">9/9</td>
<td style="text-align: left;">9/9</td>
<td style="text-align: left;">9/9</td>
<td style="text-align: left;">0/3</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">SeqVAE</td>
<td style="text-align: left;">9</td>
<td style="text-align: left;">9/9</td>
<td style="text-align: left;">9/9</td>
<td style="text-align: left;">9/9</td>
<td style="text-align: left;">0/3</td>
</tr>
<tr>
<td style="text-align: left;">OULAD</td>
<td style="text-align: left;">TimeGAN</td>
<td style="text-align: left;">9</td>
<td style="text-align: left;">6/9</td>
<td style="text-align: left;">6/9</td>
<td style="text-align: left;">6/9</td>
<td style="text-align: left;">3/3</td>
</tr>
</tbody>
</table>

<table id="tab:supp-applicability">
<caption>Complete dataset-level applicability grid for the publication
metrics. “Reported” means a numeric row is emitted when estimable;
support- and model-dependent cases retain explicit statuses. N/A denotes
a missing or constant required signal, whereas excluded quantities exist
diagnostically but are not generator-comparable under the declared
postprocessing contract.</caption>
<thead>
<tr>
<th style="text-align: left;">Construct</th>
<th style="text-align: left;">Metric</th>
<th style="text-align: left;">ASSISTments</th>
<th style="text-align: left;">EdNet</th>
<th style="text-align: left;">OULAD</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="5" style="text-align: left;"><em>Continued from previous
page</em></td>
</tr>
<tr>
<td style="text-align: left;">Construct</td>
<td style="text-align: left;">Metric</td>
<td style="text-align: left;">ASSISTments</td>
<td style="text-align: left;">EdNet</td>
<td style="text-align: left;">OULAD</td>
</tr>
<tr>
<td style="text-align: left;">Population composition</td>
<td style="text-align: left;">Behavior-rate error</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
</tr>
<tr>
<td style="text-align: left;">Population composition</td>
<td style="text-align: left;">Event-type frequency divergence</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
</tr>
<tr>
<td style="text-align: left;">Population composition</td>
<td style="text-align: left;">Average feature-distribution
divergence</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
</tr>
<tr>
<td style="text-align: left;">Population composition</td>
<td style="text-align: left;">Real-versus-synthetic detectability
AUROC</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Excluded: postprocessing confounds
comparison</td>
</tr>
<tr>
<td style="text-align: left;">Learning process</td>
<td style="text-align: left;">Behavior-transition divergence</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
</tr>
<tr>
<td style="text-align: left;">Learning process</td>
<td style="text-align: left;">Three-event sequence divergence</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
</tr>
<tr>
<td style="text-align: left;">Learning process</td>
<td style="text-align: left;">Trajectory-shape error</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
</tr>
<tr>
<td style="text-align: left;">Learning process</td>
<td style="text-align: left;">Behavior-persistence error</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
</tr>
<tr>
<td style="text-align: left;">Learning process</td>
<td style="text-align: left;">Response/inactivity-gap distribution
divergence</td>
<td style="text-align: left;">N/A: no declared gap signal</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
</tr>
<tr>
<td style="text-align: left;">Learning process</td>
<td style="text-align: left;">Behavior–hint lag error</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">N/A: hint is constant zero</td>
<td style="text-align: left;">N/A: no hint signal</td>
</tr>
<tr>
<td style="text-align: left;">Learning process</td>
<td style="text-align: left;">Feature-dependence error</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
</tr>
<tr>
<td style="text-align: left;">Learner-pathway representation</td>
<td style="text-align: left;">Subgroup behavior error</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
</tr>
<tr>
<td style="text-align: left;">Learner-pathway representation</td>
<td style="text-align: left;">Subgroup-size error</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
</tr>
<tr>
<td style="text-align: left;">Learner-pathway representation</td>
<td style="text-align: left;">Subgroup hint-use error</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">N/A: hint is constant zero</td>
<td style="text-align: left;">N/A: no hint signal</td>
</tr>
<tr>
<td style="text-align: left;">Learner-pathway representation</td>
<td style="text-align: left;">Rare-group prevalence error</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
</tr>
<tr>
<td style="text-align: left;">Learner-pathway representation</td>
<td style="text-align: left;">Signed rare-group prevalence
difference</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
</tr>
<tr>
<td style="text-align: left;">Learner-pathway representation</td>
<td style="text-align: left;">Rare-group trajectory-shape error</td>
<td style="text-align: left;">Reported when support permits</td>
<td style="text-align: left;">Reported when support permits</td>
<td style="text-align: left;">Reported when support permits</td>
</tr>
<tr>
<td style="text-align: left;">Analytic usefulness</td>
<td style="text-align: left;">AUPRC</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
</tr>
<tr>
<td style="text-align: left;">Analytic usefulness</td>
<td style="text-align: left;">AUROC</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
</tr>
<tr>
<td style="text-align: left;">Analytic usefulness</td>
<td style="text-align: left;">Standard synthetic utility loss</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
</tr>
<tr>
<td style="text-align: left;">Analytic usefulness</td>
<td style="text-align: left;">Standard augmentation gain</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
</tr>
<tr>
<td style="text-align: left;">Analytic usefulness</td>
<td style="text-align: left;">Targeted-vs-standard TSTR gain</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
</tr>
<tr>
<td style="text-align: left;">Analytic usefulness</td>
<td style="text-align: left;">Targeted augmentation gain</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
</tr>
<tr>
<td style="text-align: left;">Analytic usefulness</td>
<td style="text-align: left;">Pathway-level predictive usefulness</td>
<td style="text-align: left;">Reported when support permits</td>
<td style="text-align: left;">Reported when support permits</td>
<td style="text-align: left;">Reported when support permits</td>
</tr>
<tr>
<td style="text-align: left;">Analytic usefulness</td>
<td style="text-align: left;">Rare-group prevalence-match change</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
<td style="text-align: left;">Reported</td>
</tr>
<tr>
<td style="text-align: left;">Analytic usefulness</td>
<td style="text-align: left;">Rare-group shape-error reduction</td>
<td style="text-align: left;">Reported when support permits</td>
<td style="text-align: left;">Reported when support permits</td>
<td style="text-align: left;">Reported when support permits</td>
</tr>
<tr>
<td style="text-align: left;">Analytic usefulness</td>
<td style="text-align: left;">Learner-level outcome-task AUPRC/AUROC and
contrasts</td>
<td style="text-align: left;">Reported or status if model collapses</td>
<td style="text-align: left;">Reported or status if model collapses</td>
<td style="text-align: left;">Reported or status if model collapses</td>
</tr>
<tr>
<td style="text-align: left;">Disclosure risk</td>
<td style="text-align: left;">Exact-duplicate rate</td>
<td style="text-align: left;">Reported on full modeled trajectory</td>
<td style="text-align: left;">Reported on full modeled trajectory</td>
<td style="text-align: left;">Excluded: postprocessing confounds
comparison</td>
</tr>
<tr>
<td style="text-align: left;">Disclosure risk</td>
<td style="text-align: left;">Near-duplicate rate</td>
<td style="text-align: left;">Reported on full modeled trajectory</td>
<td style="text-align: left;">Reported on full modeled trajectory</td>
<td style="text-align: left;">Excluded: postprocessing confounds
comparison</td>
</tr>
<tr>
<td style="text-align: left;">Disclosure risk</td>
<td style="text-align: left;">Mean nearest-training distance</td>
<td style="text-align: left;">Reported on full modeled trajectory</td>
<td style="text-align: left;">Reported on full modeled trajectory</td>
<td style="text-align: left;">Excluded: postprocessing confounds
comparison</td>
</tr>
<tr>
<td style="text-align: left;">Disclosure risk</td>
<td style="text-align: left;">Membership-inference advantage</td>
<td style="text-align: left;">Reported on full modeled trajectory</td>
<td style="text-align: left;">Reported on full modeled trajectory</td>
<td style="text-align: left;">Excluded: postprocessing confounds
comparison</td>
</tr>
<tr>
<td style="text-align: left;">Disclosure risk</td>
<td style="text-align: left;">Behavior-projection exact-duplicate
rate</td>
<td style="text-align: left;">N/A: full-trajectory audit used</td>
<td style="text-align: left;">N/A: full-trajectory audit used</td>
<td style="text-align: left;">Reported on behavior projection only</td>
</tr>
<tr>
<td style="text-align: left;">Disclosure risk</td>
<td style="text-align: left;">Targeted-minus-standard privacy-risk
change</td>
<td style="text-align: left;">Reported on full modeled trajectory</td>
<td style="text-align: left;">Reported on full modeled trajectory</td>
<td style="text-align: left;">Reported on behavior projection only</td>
</tr>
</tbody>
</table>

</div>

# Supporting Disclosure-Audit Figure

<figure id="fig:rq2-disclosure-risk" data-latex-placement="!htbp">

<figcaption>Supporting dataset-level disclosure-risk audit. Panel (a)
reports membership-inference advantage for full normalized tutoring
trajectories over the declared audit columns. Panel (b) reports exact
copying for the OULAD dominant-activity/engagement projection;
full-record membership inference is not generator-comparable there. The
change in audit signal is targeted minus standard; positive values
indicate a larger tested indicator after targeting. Points are means and
whiskers show the full mean
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>±</mi><annotation encoding="application/x-tex">\pm</annotation></semantics></math>
population SD across three seeds; a descriptive SD whisker may extend
below zero. The audit does not estimate disclosure risk specifically for
learners on uncommon pathways. These are empirical memorization
indicators, not formal privacy guarantees.</figcaption>
</figure>

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-davis2006precision" class="csl-entry">

Davis, Jesse, and Mark Goadrich. 2006. “The Relationship Between
Precision-Recall and ROC Curves.” *<span class="nocase">Proceedings of
the 23rd International Conference on Machine Learning</span>*, 233–40.
<https://doi.org/10.1145/1143844.1143874>.

</div>

<div id="ref-esteban2017realvalued" class="csl-entry">

Esteban, Cristóbal, Stephanie L. Hyland, and Gunnar Rätsch. 2017.
“Real-Valued (Medical) Time Series Generation with Recurrent Conditional
GANs.” *arXiv Preprint arXiv:1706.02633*, ahead of print.
<https://doi.org/10.48550/arXiv.1706.02633>.

</div>

<div id="ref-fawcett2006roc" class="csl-entry">

Fawcett, Tom. 2006. “An Introduction to ROC Analysis.” *Pattern
Recognition Letters* 27 (8): 861–74.
<https://doi.org/10.1016/j.patrec.2005.10.010>.

</div>

<div id="ref-lin1991divergence" class="csl-entry">

Lin, Jianhua. 1991. “Divergence Measures Based on the Shannon Entropy.”
*IEEE Transactions on Information Theory* 37 (1): 145–51.
<https://doi.org/10.1109/18.61115>.

</div>

<div id="ref-lopezpaz2017c2st" class="csl-entry">

Lopez-Paz, David, and Maxime Oquab. 2017. “Revisiting Classifier
Two-Sample Tests.” *<span class="nocase">International Conference on
Learning Representations</span>*.
<https://openreview.net/forum?id=SJkXfE5xx>.

</div>

<div id="ref-saito2015precision" class="csl-entry">

Saito, Takaya, and Marc Rehmsmeier. 2015. “The Precision-Recall Plot Is
More Informative Than the ROC Plot When Evaluating Binary Classifiers on
Imbalanced Datasets.” *PLOS ONE* 10 (3): e0118432.
<https://doi.org/10.1371/journal.pone.0118432>.

</div>

<div id="ref-shokri2017membership" class="csl-entry">

Shokri, Reza, Marco Stronati, Congzheng Song, and Vitaly Shmatikov.
2017. “Membership Inference Attacks Against Machine Learning Models.”
*<span class="nocase">2017 IEEE Symposium on Security and
Privacy</span>*, 3–18. <https://doi.org/10.1109/SP.2017.41>.

</div>

<div id="ref-stadler2022synthetic" class="csl-entry">

Stadler, Theresa, Bristena Oprisanu, and Carmela Troncoso. 2022.
“Synthetic Data—Anonymisation Groundhog Day.” *31st USENIX Security
Symposium*, 1451–68.

</div>

<div id="ref-strehl2002cluster" class="csl-entry">

Strehl, Alexander, and Joydeep Ghosh. 2002. “Cluster Ensembles—a
Knowledge Reuse Framework for Combining Multiple Partitions.”
*<span class="nocase">Journal of Machine Learning Research</span>* 3:
583–617.

</div>

<div id="ref-yoon2019timegan" class="csl-entry">

Yoon, Jinsung, Daniel Jarrett, and Mihaela van der Schaar. 2019.
“Time-Series Generative Adversarial Networks.”
*<span class="nocase">Advances in Neural Information Processing
Systems</span>* 32: 5508–18.

</div>

</div>
