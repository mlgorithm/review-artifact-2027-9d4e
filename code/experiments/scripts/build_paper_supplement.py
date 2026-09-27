#!/usr/bin/env python3
"""Build complete supplementary result tables from publication-schema reports.

Every numeric row in each final report's ``metric_summary`` is emitted exactly
once, with all source summary fields retained.  Outcome-model collapse,
rare-pathway shape support, and dataset-level applicability are exported as
explicit non-numeric results rather than being represented by absent or zero
values.  The accompanying CSVs make the appendix mechanically auditable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev


DATASETS = {
    "assistments_2009_2010_skill_builder": "ASSISTments",
    "ednet_kt1": "EdNet",
    "oulad_weekly_engagement": "OULAD",
}
MODELS = {
    "markov_ngram": "Markov",
    "sequence_vae_timevae": "SeqVAE",
    "timegan": "TimeGAN",
}
SEEDS = (20260703, 20260704, 20260705)
CONSTRUCT_ORDER = (
    "Population composition",
    "Learning process",
    "Learner-pathway representation",
    "Analytic usefulness",
    "Disclosure risk",
)

TOKEN_LABELS = {
    "real_train": "Real only (TRTR)",
    "synthetic_train": "Standard synthetic only (TSTR)",
    "real_plus_synthetic": "Real + standard synthetic",
    "tail_targeted_synthetic": "Targeted synthetic only",
    "real_plus_tail_targeted_synthetic": "Real + targeted synthetic",
    "synthetic_utility_gap": "Standard synthetic utility loss",
    "real_plus_synthetic_gain_over_real": "Standard augmentation gain",
    "tail_targeted_tstr_gain": "Targeted-vs-standard TSTR gain",
    "targeted_augmentation_gain_over_real": "Targeted augmentation gain",
    "standard": "Standard synthetic",
    "tail_targeted": "Targeted synthetic",
    "tail_targeted_minus_standard_risk": "Targeted-minus-standard risk",
    # Keep machine-readable report keys stable while presenting constructs at
    # the level supported by their observable behavioral rules.
    "persistent_failure": "Persistent Low Correctness",
    "persistent_failure_prediction": "Persistent Low-Correctness Prediction",
    "persistent_misconception": "Persistent Skill Difficulty",
    "late_failure": "Late Correctness Decline",
    "rapid_guessing": "Rapid Low-Accuracy Responding",
}
METRIC_LABELS = {
    "primary_signal_rate_error": "Behavior-rate error",
    "skill_frequency_js": "Event-type frequency divergence",
    "modeled_value_marginal_js_mean": "Average feature-distribution divergence",
    "detectability_auc": "Real-versus-synthetic detectability AUROC",
    "correctness_transition_js": "Behavior-transition divergence",
    "skill_trigram_js": "Three-event sequence divergence",
    "correctness_curve_mae": "Trajectory-shape error",
    "correctness_autocorrelation_mae": "Behavior-persistence error",
    "correct_hint_cross_correlation_mae": "Behavior--hint lag error",
    "inter_event_time_js": "Response/inactivity-gap distribution divergence",
    "prespecified_signal_nmi_mae": "Feature-dependence error",
    "future_primary_signal_rate_mae": "Subgroup behavior error",
    "learner_share_mae": "Subgroup-size error",
    "future_hint_rate_mae": "Subgroup hint-use error",
    "prevalence_error": "Rare-group prevalence error",
    "primary_signal_curve_mae": "Rare-group trajectory-shape error",
    "prevalence_error_reduction": "Rare-group prevalence-error reduction",
    "primary_signal_curve_mae_reduction": "Rare-group trajectory-shape-error reduction",
    "exact_duplicate_rate": "Exact-duplicate rate",
    "behavior_projection_exact_duplicate_rate": "Behavior-projection exact-duplicate rate",
    "near_duplicate_rate": "Near-duplicate rate",
    "mean_nearest_neighbor_distance": "Mean nearest-training distance",
    "membership_inference_attack_advantage": "Membership-inference advantage",
    "exact_duplicate_rate_risk_delta": "Exact-duplicate risk change",
    "behavior_projection_exact_duplicate_rate_risk_delta": "Behavior-projection exact-copy risk change",
    "near_duplicate_rate_risk_delta": "Near-duplicate risk change",
    "mean_nearest_exposure_risk_delta": "Nearest-neighbor exposure risk change",
    "membership_inference_attack_advantage_risk_delta": "Membership-inference risk change",
    "auprc": "AUPRC",
    "auroc": "AUROC",
}

# This is the declared publication-metric contract, not a list inferred from
# whichever values happened to be estimable in a particular run.  ``reported``
# means that numeric rows are emitted whenever the metric is estimable;
# ``support_dependent`` and ``model_dependent`` point readers to the explicit
# status exports; ``not_applicable`` means the dataset lacks the required
# observable signal; and ``excluded`` identifies a deliberately non-comparable
# quantity.
APPLICABILITY_ROWS = (
    ("Population composition", "Behavior-rate error", "reported", "reported", "reported"),
    ("Population composition", "Event-type frequency divergence", "reported", "reported", "reported"),
    ("Population composition", "Average feature-distribution divergence", "reported", "reported", "reported"),
    ("Population composition", "Real-versus-synthetic detectability AUROC", "reported", "reported", "excluded_postprocessing"),
    ("Learning process", "Behavior-transition divergence", "reported", "reported", "reported"),
    ("Learning process", "Three-event sequence divergence", "reported", "reported", "reported"),
    ("Learning process", "Trajectory-shape error", "reported", "reported", "reported"),
    ("Learning process", "Behavior-persistence error", "reported", "reported", "reported"),
    ("Learning process", "Response/inactivity-gap distribution divergence", "not_applicable_no_gap", "reported", "reported"),
    ("Learning process", "Behavior--hint lag error", "reported", "not_applicable_constant_hint", "not_applicable_no_hint"),
    ("Learning process", "Feature-dependence error", "reported", "reported", "reported"),
    ("Learner-pathway representation", "Subgroup behavior error", "reported", "reported", "reported"),
    ("Learner-pathway representation", "Subgroup-size error", "reported", "reported", "reported"),
    ("Learner-pathway representation", "Subgroup hint-use error", "reported", "not_applicable_constant_hint", "not_applicable_no_hint"),
    ("Learner-pathway representation", "Rare-group prevalence error", "reported", "reported", "reported"),
    ("Learner-pathway representation", "Signed rare-group prevalence difference", "reported", "reported", "reported"),
    ("Learner-pathway representation", "Rare-group trajectory-shape error", "support_dependent", "support_dependent", "support_dependent"),
    ("Analytic usefulness", "AUPRC", "reported", "reported", "reported"),
    ("Analytic usefulness", "AUROC", "reported", "reported", "reported"),
    ("Analytic usefulness", "Standard synthetic utility loss", "reported", "reported", "reported"),
    ("Analytic usefulness", "Standard augmentation gain", "reported", "reported", "reported"),
    ("Analytic usefulness", "Targeted-vs-standard TSTR gain", "reported", "reported", "reported"),
    ("Analytic usefulness", "Targeted augmentation gain", "reported", "reported", "reported"),
    ("Analytic usefulness", "Tail learnability", "support_dependent", "support_dependent", "support_dependent"),
    ("Analytic usefulness", "Rare-group prevalence-match change", "reported", "reported", "reported"),
    ("Analytic usefulness", "Rare-group shape-error reduction", "support_dependent", "support_dependent", "support_dependent"),
    ("Analytic usefulness", "Learner-level outcome-task AUPRC/AUROC and contrasts", "model_dependent", "model_dependent", "model_dependent"),
    ("Disclosure risk", "Exact-duplicate rate", "full_trajectory", "full_trajectory", "excluded_postprocessing"),
    ("Disclosure risk", "Near-duplicate rate", "full_trajectory", "full_trajectory", "excluded_postprocessing"),
    ("Disclosure risk", "Mean nearest-training distance", "full_trajectory", "full_trajectory", "excluded_postprocessing"),
    ("Disclosure risk", "Membership-inference advantage", "full_trajectory", "full_trajectory", "excluded_postprocessing"),
    ("Disclosure risk", "Behavior-projection exact-duplicate rate", "not_applicable_full_trajectory", "not_applicable_full_trajectory", "behavior_projection"),
    ("Disclosure risk", "Targeted-minus-standard privacy-risk change", "full_trajectory", "full_trajectory", "behavior_projection"),
)

APPLICABILITY_TEXT = {
    "reported": "Reported",
    "support_dependent": "Reported when support permits",
    "model_dependent": "Reported or status if model collapses",
    "full_trajectory": "Reported on full modeled trajectory",
    "behavior_projection": "Reported on behavior projection only",
    "not_applicable_no_gap": "N/A: no declared gap signal",
    "not_applicable_constant_hint": "N/A: hint is constant zero",
    "not_applicable_no_hint": "N/A: no hint signal",
    "not_applicable_full_trajectory": "N/A: full-trajectory audit used",
    "excluded_postprocessing": "Excluded: postprocessing confounds comparison",
}


def _escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _title(token: str) -> str:
    return TOKEN_LABELS.get(token, token.replace("_", " ").title())


def _construct(path: str) -> str:
    if path.startswith(("rq1.average_fidelity.", "rq1.detectability.")):
        return "Population composition"
    if path.startswith(("rq1.temporal_fidelity.", "rq1.cross_signal_dependence.")):
        return "Learning process"
    if path.startswith(("rq1.subgroup_fidelity.", "rq1.tail_fidelity.")):
        return "Learner-pathway representation"
    if path.startswith("rq2.privacy_risk."):
        return "Disclosure risk"
    if path.startswith((
        "rq2.downstream_utility.",
        "rq2.tail_learnability.",
        "rq2.tail_targeting_fidelity_gain.",
        "rq2.learner_level_outcome_tasks.",
    )):
        return "Analytic usefulness"
    raise ValueError(f"Unmapped publication metric: {path}")


def _direction(path: str) -> str:
    if path.endswith("detectability_auc"):
        return "0.5 best"
    if "mean_nearest_neighbor_distance" in path:
        return "Higher"
    if path.endswith(("auprc", "auroc")):
        if ".synthetic_utility_gap." in path:
            return "Lower"
        return "Higher"
    if path.endswith(("_reduction", "_gain")):
        return "Higher"
    if "gain_over_real" in path or "tail_targeted_tstr_gain" in path:
        return "Higher"
    return "Lower"


def _scope_and_measure(path: str) -> tuple[str, str]:
    parts = path.split(".")
    leaf = parts[-1]
    if path.startswith("rq1.tail_fidelity."):
        return _title(parts[2]), METRIC_LABELS[leaf]
    if path.startswith("rq2.tail_targeting_fidelity_gain."):
        return _title(parts[2]), METRIC_LABELS[leaf]
    if path.startswith("rq2.tail_learnability."):
        group = _title(parts[2])
        if parts[3] == "arms":
            return f"{group}; {_title(parts[4])}", METRIC_LABELS[leaf]
        return f"{group}; {_title(parts[3])}", METRIC_LABELS[leaf]
    if path.startswith("rq2.downstream_utility."):
        if parts[2] == "arms":
            return _title(parts[3]), METRIC_LABELS[leaf]
        return _title(parts[2]), METRIC_LABELS[leaf]
    if path.startswith("rq2.learner_level_outcome_tasks."):
        task = _title(parts[2])
        if parts[3] == "arms":
            return f"{task}; {_title(parts[4])}", METRIC_LABELS[leaf]
        return f"{task}; {_title(parts[3])}", METRIC_LABELS[leaf]
    if path.startswith("rq2.privacy_risk."):
        return _title(parts[2]), METRIC_LABELS[leaf]
    return "All eligible learners/events", METRIC_LABELS.get(leaf, _title(leaf))


def _load_rows(report_root: Path) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    statuses: list[dict] = []
    for dataset_key, dataset in DATASETS.items():
        for model_key, model in MODELS.items():
            path = report_root / dataset_key / model_key / "multi_seed_summary.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("publication_schema_version") != "publication_metrics_v2":
                raise ValueError(f"Unexpected publication schema in {path}")
            if report.get("seeds") != list(SEEDS):
                raise ValueError(f"Unexpected seed set in {path}")
            for metric_path, summary in sorted(report["metric_summary"].items()):
                required_fields = {"mean", "std", "min", "max", "n"}
                missing_fields = required_fields - set(summary)
                if missing_fields:
                    raise ValueError(
                        f"Metric summary {metric_path} in {path} is missing "
                        f"{sorted(missing_fields)}"
                    )
                scope, measure = _scope_and_measure(metric_path)
                rows.append(
                    {
                        "construct": _construct(metric_path),
                        "dataset": dataset,
                        "model": model,
                        "scope": scope,
                        "measure": measure,
                        "direction": _direction(metric_path),
                        "mean": summary["mean"],
                        "std": summary["std"],
                        "min": summary["min"],
                        "max": summary["max"],
                        "n": summary["n"],
                        "missing_runs": summary.get("missing_runs", 0),
                        "metric_path": metric_path,
                    }
                )
            for status_path, status in sorted(
                report.get("publication_status_summary", {}).items()
            ):
                statuses.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "status_path": status_path,
                        "counts": json.dumps(status.get("counts", {}), sort_keys=True),
                        "seeds_by_status": json.dumps(
                            status.get("seeds_by_status", {}), sort_keys=True
                        ),
                    }
                )
    return rows, statuses


def _load_rare_shape_status_rows(report_root: Path) -> list[dict]:
    """Export seed-level support for standard, targeted, and paired shape metrics."""
    rows: list[dict] = []
    for dataset_key, dataset in DATASETS.items():
        for model_key, model in MODELS.items():
            for seed in SEEDS:
                path = (
                    report_root
                    / dataset_key
                    / model_key
                    / f"seed_{seed}"
                    / "generation_evaluation_report.json"
                )
                report = json.loads(path.read_text(encoding="utf-8"))
                publication_metrics = report["publication"]["metrics"]
                standard_groups = publication_metrics["rq1"]["tail_fidelity"]
                reductions = publication_metrics["rq2"][
                    "tail_targeting_fidelity_gain"
                ]
                targeted_shape = report["tail_targeted_evaluation"][
                    "tail_fidelity"
                ]["tail_shape"]["by_tail_group"]

                for published_name, standard in sorted(standard_groups.items()):
                    raw_name = standard["source_tail_group"]
                    targeted = targeted_shape[raw_name]
                    standard_status = standard.get(
                        "primary_signal_curve_mae_status",
                        "estimable"
                        if "primary_signal_curve_mae" in standard
                        else "not_estimable",
                    )
                    # The publication contract requires at least 10 qualifying
                    # learners in both sets.  The raw diagnostic's
                    # ``exploratory`` flag is intentionally looser and must not
                    # be used as the publication estimability decision.
                    targeted_status = (
                        "estimable"
                        if int(targeted["real_learners"]) >= 10
                        and int(targeted["synthetic_learners"]) >= 10
                        else "not_estimable"
                    )
                    reduction_status = (
                        "estimable"
                        if "primary_signal_curve_mae_reduction"
                        in reductions[published_name]
                        else "not_estimable"
                    )
                    if reduction_status == "estimable" and (
                        standard_status != "estimable"
                        or targeted_status != "estimable"
                    ):
                        raise AssertionError(
                            f"Paired reduction is present without two estimable arms: {path}, "
                            f"{published_name}"
                        )
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "seed": seed,
                            "rare_group": _title(published_name),
                            "source_tail_group": raw_name,
                            "standard_status": standard_status,
                            "standard_real_learners": standard["real_learners"],
                            "standard_synthetic_learners": standard[
                                "synthetic_learners"
                            ],
                            "targeted_status": targeted_status,
                            "targeted_real_learners": targeted["real_learners"],
                            "targeted_synthetic_learners": targeted[
                                "synthetic_learners"
                            ],
                            "paired_reduction_status": reduction_status,
                        }
                    )
    return rows


def _applicability_rows() -> list[dict]:
    rows: list[dict] = []
    dataset_columns = (
        ("ASSISTments", 2),
        ("EdNet", 3),
        ("OULAD", 4),
    )
    for contract_row in APPLICABILITY_ROWS:
        construct, measure = contract_row[:2]
        for dataset, index in dataset_columns:
            status = contract_row[index]
            rows.append(
                {
                    "construct": construct,
                    "measure": measure,
                    "dataset": dataset,
                    "status": status,
                    "display": APPLICABILITY_TEXT[status],
                }
            )
    return rows


def _validate_applicability_contract(rows: list[dict]) -> None:
    """Ensure schema-dependent applicability cells agree with numeric exports."""
    present = {(row["dataset"], row["measure"]) for row in rows}
    expected_presence = {
        "Response/inactivity-gap distribution divergence": {"EdNet", "OULAD"},
        "Behavior--hint lag error": {"ASSISTments"},
        "Subgroup hint-use error": {"ASSISTments"},
        "Real-versus-synthetic detectability AUROC": {"ASSISTments", "EdNet"},
        "Exact-duplicate rate": {"ASSISTments", "EdNet"},
        "Near-duplicate rate": {"ASSISTments", "EdNet"},
        "Mean nearest-training distance": {"ASSISTments", "EdNet"},
        "Membership-inference advantage": {"ASSISTments", "EdNet"},
        "Behavior-projection exact-duplicate rate": {"OULAD"},
    }
    for measure, expected_datasets in expected_presence.items():
        observed_datasets = {
            dataset
            for dataset in DATASETS.values()
            if (dataset, measure) in present
        }
        if observed_datasets != expected_datasets:
            raise AssertionError(
                f"Applicability contract mismatch for {measure}: expected "
                f"{sorted(expected_datasets)}, observed {sorted(observed_datasets)}"
            )


def _load_signed_prevalence_rows(report_root: Path) -> list[dict]:
    collected: dict[tuple[str, str, str, str], dict[str, list[float]]] = {}
    for dataset_key, dataset in DATASETS.items():
        for model_key, model in MODELS.items():
            for seed in (20260703, 20260704, 20260705):
                path = (
                    report_root
                    / dataset_key
                    / model_key
                    / f"seed_{seed}"
                    / "generation_evaluation_report.json"
                )
                report = json.loads(path.read_text(encoding="utf-8"))
                publication_groups = (
                    report.get("publication", {})
                    .get("metrics", {})
                    .get("rq1", {})
                    .get("tail_fidelity", {})
                )
                standard = report["tail_fidelity"]
                targeted = report["tail_targeted_evaluation"]["tail_fidelity"]
                for published_name, group in publication_groups.items():
                    raw_name = group["source_tail_group"]
                    real = float(standard["tail_prevalence_real"][raw_name])
                    standard_synthetic = float(
                        standard["tail_prevalence_synthetic"][raw_name]
                    )
                    targeted_synthetic = float(
                        targeted["tail_prevalence_synthetic"][raw_name]
                    )
                    values = collected.setdefault(
                        (dataset, model, published_name, raw_name),
                        {
                            "real": [],
                            "standard": [],
                            "targeted": [],
                            "standard_signed": [],
                            "targeted_signed": [],
                        },
                    )
                    values["real"].append(real)
                    values["standard"].append(standard_synthetic)
                    values["targeted"].append(targeted_synthetic)
                    values["standard_signed"].append(standard_synthetic - real)
                    values["targeted_signed"].append(targeted_synthetic - real)

    rows = []
    for (dataset, model, published_name, raw_name), values in collected.items():
        row = {
            "dataset": dataset,
            "model": model,
            "rare_group": _title(published_name),
            "source_tail_group": raw_name,
        }
        for key, observations in values.items():
            row[f"{key}_mean"] = mean(observations)
            row[f"{key}_std"] = pstdev(observations)
        rows.append(row)
    return rows


def _result_text(row: dict) -> str:
    return rf"${float(row['mean']):.4f} \mathbin{{\pm}} {float(row['std']):.4f}$"


def _write_construct_table(path: Path, construct: str, rows: list[dict]) -> None:
    selected = [row for row in rows if row["construct"] == construct]
    lines = [
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{@{}p{2.25cm}p{2.05cm}p{4.7cm}p{5.0cm}p{2.6cm}p{1.25cm}@{}}",
        rf"\caption{{Complete publication results for {_escape(construct.lower())}. Every numeric metric-summary entry is shown; SD is the population SD across the fixed seeds and $n$ is the number of estimable seeds.}}\label{{tab:supp-{construct.lower().replace(' ', '-')}}}\\",
        r"\toprule",
        r"Dataset & Generator & Scope / comparison & Metric & Mean $\pm$ SD ($n$) & Direction \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{6}{l}{\textit{Continued from previous page}}\\",
        r"\toprule",
        r"Dataset & Generator & Scope / comparison & Metric & Mean $\pm$ SD ($n$) & Direction \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{6}{r}{\textit{Continued on next page}}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for row in selected:
        n_text = str(row["n"])
        if row["missing_runs"]:
            n_text += rf"; {row['missing_runs']} missing"
        lines.append(
            " & ".join(
                [
                    _escape(row["dataset"]),
                    _escape(row["model"]),
                    _escape(row["scope"]),
                    _escape(row["measure"]),
                    f"{_result_text(row)} ({n_text})",
                    _escape(row["direction"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\end{longtable}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_status_table(path: Path, statuses: list[dict]) -> None:
    grouped: dict[tuple[str, str, str], dict] = {}
    for row in statuses:
        path_parts = row["status_path"].split(".")
        task = _title(path_parts[2])
        group = grouped.setdefault(
            (row["dataset"], row["model"], task),
            {"overall": "", "arms": []},
        )
        counts = json.loads(row["counts"])
        seeds = json.loads(row["seeds_by_status"])
        detail = "; ".join(
            f"{_title(status).lower()} ({counts[status]}): "
            f"{', '.join(str(seed) for seed in seed_list)}"
            for status, seed_list in seeds.items()
        )
        if "non_estimable_arms" in path_parts:
            group["arms"].append(f"{_title(path_parts[-1])}: {detail}")
        else:
            group["overall"] = detail

    lines = [
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{@{}p{2.2cm}p{2.0cm}p{3.6cm}p{7.05cm}p{7.65cm}@{}}",
        r"\caption{Learner-level outcome-task statuses. Every source status entry is represented: overall task status and any non-estimable arm are shown together rather than replacing undefined metrics with zero.}\label{tab:supp-statuses}\\",
        r"\toprule",
        r"Dataset & Generator & Task & Overall task status and seeds & Non-estimable arms and seeds \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Dataset & Generator & Task & Overall task status and seeds & Non-estimable arms and seeds \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for (dataset, model, task), group in grouped.items():
        lines.append(
            " & ".join(
                _escape(str(value))
                for value in (
                    dataset,
                    model,
                    task,
                    group["overall"],
                    " ".join(group["arms"]) if group["arms"] else "None",
                )
            )
            + r" \\"
        )
    lines.extend([r"\end{longtable}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _shape_support_cell(rows: list[dict], arm: str) -> str:
    status_key = f"{arm}_status"
    real_key = f"{arm}_real_learners"
    synthetic_key = f"{arm}_synthetic_learners"
    estimable = [row for row in rows if row[status_key] == "estimable"]
    not_estimable = [row for row in rows if row[status_key] != "estimable"]
    text = f"{len(estimable)}/{len(SEEDS)} estimable"
    if not_estimable:
        detail = "; ".join(
            f"{row['seed']} ({row[real_key]}/{row[synthetic_key]})"
            for row in not_estimable
        )
        text += f"; not estimable: {detail}"
    return text


def _reduction_support_cell(rows: list[dict]) -> str:
    estimable = [
        row for row in rows if row["paired_reduction_status"] == "estimable"
    ]
    missing = [
        str(row["seed"])
        for row in rows
        if row["paired_reduction_status"] != "estimable"
    ]
    text = f"{len(estimable)}/{len(SEEDS)} estimable"
    if missing:
        text += f"; not estimable: {', '.join(missing)}"
    return text


def _write_rare_shape_status_table(path: Path, rows: list[dict]) -> int:
    pathway_groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        pathway_groups.setdefault(
            (row["dataset"], row["model"], row["rare_group"]), []
        ).append(row)
    incomplete = {
        key: sorted(value, key=lambda row: row["seed"])
        for key, value in pathway_groups.items()
        if any(
            row[status_key] != "estimable"
            for row in value
            for status_key in (
                "standard_status",
                "targeted_status",
                "paired_reduction_status",
            )
        )
    }
    dataset_model_groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        dataset_model_groups.setdefault((row["dataset"], row["model"]), []).append(row)
    lines = [
        r"\scriptsize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{longtable}{@{}p{2.3cm}p{1.8cm}p{2.7cm}p{3.0cm}p{3.0cm}p{3.1cm}p{3.3cm}@{}}",
        r"\caption{Rare-pathway conditional-shape support summarized by dataset and generator. Each fraction is the number of estimable pathway--seed cells over all eligible cells; each arm requires at least 10 qualifying real and synthetic learners. ``Affected pathways'' counts pathways with at least one non-estimable standard, targeted, or paired cell. The accompanying \texttt{rare-shape-estimability.csv} retains all 126 seed-level records, exact statuses, and real/synthetic learner counts.}\label{tab:supp-shape-statuses}\\",
        r"\toprule",
        r"Dataset & Generator & Eligible cells & Standard shape & Targeted shape & Paired comparison & Affected pathways \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{7}{l}{\textit{Continued from previous page}}\\",
        r"\toprule",
        r"Dataset & Generator & Eligible cells & Standard shape & Targeted shape & Paired comparison & Affected pathways \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for (dataset, model), group_rows in dataset_model_groups.items():
        eligible = len(group_rows)
        pathway_count = len({row["rare_group"] for row in group_rows})
        affected_count = sum(
            (dataset, model, rare_group) in incomplete
            for rare_group in {row["rare_group"] for row in group_rows}
        )
        standard_estimable = sum(
            row["standard_status"] == "estimable" for row in group_rows
        )
        targeted_estimable = sum(
            row["targeted_status"] == "estimable" for row in group_rows
        )
        paired_estimable = sum(
            row["paired_reduction_status"] == "estimable" for row in group_rows
        )
        lines.append(
            " & ".join(
                _escape(value)
                for value in (
                    dataset,
                    model,
                    str(eligible),
                    f"{standard_estimable}/{eligible}",
                    f"{targeted_estimable}/{eligible}",
                    f"{paired_estimable}/{eligible}",
                    f"{affected_count}/{pathway_count}",
                )
            )
            + r" \\"
        )
    lines.extend([r"\end{longtable}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return len(incomplete)


def _write_applicability_table(path: Path) -> None:
    lines = [
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{@{}p{2.8cm}p{5.2cm}p{4.65cm}p{4.65cm}p{4.65cm}@{}}",
        r"\caption{Complete dataset-level applicability grid for the publication metrics. ``Reported'' means a numeric row is emitted when estimable; support- and model-dependent cases retain explicit statuses. N/A denotes a missing or constant required signal, whereas excluded quantities exist diagnostically but are not generator-comparable under the declared postprocessing contract.}\label{tab:supp-applicability}\\",
        r"\toprule",
        r"Construct & Metric & ASSISTments & EdNet & OULAD \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{5}{l}{\textit{Continued from previous page}}\\",
        r"\toprule",
        r"Construct & Metric & ASSISTments & EdNet & OULAD \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for construct, measure, assistments, ednet, oulad in APPLICABILITY_ROWS:
        lines.append(
            " & ".join(
                _escape(value)
                for value in (
                    construct,
                    measure,
                    APPLICABILITY_TEXT[assistments],
                    APPLICABILITY_TEXT[ednet],
                    APPLICABILITY_TEXT[oulad],
                )
            )
            + r" \\"
        )
    lines.extend([r"\end{longtable}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _percent_result(row: dict, key: str, *, signed: bool = False) -> str:
    mean_value = 100 * float(row[f"{key}_mean"])
    std_value = 100 * float(row[f"{key}_std"])
    sign = "+" if signed and mean_value > 0 else ""
    return rf"${sign}{mean_value:.2f} \mathbin{{\pm}} {std_value:.2f}$"


def _write_signed_prevalence_table(path: Path, rows: list[dict]) -> None:
    lines = [
        r"\begingroup",
        r"\fontsize{6.8}{7.6}\selectfont",
        r"\renewcommand{\arraystretch}{0.90}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\setlength{\LTpre}{2pt}",
        r"\setlength{\LTpost}{2pt}",
        r"\begin{longtable}{@{}p{2.0cm}p{1.8cm}p{3.1cm}p{2.0cm}p{2.25cm}p{2.25cm}p{2.35cm}p{2.35cm}@{}}",
        r"\caption{Signed rare-pathway prevalence results. Prevalences are learner percentages and differences are synthetic minus real in percentage points. Positive differences mean overrepresentation; negative differences mean underrepresentation. Values are mean $\pm$ population SD across the three fixed generator seeds.}\label{tab:supp-signed-prevalence}\\",
        r"\toprule",
        r"Dataset & Generator & Pathway & Real (\%) & Standard synth. (\%) & Targeted synth. (\%) & Standard signed diff. (pp) & Targeted signed diff. (pp) \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Dataset & Generator & Pathway & Real (\%) & Standard synth. (\%) & Targeted synth. (\%) & Standard signed diff. (pp) & Targeted signed diff. (pp) \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{8}{r}{\textit{Continued on next page}}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _escape(row["dataset"]),
                    _escape(row["model"]),
                    _escape(row["rare_group"]),
                    _percent_result(row, "real"),
                    _percent_result(row, "standard"),
                    _percent_result(row, "targeted"),
                    _percent_result(row, "standard_signed", signed=True),
                    _percent_result(row, "targeted_signed", signed=True),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\end{longtable}", r"\endgroup", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("experiments/reports/standard_vs_tail_targeted"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, statuses = _load_rows(args.report_root)
    prevalence_rows = _load_signed_prevalence_rows(args.report_root)
    rare_shape_status_rows = _load_rare_shape_status_rows(args.report_root)
    applicability_rows = _applicability_rows()
    expected = sum(
        len(
            json.loads(path.read_text(encoding="utf-8"))["metric_summary"]
        )
        for path in args.report_root.glob("*/*/multi_seed_summary.json")
    )
    if len(rows) != expected:
        raise AssertionError(f"Emitted {len(rows)} rows but expected {expected}")
    counts = Counter(row["construct"] for row in rows)
    if set(counts) != set(CONSTRUCT_ORDER):
        raise AssertionError(f"Construct coverage mismatch: {counts}")
    _validate_applicability_contract(rows)
    if len({(row["dataset"], row["model"], row["metric_path"]) for row in rows}) != len(rows):
        raise AssertionError("Duplicate dataset/model/metric paths in numeric export")
    expected_shape_rows = len(prevalence_rows) * len(SEEDS)
    if len(rare_shape_status_rows) != expected_shape_rows:
        raise AssertionError(
            f"Emitted {len(rare_shape_status_rows)} rare-shape rows; "
            f"expected {expected_shape_rows}"
        )

    numeric_csv = args.output_dir / "complete-publication-results.csv"
    _write_csv(numeric_csv, rows)
    _write_csv(args.output_dir / "complete-publication-statuses.csv", statuses)
    _write_csv(args.output_dir / "signed-rare-pathway-prevalence.csv", prevalence_rows)
    _write_csv(
        args.output_dir / "rare-shape-estimability.csv",
        rare_shape_status_rows,
    )
    _write_csv(
        args.output_dir / "metric-applicability.csv",
        applicability_rows,
    )
    for index, construct in enumerate(CONSTRUCT_ORDER, start=1):
        slug = construct.lower().replace(" ", "-")
        _write_construct_table(
            args.output_dir / f"s{index}-{slug}-results.tex", construct, rows
        )
    _write_status_table(args.output_dir / "s6-publication-statuses.tex", statuses)
    incomplete_shape_groups = _write_rare_shape_status_table(
        args.output_dir / "s6b-rare-shape-statuses.tex",
        rare_shape_status_rows,
    )
    _write_applicability_table(
        args.output_dir / "s6c-metric-applicability.tex"
    )
    _write_signed_prevalence_table(
        args.output_dir / "s3b-signed-rare-pathway-prevalence.tex",
        prevalence_rows,
    )
    audit = {
        "publication_schema": "publication_metrics_v2",
        "source_report_root": str(args.report_root),
        "numeric_rows_emitted": len(rows),
        "numeric_rows_expected": expected,
        "numeric_summary_fields_emitted": [
            "mean",
            "std",
            "min",
            "max",
            "n",
            "missing_runs",
        ],
        "numeric_csv_sha256": _sha256(numeric_csv),
        "outcome_status_rows_emitted": len(statuses),
        "rare_shape_seed_status_rows_emitted": len(rare_shape_status_rows),
        "rare_shape_seed_status_rows_expected": expected_shape_rows,
        "rare_shape_incomplete_dataset_model_pathway_groups": incomplete_shape_groups,
        "applicability_rows_emitted": len(applicability_rows),
        "applicability_contract_validated_against_numeric_export": True,
        "signed_prevalence_rows_emitted": len(prevalence_rows),
        "rows_by_construct": dict(counts),
        "datasets": list(DATASETS.values()),
        "generators": list(MODELS.values()),
        "seeds": list(SEEDS),
    }
    (args.output_dir / "appendix-generation-audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
