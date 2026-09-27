#!/usr/bin/env python3
"""Build the LAK paper's construct-level summary figures from final reports.

The script reads the completed ``standard_vs_tail_targeted`` report snapshots.
It never reruns a generator or evaluator and never treats the standalone
``standard`` snapshot as extra runs.  Multi-seed summaries supply paper values;
the corresponding seed reports supply the three plotted seed points and the
descriptive real/synthetic pathway prevalences.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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

RQ1_FACETS = (
    "Marginal\ncomposition",
    "Holistic\ndetectability",
    "Local process\nstructure",
    "Global trajectory\nshape",
    "Pathway prevalence\nmatch",
)

COLORS = {"Markov": "paperblue", "SeqVAE": "paperpurple", "TimeGAN": "paperred"}


def _load_reports(report_root: Path) -> dict[tuple[str, str], dict]:
    reports: dict[tuple[str, str], dict] = {}
    for dataset_key in DATASETS:
        for model_key in MODELS:
            path = report_root / dataset_key / model_key / "multi_seed_summary.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing final report: {path}")
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("publication_schema_version") != "publication_metrics_v2":
                raise ValueError(f"Unexpected publication schema in {path}")
            if report.get("seeds") != list(SEEDS):
                raise ValueError(f"Unexpected seed set in {path}: {report.get('seeds')}")
            reports[(dataset_key, model_key)] = report["metric_summary"]
    return reports


def _load_seed_metrics(report_root: Path) -> dict[tuple[str, str], list[dict]]:
    reports: dict[tuple[str, str], list[dict]] = {}
    for dataset_key in DATASETS:
        for model_key in MODELS:
            runs = []
            for seed in SEEDS:
                path = (
                    report_root
                    / dataset_key
                    / model_key
                    / f"seed_{seed}"
                    / "generation_evaluation_report.json"
                )
                if not path.exists():
                    raise FileNotFoundError(f"Missing final seed report: {path}")
                report = json.loads(path.read_text(encoding="utf-8"))
                publication = report.get("publication", {})
                if publication.get("schema_version") != "publication_metrics_v2":
                    raise ValueError(f"Unexpected publication schema in {path}")
                if int(report.get("generation_seed", seed)) != seed:
                    raise ValueError(f"Unexpected generation seed in {path}")
                runs.append(publication["metrics"])
            reports[(dataset_key, model_key)] = runs
    return reports


def _path_value(mapping: dict, path: str) -> float | None:
    value: object = mapping
    for token in path.split("."):
        if not isinstance(value, dict) or token not in value:
            return None
        value = value[token]
    return float(value) if isinstance(value, (int, float)) else None


def _mean(metrics: dict, key: str) -> float | None:
    summary = metrics.get(key)
    if not summary or summary.get("n", 0) == 0:
        return None
    value = summary.get("mean")
    return float(value) if value is not None else None


def _std(metrics: dict, key: str) -> float | None:
    summary = metrics.get(key)
    if not summary or summary.get("n", 0) == 0:
        return None
    value = summary.get("std")
    return float(value) if value is not None else None


def _average_ranks(values: dict[str, float]) -> dict[str, float]:
    """Return lower-is-better ranks with average ranks for exact ties."""
    ordered = sorted(values.items(), key=lambda item: item[1])
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        rank = ((i + 1) + j) / 2.0
        for model, _ in ordered[i:j]:
            ranks[model] = rank
        i = j
    return ranks


def _rq1_facet_keys(all_keys: set[str], facet: str) -> list[str]:
    if facet == "Marginal\ncomposition":
        return sorted(key for key in all_keys if key.startswith("rq1.average_fidelity."))
    if facet == "Holistic\ndetectability":
        return sorted(key for key in all_keys if key.startswith("rq1.detectability."))
    if facet == "Local process\nstructure":
        return sorted(
            key
            for key in all_keys
            if (
                key.startswith("rq1.cross_signal_dependence.")
                or (
                    key.startswith("rq1.temporal_fidelity.")
                    and not key.endswith("correctness_curve_mae")
                )
            )
        )
    if facet == "Global trajectory\nshape":
        return sorted(
            key
            for key in all_keys
            if key == "rq1.temporal_fidelity.correctness_curve_mae"
        )
    if facet == "Pathway prevalence\nmatch":
        return sorted(
            key
            for key in all_keys
            if key.startswith("rq1.tail_fidelity.")
            and key.endswith(".prevalence_error")
        )
    raise ValueError(f"Unknown RQ1 facet: {facet}")


def _rq1_rank_rows(reports: dict[tuple[str, str], dict]) -> list[dict]:
    rows: list[dict] = []
    for dataset_key, dataset_name in DATASETS.items():
        construct_metrics: dict[str, list[str]] = {}
        all_keys = set().union(
            *(reports[(dataset_key, model_key)].keys() for model_key in MODELS)
        )
        for facet in RQ1_FACETS:
            keys = _rq1_facet_keys(all_keys, facet)
            keys = [
                key
                for key in keys
                if all(
                    _mean(reports[(dataset_key, model_key)], key) is not None
                    and reports[(dataset_key, model_key)][key].get("n") == 3
                    for model_key in MODELS
                )
            ]
            construct_metrics[facet] = sorted(keys)

        per_model_ranks = {
            model_key: {facet: [] for facet in RQ1_FACETS}
            for model_key in MODELS
        }
        for facet, keys in construct_metrics.items():
            for key in keys:
                values = {
                    model_key: _mean(reports[(dataset_key, model_key)], key)
                    for model_key in MODELS
                }
                ranks = _average_ranks({k: float(v) for k, v in values.items()})
                for model_key, rank in ranks.items():
                    per_model_ranks[model_key][facet].append(rank)

        for model_key, model_name in MODELS.items():
            for facet in RQ1_FACETS:
                ranks = per_model_ranks[model_key][facet]
                if not ranks:
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model_name,
                            "facet": facet.replace("\n", " "),
                            "mean_within_dataset_rank": None,
                            "metric_count": 0,
                        }
                    )
                    continue
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "facet": facet.replace("\n", " "),
                        "mean_within_dataset_rank": sum(ranks) / len(ranks),
                        "metric_count": len(ranks),
                    }
                )
    return rows


def _rq1_shape_support_rows(reports: dict[tuple[str, str], dict]) -> list[dict]:
    """Count estimable rare-group-by-seed conditional-shape results."""
    rows = []
    for dataset_key, dataset_name in DATASETS.items():
        for model_key, model_name in MODELS.items():
            metrics = reports[(dataset_key, model_key)]
            prevalence_keys = [
                key
                for key in metrics
                if key.startswith("rq1.tail_fidelity.")
                and key.endswith(".prevalence_error")
            ]
            shape_keys = {
                key.removesuffix(".primary_signal_curve_mae"): summary
                for key, summary in metrics.items()
                if key.startswith("rq1.tail_fidelity.")
                and key.endswith(".primary_signal_curve_mae")
            }
            estimable = sum(
                int(
                    shape_keys.get(
                        key.removesuffix(".prevalence_error"), {}
                    ).get("n", 0)
                )
                for key in prevalence_keys
            )
            total = 3 * len(prevalence_keys)
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "estimable_group_seed_shapes": estimable,
                    "eligible_group_seed_shapes": total,
                    "estimable_share": estimable / total if total else 0.0,
                }
            )
    return rows


def _latex_escape(value: str) -> str:
    return (
        value.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("—", "--")
    )


def _write_rq1_figure(
    rows: list[dict], support_rows: list[dict], output_dir: Path
) -> None:
    row_labels = [
        (dataset, model)
        for dataset in DATASETS.values()
        for model in MODELS.values()
    ]
    columns = [name.replace("\n", " ") for name in RQ1_FACETS]
    lookup = {
        (row["dataset"], row["model"], row["facet"]): row for row in rows
    }
    support_lookup = {
        (row["dataset"], row["model"]): row for row in support_rows
    }
    lines = [
        r"\begin{tikzpicture}[x=1cm,y=1cm,font=\sffamily]",
        r"\definecolor{rankbest}{HTML}{D8ECF6}",
        r"\definecolor{rankworst}{HTML}{F7D9C4}",
        r"\definecolor{supportbest}{HTML}{D9EAD3}",
        r"\definecolor{supportworst}{HTML}{F2F2F2}",
    ]
    left = 4.00
    cell_w = 1.83
    cell_h = 0.63
    top = 0.0
    lens_groups = (
        (r"Lens 1\\What occurs?", left, left + 2 * cell_w),
        (
            r"Lens 2\\How does learning unfold?",
            left + 2 * cell_w,
            left + 4 * cell_w,
        ),
    )
    support_left = left + len(columns) * cell_w + 0.16
    support_w = 2.30
    lens_groups += (
        (
            r"Lens 3\\Who is represented?",
            left + 4 * cell_w,
            support_left + support_w,
        ),
    )
    for label, x0, x1 in lens_groups:
        lines.extend(
            [
                rf"\fill[black!6,rounded corners=1.5pt] ({x0 + 0.05:.3f},{top + 1.76:.3f}) rectangle ({x1 - 0.05:.3f},{top + 1.12:.3f});",
                rf"\node[align=center,font=\sffamily\bfseries\scriptsize,text=black!78] at ({(x0 + x1) / 2:.3f},{top + 1.42:.3f}) {{{label}}};",
            ]
        )
    for j, column in enumerate(columns):
        x = left + (j + 0.5) * cell_w
        display_column = column.replace(" ", r"\\")
        lines.append(
            rf"\node[align=center,font=\sffamily\bfseries\scriptsize] at ({x:.3f},{top + 0.53:.3f}) {{{_latex_escape(display_column)}}};"
        )
    lines.append(
        rf"\node[align=center,font=\sffamily\bfseries\footnotesize] at ({support_left + support_w / 2:.3f},{top + 0.53:.3f}) {{Rare-shape\\support}};"
    )
    for i, (dataset, model) in enumerate(row_labels):
        y_top = top - i * cell_h
        y_mid = y_top - cell_h / 2
        label = rf"{_latex_escape(dataset)} -- {_latex_escape(model)}"
        lines.append(
            rf"\node[anchor=east,font=\sffamily\footnotesize] at ({left - 0.16:.3f},{y_mid:.3f}) {{{label}}};"
        )
        for j, column in enumerate(columns):
            item = lookup[(dataset, model, column)]
            x0 = left + j * cell_w
            value = item["mean_within_dataset_rank"]
            if value is None:
                lines.append(
                    rf"\fill[gray!16] ({x0:.3f},{y_top:.3f}) rectangle ({x0 + cell_w:.3f},{y_top - cell_h:.3f});"
                )
                value_text = r"n/a"
            else:
                numeric_value = float(value)
                percentage = round(100 * (3.0 - numeric_value) / 2.0)
                lines.append(
                    rf"\fill[rankbest!{percentage}!rankworst] ({x0:.3f},{y_top:.3f}) rectangle ({x0 + cell_w:.3f},{y_top - cell_h:.3f});"
                )
                value_text = rf"{numeric_value:.2f}\;\textnormal{{({item['metric_count']})}}"
            lines.append(
                rf"\draw[white,line width=0.7pt] ({x0:.3f},{y_top:.3f}) rectangle ({x0 + cell_w:.3f},{y_top - cell_h:.3f});"
            )
            lines.append(
                rf"\node[align=center,font=\sffamily\scriptsize] at ({x0 + cell_w / 2:.3f},{y_mid:.3f}) {{{value_text}}};"
            )
        support = support_lookup[(dataset, model)]
        support_pct = round(100 * float(support["estimable_share"]))
        lines.append(
            rf"\fill[supportbest!{support_pct}!supportworst] ({support_left:.3f},{y_top:.3f}) rectangle ({support_left + support_w:.3f},{y_top - cell_h:.3f});"
        )
        lines.append(
            rf"\draw[white,line width=0.7pt] ({support_left:.3f},{y_top:.3f}) rectangle ({support_left + support_w:.3f},{y_top - cell_h:.3f});"
        )
        lines.append(
            rf"\node[align=center,font=\sffamily\scriptsize] at ({support_left + support_w / 2:.3f},{y_mid:.3f}) {{{support['estimable_group_seed_shapes']}/{support['eligible_group_seed_shapes']}\;\textnormal{{seed--groups}}}};"
        )
        if i in (2, 5):
            lines.append(
                rf"\draw[white,line width=2.2pt] ({left:.3f},{y_top - cell_h:.3f}) -- ({support_left + support_w:.3f},{y_top - cell_h:.3f});"
            )
    legend_y = top - len(row_labels) * cell_h - 0.40
    lines.extend(
        [
            rf"\fill[rankbest] ({left:.3f},{legend_y:.3f}) rectangle ({left + 0.42:.3f},{legend_y - 0.25:.3f});",
            rf"\node[anchor=west,font=\sffamily\scriptsize] at ({left + 0.52:.3f},{legend_y - 0.125:.3f}) {{lower relative error rank}};",
            rf"\fill[rankworst] ({left + 3.72:.3f},{legend_y:.3f}) rectangle ({left + 4.14:.3f},{legend_y - 0.25:.3f});",
            rf"\node[anchor=west,font=\sffamily\scriptsize] at ({left + 4.24:.3f},{legend_y - 0.125:.3f}) {{higher relative error rank}};",
            rf"\fill[supportbest] ({left + 7.58:.3f},{legend_y:.3f}) rectangle ({left + 8.00:.3f},{legend_y - 0.25:.3f});",
            rf"\node[anchor=west,font=\sffamily\scriptsize] at ({left + 8.10:.3f},{legend_y - 0.125:.3f}) {{more conditional-shape estimates available}};",
            r"\end{tikzpicture}",
            "",
        ]
    )
    (output_dir / "rq1-construct-profile.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _utility_rows(
    reports: dict[tuple[str, str], dict],
    seed_metrics: dict[tuple[str, str], list[dict]],
) -> list[dict]:
    rows = []
    for dataset_key, dataset_name in DATASETS.items():
        for model_key, model_name in MODELS.items():
            metrics = reports[(dataset_key, model_key)]
            standard_path = "rq2.downstream_utility.synthetic_utility_gap.auprc"
            targeting_path = "rq2.downstream_utility.tail_targeted_tstr_gain.auprc"
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "standard_loss": _mean(
                        metrics,
                        standard_path,
                    ),
                    "standard_loss_sd": _std(
                        metrics,
                        standard_path,
                    ),
                    "standard_loss_seed_values": [
                        _path_value(run, standard_path)
                        for run in seed_metrics[(dataset_key, model_key)]
                    ],
                    "targeting_gain": _mean(
                        metrics,
                        targeting_path,
                    ),
                    "targeting_gain_sd": _std(
                        metrics,
                        targeting_path,
                    ),
                    "targeting_gain_seed_values": [
                        _path_value(run, targeting_path)
                        for run in seed_metrics[(dataset_key, model_key)]
                    ],
                    "standard_augmentation_gain": _mean(
                        metrics,
                        "rq2.downstream_utility.real_plus_synthetic_gain_over_real.auprc",
                    ),
                    "targeted_augmentation_gain": _mean(
                        metrics,
                        "rq2.downstream_utility.targeted_augmentation_gain_over_real.auprc",
                    ),
                }
            )
    return rows


def _tail_effect_rows(reports: dict[tuple[str, str], dict]) -> list[dict]:
    rows = []
    for dataset_key, dataset_name in DATASETS.items():
        for model_key, model_name in MODELS.items():
            metrics = reports[(dataset_key, model_key)]
            prefix = "rq2.tail_targeting_fidelity_gain."
            suffix = ".prevalence_error_reduction"
            groups = sorted(
                key.removeprefix(prefix).removesuffix(suffix)
                for key in metrics
                if key.startswith(prefix) and key.endswith(suffix)
            )
            for group in groups:
                prevalence_key = f"{prefix}{group}{suffix}"
                shape_key = (
                    f"{prefix}{group}.primary_signal_curve_mae_reduction"
                )
                prediction_key = (
                    f"rq2.tail_learnability.{group}.tail_targeted_tstr_gain.auprc"
                )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "rare_group": group,
                        "prevalence_match_change": _mean(metrics, prevalence_key),
                        "conditional_shape_change": _mean(metrics, shape_key),
                        "rare_group_auprc_change": _mean(metrics, prediction_key),
                    }
                )
    return rows


def _targeting_concordance(rows: list[dict]) -> list[dict]:
    counts = {
        "Representation and tail AUPRC both improved": 0,
        "Representation improved; tail AUPRC did not": 0,
        "Tail AUPRC improved; representation did not": 0,
        "Neither improved": 0,
    }
    for row in rows:
        representation = float(row["prevalence_match_change"]) > 0
        prediction = float(row["rare_group_auprc_change"]) > 0
        if representation and prediction:
            label = "Representation and tail AUPRC both improved"
        elif representation:
            label = "Representation improved; tail AUPRC did not"
        elif prediction:
            label = "Tail AUPRC improved; representation did not"
        else:
            label = "Neither improved"
        counts[label] += 1
    return [{"outcome": label, "count": count} for label, count in counts.items()]


def _write_bar_panel(
    lines: list[str],
    *,
    origin_x: float,
    title: str,
    values: list[dict],
    value_key: str,
    std_key: str,
    minimum: float,
    maximum: float,
    ticks: list[float],
    ylabel: str,
) -> None:
    width = 4.55
    height = 2.45
    base_y = 0.45
    lines.append(
        rf"\node[anchor=west,font=\sffamily\bfseries\footnotesize] at ({origin_x:.3f},{base_y + height + 0.70:.3f}) {{{_latex_escape(title)}}};"
    )
    lines.append(
        rf"\node[anchor=west,font=\sffamily\tiny] at ({origin_x:.3f},{base_y + height + 0.40:.3f}) {{{_latex_escape(ylabel)}}};"
    )
    for tick in ticks:
        y = base_y + height * (tick - minimum) / (maximum - minimum)
        lines.append(
            rf"\draw[gray!30,line width=0.35pt] ({origin_x:.3f},{y:.3f}) -- ({origin_x + width:.3f},{y:.3f});"
        )
        lines.append(
            rf"\node[anchor=east,font=\sffamily\tiny] at ({origin_x - 0.07:.3f},{y:.3f}) {{{tick:.2f}}};"
        )
    zero_y = base_y + height * (0 - minimum) / (maximum - minimum)
    lines.append(
        rf"\draw[black!65,line width=0.55pt] ({origin_x:.3f},{zero_y:.3f}) -- ({origin_x + width:.3f},{zero_y:.3f});"
    )
    group_w = width / len(DATASETS)
    bar_w = 0.28
    for dataset_index, dataset in enumerate(DATASETS.values()):
        center = origin_x + group_w * (dataset_index + 0.5)
        lines.append(
            rf"\node[align=center,font=\sffamily\tiny] at ({center:.3f},{base_y - 0.27:.3f}) {{{_latex_escape(dataset)}}};"
        )
        for model_index, model in enumerate(MODELS.values()):
            row = next(
                row for row in values if row["dataset"] == dataset and row["model"] == model
            )
            value = float(row[value_key])
            std = float(row[std_key])
            value_y = base_y + height * (value - minimum) / (maximum - minimum)
            x0 = center + (model_index - 1) * (bar_w + 0.06) - bar_w / 2
            low_y, high_y = sorted((zero_y, value_y))
            lines.append(
                rf"\fill[{COLORS[model]}] ({x0:.3f},{low_y:.3f}) rectangle ({x0 + bar_w:.3f},{high_y:.3f});"
            )
            whisker_low = max(minimum, value - std)
            whisker_high = min(maximum, value + std)
            whisker_low_y = base_y + height * (whisker_low - minimum) / (maximum - minimum)
            whisker_high_y = base_y + height * (whisker_high - minimum) / (maximum - minimum)
            center_x = x0 + bar_w / 2
            lines.append(
                rf"\draw[black!75,line width=0.45pt] ({center_x:.3f},{whisker_low_y:.3f}) -- ({center_x:.3f},{whisker_high_y:.3f});"
            )
            lines.append(
                rf"\draw[black!75,line width=0.45pt] ({center_x - 0.07:.3f},{whisker_low_y:.3f}) -- ({center_x + 0.07:.3f},{whisker_low_y:.3f});"
            )
            lines.append(
                rf"\draw[black!75,line width=0.45pt] ({center_x - 0.07:.3f},{whisker_high_y:.3f}) -- ({center_x + 0.07:.3f},{whisker_high_y:.3f});"
            )
            seed_values = row[f"{value_key}_seed_values"]
            for seed_index, seed_value in enumerate(seed_values):
                if seed_value is None:
                    continue
                seed_y = base_y + height * (
                    float(seed_value) - minimum
                ) / (maximum - minimum)
                seed_x = center_x + (seed_index - 1) * 0.055
                lines.append(
                    rf"\fill[white] ({seed_x:.3f},{seed_y:.3f}) circle (0.030);"
                )
                lines.append(
                    rf"\draw[black!80,line width=0.35pt] ({seed_x:.3f},{seed_y:.3f}) circle (0.030);"
                )


def _write_rq2_figure(utility_rows: list[dict], output_dir: Path) -> None:
    lines = [
        r"\begin{tikzpicture}[x=1cm,y=1cm,font=\sffamily]",
        r"\definecolor{paperblue}{HTML}{2166AC}",
        r"\definecolor{paperpurple}{HTML}{7B3294}",
        r"\definecolor{paperred}{HTML}{D6604D}",
        r"\definecolor{papergreen}{HTML}{4D9221}",
        r"\definecolor{paperpink}{HTML}{C51B7D}",
    ]
    _write_bar_panel(
        lines,
        origin_x=1.02,
        title="a  Synthetic-only loss",
        values=utility_rows,
        value_key="standard_loss",
        std_key="standard_loss_sd",
        minimum=0.0,
        maximum=0.25,
        ticks=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25],
        ylabel="AUPRC loss; lower is better",
    )
    _write_bar_panel(
        lines,
        origin_x=6.55,
        title="b  Targeting effect",
        values=utility_rows,
        value_key="targeting_gain",
        std_key="targeting_gain_sd",
        minimum=-0.10,
        maximum=0.03,
        ticks=[-0.10, -0.05, 0.00],
        ylabel="AUPRC change; higher is better",
    )
    legend_y = -0.78
    for index, model in enumerate(MODELS.values()):
        x = 1.30 + index * 1.72
        lines.append(
            rf"\fill[{COLORS[model]}] ({x:.3f},{legend_y:.3f}) rectangle ({x + 0.26:.3f},{legend_y + 0.18:.3f});"
        )
        lines.append(
            rf"\node[anchor=west,font=\sffamily\tiny] at ({x + 0.34:.3f},{legend_y + 0.09:.3f}) {{{model}}};"
        )
    lines.extend([r"\end{tikzpicture}", ""])
    (output_dir / "rq2-utility-and-targeting.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _effect_x(value: float, origin: float, width: float, minimum: float, maximum: float) -> float:
    return origin + width * (value - minimum) / (maximum - minimum)


def _write_targeting_effect_figure(rows: list[dict], output_dir: Path) -> None:
    specifications = (
        (
            "a  Pathway prevalence match",
            "Standard prevalence error minus targeted prevalence error; right is better",
            "prevalence_match_change",
            -0.04,
            0.07,
            (-0.04, 0.00, 0.04),
        ),
        (
            "b  Conditional trajectory shape",
            "Standard shape error minus targeted shape error; right is better",
            "conditional_shape_change",
            -0.05,
            0.12,
            (-0.05, 0.00, 0.05, 0.10),
        ),
        (
            "c  AUPRC within rare pathways",
            "Targeted AUPRC minus standard AUPRC; right is better",
            "rare_group_auprc_change",
            -0.11,
            0.07,
            (-0.10, -0.05, 0.00, 0.05),
        ),
    )
    lines = [
        r"\begin{tikzpicture}[x=1cm,y=1cm,font=\sffamily]",
        r"\definecolor{paperblue}{HTML}{2166AC}",
        r"\definecolor{paperpurple}{HTML}{7B3294}",
        r"\definecolor{paperred}{HTML}{D6604D}",
    ]
    panel_width = 3.55
    panel_gap = 1.05
    panel_origins = [2.05 + index * (panel_width + panel_gap) for index in range(3)]
    dataset_y = {"ASSISTments": 2.60, "EdNet": 1.65, "OULAD": 0.70}
    model_offsets = {"Markov": 0.19, "SeqVAE": 0.0, "TimeGAN": -0.19}
    group_offsets: dict[tuple[str, str], float] = {}
    for dataset in DATASETS.values():
        groups = sorted({row["rare_group"] for row in rows if row["dataset"] == dataset})
        midpoint = (len(groups) - 1) / 2
        for index, group in enumerate(groups):
            group_offsets[(dataset, group)] = (index - midpoint) * 0.035

    for panel_index, (title, subtitle, key, minimum, maximum, ticks) in enumerate(
        specifications
    ):
        origin = panel_origins[panel_index]
        lines.append(
            rf"\node[anchor=west,font=\sffamily\bfseries\footnotesize] at ({origin:.3f},3.78) {{{_latex_escape(title)}}};"
        )
        lines.append(
            rf"\node[anchor=north west,align=left,text width={panel_width:.2f}cm,font=\sffamily\tiny] at ({origin:.3f},3.57) {{{_latex_escape(subtitle)}}};"
        )
        for tick in ticks:
            x = _effect_x(float(tick), origin, panel_width, minimum, maximum)
            style = "black!70,line width=0.65pt" if tick == 0 else "gray!28,line width=0.35pt"
            lines.append(rf"\draw[{style}] ({x:.3f},0.30) -- ({x:.3f},3.02);")
            lines.append(
                rf"\node[anchor=north,font=\sffamily\tiny] at ({x:.3f},0.24) {{{tick:.2f}}};"
            )
        for dataset, y in dataset_y.items():
            lines.append(
                rf"\draw[gray!18,line width=0.35pt] ({origin:.3f},{y:.3f}) -- ({origin + panel_width:.3f},{y:.3f});"
            )
            if panel_index == 0:
                lines.append(
                    rf"\node[anchor=east,font=\sffamily\tiny] at ({origin - 0.12:.3f},{y:.3f}) {{{_latex_escape(dataset)}}};"
                )
        for row in rows:
            value = row[key]
            if value is None:
                continue
            x = _effect_x(float(value), origin, panel_width, minimum, maximum)
            y = (
                dataset_y[row["dataset"]]
                + model_offsets[row["model"]]
                + group_offsets[(row["dataset"], row["rare_group"])]
            )
            marker = {"Markov": "circle", "SeqVAE": "diamond", "TimeGAN": "rectangle"}[
                row["model"]
            ]
            lines.append(
                rf"\node[{marker},draw={COLORS[row['model']]},fill={COLORS[row['model']]},inner sep=1.25pt] at ({x:.3f},{y:.3f}) {{}};"
            )

    legend_y = -0.48
    for index, model in enumerate(MODELS.values()):
        x = 2.10 + index * 1.62
        marker = {"Markov": "circle", "SeqVAE": "diamond", "TimeGAN": "rectangle"}[model]
        lines.append(
            rf"\node[{marker},draw={COLORS[model]},fill={COLORS[model]},inner sep=1.25pt] at ({x:.3f},{legend_y:.3f}) {{}};"
        )
        lines.append(
            rf"\node[anchor=west,font=\sffamily\tiny] at ({x + 0.20:.3f},{legend_y:.3f}) {{{model}}};"
        )
    lines.extend([r"\end{tikzpicture}", ""])
    (output_dir / "rq2-targeting-effects.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _privacy_rows(reports: dict[tuple[str, str], dict]) -> list[dict]:
    rows = []
    for dataset_key, dataset_name in DATASETS.items():
        for model_key, model_name in MODELS.items():
            metrics = reports[(dataset_key, model_key)]
            if dataset_name == "OULAD":
                metric = "behavior_projection_exact_duplicate_rate"
                measure = "behavior_copy_rate"
            else:
                metric = "membership_inference_attack_advantage"
                measure = "membership_advantage"
            row = {
                "dataset": dataset_name,
                "model": model_name,
                "measure": measure,
            }
            for arm, prefix in (
                ("standard", "standard"),
                ("targeted", "tail_targeted"),
            ):
                key = f"rq2.privacy_risk.{prefix}.{metric}"
                row[f"{arm}_mean"] = _mean(metrics, key)
                row[f"{arm}_sd"] = _std(metrics, key)
            row["targeted_minus_standard_mean"] = (
                float(row["targeted_mean"]) - float(row["standard_mean"])
            )
            rows.append(row)
    return rows


def _write_horizontal_privacy_panel(
    lines: list[str],
    *,
    rows: list[dict],
    origin_x: float,
    top_y: float,
    width: float,
    maximum: float,
    ticks: list[float],
    title: str,
    subtitle: str,
) -> None:
    row_h = 0.48
    lines.append(
        rf"\node[anchor=west,font=\sffamily\bfseries\footnotesize] at ({origin_x - 2.45:.3f},{top_y + 0.72:.3f}) {{{_latex_escape(title)}}};"
    )
    lines.append(
        rf"\node[anchor=west,font=\sffamily\tiny] at ({origin_x - 2.45:.3f},{top_y + 0.44:.3f}) {{{_latex_escape(subtitle)}}};"
    )
    lines.append(
        rf"\node[font=\sffamily\bfseries\tiny] at ({origin_x + width + 0.40:.3f},{top_y + 0.44:.3f}) {{$\Delta$ risk}};"
    )
    bottom_y = top_y - (len(rows) - 1) * row_h - 0.28
    for tick in ticks:
        x = origin_x + width * tick / maximum
        lines.append(
            rf"\draw[gray!28,line width=0.35pt] ({x:.3f},{top_y + 0.16:.3f}) -- ({x:.3f},{bottom_y:.3f});"
        )
        lines.append(
            rf"\node[anchor=north,font=\sffamily\tiny] at ({x:.3f},{bottom_y - 0.05:.3f}) {{{tick:.2f}}};"
        )
    for index, row in enumerate(rows):
        y = top_y - index * row_h
        label = (
            f"{row['dataset']} -- {row['model']}"
            if row["dataset"] != "OULAD"
            else row["model"]
        )
        lines.append(
            rf"\node[anchor=east,font=\sffamily\tiny] at ({origin_x - 0.10:.3f},{y:.3f}) {{{_latex_escape(label)}}};"
        )
        for arm, dy, color, shape in (
            ("standard", 0.09, "paperblue", "circle"),
            ("targeted", -0.09, "paperorange", "diamond"),
        ):
            mean = float(row[f"{arm}_mean"])
            std = float(row[f"{arm}_sd"])
            x = origin_x + width * min(maximum, max(0.0, mean)) / maximum
            x_low = origin_x + width * min(maximum, max(0.0, mean - std)) / maximum
            x_high = origin_x + width * min(maximum, max(0.0, mean + std)) / maximum
            point_y = y + dy
            lines.append(
                rf"\draw[{color},line width=0.6pt] ({x_low:.3f},{point_y:.3f}) -- ({x_high:.3f},{point_y:.3f});"
            )
            lines.append(
                rf"\draw[{color},line width=0.6pt] ({x_low:.3f},{point_y - 0.045:.3f}) -- ({x_low:.3f},{point_y + 0.045:.3f});"
            )
            lines.append(
                rf"\draw[{color},line width=0.6pt] ({x_high:.3f},{point_y - 0.045:.3f}) -- ({x_high:.3f},{point_y + 0.045:.3f});"
            )
            lines.append(
                rf"\node[{shape},draw={color},fill={color},inner sep=1.4pt] at ({x:.3f},{point_y:.3f}) {{}};"
            )
        change = float(row["targeted_minus_standard_mean"])
        sign = "+" if change > 0 else ""
        lines.append(
            rf"\node[anchor=west,font=\sffamily\tiny] at ({origin_x + width + 0.13:.3f},{y:.3f}) {{{sign}{change:.4f}}};"
        )


def _write_privacy_figure(rows: list[dict], output_dir: Path) -> None:
    tutoring = [row for row in rows if row["measure"] == "membership_advantage"]
    oulad = [row for row in rows if row["measure"] == "behavior_copy_rate"]
    lines = [
        r"\begin{tikzpicture}[x=1cm,y=1cm,font=\sffamily]",
        r"\definecolor{paperblue}{HTML}{2166AC}",
        r"\definecolor{paperorange}{HTML}{D95F02}",
    ]
    _write_horizontal_privacy_panel(
        lines,
        rows=tutoring,
        origin_x=3.55,
        top_y=2.85,
        width=4.35,
        maximum=0.06,
        ticks=[0.00, 0.02, 0.04, 0.06],
        title="a  Tutoring-data membership advantage",
        subtitle=r"Lower is safer; point and whisker show mean $\pm$ population SD",
    )
    _write_horizontal_privacy_panel(
        lines,
        rows=oulad,
        origin_x=11.20,
        top_y=2.85,
        width=3.75,
        maximum=0.18,
        ticks=[0.00, 0.05, 0.10, 0.15],
        title="b  OULAD behavior-projection exact-copy rate",
        subtitle="Lower is safer; behavior projection only",
    )
    lines.extend(
        [
            r"\node[circle,draw=paperblue,fill=paperblue,inner sep=1.4pt] at (3.55,-0.42) {};",
            r"\node[anchor=west,font=\sffamily\tiny] at (3.75,-0.42) {standard synthetic};",
            r"\node[diamond,draw=paperorange,fill=paperorange,inner sep=1.4pt] at (5.75,-0.42) {};",
            r"\node[anchor=west,font=\sffamily\tiny] at (5.98,-0.42) {tail-targeted synthetic};",
            r"\node[anchor=west,font=\sffamily\tiny] at (9.15,-0.42) {Exact- and near-copy rates were 0 for every ASSISTments and EdNet arm.};",
            r"\end{tikzpicture}",
            "",
        ]
    )
    (output_dir / "rq2-disclosure-risk.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    reports = _load_reports(args.report_root)
    seed_metrics = _load_seed_metrics(args.report_root)

    rq1_rows = _rq1_rank_rows(reports)
    rq1_support_rows = _rq1_shape_support_rows(reports)
    utility_rows = _utility_rows(reports, seed_metrics)
    tail_effect_rows = _tail_effect_rows(reports)
    concordance = _targeting_concordance(tail_effect_rows)
    privacy_rows = _privacy_rows(reports)
    _write_rq1_figure(rq1_rows, rq1_support_rows, args.output_dir)
    _write_rq2_figure(utility_rows, args.output_dir)
    _write_targeting_effect_figure(tail_effect_rows, args.output_dir)
    _write_privacy_figure(privacy_rows, args.output_dir)
    _write_csv(args.output_dir / "rq1-construct-profile.csv", rq1_rows)
    _write_csv(args.output_dir / "rq1-shape-support.csv", rq1_support_rows)
    _write_csv(args.output_dir / "rq2-utility-summary.csv", utility_rows)
    _write_csv(args.output_dir / "rq2-tail-targeting-effects.csv", tail_effect_rows)
    _write_csv(args.output_dir / "rq2-tail-targeting-concordance.csv", concordance)
    _write_csv(args.output_dir / "rq2-disclosure-risk.csv", privacy_rows)


if __name__ == "__main__":
    main()
