"""Dataset adapter registry."""

from __future__ import annotations

import importlib


ALIASES = {
    "assistments": "assistments",
    "assistments_2009_2010_skill_builder": "assistments",
    "ednet": "ednet",
    "ednet_kt1": "ednet",
    "kdd": "kddcup2010",
    "kddcup2010": "kddcup2010",
    "kddcup2010_bridge_to_algebra_2008_2009": "kddcup2010",
    "oulad": "oulad",
    "oulad_assessment_trajectory": "oulad",
    "oulad_weekly": "oulad_weekly",
    "oulad_weekly_engagement": "oulad_weekly",
}


def get_dataset(dataset_name: str):
    module_name = ALIASES.get(dataset_name, dataset_name)
    try:
        return importlib.import_module(f"data.dataset_scripts.{module_name}.dataset")
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. Known names: {sorted(ALIASES)}."
        ) from exc
