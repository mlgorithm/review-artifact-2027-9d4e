#!/usr/bin/env python3
"""Preflight, run, resume, and audit the locked final experiment matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from data.dataset_scripts.registry import get_dataset
from pipeline import (
    EXPERIMENT_DESIGN,
    LOCAL_RUN_ROOT,
    occupied_experiment_seed_paths,
    run_pipeline,
    sha256_file,
    write_json,
)
from synthetic_generation.run_generator import load_generator_script


DEFAULT_CONFIG = EXPERIMENTS / "configs" / "final_experiments.yaml"
DEFAULT_MANIFEST = LOCAL_RUN_ROOT / "manifests" / "final_experiment_manifest.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_bundle_hash() -> tuple[str, dict[str, str]]:
    paths = sorted(
        [
            *EXPERIMENTS.glob("*.py"),
            *EXPERIMENTS.glob("configs/*.yaml"),
            *EXPERIMENTS.glob("evaluation/*.py"),
            *EXPERIMENTS.glob("data/dataset_scripts/**/*.py"),
            *EXPERIMENTS.glob("synthetic_generation/**/*.py"),
            Path(__file__),
        ]
    )
    files = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in paths
        if path.is_file() and "__pycache__" not in path.parts
    }
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest(), files


def load_config(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    required = ("experiment", "datasets", "core_models")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Final experiment config is missing: {missing}")
    return config


def select_matrix(config, args):
    experiment = dict(config["experiment"])
    configured_datasets = [str(value) for value in config["datasets"]]
    datasets = [str(value) for value in (args.datasets or configured_datasets)]
    unknown_datasets = sorted(set(datasets) - set(configured_datasets))
    if unknown_datasets:
        raise ValueError(
            "Requested datasets are outside the locked experiment config: "
            f"{unknown_datasets}. Allowed: {configured_datasets}."
        )

    core_models = [str(value) for value in config["core_models"]]
    auxiliary_models = [str(value) for value in config.get("auxiliary_models", [])]
    include_auxiliary = bool(getattr(args, "include_auxiliary", False))
    models = [str(value) for value in (args.models or core_models)]
    if include_auxiliary:
        models.extend(model for model in auxiliary_models if model not in models)
    allowed_models = set(core_models)
    if include_auxiliary:
        allowed_models.update(auxiliary_models)
    unknown_models = sorted(set(models) - allowed_models)
    if unknown_models:
        hint = " Pass --include-auxiliary for configured auxiliary models." if auxiliary_models else ""
        raise ValueError(
            f"Requested models are outside the locked experiment config: {unknown_models}."
            + hint
        )

    locked_seeds = [int(seed) for seed in experiment["seeds"]]
    seeds = [int(seed) for seed in (args.seeds or locked_seeds)]
    development_seed_override = bool(getattr(args, "allow_fewer_seeds", False))
    if not development_seed_override and seeds != locked_seeds:
        raise ValueError(
            f"Final experiments require the locked seeds {locked_seeds}; received {seeds}. "
            "Use --allow-fewer-seeds only for an explicitly non-final smoke test."
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError("Generation seeds must be unique.")
    if not seeds:
        raise ValueError("At least one generation seed is required.")
    return datasets, models, seeds, int(experiment.get("tail_oversample", 3))


def neural_fairness_issues(environment: dict[str, object]) -> list[str]:
    """Reject drift in the shared VAE/TimeGAN exposure and representation."""
    pairs = {
        "window": ("VAE_WINDOW", "TIMEGAN_WINDOW"),
        "vocabulary cap": ("VAE_MAX_VOCAB", "TIMEGAN_MAX_VOCAB"),
        "batch size": ("VAE_BATCH", "TIMEGAN_BATCH"),
        "target batch steps": ("VAE_TARGET_STEPS", "TIMEGAN_TARGET_STEPS"),
        "epoch ceiling": ("VAE_EPOCHS", "TIMEGAN_N_ITER"),
    }
    issues = []
    for label, (vae_key, timegan_key) in pairs.items():
        if vae_key not in environment or timegan_key not in environment:
            issues.append(f"neural fairness config is missing {vae_key} or {timegan_key}")
            continue
        if int(environment[vae_key]) != int(environment[timegan_key]):
            issues.append(
                f"neural fairness {label} differs: "
                f"{vae_key}={environment[vae_key]}, {timegan_key}={environment[timegan_key]}"
            )
    return issues


def preflight(config, datasets, models, seeds, preprocess: bool):
    issues = []
    if len(set(seeds)) != len(seeds):
        issues.append("generation seeds are not unique")
    if int(dict(config["experiment"]).get("tail_oversample", 3)) <= 1:
        issues.append("tail_oversample must be greater than one for the RQ2 arm")
    environment = dict(config.get("environment", {}))
    issues.extend(neural_fairness_issues(environment))

    torch_version = None
    accelerator = {"mps_available": False, "cuda_available": False}
    neural_devices = {
        "sequence_vae_timevae": "VAE_DEVICE",
        "timegan": "TIMEGAN_DEVICE",
    }
    if any(model in neural_devices for model in models):
        try:
            torch = importlib.import_module("torch")
            torch_version = torch.__version__
            accelerator = {
                "mps_available": bool(torch.backends.mps.is_available()),
                "cuda_available": bool(torch.cuda.is_available()),
            }
            requested_devices = {
                str(environment.get(neural_devices[model], "cpu"))
                for model in models
                if model in neural_devices
            }
            if "mps" in requested_devices and not torch.backends.mps.is_available():
                issues.append("locked config requests MPS, but torch.backends.mps.is_available() is false")
            if "cuda" in requested_devices and not torch.cuda.is_available():
                issues.append("locked config requests CUDA, but torch.cuda.is_available() is false")
        except ImportError:
            issues.append("PyTorch is required by the locked core model matrix")
    for model in models:
        try:
            generator = load_generator_script(model)
            dependency_probe = getattr(generator, "dependency_issues", None)
            if callable(dependency_probe):
                issues.extend(
                    f"generator {model}: {issue}" for issue in dependency_probe()
                )
        except Exception as exc:
            issues.append(f"generator {model}: {exc}")

    dataset_summary = {}
    for name in datasets:
        try:
            adapter = get_dataset(name)
            if preprocess:
                package = adapter.__name__.rsplit(".", 1)[0]
                module = importlib.import_module(f"{package}.preprocess")
                if not callable(getattr(module, "run_preprocessing", None)):
                    raise RuntimeError("missing run_preprocessing entrypoint")
                raw_candidates = {
                    value
                    for key, value in vars(module).items()
                    if isinstance(value, Path)
                    and ("RAW" in key.upper() or "SOURCE" in key.upper())
                    and value.suffix.lower() in {".zip", ".csv", ".txt"}
                }
                missing_raw = [str(path) for path in raw_candidates if not path.exists()]
                if missing_raw:
                    raise FileNotFoundError(f"missing raw inputs: {missing_raw}")
            else:
                split = adapter.load_split()
                dataset_summary[name] = {
                    "dataset": split.name,
                    "train_path": str(split.train_path),
                    "test_path": str(split.test_path),
                    "train_sha256": sha256_file(split.train_path),
                    "test_sha256": sha256_file(split.test_path),
                    "train_rows": int(len(split.train)),
                    "test_rows": int(len(split.test)),
                    "train_learners": int(split.train[split.columns["learner"]].nunique()),
                    "test_learners": int(split.test[split.columns["learner"]].nunique()),
                }
        except Exception as exc:
            issues.append(f"dataset {name}: {exc}")
    return issues, {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch_version,
        "accelerator": accelerator,
        "datasets": dataset_summary,
    }


def initial_manifest(
    config_path,
    config,
    datasets,
    models,
    seeds,
    tail_oversample,
    reuse_standard_if_valid,
):
    bundle_hash, files = source_bundle_hash()
    return {
        "experiment": dict(config["experiment"]).get("name", "final_experiment"),
        "experiment_design": EXPERIMENT_DESIGN,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "initialized",
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "source_bundle_sha256": bundle_hash,
        "source_files_sha256": files,
        "datasets": datasets,
        "models": models,
        "seeds": seeds,
        "tail_oversample": tail_oversample,
        "reuse_standard_if_valid": bool(reuse_standard_if_valid),
        "environment": {str(k): str(v) for k, v in dict(config.get("environment", {})).items()},
        "cells": {},
    }


def load_or_create_manifest(path, expected, resume):
    if resume and path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for key in (
            "config_sha256",
            "source_bundle_sha256",
            "experiment_design",
            "reuse_standard_if_valid",
            "dataset_artifacts",
            "datasets",
            "models",
            "seeds",
        ):
            if manifest.get(key) != expected.get(key):
                raise RuntimeError(
                    f"Cannot resume: manifest {key} differs from the current locked experiment."
                )
        return manifest
    if path.exists() and not resume:
        raise FileExistsError(
            f"Manifest already exists: {path}. Use --resume or choose --manifest."
        )
    write_json(path, expected)
    return expected


def selected_occupied_paths(
    datasets,
    models,
    seeds,
    *,
    allow_standard_reuse: bool = False,
) -> list[Path]:
    """Find local artifacts that a fresh immutable run would overwrite."""
    occupied = []
    for dataset in datasets:
        split = get_dataset(dataset).load_split()
        for model in models:
            for seed in seeds:
                occupied.extend(
                    occupied_experiment_seed_paths(
                        split,
                        model,
                        int(seed),
                        allow_standard_reuse=allow_standard_reuse,
                    )
                )
    return sorted(set(occupied))


def run(args) -> dict[str, object]:
    config = load_config(args.config)
    datasets, models, seeds, tail_oversample = select_matrix(config, args)
    if not (args.preflight_only or args.dry_run) and (
        len(datasets) != 1 or len(models) != 1
    ):
        raise ValueError(
            "Final generation must run exactly one dataset/model cell at a time. "
            "Pass one --datasets value and one --models value."
        )
    environment = {str(key): str(value) for key, value in dict(config.get("environment", {})).items()}
    os.environ.update(environment)
    issues, runtime = preflight(config, datasets, models, seeds, args.preprocess)
    if issues:
        raise RuntimeError("Final experiment preflight failed:\n- " + "\n- ".join(issues))

    if args.preprocess and not (args.preflight_only or args.dry_run):
        for dataset in datasets:
            adapter = get_dataset(dataset)
            package = adapter.__name__.rsplit(".", 1)[0]
            module = importlib.import_module(f"{package}.preprocess")
            module.run_preprocessing()
            adapter.load_split()
        issues, runtime = preflight(config, datasets, models, seeds, preprocess=False)
        if issues:
            raise RuntimeError("Post-preprocessing final preflight failed:\n- " + "\n- ".join(issues))

    expected = initial_manifest(
        args.config,
        config,
        datasets,
        models,
        seeds,
        tail_oversample,
        args.reuse_standard_if_valid,
    )
    expected["runtime"] = runtime
    expected["dataset_artifacts"] = runtime.get("datasets", {})
    if args.preflight_only or args.dry_run:
        expected["status"] = "preflight_passed"
        expected["matrix"] = [f"{dataset}::{model}" for dataset in datasets for model in models]
        return expected

    if not args.resume:
        occupied = selected_occupied_paths(
            datasets,
            models,
            seeds,
            allow_standard_reuse=args.reuse_standard_if_valid,
        )
        if occupied:
            rendered = "\n- ".join(str(path) for path in occupied)
            raise FileExistsError(
                "Refusing to start because the selected immutable run already has local "
                f"artifacts:\n- {rendered}\nUse --resume to validate completed seeds, or "
                "archive the existing cell before changing the design. No manifest was written."
            )

    manifest = load_or_create_manifest(args.manifest, expected, args.resume)
    manifest["status"] = "running"
    manifest["updated_at"] = utc_now()
    write_json(args.manifest, manifest)

    for dataset in datasets:
        for model in models:
            cell = f"{dataset}::{model}"
            manifest["cells"][cell] = {"status": "running", "started_at": utc_now()}
            manifest["updated_at"] = utc_now()
            write_json(args.manifest, manifest)
            try:
                summary = run_pipeline(
                    dataset,
                    model,
                    seeds,
                    preprocess=False,
                    tail_oversample=tail_oversample,
                    strict=True,
                    reuse_standard_if_valid=args.reuse_standard_if_valid,
                    resume_existing=args.resume,
                )
                manifest["cells"][cell] = {
                    "status": "completed",
                    "started_at": manifest["cells"][cell]["started_at"],
                    "completed_at": utc_now(),
                    "summary_report": summary["summary_report"],
                    "run_count": len(summary["runs"]),
                }
            except Exception as exc:
                manifest["cells"][cell] = {
                    "status": "failed",
                    "started_at": manifest["cells"][cell]["started_at"],
                    "failed_at": utc_now(),
                    "error": str(exc),
                }
                manifest["status"] = "failed"
                manifest["updated_at"] = utc_now()
                write_json(args.manifest, manifest)
                raise
            manifest["updated_at"] = utc_now()
            write_json(args.manifest, manifest)
    manifest["status"] = "completed"
    manifest["completed_at"] = utc_now()
    manifest["updated_at"] = utc_now()
    write_json(args.manifest, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reuse-standard-if-valid",
        action="store_true",
        help=(
            "Reuse a provenance-matched existing standard arm; "
            "the tail-targeted arm is still generated under the current contract."
        ),
    )
    parser.add_argument("--include-auxiliary", action="store_true")
    parser.add_argument(
        "--allow-fewer-seeds",
        action="store_true",
        help="Allow fewer or nonlocked seeds only for smoke tests; never final results.",
    )
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
