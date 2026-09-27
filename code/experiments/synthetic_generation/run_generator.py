#!/usr/bin/env python3
"""Run a synthetic generator on boundary-preserving training trajectories."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))


GENERATOR_PACKAGE = "synthetic_generation.generator_scripts"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def generator_source_provenance(generator_script: ModuleType) -> tuple[dict[str, str], str]:
    """Hash the runner and complete local generator source bundle.

    Hashing the package rather than only the selected leaf script includes shared
    helpers and cross-generator imports that can change the generated artifact.
    """
    source_path = Path(generator_script.__file__).resolve()
    source_paths = sorted({Path(__file__).resolve(), *source_path.parent.glob("*.py")})
    file_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in source_paths
    }
    bundle_payload = json.dumps(file_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return file_hashes, hashlib.sha256(bundle_payload).hexdigest()


def validate_generator_module(
    module: ModuleType,
    source: str,
    expected_name: str,
) -> ModuleType:
    declared_name = getattr(module, "MODEL_NAME", None)
    if declared_name != expected_name:
        raise ValueError(
            f"Generator {source} declares MODEL_NAME={declared_name!r}; "
            f"expected {expected_name!r}."
        )
    if not hasattr(module, "generate"):
        raise ValueError(
            f"Generator {source} does not define "
            "generate(train_real, output_dir, seed, generation_shape=None)."
        )
    return module


def load_generator_script(model_name: str) -> ModuleType:
    module_name = f"{GENERATOR_PACKAGE}.{model_name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise ValueError(f"Unsupported generator model: {model_name}") from exc
        raise
    return validate_generator_module(module, module_name, model_name)


def run_generator(
    model_name: str,
    train: pd.DataFrame,
    output_dir: Path,
    seed: int,
    generation_shape: dict[str, int] | None = None,
) -> dict[str, object]:
    generator_script = load_generator_script(model_name)
    metadata = generator_script.generate(
        train_real=train,
        output_dir=output_dir,
        seed=seed,
        generation_shape=generation_shape,
    )
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("Generator generate(...) must return a metadata dictionary or None.")
    source_path = Path(generator_script.__file__).resolve()
    source_files, source_bundle_hash = generator_source_provenance(generator_script)
    metadata["generator_script_path"] = str(source_path)
    metadata["generator_script_sha256"] = sha256_file(source_path)
    metadata["generator_source_files_sha256"] = source_files
    metadata["generator_source_bundle_sha256"] = source_bundle_hash
    synthetic_path = output_dir / "synth_generation.csv"
    if synthetic_path.exists():
        metadata["synthetic_output_sha256"] = sha256_file(synthetic_path)
    metadata_path = output_dir / "generation_metadata.json"
    write_json_atomic(metadata_path, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic trajectory generator.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260703)
    args = parser.parse_args()

    train = pd.read_csv(args.train, low_memory=False)
    metadata = run_generator(args.model, train, args.output_dir, args.seed)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
