#!/usr/bin/env python3
"""Preprocess OULAD into assessment trajectory train/test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DATASET_DIR = ROOT / "experiments" / "data" / "datasets" / "oulad"
RAW_ZIP = DATASET_DIR / "raw" / "oulad.zip"
PROCESSED_DIR = DATASET_DIR
SEED = 20260703
TRAIN_RATIO = 0.8
OUTPUT_COLUMNS = [
    "learner_id",
    "order",
    "assessment_id",
    "skill_id",
    "assessment_type",
    "module_id",
    "presentation_id",
    "correct",
    "score_bin",
    "submission_bin",
    "click_bin",
    "active_days_bin",
    "attempt_bin",
    "hint_bin",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_train_assignment(learner_id: str, train_ratio: float = TRAIN_RATIO, seed: int = SEED) -> bool:
    digest = hashlib.sha256(f"{seed}:{learner_id}".encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12)
    return value < train_ratio


def read_csv_from_zip(archive: zipfile.ZipFile, name: str, **kwargs) -> pd.DataFrame:
    with archive.open(name) as handle:
        return pd.read_csv(handle, **kwargs)


def bin_score(score: float) -> str:
    if score < 40:
        return "0-39"
    if score < 55:
        return "40-54"
    if score < 70:
        return "55-69"
    if score < 85:
        return "70-84"
    return "85-100"


def bin_submission(days_late: float) -> str:
    if days_late <= -7:
        return "early_7d+"
    if days_late < 0:
        return "early_1-6d"
    if days_late == 0:
        return "on_time"
    if days_late <= 7:
        return "late_1-7d"
    return "late_7d+"


def bin_clicks(clicks: float) -> str:
    if clicks <= 0:
        return "0"
    if clicks < 10:
        return "1-9"
    if clicks < 50:
        return "10-49"
    if clicks < 200:
        return "50-199"
    return "200+"


def bin_active_days(days: float) -> str:
    if days <= 0:
        return "0"
    if days < 3:
        return "1-2"
    if days < 7:
        return "3-6"
    if days < 14:
        return "7-13"
    return "14+"


def learner_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["code_module"].astype(str)
        + "_"
        + frame["code_presentation"].astype(str)
        + "_"
        + frame["id_student"].astype(str)
    )


def student_train_mask(
    frame: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    seed: int = SEED,
) -> pd.Series:
    """Keep every enrollment for one physical student in the same split."""
    return frame["id_student"].astype(str).map(
        lambda student_id: stable_train_assignment(student_id, train_ratio, seed)
    )


def prior_vle_features(student_vle: pd.DataFrame, assessment_rows: pd.DataFrame) -> pd.DataFrame:
    vle = student_vle.copy()
    vle["learner_id"] = learner_key(vle)
    vle["date"] = pd.to_numeric(vle["date"], errors="coerce")
    vle["sum_click"] = pd.to_numeric(vle["sum_click"], errors="coerce").fillna(0)
    vle = vle.dropna(subset=["learner_id", "date"])
    daily = (
        vle.groupby(["learner_id", "date"], as_index=False)
        .agg(day_clicks=("sum_click", "sum"), active=("sum_click", "size"))
        .sort_values(["learner_id", "date"])
    )
    daily["cum_clicks"] = daily.groupby("learner_id")["day_clicks"].cumsum()
    daily["cum_active_days"] = daily.groupby("learner_id").cumcount() + 1
    daily_by_learner = {
        learner_id: group
        for learner_id, group in daily.groupby("learner_id", sort=False)
    }

    features = []
    for learner_id, group in assessment_rows.groupby("learner_id", sort=False):
        vle_group = daily_by_learner.get(learner_id)
        if vle_group is None:
            for _, row in group.iterrows():
                features.append({"index": row.name, "prior_clicks": 0.0, "prior_active_days": 0.0})
            continue
        dates = vle_group["date"].to_numpy()
        clicks = vle_group["cum_clicks"].to_numpy()
        active_days = vle_group["cum_active_days"].to_numpy()
        for _, row in group.iterrows():
            cutoff = int(row["date_submitted"])
            index = dates.searchsorted(cutoff, side="left") - 1
            if index >= 0:
                features.append(
                    {
                        "index": row.name,
                        "prior_clicks": float(clicks[index]),
                        "prior_active_days": float(active_days[index]),
                    }
                )
            else:
                features.append({"index": row.name, "prior_clicks": 0.0, "prior_active_days": 0.0})
    return pd.DataFrame(features).set_index("index")


def preprocess(raw_zip: Path = RAW_ZIP, output_dir: Path = PROCESSED_DIR) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw_zip) as archive:
        assessments = read_csv_from_zip(archive, "assessments.csv")
        student_assessment = read_csv_from_zip(archive, "studentAssessment.csv")
        student_info = read_csv_from_zip(archive, "studentInfo.csv")
        student_vle = read_csv_from_zip(archive, "studentVle.csv")

    assessments = assessments.rename(columns={"date": "assessment_date"})
    frame = student_assessment.merge(assessments, on="id_assessment", how="inner")
    frame = frame.merge(
        student_info[["code_module", "code_presentation", "id_student", "num_of_prev_attempts", "final_result"]],
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame["date_submitted"] = pd.to_numeric(frame["date_submitted"], errors="coerce")
    frame["assessment_date"] = pd.to_numeric(frame["assessment_date"], errors="coerce")
    frame = frame.dropna(subset=["score", "date_submitted", "assessment_date"])
    frame = frame[(frame["score"] >= 0) & (frame["score"] <= 100)]
    frame["learner_id"] = learner_key(frame)
    frame = frame.sort_values(["learner_id", "date_submitted", "id_assessment"]).reset_index(drop=True)
    frame["order"] = frame.groupby("learner_id").cumcount()

    vle_features = prior_vle_features(student_vle, frame)
    frame = frame.join(vle_features, how="left")
    frame["prior_clicks"] = frame["prior_clicks"].fillna(0)
    frame["prior_active_days"] = frame["prior_active_days"].fillna(0)

    frame["correct"] = (frame["score"] >= 40).astype(int)
    frame["score_bin"] = frame["score"].map(bin_score)
    frame["submission_bin"] = (frame["date_submitted"] - frame["assessment_date"]).map(bin_submission)
    frame["click_bin"] = frame["prior_clicks"].map(bin_clicks)
    frame["active_days_bin"] = frame["prior_active_days"].map(bin_active_days)
    previous_attempts = pd.to_numeric(frame["num_of_prev_attempts"], errors="coerce").fillna(0).astype(int)
    frame["attempt_bin"] = previous_attempts.clip(upper=3).astype(str).replace({"3": "3+"})
    frame["hint_bin"] = "0"
    frame["assessment_id"] = "a" + frame["id_assessment"].astype(str)
    frame["assessment_type"] = frame["assessment_type"].astype(str)
    frame["module_id"] = frame["code_module"].astype(str)
    frame["presentation_id"] = frame["code_presentation"].astype(str)
    frame["skill_id"] = frame["module_id"] + ":" + frame["assessment_type"]

    output = frame[OUTPUT_COLUMNS].copy()
    main_path = output_dir / "main.csv"
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    output.to_csv(main_path, index=False)

    train_mask = student_train_mask(frame)
    train = output[train_mask].copy()
    test = output[~train_mask].copy()

    train_students = set(frame.loc[train_mask, "id_student"].astype(str))
    test_students = set(frame.loc[~train_mask, "id_student"].astype(str))
    physical_student_overlap = train_students & test_students
    if physical_student_overlap:
        raise RuntimeError(
            "Physical-student leakage across OULAD splits: "
            f"{len(physical_student_overlap)} overlapping id_student values."
        )

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    metadata = {
        "dataset": "oulad_assessment_trajectory",
        "raw_zip": str(raw_zip),
        "raw_zip_sha256": sha256_file(raw_zip),
        "preprocessing_script_sha256": sha256_file(Path(__file__)),
        "columns": OUTPUT_COLUMNS,
        "target": "correct",
        "split_seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "clean_rows": len(output),
        "train_rows": len(train),
        "test_rows": len(test),
        "clean_learners": int(output["learner_id"].nunique()),
        "train_learners": int(train["learner_id"].nunique()),
        "test_learners": int(test["learner_id"].nunique()),
        "split_unit": "physical_student_id",
        "train_physical_students": len(train_students),
        "test_physical_students": len(test_students),
        "physical_student_overlap": 0,
        "main_sha256": sha256_file(main_path),
        "train_sha256": sha256_file(train_path),
        "test_sha256": sha256_file(test_path),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def run_preprocessing() -> dict[str, object]:
    """Pipeline entry point using the canonical raw and processed paths."""
    return preprocess()


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess OULAD assessment trajectories.")
    parser.add_argument("--raw-zip", type=Path, default=RAW_ZIP)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    metadata = preprocess(args.raw_zip, args.output_dir)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
