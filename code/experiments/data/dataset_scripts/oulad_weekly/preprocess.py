#!/usr/bin/env python3
"""Build explicit weekly OULAD engagement trajectories.

Every enrollment receives one row for every course week, including zero-activity
weeks. This preserves non-activity instead of silently deleting it. Terminal
outcomes and demographics are repeated learner-level attributes; downstream
features never include the terminal outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
SOURCE_ZIP = ROOT / "experiments" / "data" / "datasets" / "oulad" / "raw" / "oulad.zip"
PROCESSED_DIR = ROOT / "experiments" / "data" / "datasets" / "oulad_weekly"
SEED = 20260703
TRAIN_RATIO = 0.8
PREFIX_WEEKS = 4
VLE_CHUNK_ROWS = 500_000

DEMOGRAPHIC_COLUMNS = (
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
)
OUTPUT_COLUMNS = [
    "learner_id",
    "order",
    "course_id",
    "module_id",
    "presentation_id",
    "dominant_activity",
    "engaged",
    "registered",
    "click_bin",
    "active_days_bin",
    "gap_bin",
    "dropout",
    "failure",
    "final_result",
    *DEMOGRAPHIC_COLUMNS,
    "previous_attempt_bin",
    "studied_credits_bin",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_train_assignment(student_id: object, train_ratio: float = TRAIN_RATIO, seed: int = SEED) -> bool:
    digest = hashlib.sha256(f"{seed}:{student_id}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12) < train_ratio


def learner_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["code_module"].astype(str)
        + "_"
        + frame["code_presentation"].astype(str)
        + "_"
        + frame["id_student"].astype(str)
    )


def course_key(frame: pd.DataFrame) -> pd.Series:
    return frame["code_module"].astype(str) + "_" + frame["code_presentation"].astype(str)


def bin_clicks(value: float) -> str:
    if value <= 0:
        return "0"
    if value < 10:
        return "1-9"
    if value < 50:
        return "10-49"
    if value < 200:
        return "50-199"
    return "200+"


def bin_active_days(value: float) -> str:
    if value <= 0:
        return "0"
    if value <= 2:
        return "1-2"
    if value <= 4:
        return "3-4"
    return "5-7"


def bin_previous_attempts(value: object) -> str:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0).iloc[0]
    return str(min(3, max(0, int(parsed)))) + ("+" if parsed >= 3 else "")


def bin_credits(value: object) -> str:
    parsed = float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0).iloc[0])
    if parsed < 60:
        return "0-59"
    if parsed < 120:
        return "60-119"
    if parsed < 180:
        return "120-179"
    return "180+"


def gap_bins(frame: pd.DataFrame) -> pd.Series:
    """Weeks since the most recent active week, computed within enrollment."""
    values = pd.Series(index=frame.index, dtype="object")
    for _, group in frame.groupby("learner_id", sort=False):
        last_active: int | None = None
        for index, row in group.sort_values("order").iterrows():
            week = int(row["order"])
            if last_active is None:
                label = "start"
            else:
                gap = max(1, week - last_active)
                label = "1w" if gap == 1 else ("2-3w" if gap <= 3 else "4w+")
            values.loc[index] = label
            if int(row["engaged"]) == 1:
                last_active = week
    return values


def _read(archive: zipfile.ZipFile, name: str, **kwargs) -> pd.DataFrame:
    with archive.open(name) as handle:
        return pd.read_csv(handle, na_values=["?"], **kwargs)


def aggregate_weekly_vle(archive: zipfile.ZipFile, activity_lookup: pd.DataFrame) -> pd.DataFrame:
    """Chunked exact aggregation of clicks, active days, and dominant activity."""
    daily_parts: list[pd.DataFrame] = []
    activity_parts: list[pd.DataFrame] = []
    with archive.open("studentVle.csv") as handle:
        chunks = pd.read_csv(
            handle,
            na_values=["?"],
            usecols=[
                "code_module",
                "code_presentation",
                "id_student",
                "id_site",
                "date",
                "sum_click",
            ],
            chunksize=VLE_CHUNK_ROWS,
        )
        for chunk in chunks:
            chunk["date"] = pd.to_numeric(chunk["date"], errors="coerce")
            chunk["sum_click"] = pd.to_numeric(chunk["sum_click"], errors="coerce").fillna(0)
            chunk = chunk[chunk["date"].notna() & (chunk["date"] >= 0)].copy()
            if chunk.empty:
                continue
            chunk["learner_id"] = learner_key(chunk)
            chunk["order"] = (chunk["date"] // 7).astype(int)
            chunk = chunk.merge(
                activity_lookup,
                on=["code_module", "code_presentation", "id_site"],
                how="left",
            )
            chunk["activity_type"] = chunk["activity_type"].fillna("unknown").astype(str)
            daily_parts.append(
                chunk.groupby(["learner_id", "order", "date"], as_index=False)["sum_click"].sum()
            )
            activity_parts.append(
                chunk.groupby(
                    ["learner_id", "order", "activity_type"], as_index=False
                )["sum_click"].sum()
            )

    if not daily_parts:
        return pd.DataFrame(
            columns=["learner_id", "order", "weekly_clicks", "active_days", "dominant_activity"]
        )
    daily = pd.concat(daily_parts, ignore_index=True)
    # A learner-day may straddle input chunks, so collapse it again before
    # counting distinct active days.
    daily = daily.groupby(["learner_id", "order", "date"], as_index=False)["sum_click"].sum()
    weekly = daily.groupby(["learner_id", "order"], as_index=False).agg(
        weekly_clicks=("sum_click", "sum"),
        active_days=("date", "nunique"),
    )

    activity = pd.concat(activity_parts, ignore_index=True)
    activity = activity.groupby(
        ["learner_id", "order", "activity_type"], as_index=False
    )["sum_click"].sum()
    activity = activity.sort_values(
        ["learner_id", "order", "sum_click", "activity_type"],
        ascending=[True, True, False, True],
    ).drop_duplicates(["learner_id", "order"])
    activity = activity.rename(columns={"activity_type": "dominant_activity"})[
        ["learner_id", "order", "dominant_activity"]
    ]
    return weekly.merge(activity, on=["learner_id", "order"], how="left")


def build_weekly_panel(enrollments: pd.DataFrame, weekly_activity: pd.DataFrame) -> pd.DataFrame:
    enrollments = enrollments.copy()
    enrollments["course_weeks"] = (
        pd.to_numeric(enrollments["module_presentation_length"], errors="coerce")
        .fillna(0)
        .map(lambda value: max(PREFIX_WEEKS + 1, int(math.ceil(float(value) / 7.0))))
    )
    panel = enrollments.loc[enrollments.index.repeat(enrollments["course_weeks"])].copy()
    panel["order"] = panel.groupby("learner_id", sort=False).cumcount()
    panel = panel.merge(weekly_activity, on=["learner_id", "order"], how="left")
    panel["weekly_clicks"] = pd.to_numeric(panel["weekly_clicks"], errors="coerce").fillna(0)
    panel["active_days"] = pd.to_numeric(panel["active_days"], errors="coerce").fillna(0)
    panel["dominant_activity"] = panel["dominant_activity"].fillna("none").astype(str)

    unregister = pd.to_numeric(panel["date_unregistration"], errors="coerce")
    panel["registered"] = (unregister.isna() | (panel["order"] * 7 < unregister)).astype(int)
    inactive = panel["registered"].eq(0)
    panel.loc[inactive, ["weekly_clicks", "active_days"]] = 0
    panel.loc[inactive, "dominant_activity"] = "none"
    panel["engaged"] = ((panel["weekly_clicks"] > 0) & panel["registered"].eq(1)).astype(int)
    panel["click_bin"] = panel["weekly_clicks"].map(bin_clicks)
    panel["active_days_bin"] = panel["active_days"].map(bin_active_days)
    # A trajectory ends when registration ends. Weeks with no activity *before*
    # that point remain explicit zero rows; post-withdrawal time is outside the
    # learner trajectory. Retain week zero for pre-course withdrawals so the
    # explicit dropout outcome is not silently discarded.
    panel = panel[panel["registered"].eq(1) | panel["order"].eq(0)].copy()
    panel["gap_bin"] = gap_bins(panel)

    panel["final_result"] = panel["final_result"].fillna("Unknown").astype(str)
    panel["dropout"] = panel["final_result"].eq("Withdrawn").astype(int)
    panel["failure"] = panel["final_result"].eq("Fail").astype(int)
    for column in DEMOGRAPHIC_COLUMNS:
        panel[column] = panel[column].fillna("Unknown").astype(str)
    panel["previous_attempt_bin"] = panel["num_of_prev_attempts"].map(bin_previous_attempts)
    panel["studied_credits_bin"] = panel["studied_credits"].map(bin_credits)
    return panel[OUTPUT_COLUMNS].copy()


def preprocess(raw_zip: Path = SOURCE_ZIP, output_dir: Path = PROCESSED_DIR) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw_zip) as archive:
        info = _read(archive, "studentInfo.csv")
        registration = _read(archive, "studentRegistration.csv")
        courses = _read(archive, "courses.csv")
        vle = _read(
            archive,
            "vle.csv",
            usecols=["code_module", "code_presentation", "id_site", "activity_type"],
        )
        weekly_activity = aggregate_weekly_vle(archive, vle)

    key = ["code_module", "code_presentation", "id_student"]
    enrollments = info.merge(registration[key + ["date_unregistration"]], on=key, how="left")
    enrollments = enrollments.merge(
        courses[["code_module", "code_presentation", "module_presentation_length"]],
        on=["code_module", "code_presentation"],
        how="left",
    )
    enrollments["learner_id"] = learner_key(enrollments)
    enrollments["course_id"] = course_key(enrollments)
    enrollments["module_id"] = enrollments["code_module"].astype(str)
    enrollments["presentation_id"] = enrollments["code_presentation"].astype(str)
    if enrollments["learner_id"].duplicated().any():
        raise RuntimeError("OULAD enrollment keys are not unique before weekly expansion.")

    output = build_weekly_panel(enrollments, weekly_activity)
    physical_train = enrollments["id_student"].astype(str).map(stable_train_assignment)
    train_ids = set(enrollments.loc[physical_train, "learner_id"].astype(str))
    test_ids = set(enrollments.loc[~physical_train, "learner_id"].astype(str))
    train = output[output["learner_id"].isin(train_ids)].copy()
    test = output[output["learner_id"].isin(test_ids)].copy()

    train_students = set(enrollments.loc[physical_train, "id_student"].astype(str))
    test_students = set(enrollments.loc[~physical_train, "id_student"].astype(str))
    if train_students & test_students:
        raise RuntimeError("Physical OULAD students overlap across weekly train/test splits.")

    main_path, train_path, test_path = (
        output_dir / "main.csv",
        output_dir / "train.csv",
        output_dir / "test.csv",
    )
    output.to_csv(main_path, index=False)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    metadata = {
        "dataset": "oulad_weekly_engagement",
        "representation": "explicit_course_week_panel",
        "zero_activity_weeks_retained": True,
        "post_withdrawal_weeks_excluded": True,
        "raw_zip": str(raw_zip),
        "raw_zip_sha256": sha256_file(raw_zip),
        "preprocessing_script_sha256": sha256_file(Path(__file__)),
        "columns": OUTPUT_COLUMNS,
        "primary_binary_signal": "engaged",
        "terminal_outcomes": ["dropout", "failure", "final_result"],
        "split_seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "split_unit": "physical_student_id",
        "prefix_weeks_for_outcome_tasks": PREFIX_WEEKS,
        "clean_rows": int(len(output)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "clean_learners": int(output["learner_id"].nunique()),
        "train_learners": int(train["learner_id"].nunique()),
        "test_learners": int(test["learner_id"].nunique()),
        "dropout_learners": int(output.groupby("learner_id")["dropout"].first().sum()),
        "failure_learners": int(output.groupby("learner_id")["failure"].first().sum()),
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
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def run_preprocessing() -> dict[str, object]:
    return preprocess()


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess weekly OULAD engagement trajectories.")
    parser.add_argument("--raw-zip", type=Path, default=SOURCE_ZIP)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    print(json.dumps(preprocess(args.raw_zip, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
