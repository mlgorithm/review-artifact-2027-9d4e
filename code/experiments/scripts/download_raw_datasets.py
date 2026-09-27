#!/usr/bin/env python3
"""Download raw datasets used by the experiment pipeline.

Dependencies are declared in `experiments/requirements.txt`. Install them with:

    python3 -m pip install --user -r experiments/requirements.txt
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = ROOT / "experiments" / "data" / "datasets"


def ensure_dirs() -> None:
    for name in ["assistments", "ednet", "oulad"]:
        (DATASETS_DIR / name / "raw").mkdir(parents=True, exist_ok=True)


def download_url(url: str, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        print(f"exists: {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {output_path}")
    with urllib.request.urlopen(url) as response, output_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def download_gdrive(file_id: str, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        print(f"exists: {output_path}")
        return
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit("Install dependencies first: python3 -m pip install --user -r experiments/requirements.txt") from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading Google Drive id {file_id} -> {output_path}")
    gdown.download(id=file_id, output=str(output_path), quiet=False)


def download_kdd(overwrite: bool) -> None:
    try:
        from EduData.DataSet.download_data.download_data import get_data
    except ImportError as exc:
        raise SystemExit("Install dependencies first: python3 -m pip install --user -r experiments/requirements.txt") from exc
    output_dir = DATASETS_DIR / "kddcup2010" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "KDD_Cup_2010").exists() and not overwrite:
        print(f"exists: {output_dir / 'KDD_Cup_2010'}")
        return
    print(f"downloading KDD-CUP-2010 -> {output_dir}")
    get_data("KDD-CUP-2010", str(output_dir), override=overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw experiment datasets.")
    parser.add_argument("--overwrite", action="store_true", help="Redownload existing files.")
    parser.add_argument(
        "--dataset",
        choices=["all", "assistments", "ednet", "kddcup2010", "oulad"],
        default="all",
    )
    args = parser.parse_args()

    ensure_dirs()
    # "all" means the locked paper benchmark. KDD remains explicitly
    # downloadable for optional development, but is not selected by default.
    selected = {args.dataset} if args.dataset != "all" else {"assistments", "ednet", "oulad"}

    if "oulad" in selected:
        download_url(
            "https://archive.ics.uci.edu/static/public/349/open%2Buniversity%2Blearning%2Banalytics%2Bdataset.zip",
            DATASETS_DIR / "oulad" / "raw" / "oulad.zip",
            args.overwrite,
        )

    if "assistments" in selected:
        download_gdrive(
            "1NNXHFRxcArrU0ZJSb9BIL56vmUt5FhlE",
            DATASETS_DIR / "assistments" / "raw" / "assistments_2009_2010_skill_builder_corrected.csv",
            args.overwrite,
        )

    if "ednet" in selected:
        download_gdrive("117aYJAWG3GU48suS66NPaB82HwFj6xWS", DATASETS_DIR / "ednet" / "raw" / "contents.zip", args.overwrite)
        download_gdrive("1AmGcOs5U31wIIqvthn9ARqJMrMTFTcaw", DATASETS_DIR / "ednet" / "raw" / "ednet_kt1.zip", args.overwrite)

    if "kddcup2010" in selected:
        download_kdd(args.overwrite)


if __name__ == "__main__":
    sys.exit(main())
