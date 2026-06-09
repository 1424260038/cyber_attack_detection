"""Prepare raw traffic CSV files for CyberDD training."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from utils.attack_taxonomy import normalize_label


DEFAULT_LABEL_COLUMNS = ["Label", "label", "class", "attack", "Attack", "target"]
DEFAULT_DROP_COLUMNS = {"Flow ID", "Src IP", "Dst IP", "Timestamp"}


def find_label_column(df: pd.DataFrame, explicit_label: str | None = None) -> str:
    if explicit_label and explicit_label in df.columns:
        return explicit_label
    for column in DEFAULT_LABEL_COLUMNS:
        if column in df.columns:
            return column
    raise ValueError(f"No label column found. Tried: {DEFAULT_LABEL_COLUMNS}")


def prepare_file(path: Path, label_column: str | None, max_rows: int | None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    label_col = find_label_column(df, label_column)

    labels = df[label_col].map(normalize_label)
    drop_columns = set(DEFAULT_DROP_COLUMNS) | {label_col}
    feature_columns = [col for col in df.columns if col not in drop_columns]
    numeric = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")
    numeric = numeric.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)

    prepared = numeric.copy()
    prepared["Label"] = labels
    if max_rows is not None and len(prepared) > max_rows:
        prepared = prepared.sample(n=max_rows, random_state=42)
    return prepared


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare raw network traffic CSV files.")
    parser.add_argument("--input", required=True, help="Input CSV file or directory")
    parser.add_argument("--output", default="data/prepared_traffic.csv", help="Output CSV path")
    parser.add_argument("--summary", default="outputs/dataset_summary.json", help="Output summary JSON")
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--max-rows-per-file", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        csv_files = sorted(input_path.glob("*.csv"))
    else:
        csv_files = [input_path]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {input_path}")

    frames = [prepare_file(path, args.label_column, args.max_rows_per_file) for path in csv_files]
    combined = pd.concat(frames, ignore_index=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    summary = {
        "input": str(input_path),
        "files": [str(path) for path in csv_files],
        "output": str(output_path),
        "rows": int(len(combined)),
        "feature_columns": int(len(combined.columns) - 1),
        "label_distribution": dict(Counter(combined["Label"])),
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Prepared {summary['rows']} rows -> {output_path}")
    print(f"Summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
