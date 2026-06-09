"""Generate a data quality profile for traffic CSV files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from tools.prepare_dataset import DEFAULT_DROP_COLUMNS, find_label_column
from utils.attack_taxonomy import normalize_label


def csv_files_from_path(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.csv"))
    return [input_path]


def profile_file(path: Path, label_column: str | None, max_rows: int | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(path, low_memory=False)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    try:
        label_col = find_label_column(df, label_column)
        labels = df[label_col].map(normalize_label)
    except ValueError:
        label_col = None
        labels = pd.Series([], dtype=str)

    drop_columns = set(DEFAULT_DROP_COLUMNS)
    if label_col:
        drop_columns.add(label_col)

    feature_columns = [column for column in df.columns if column not in drop_columns]
    numeric = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([float("inf"), float("-inf")], np.nan)
    non_numeric_columns = [
        column
        for column in feature_columns
        if numeric[column].isna().all() and df[column].notna().any()
    ]

    frame_profile = {
        "file": str(path),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "label_column": label_col,
        "feature_columns": int(len(feature_columns)),
        "numeric_feature_columns": int(numeric.dropna(axis=1, how="all").shape[1]),
        "non_numeric_columns": non_numeric_columns[:20],
        "missing_values": int(df.isna().sum().sum()),
        "numeric_missing_or_invalid_values": int(numeric.isna().sum().sum()),
        "label_distribution": dict(Counter(labels)) if label_col else {},
    }
    return numeric, frame_profile


def build_profile(input_path: Path, label_column: str | None, max_rows_per_file: int | None) -> dict[str, Any]:
    files = csv_files_from_path(input_path)
    if not files:
        raise FileNotFoundError(f"No CSV files found under {input_path}")

    numeric_frames = []
    file_profiles = []
    label_counter: Counter[str] = Counter()
    for path in files:
        numeric, file_profile = profile_file(path, label_column, max_rows_per_file)
        numeric_frames.append(numeric)
        file_profiles.append(file_profile)
        label_counter.update(file_profile["label_distribution"])

    combined_numeric = pd.concat(numeric_frames, ignore_index=True)
    valid_numeric = combined_numeric.dropna(axis=1, how="all")
    missing_rate = float(combined_numeric.isna().sum().sum() / max(combined_numeric.size, 1))
    label_total = sum(label_counter.values())
    class_count = len(label_counter)
    imbalance_ratio = 0.0
    if label_counter:
        counts = list(label_counter.values())
        imbalance_ratio = float(max(counts) / max(min(counts), 1))

    low_variance_columns = []
    if not valid_numeric.empty:
        variance = valid_numeric.fillna(0.0).var(axis=0)
        low_variance_columns = [str(column) for column, value in variance.items() if float(value) <= 1e-12][:20]

    warnings = []
    if not label_counter:
        warnings.append("未识别到标签列，无法计算类别分布。")
    if class_count and class_count < 2:
        warnings.append("只识别到一个类别，无法训练有效分类模型。")
    if imbalance_ratio >= 5:
        warnings.append(f"类别分布不均衡，最大/最小类别样本比为 {imbalance_ratio:.2f}。")
    if missing_rate >= 0.1:
        warnings.append(f"数值特征缺失或非法值比例较高：{missing_rate:.2%}。")
    if len(valid_numeric.columns) < 64:
        warnings.append(f"可用数值特征少于默认模型输入 64 维：当前 {len(valid_numeric.columns)} 维。")
    if low_variance_columns:
        warnings.append(f"存在低方差特征列，建议训练前检查：{', '.join(low_variance_columns[:5])}。")

    return {
        "input": str(input_path),
        "files": [str(path) for path in files],
        "rows": int(sum(item["rows"] for item in file_profiles)),
        "columns": int(max((item["columns"] for item in file_profiles), default=0)),
        "numeric_feature_columns": int(len(valid_numeric.columns)),
        "missing_rate": missing_rate,
        "label_distribution": dict(label_counter),
        "class_count": class_count,
        "imbalance_ratio": imbalance_ratio,
        "low_variance_columns": low_variance_columns,
        "file_profiles": file_profiles,
        "warnings": warnings,
        "quality_score": quality_score(missing_rate, imbalance_ratio, class_count, len(valid_numeric.columns)),
    }


def quality_score(missing_rate: float, imbalance_ratio: float, class_count: int, numeric_features: int) -> int:
    score = 100
    score -= min(int(missing_rate * 100), 35)
    if imbalance_ratio >= 10:
        score -= 25
    elif imbalance_ratio >= 5:
        score -= 15
    elif imbalance_ratio >= 2:
        score -= 5
    if class_count < 2:
        score -= 30
    if numeric_features < 64:
        score -= 15
    return max(score, 0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile traffic CSV data quality.")
    parser.add_argument("--input", required=True, help="Input CSV file or directory")
    parser.add_argument("--output", default="outputs/data_profile.json", help="Output JSON path")
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--max-rows-per-file", type=int, default=None)
    args = parser.parse_args()

    profile = build_profile(Path(args.input), args.label_column, args.max_rows_per_file)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Profiled {profile['rows']} rows, quality score={profile['quality_score']} -> {output_path}")
    if profile["warnings"]:
        print("Warnings:")
        for warning in profile["warnings"]:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
