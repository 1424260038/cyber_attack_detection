"""Fit and save the traffic CSV preprocessor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from data.preprocessing import TabularPreprocessor


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit a reusable CSV preprocessor.")
    parser.add_argument("--input", default="data/demo_traffic.csv", help="Input CSV file")
    parser.add_argument("--output", default="artifacts/preprocessor.json", help="Output preprocessor JSON")
    parser.add_argument("--input-dim", type=int, default=64)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    preprocessor = TabularPreprocessor.fit(df, input_dim=args.input_dim)
    preprocessor.save(args.output)
    print(f"Saved preprocessor to {args.output}")
    print(f"Feature columns: {len(preprocessor.feature_columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
