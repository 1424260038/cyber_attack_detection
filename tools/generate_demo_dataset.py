"""Generate a deterministic demo traffic CSV for CyberDD."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


LABELS = [
    "normal",
    "dos",
    "probe",
    "r2l",
    "u2r",
    "malware",
    "phishing",
    "apt",
]


def build_class_profile(label_index: int, input_dim: int) -> np.ndarray:
    if label_index == 0:
        return np.zeros(input_dim)

    profile = np.zeros(input_dim)
    start = ((label_index - 1) * 8) % input_dim
    end = min(start + 8, input_dim)
    profile[start:end] = 2.5 + label_index * 0.15
    profile += label_index * 0.12
    return profile


def generate_rows(samples_per_class: int, input_dim: int, seed: int) -> list[dict[str, float | str]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str]] = []

    for label_index, label in enumerate(LABELS):
        center = build_class_profile(label_index, input_dim)
        scale = 0.25 if label_index == 0 else 0.45
        for _ in range(samples_per_class):
            features = center + rng.normal(0, scale, input_dim)
            row: dict[str, float | str] = {
                f"f{i}": round(float(features[i]), 6)
                for i in range(input_dim)
            }
            row["Label"] = label
            rows.append(row)

    rng.shuffle(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate demo traffic CSV data.")
    parser.add_argument("--output", default="data/demo_traffic.csv", help="Output CSV path")
    parser.add_argument("--samples-per-class", type=int, default=120)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = generate_rows(args.samples_per_class, args.input_dim, args.seed)
    fieldnames = [f"f{i}" for i in range(args.input_dim)] + ["Label"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows at {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
