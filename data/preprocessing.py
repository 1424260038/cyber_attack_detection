"""Reusable tabular preprocessing for traffic feature CSV files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_EXCLUDED_COLUMNS = {
    "Label",
    "label",
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp",
}


@dataclass
class TabularPreprocessor:
    feature_columns: list[str]
    mean: list[float]
    std: list[float]

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        input_dim: int = 64,
        excluded_columns: Iterable[str] = DEFAULT_EXCLUDED_COLUMNS,
    ) -> "TabularPreprocessor":
        excluded = set(excluded_columns)
        feature_candidates = [col for col in df.columns if col not in excluded]
        numeric_df = df[feature_candidates].apply(pd.to_numeric, errors="coerce")
        numeric_df = numeric_df.dropna(axis=1, how="all")

        if numeric_df.shape[1] < input_dim:
            raise ValueError(f"Need at least {input_dim} numeric feature columns, got {numeric_df.shape[1]}")

        feature_columns = list(numeric_df.columns[:input_dim])
        features = numeric_df[feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std = np.where(std == 0, 1.0, std)

        return cls(
            feature_columns=feature_columns,
            mean=mean.astype(float).tolist(),
            std=std.astype(float).tolist(),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TabularPreprocessor":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(
            feature_columns=list(payload["feature_columns"]),
            mean=list(payload["mean"]),
            std=list(payload["std"]),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "feature_columns": self.feature_columns,
                    "mean": self.mean,
                    "std": self.std,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def transform_frame(self, df: pd.DataFrame) -> np.ndarray:
        missing = [col for col in self.feature_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing[:8]}")

        features = df[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return self.transform_array(features.to_numpy(dtype=np.float32))

    def transform_array(self, features: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        features = np.asarray(features, dtype=np.float32)

        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[1] != len(self.feature_columns):
            raise ValueError(f"Expected {len(self.feature_columns)} features, got {features.shape[1]}")

        return (features - mean) / std
