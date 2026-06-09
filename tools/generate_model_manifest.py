"""Generate a machine-readable manifest for model artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from data.preprocessing import TabularPreprocessor
from utils.model_loader import load_checkpoint_model


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    checkpoint_path = PROJECT_DIR / "checkpoints" / "best_model.pth"
    preprocessor_path = PROJECT_DIR / "artifacts" / "preprocessor.json"
    torchscript_path = PROJECT_DIR / "artifacts" / "model.pt"
    metrics_path = PROJECT_DIR / "outputs" / "test_results.json"
    dataset_summary_path = PROJECT_DIR / "outputs" / "dataset_summary.json"
    data_profile_path = PROJECT_DIR / "outputs" / "data_profile.json"
    output_path = PROJECT_DIR / "artifacts" / "model_manifest.json"

    _, model_info = load_checkpoint_model(checkpoint_path)
    preprocessor = TabularPreprocessor.load(preprocessor_path)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": {
            "architecture": model_info.get("architecture"),
            "input_dim": model_info.get("input_dim"),
            "num_classes": model_info.get("num_classes"),
            "class_names": model_info.get("class_names"),
            "best_val_acc": model_info.get("best_val_acc"),
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_DIR)),
            "checkpoint_sha256": sha256(checkpoint_path),
        },
        "preprocessor": {
            "path": str(preprocessor_path.relative_to(PROJECT_DIR)),
            "sha256": sha256(preprocessor_path),
            "feature_columns": preprocessor.feature_columns,
        },
        "torchscript": {
            "path": str(torchscript_path.relative_to(PROJECT_DIR)),
            "sha256": sha256(torchscript_path),
        },
        "metrics": load_json(metrics_path),
        "dataset_summary": load_json(dataset_summary_path),
        "data_profile": {
            "path": str(data_profile_path.relative_to(PROJECT_DIR)),
            "sha256": sha256(data_profile_path),
            "profile": load_json(data_profile_path),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated manifest: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
