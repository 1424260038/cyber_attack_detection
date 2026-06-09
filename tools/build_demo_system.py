"""Build or refresh all demo artifacts for the CyberDD system."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str]) -> None:
    print(f"\n==> {name}")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh CyberDD demo data, artifacts, report and checks.")
    parser.add_argument("--samples-per-class", type=int, default=120)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--train", action="store_true", help="Retrain the demo model before exporting artifacts.")
    parser.add_argument("--skip-checks", action="store_true", help="Skip the final quality check suite.")
    parser.add_argument("--skip-frontend", action="store_true", help="Skip frontend checks in the final suite.")
    args = parser.parse_args()

    python = sys.executable
    run_step(
        "Generate deterministic demo dataset",
        [
            python,
            "tools/generate_demo_dataset.py",
            "--output",
            "data/demo_traffic.csv",
            "--samples-per-class",
            str(args.samples_per_class),
            "--input-dim",
            str(args.input_dim),
        ],
    )
    run_step(
        "Prepare normalized training CSV",
        [
            python,
            "tools/prepare_dataset.py",
            "--input",
            "data/demo_traffic.csv",
            "--output",
            "data/prepared_traffic.csv",
            "--summary",
            "outputs/dataset_summary.json",
        ],
    )
    run_step(
        "Profile data quality",
        [
            python,
            "tools/profile_dataset.py",
            "--input",
            "data/demo_traffic.csv",
            "--output",
            "outputs/data_profile.json",
        ],
    )
    run_step(
        "Fit tabular preprocessor",
        [
            python,
            "tools/fit_preprocessor.py",
            "--input",
            "data/demo_traffic.csv",
            "--output",
            "artifacts/preprocessor.json",
            "--input-dim",
            str(args.input_dim),
        ],
    )

    if args.train:
        run_step(
            "Train demo classifier",
            [
                python,
                "main.py",
                "--mode",
                "train",
                "--model",
                "cnn_lstm",
                "--data",
                "data",
                "--epochs",
                str(args.epochs),
                "--device",
                "cpu",
                "--output_dir",
                "checkpoints",
            ],
        )
        run_step(
            "Evaluate demo classifier",
            [
                python,
                "main.py",
                "--mode",
                "test",
                "--model",
                "cnn_lstm",
                "--data",
                "data",
                "--checkpoint",
                "checkpoints/best_model.pth",
                "--device",
                "cpu",
                "--output_dir",
                "outputs",
            ],
        )
    else:
        print("\nModel training skipped. Existing checkpoints/best_model.pth and outputs/test_results.json are reused.")

    run_step(
        "Export TorchScript model",
        [python, "tools/export_model.py", "--checkpoint", "checkpoints/best_model.pth", "--output", "artifacts/model.pt"],
    )
    run_step("Generate model manifest", [python, "tools/generate_model_manifest.py"])
    run_step("Generate project report", [python, "tools/generate_project_report.py"])
    run_step("Export OpenAPI schema", [python, "tools/export_openapi.py"])
    run_step("Generate acceptance checklist", [python, "tools/generate_acceptance_checklist.py"])
    run_step("Audit system completion", [python, "tools/audit_completion.py"])
    run_step("Package release deliverables", [python, "tools/package_release.py"])

    if not args.skip_checks:
        check_command = [python, "tools/run_all_checks.py"]
        if args.skip_frontend:
            check_command.append("--skip-frontend")
        run_step("Run quality checks", check_command)

    print("\nDemo system artifacts refreshed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
