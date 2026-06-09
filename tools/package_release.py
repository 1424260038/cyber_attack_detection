"""Create a zip release package for CyberDD deliverables."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_DIR / "release" / "cyberdd_release.zip"
DEFAULT_MANIFEST = PROJECT_DIR / "release" / "release_manifest.json"

INCLUDE_PATHS = [
    "api.py",
    "main.py",
    "detector.py",
    "requirements.txt",
    "README.md",
    "启动器.cmd",
    "run.cmd",
    "run.sh",
    "system_check.py",
    "Dockerfile",
    "docker-compose.yml",
    ".dockerignore",
    "configs",
    "data",
    "evaluation",
    "kg",
    "models",
    "training",
    "utils",
    "tools",
    "tests",
    "artifacts",
    "checkpoints/best_model.pth",
    "checkpoints/training_history.json",
    "outputs/test_results.json",
    "outputs/dataset_summary.json",
    "outputs/data_profile.json",
    "outputs/openapi.json",
    "outputs/acceptance_checklist.json",
    "outputs/acceptance_checklist.md",
    "outputs/completion_audit.json",
    "outputs/completion_audit.md",
    "outputs/project_report.md",
    "outputs/demo_runbook.md",
    "web/dist",
    "web/package.json",
    "web/pnpm-lock.yaml",
    "web/src",
    "web/index.html",
    "web/vite.config.ts",
    "web/tsconfig.json",
    "web/tsconfig.app.json",
    "web/tsconfig.node.json",
    "web/tailwind.config.js",
    "web/postcss.config.js",
    "web/eslint.config.js",
]

EXCLUDE_PARTS = {
    "__pycache__",
    "node_modules",
    ".pnpm-store",
    ".git",
    ".venv",
    "venv",
    "release",
    "大创",
}
EXCLUDE_SUFFIXES = {".pyc", ".log"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def should_exclude(path: Path) -> bool:
    relative_parts = set(path.relative_to(PROJECT_DIR).parts)
    if relative_parts & EXCLUDE_PARTS:
        return True
    return path.suffix.lower() in EXCLUDE_SUFFIXES


def collect_files() -> list[Path]:
    files: list[Path] = []
    for item in INCLUDE_PATHS:
        path = PROJECT_DIR / item
        if not path.exists():
            continue
        if path.is_file() and not should_exclude(path):
            files.append(path)
            continue
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and not should_exclude(child):
                    files.append(child)
    return sorted(set(files), key=lambda p: str(p.relative_to(PROJECT_DIR)).lower())


def main() -> int:
    parser = argparse.ArgumentParser(description="Package CyberDD release deliverables.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()

    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    files = collect_files()
    if not files:
        raise RuntimeError("No files collected for release package")

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(PROJECT_DIR).as_posix())

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "package": str(output_path.relative_to(PROJECT_DIR)),
        "package_sha256": sha256(output_path),
        "file_count": len(files),
        "package_size": output_path.stat().st_size,
        "included_files": [str(path.relative_to(PROJECT_DIR)) for path in files],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Packaged {len(files)} files -> {output_path}")
    print(f"Release manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
