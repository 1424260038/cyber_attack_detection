"""Run project quality checks in a predictable order."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = PROJECT_DIR / "web"


def run_step(name: str, command: list[str], cwd: Path) -> None:
    print(f"\n==> {name}")
    print(" ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CyberDD backend and frontend checks.")
    parser.add_argument(
        "--skip-frontend",
        action="store_true",
        help="Skip TypeScript, ESLint and Vite build checks.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Run frontend TypeScript and ESLint checks but skip Vite production build.",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip temporary HTTP service smoke test.",
    )
    args = parser.parse_args()

    run_step("Python compile", [sys.executable, "-m", "compileall", "-q", "."], PROJECT_DIR)
    run_step("Unit tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests"], PROJECT_DIR)
    run_step("System check", [sys.executable, "system_check.py"], PROJECT_DIR)
    if not args.skip_smoke:
        run_step("HTTP smoke test", [sys.executable, "tools/smoke_test_service.py"], PROJECT_DIR)

    if args.skip_frontend:
        print("\nFrontend checks skipped.")
        return 0

    pnpm = shutil.which("pnpm") or shutil.which("pnpm.cmd")
    if pnpm is None:
        raise FileNotFoundError("pnpm not found. Install pnpm or run with --skip-frontend.")

    run_step("Frontend TypeScript", [pnpm, "exec", "tsc", "-b"], WEB_DIR)
    run_step("Frontend ESLint", [pnpm, "exec", "eslint", "."], WEB_DIR)
    if not args.skip_build:
        run_step("Frontend build", [pnpm, "exec", "vite", "build"], WEB_DIR)

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
