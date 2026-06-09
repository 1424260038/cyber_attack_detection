"""Build the CyberDD windowed launcher as a Windows executable."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
LAUNCHER_SOURCE = PROJECT_DIR / "CyberDD启动器.pyw"
OUTPUT_EXE = PROJECT_DIR / "CyberDD启动器.exe"
BUILD_DIR = PROJECT_DIR / "build" / "launcher"


def pyinstaller_available() -> bool:
    return importlib.util.find_spec("PyInstaller") is not None


def main() -> int:
    parser = argparse.ArgumentParser(description="Build CyberDD启动器.exe with PyInstaller.")
    parser.add_argument(
        "--install-pyinstaller",
        action="store_true",
        help="Install PyInstaller first if it is not available.",
    )
    args = parser.parse_args()

    if not LAUNCHER_SOURCE.exists():
        print(f"Launcher source not found: {LAUNCHER_SOURCE}")
        return 1

    if not pyinstaller_available():
        if not args.install_pyinstaller:
            print("PyInstaller is not installed.")
            print("Run: python tools/build_windows_launcher.py --install-pyinstaller")
            return 1
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--windowed",
        "--clean",
        "--noconfirm",
        "--name",
        "CyberDD启动器",
        "--distpath",
        str(PROJECT_DIR),
        "--workpath",
        str(BUILD_DIR),
        "--specpath",
        str(BUILD_DIR),
        str(LAUNCHER_SOURCE),
    ]
    subprocess.check_call(cmd, cwd=PROJECT_DIR)

    print(f"Launcher executable generated: {OUTPUT_EXE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
