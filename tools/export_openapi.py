"""Export the FastAPI OpenAPI schema for CyberDD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from api import app


def main() -> int:
    parser = argparse.ArgumentParser(description="Export CyberDD OpenAPI schema.")
    parser.add_argument("--output", default="outputs/openapi.json")
    args = parser.parse_args()

    output_path = PROJECT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(app.openapi(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported OpenAPI schema -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
