"""Start the FastAPI service and run HTTP smoke tests."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(url: str, method: str = "GET", headers: dict[str, str] | None = None) -> dict:
    request = Request(url, method=method)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def request_bytes(url: str) -> bytes:
    with urlopen(Request(url, method="GET"), timeout=5) as response:
        return response.read()


def wait_for_health(base_url: str, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            health = request_json(f"{base_url}/health")
            if health.get("model_loaded"):
                return
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"Service did not become healthy: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HTTP smoke tests against a temporary CyberDD API service.")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        wait_for_health(base_url, args.timeout)
        metadata = request_json(f"{base_url}/metadata")
        if metadata.get("input_dim") != 64:
            raise RuntimeError(f"Unexpected input_dim: {metadata.get('input_dim')}")

        admin_headers = {}
        if os.getenv("CYBERDD_ADMIN_TOKEN"):
            admin_headers["X-Admin-Token"] = os.environ["CYBERDD_ADMIN_TOKEN"]

        runtime = request_json(f"{base_url}/admin/runtime", headers=admin_headers)
        if not runtime["artifacts"]["checkpoint"]["exists"]:
            raise RuntimeError("Runtime checkpoint artifact missing")

        report = request_bytes(f"{base_url}/artifacts/report")
        if b"CyberDD" not in report:
            raise RuntimeError("Project report download did not contain expected content")

        release_manifest = request_json(f"{base_url}/artifacts/release-manifest.json")
        if release_manifest.get("file_count", 0) <= 0:
            raise RuntimeError(f"Release manifest is invalid: {release_manifest}")

        openapi = request_json(f"{base_url}/artifacts/openapi.json")
        if "paths" not in openapi:
            raise RuntimeError("OpenAPI download is invalid")

        acceptance = request_bytes(f"{base_url}/artifacts/acceptance-checklist")
        if "CyberDD".encode("utf-8") not in acceptance:
            raise RuntimeError("Acceptance checklist download is invalid")

        completion = request_json(f"{base_url}/artifacts/completion-audit.json")
        if completion.get("status") != "complete":
            raise RuntimeError(f"Completion audit is not complete: {completion}")

        replay = request_json(f"{base_url}/demo/replay?max_rows=3", method="POST")
        if replay.get("processed_rows") != 3:
            raise RuntimeError(f"Replay failed: {replay}")

        summary = request_json(f"{base_url}/events/summary")
        if summary.get("total_events", 0) < 3:
            raise RuntimeError(f"Event summary did not record replay events: {summary}")

        print(f"HTTP smoke test passed at {base_url}")
        return 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
