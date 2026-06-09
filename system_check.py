"""End-to-end system check for the CyberDD project."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from fastapi.testclient import TestClient

import api
from data.preprocessing import TabularPreprocessor


PROJECT_DIR = Path(__file__).resolve().parent


def check_model() -> dict:
    api.load_model()
    api.load_knowledge_graph()
    if not api.model_info:
        raise RuntimeError("Model metadata is empty")

    input_dim = int(api.model_info["input_dim"])
    sample = [2.0] * input_dim
    result = api.predict_features(sample)
    if "prediction" not in result or "probabilities" not in result:
        raise RuntimeError("Prediction response is incomplete")

    chain = api.explain_attack("dos", 0.9)
    if not chain["attack_chain_cn"]:
        raise RuntimeError("Knowledge graph explanation returned an empty attack chain")

    return {
        "architecture": api.model_info["architecture"],
        "input_dim": input_dim,
        "num_classes": api.model_info["num_classes"],
        "prediction": result["prediction"],
        "risk_level": result["risk_level"],
    }


def check_metrics() -> dict:
    results_path = PROJECT_DIR / "outputs" / "test_results.json"
    if not results_path.exists():
        raise FileNotFoundError("outputs/test_results.json not found")

    with open(results_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    required = ["accuracy", "precision", "recall", "f1_score", "auc_roc", "average_precision"]
    missing = [key for key in required if key not in metrics]
    if missing:
        raise RuntimeError(f"Missing metrics: {missing}")

    return {key: metrics[key] for key in required}


def check_artifacts() -> dict:
    preprocessor_path = PROJECT_DIR / "artifacts" / "preprocessor.json"
    model_path = PROJECT_DIR / "artifacts" / "model.pt"
    report_path = PROJECT_DIR / "outputs" / "project_report.md"
    dataset_summary_path = PROJECT_DIR / "outputs" / "dataset_summary.json"
    data_profile_path = PROJECT_DIR / "outputs" / "data_profile.json"
    openapi_path = PROJECT_DIR / "outputs" / "openapi.json"
    acceptance_md_path = PROJECT_DIR / "outputs" / "acceptance_checklist.md"
    acceptance_json_path = PROJECT_DIR / "outputs" / "acceptance_checklist.json"
    completion_audit_path = PROJECT_DIR / "outputs" / "completion_audit.md"
    completion_audit_json_path = PROJECT_DIR / "outputs" / "completion_audit.json"
    manifest_path = PROJECT_DIR / "artifacts" / "model_manifest.json"
    release_package_path = PROJECT_DIR / "release" / "cyberdd_release.zip"
    release_manifest_path = PROJECT_DIR / "release" / "release_manifest.json"
    dockerfile_path = PROJECT_DIR / "Dockerfile"
    compose_path = PROJECT_DIR / "docker-compose.yml"

    if not preprocessor_path.exists():
        raise FileNotFoundError("artifacts/preprocessor.json not found")
    if not model_path.exists():
        raise FileNotFoundError("artifacts/model.pt not found")
    if not report_path.exists():
        raise FileNotFoundError("outputs/project_report.md not found")
    if not dataset_summary_path.exists():
        raise FileNotFoundError("outputs/dataset_summary.json not found")
    if not data_profile_path.exists():
        raise FileNotFoundError("outputs/data_profile.json not found")
    if not openapi_path.exists():
        raise FileNotFoundError("outputs/openapi.json not found")
    if not acceptance_md_path.exists() or not acceptance_json_path.exists():
        raise FileNotFoundError("Acceptance checklist files not found")
    if not completion_audit_path.exists() or not completion_audit_json_path.exists():
        raise FileNotFoundError("Completion audit files not found")
    if not manifest_path.exists():
        raise FileNotFoundError("artifacts/model_manifest.json not found")
    if not release_package_path.exists() or not release_manifest_path.exists():
        raise FileNotFoundError("Release package files not found")
    if not dockerfile_path.exists() or not compose_path.exists():
        raise FileNotFoundError("Docker deployment files not found")

    preprocessor = TabularPreprocessor.load(preprocessor_path)
    scripted_model = torch.jit.load(str(model_path), map_location="cpu")
    example = torch.randn(1, len(preprocessor.feature_columns))
    output = scripted_model(example)
    if output.shape[0] != 1:
        raise RuntimeError("TorchScript model returned invalid output")

    return {
        "preprocessor_features": len(preprocessor.feature_columns),
        "torchscript_output_dim": int(output.shape[1]),
        "report": report_path.name,
        "manifest": manifest_path.name,
        "data_profile": data_profile_path.name,
        "openapi": openapi_path.name,
        "acceptance": acceptance_md_path.name,
        "completion_audit": completion_audit_path.name,
        "release_package": release_package_path.name,
    }


def check_api_routes() -> dict:
    with TestClient(api.app) as client:
        metadata = client.get("/metadata")
        if metadata.status_code != 200:
            raise RuntimeError(f"/metadata failed: {metadata.status_code}")

        samples = client.get("/demo-samples")
        if samples.status_code != 200:
            raise RuntimeError(f"/demo-samples failed: {samples.status_code}")

        features = samples.json()["attack"]
        manifest = client.get("/manifest")
        if manifest.status_code != 200:
            raise RuntimeError(f"/manifest failed: {manifest.status_code}")
        if "data_profile" not in manifest.json():
            raise RuntimeError("Manifest does not include data_profile")

        runtime = client.get("/admin/runtime")
        if runtime.status_code != 200:
            raise RuntimeError(f"/admin/runtime failed: {runtime.status_code}")
        if not runtime.json()["artifacts"]["checkpoint"]["exists"]:
            raise RuntimeError("Runtime artifact check did not find checkpoint")
        if not runtime.json()["artifacts"]["demo_runbook"]["exists"]:
            raise RuntimeError("Runtime artifact check did not find demo runbook")
        if not runtime.json()["artifacts"]["release_package"]["exists"]:
            raise RuntimeError("Runtime artifact check did not find release package")
        if not runtime.json()["artifacts"]["acceptance_checklist"]["exists"]:
            raise RuntimeError("Runtime artifact check did not find acceptance checklist")
        if not runtime.json()["artifacts"]["completion_audit"]["exists"]:
            raise RuntimeError("Runtime artifact check did not find completion audit")

        reload_response = client.post("/admin/reload")
        if reload_response.status_code != 200 or not reload_response.json()["model_loaded"]:
            raise RuntimeError(f"/admin/reload failed: {reload_response.status_code}")

        for artifact_route in [
            "/artifacts/report",
            "/artifacts/runbook",
            "/artifacts/manifest.json",
            "/artifacts/data-profile.json",
            "/artifacts/openapi.json",
            "/artifacts/acceptance-checklist",
            "/artifacts/acceptance-checklist.json",
            "/artifacts/completion-audit",
            "/artifacts/completion-audit.json",
            "/artifacts/release.zip",
            "/artifacts/release-manifest.json",
        ]:
            artifact_response = client.get(artifact_route)
            if artifact_response.status_code != 200:
                raise RuntimeError(f"{artifact_route} failed: {artifact_response.status_code}")

        prediction = client.post("/predict", json={"features": features})
        if prediction.status_code != 200:
            raise RuntimeError(f"/predict failed: {prediction.status_code}")
        if not prediction.json().get("feature_contributions"):
            raise RuntimeError("/predict returned no feature contributions")

        replay = client.post("/demo/replay?max_rows=4")
        if replay.status_code != 200 or replay.json()["processed_rows"] != 4:
            raise RuntimeError(f"/demo/replay failed: {replay.status_code}")

        events_summary = client.get("/events/summary")
        if events_summary.status_code != 200:
            raise RuntimeError(f"/events/summary failed: {events_summary.status_code}")

        dataset_summary = client.get("/dataset/summary")
        if dataset_summary.status_code != 200:
            raise RuntimeError(f"/dataset/summary failed: {dataset_summary.status_code}")

        dataset_profile = client.get("/dataset/profile")
        if dataset_profile.status_code != 200:
            raise RuntimeError(f"/dataset/profile failed: {dataset_profile.status_code}")

        csv_path = PROJECT_DIR / "data" / "demo_traffic.csv"
        with open(csv_path, "rb") as f:
            upload = client.post(
                "/predict/upload",
                files={"file": ("demo_traffic.csv", f, "text/csv")},
                data={"max_rows": "3"},
            )
        if upload.status_code != 200:
            raise RuntimeError(f"/predict/upload failed: {upload.status_code} {upload.text}")

        csv_text = csv_path.read_text(encoding="utf-8")
        prediction_export = client.post(
            "/predict/csv/export",
            json={"csv_text": csv_text, "max_rows": 3},
        )
        if prediction_export.status_code != 200 or "top_feature" not in prediction_export.text.splitlines()[0]:
            raise RuntimeError(f"/predict/csv/export failed: {prediction_export.status_code}")

        event_export = client.get("/events/export.csv?limit=10")
        if event_export.status_code != 200 or "text/csv" not in event_export.headers["content-type"]:
            raise RuntimeError(f"/events/export.csv failed: {event_export.status_code}")

        return {
            "metadata": metadata.json()["input_dim"],
            "manifest_model": manifest.json()["model"]["architecture"],
            "prediction": prediction.json()["prediction"],
            "replay_rows": replay.json()["processed_rows"],
            "events": events_summary.json()["total_events"],
            "upload_rows": upload.json()["processed_rows"],
            "dataset_rows": dataset_summary.json()["rows"],
            "data_quality": dataset_profile.json()["quality_score"],
            "prediction_export_rows": len(prediction_export.text.splitlines()) - 1,
        }


def check_frontend_build() -> bool:
    package_json = PROJECT_DIR / "web" / "package.json"
    if not package_json.exists():
        raise FileNotFoundError("web/package.json not found")

    pnpm = shutil.which("pnpm") or shutil.which("pnpm.cmd")
    if pnpm is None:
        raise FileNotFoundError("pnpm not found")

    subprocess.run(
        [pnpm, "exec", "tsc", "-b"],
        cwd=PROJECT_DIR / "web",
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return True


def check_unittests() -> bool:
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=PROJECT_DIR,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return True


def main() -> int:
    print("CyberDD system check")
    print("=" * 40)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")

    model_summary = check_model()
    print(f"Model: {model_summary}")

    metrics_summary = check_metrics()
    print(f"Metrics: {metrics_summary}")

    artifact_summary = check_artifacts()
    print(f"Artifacts: {artifact_summary}")

    api_summary = check_api_routes()
    print(f"API routes: {api_summary}")

    check_frontend_build()
    print("Frontend TypeScript check: passed")

    check_unittests()
    print("Unit tests: passed")
    print("=" * 40)
    print("System check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
