"""API regression tests for CyberDD."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import api


PROJECT_DIR = Path(__file__).resolve().parents[1]


class ApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.client.delete("/events")

    def test_metadata_and_manifest(self) -> None:
        metadata = self.client.get("/metadata")
        self.assertEqual(metadata.status_code, 200)
        self.assertEqual(metadata.json()["input_dim"], 64)

        manifest = self.client.get("/manifest")
        self.assertEqual(manifest.status_code, 200)
        self.assertEqual(manifest.json()["model"]["architecture"], "SimpleClassifier")
        self.assertIn("data_profile", manifest.json())

    def test_admin_runtime_and_reload(self) -> None:
        runtime = self.client.get("/admin/runtime")
        self.assertEqual(runtime.status_code, 200)
        self.assertTrue(runtime.json()["artifacts"]["checkpoint"]["exists"])
        self.assertTrue(runtime.json()["artifacts"]["demo_runbook"]["exists"])
        self.assertTrue(runtime.json()["artifacts"]["release_package"]["exists"])
        self.assertTrue(runtime.json()["artifacts"]["acceptance_checklist"]["exists"])
        self.assertTrue(runtime.json()["artifacts"]["completion_audit"]["exists"])

        reload_response = self.client.post("/admin/reload")
        self.assertEqual(reload_response.status_code, 200)
        self.assertTrue(reload_response.json()["model_loaded"])
        self.assertTrue(reload_response.json()["knowledge_graph_loaded"])

    def test_optional_admin_token(self) -> None:
        old_token = os.environ.get("CYBERDD_ADMIN_TOKEN")
        os.environ["CYBERDD_ADMIN_TOKEN"] = "test-token"
        try:
            denied = self.client.get("/admin/runtime")
            self.assertEqual(denied.status_code, 401)

            allowed = self.client.get("/admin/runtime", headers={"X-Admin-Token": "test-token"})
            self.assertEqual(allowed.status_code, 200)
        finally:
            if old_token is None:
                os.environ.pop("CYBERDD_ADMIN_TOKEN", None)
            else:
                os.environ["CYBERDD_ADMIN_TOKEN"] = old_token

    def test_artifact_downloads(self) -> None:
        for path in [
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
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertGreater(len(response.content), 0)

        release_manifest = self.client.get("/artifacts/release-manifest.json").json()
        self.assertGreater(release_manifest["file_count"], 0)
        self.assertIn("outputs\\acceptance_checklist.md", release_manifest["included_files"])
        self.assertIn("outputs\\completion_audit.md", release_manifest["included_files"])

    def test_predict_records_event(self) -> None:
        samples = self.client.get("/demo-samples").json()
        normal_response = self.client.post("/predict", json={"features": samples["normal"]})
        self.assertEqual(normal_response.status_code, 200)
        self.assertEqual(normal_response.json()["prediction"], "Normal")
        self.assertEqual(normal_response.json()["risk_level"], "Low")

        response = self.client.post("/predict", json={"features": samples["attack"]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prediction"], "Attack")
        self.assertGreater(len(response.json()["feature_contributions"]), 0)

        summary = self.client.get("/events/summary")
        self.assertEqual(summary.status_code, 200)
        self.assertEqual(summary.json()["total_events"], 2)

    def test_upload_records_events(self) -> None:
        csv_path = PROJECT_DIR / "data" / "demo_traffic.csv"
        with open(csv_path, "rb") as f:
            response = self.client.post(
                "/predict/upload",
                files={"file": ("demo_traffic.csv", f, "text/csv")},
                data={"max_rows": "4"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["processed_rows"], 4)

        events = self.client.get("/events?limit=10")
        self.assertEqual(events.status_code, 200)
        self.assertEqual(len(events.json()["events"]), 4)

    def test_demo_replay_records_events(self) -> None:
        response = self.client.post("/demo/replay?max_rows=5")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["processed_rows"], 5)
        self.assertEqual(response.json()["source_file"], "data\\demo_traffic.csv")

        summary = self.client.get("/events/summary")
        self.assertEqual(summary.status_code, 200)
        self.assertEqual(summary.json()["total_events"], 5)

    def test_csv_prediction_export(self) -> None:
        csv_path = PROJECT_DIR / "data" / "demo_traffic.csv"
        csv_text = csv_path.read_text(encoding="utf-8")
        export = self.client.post(
            "/predict/csv/export",
            json={"csv_text": csv_text, "max_rows": 3},
        )
        self.assertEqual(export.status_code, 200)
        self.assertIn("text/csv", export.headers["content-type"])
        header = export.text.splitlines()[0]
        self.assertIn("top_feature", header)
        self.assertEqual(len(export.text.splitlines()), 4)

    def test_dataset_summary_and_event_export(self) -> None:
        summary = self.client.get("/dataset/summary")
        self.assertEqual(summary.status_code, 200)
        self.assertEqual(summary.json()["rows"], 960)
        self.assertEqual(summary.json()["feature_columns"], 64)

        profile = self.client.get("/dataset/profile")
        self.assertEqual(profile.status_code, 200)
        self.assertEqual(profile.json()["quality_score"], 100)
        self.assertEqual(profile.json()["numeric_feature_columns"], 64)

        samples = self.client.get("/demo-samples").json()
        response = self.client.post("/predict", json={"features": samples["attack"]})
        self.assertEqual(response.status_code, 200)
        prediction_label = response.json()["prediction"]

        export = self.client.get("/events/export.csv?limit=10")
        self.assertEqual(export.status_code, 200)
        self.assertIn("text/csv", export.headers["content-type"])
        self.assertIn("prediction", export.text.splitlines()[0])
        self.assertIn(prediction_label, export.text)


if __name__ == "__main__":
    unittest.main()
