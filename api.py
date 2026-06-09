"""Cyber Attack Detection API."""

import csv
import json
from collections import Counter
from io import StringIO
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import torch
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# 获取项目根目录（api.py所在目录）
script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)  # 切换到项目目录
sys.path.insert(0, str(script_dir))

from kg.knowledge_graph import AttackDetector as KnowledgeGraphAttackDetector
from kg.knowledge_graph import create_attack_knowledge_graph
from data.preprocessing import DEFAULT_EXCLUDED_COLUMNS, TabularPreprocessor
from utils.attack_taxonomy import normalize_label
from utils.event_store import DetectionEventStore
from utils.model_loader import find_default_checkpoint, load_checkpoint_model

model = None
model_loaded = False
model_info: dict[str, Any] = {}
model_error = ""
knowledge_graph = None
attack_knowledge_detector = None
preprocessor: TabularPreprocessor | None = None
preprocessor_error = ""
event_store = DetectionEventStore(script_dir / "outputs" / "detection_events.jsonl")

RECOMMENDATIONS = {
    "normal": "当前流量未表现出明显攻击特征，建议继续监控基线变化。",
    "dos": "建议检查异常连接数、限流策略和边界防护设备日志。",
    "ddos": "建议启用流量清洗、黑洞路由或云防护，并核查源地址分布。",
    "probe": "建议关联端口扫描日志，核查暴露服务和弱口令风险。",
    "r2l": "建议审查远程登录、账号异常行为和访问控制策略。",
    "u2r": "建议核查提权痕迹、系统调用异常和高权限账户活动。",
    "malware": "建议隔离可疑主机，采集进程、文件和网络连接证据。",
    "ransomware": "建议立即隔离终端，检查加密行为、备份状态和横向移动痕迹。",
    "phishing": "建议核查邮件/URL 来源，封禁钓鱼域名并开展账号风险排查。",
    "apt": "建议开展多源日志关联，重点关注持久化、C2 通信和数据外传。",
    "attack": "建议结合日志、流量和终端告警进一步确认攻击类型。",
}

class DetectionRequest(BaseModel):
    features: list[float]

class DetectionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    class_probabilities: dict[str, float] = {}
    feature_contributions: list[dict[str, Any]] = []
    attack_type: str | None = None
    risk_level: str = "Low"
    recommendation: str = ""
    attack_chain: list[str] = []
    attack_chain_cn: list[str] = []
    detected_techniques: list[str] = []
    chain_details: list[dict[str, Any]] = []

class CsvDetectionRequest(BaseModel):
    csv_text: str
    label_column: str | None = None
    max_rows: int = 200

class CsvDetectionResponse(BaseModel):
    total_rows: int
    processed_rows: int
    used_columns: list[str]
    summary: dict[str, int]
    results: list[dict[str, Any]]

class DemoReplayResponse(CsvDetectionResponse):
    source_file: str

class ExplainRequest(BaseModel):
    attack_type: str
    confidence: float = 0.8

class EventListResponse(BaseModel):
    events: list[dict[str, Any]]

class EventSummaryResponse(BaseModel):
    total_events: int
    prediction_counts: dict[str, int]
    risk_counts: dict[str, int]
    attack_type_counts: dict[str, int]
    latest: dict[str, Any] | None = None

class DatasetSummaryResponse(BaseModel):
    input: str | None = None
    files: list[str] = []
    output: str | None = None
    rows: int = 0
    feature_columns: int = 0
    label_distribution: dict[str, int] = {}

class DatasetProfileResponse(BaseModel):
    input: str | None = None
    files: list[str] = []
    rows: int = 0
    columns: int = 0
    numeric_feature_columns: int = 0
    missing_rate: float = 0.0
    label_distribution: dict[str, int] = {}
    class_count: int = 0
    imbalance_ratio: float = 0.0
    low_variance_columns: list[str] = []
    warnings: list[str] = []
    quality_score: int = 0

class AdminReloadResponse(BaseModel):
    reloaded: bool
    model_loaded: bool
    preprocessor_loaded: bool
    knowledge_graph_loaded: bool
    model_error: str = ""
    preprocessor_error: str = ""

def require_admin_token(x_admin_token: str | None) -> None:
    expected = os.getenv("CYBERDD_ADMIN_TOKEN")
    if expected and x_admin_token != expected:
        raise HTTPException(status_code=401, detail="Invalid admin token")

def record_detection_event(
    result: dict[str, Any],
    source: str,
    row_index: int | None = None,
) -> dict[str, Any]:
    event = {
        "source": source,
        "row_index": row_index,
        "prediction": result.get("prediction"),
        "confidence": result.get("confidence"),
        "attack_type": result.get("attack_type"),
        "risk_level": result.get("risk_level"),
        "probabilities": result.get("probabilities"),
        "attack_chain_cn": result.get("attack_chain_cn", []),
    }
    return event_store.append(event)

def normalize_attack_label(label: str | None) -> str:
    if not label:
        return "normal"
    return normalize_label(label)

def risk_level(prediction: str, confidence: float) -> str:
    if prediction == "Normal":
        return "Low"
    if confidence >= 0.85:
        return "High"
    if confidence >= 0.65:
        return "Medium"
    return "Low"

def explain_attack(attack_type: str | None, confidence: float) -> dict[str, Any]:
    normalized_type = normalize_attack_label(attack_type)
    recommendation = RECOMMENDATIONS.get(normalized_type, RECOMMENDATIONS["attack"])

    if normalized_type == "normal" or attack_knowledge_detector is None:
        return {
            "recommendation": recommendation,
            "attack_chain": [],
            "attack_chain_cn": [],
            "detected_techniques": [],
            "chain_details": [],
        }

    chain = attack_knowledge_detector.generate_attack_chain([normalized_type])
    return {
        "recommendation": recommendation,
        "attack_chain": chain["attack_chain"],
        "attack_chain_cn": chain["attack_chain_cn"],
        "detected_techniques": chain["detected_techniques"],
        "chain_details": chain["chain_details"],
    }

def resolve_feature_names(feature_names: list[str] | None = None) -> list[str]:
    expected_dim = int(model_info["input_dim"])
    if feature_names and len(feature_names) == expected_dim:
        return feature_names
    if preprocessor is not None and len(preprocessor.feature_columns) == expected_dim:
        return preprocessor.feature_columns
    return [f"f{i}" for i in range(expected_dim)]

def compute_feature_contributions(
    features: list[float],
    feature_names: list[str] | None = None,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    names = resolve_feature_names(feature_names)
    ranked = []
    for index, value in enumerate(features):
        numeric_value = float(value)
        score = abs(numeric_value)
        ranked.append(
            {
                "feature": names[index] if index < len(names) else f"f{index}",
                "index": index,
                "value": numeric_value,
                "score": score,
                "direction": "above_baseline" if numeric_value >= 0 else "below_baseline",
            }
        )
    return sorted(ranked, key=lambda item: item["score"], reverse=True)[:top_k]


def load_demo_feature_sample(label: str) -> list[float]:
    """Load one real demo row and convert it to the model input space."""
    expected_dim = int(model_info["input_dim"])
    demo_path = script_dir / "data" / "demo_traffic.csv"
    if not demo_path.exists():
        raise HTTPException(status_code=404, detail="data/demo_traffic.csv not found")

    df = pd.read_csv(demo_path, low_memory=False)
    if "Label" not in df.columns:
        raise HTTPException(status_code=500, detail="Demo CSV must contain a Label column")

    normalized_labels = df["Label"].map(normalize_label)
    if label == "normal":
        matches = df[normalized_labels == "normal"]
    else:
        matches = df[normalized_labels != "normal"]

    if matches.empty:
        raise HTTPException(status_code=500, detail=f"No {label} sample found in demo CSV")

    sample = matches.iloc[[0]]
    if preprocessor is not None and len(preprocessor.feature_columns) == expected_dim:
        features = preprocessor.transform_frame(sample)[0]
    else:
        feature_columns = [col for col in sample.columns if col not in DEFAULT_EXCLUDED_COLUMNS]
        numeric = sample[feature_columns].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        if numeric.shape[1] < expected_dim:
            raise HTTPException(
                status_code=500,
                detail=f"Demo CSV needs at least {expected_dim} numeric feature columns, got {numeric.shape[1]}",
            )
        features = numeric.iloc[0, :expected_dim].fillna(0.0).to_numpy(dtype="float32")

    return [round(float(value), 6) for value in features]


def predict_features(features: list[float], feature_names: list[str] | None = None) -> dict[str, Any]:
    expected_dim = int(model_info["input_dim"])
    if len(features) != expected_dim:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {expected_dim} features, got {len(features)}",
        )

    device = next(model.parameters()).device
    features_tensor = torch.tensor(
        features,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        outputs = model(features_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    class_names = model_info.get("class_names", [])
    class_probabilities = {
        class_names[i] if i < len(class_names) else f"Class_{i}": probs[i].item()
        for i in range(len(probs))
    }

    attack_type = None
    if int(model_info["num_classes"]) == 2:
        normal_prob = probs[0].item()
        attack_prob = probs[1].item()
        if attack_prob > normal_prob:
            attack_type = "attack"
    else:
        normal_prob = probs[0].item()
        attack_prob = probs[1:].sum().item()
        if attack_prob > normal_prob:
            attack_index = int(torch.argmax(probs[1:]).item() + 1)
            attack_type = class_names[attack_index] if attack_index < len(class_names) else f"Class_{attack_index}"

    is_attack = attack_prob > normal_prob
    prediction = "Attack" if is_attack else "Normal"
    confidence = attack_prob if is_attack else normal_prob
    explanation = explain_attack(attack_type if is_attack else "normal", confidence)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": {
            "Normal": normal_prob,
            "Attack": attack_prob,
        },
        "class_probabilities": class_probabilities,
        "attack_type": attack_type if is_attack else "normal",
        "risk_level": risk_level(prediction, confidence),
        "feature_contributions": compute_feature_contributions(features, feature_names),
        **explanation,
    }

def load_model():
    global model, model_loaded, model_info, model_error
    
    try:
        checkpoint_path = find_default_checkpoint(script_dir)
        model, model_info = load_checkpoint_model(checkpoint_path)
        model_loaded = True
        model_error = ""
        print(
            "Model loaded successfully: "
            f"{model_info['architecture']} from {model_info['checkpoint_path']}"
        )
    except Exception as e:
        model = None
        model_info = {}
        model_loaded = False
        model_error = str(e)
        print(f"Model loading failed: {model_error}")
        import traceback
        traceback.print_exc()

def load_knowledge_graph():
    global knowledge_graph, attack_knowledge_detector
    knowledge_graph = create_attack_knowledge_graph()
    attack_knowledge_detector = KnowledgeGraphAttackDetector(knowledge_graph)

def load_preprocessor():
    global preprocessor, preprocessor_error
    path = script_dir / "artifacts" / "preprocessor.json"
    if not path.exists():
        preprocessor = None
        preprocessor_error = f"Preprocessor not found: {path}"
        return

    try:
        preprocessor = TabularPreprocessor.load(path)
        preprocessor_error = ""
    except Exception as e:
        preprocessor = None
        preprocessor_error = str(e)

def artifact_status() -> dict[str, Any]:
    paths = {
        "checkpoint": script_dir / "checkpoints" / "best_model.pth",
        "preprocessor": script_dir / "artifacts" / "preprocessor.json",
        "torchscript": script_dir / "artifacts" / "model.pt",
        "manifest": script_dir / "artifacts" / "model_manifest.json",
        "dataset_summary": script_dir / "outputs" / "dataset_summary.json",
        "data_profile": script_dir / "outputs" / "data_profile.json",
        "openapi_schema": script_dir / "outputs" / "openapi.json",
        "acceptance_checklist": script_dir / "outputs" / "acceptance_checklist.md",
        "acceptance_checklist_json": script_dir / "outputs" / "acceptance_checklist.json",
        "completion_audit": script_dir / "outputs" / "completion_audit.md",
        "completion_audit_json": script_dir / "outputs" / "completion_audit.json",
        "project_report": script_dir / "outputs" / "project_report.md",
        "demo_runbook": script_dir / "outputs" / "demo_runbook.md",
        "release_package": script_dir / "release" / "cyberdd_release.zip",
        "release_manifest": script_dir / "release" / "release_manifest.json",
        "frontend_build": script_dir / "web" / "dist" / "index.html",
    }
    return {
        name: {
            "path": str(path.relative_to(script_dir)),
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() else 0,
        }
        for name, path in paths.items()
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    load_knowledge_graph()
    load_preprocessor()
    yield

app = FastAPI(title="Cyber Attack Detection API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

web_dist_dir = script_dir / "web" / "dist"
if web_dist_dir.exists():
    assets_dir = web_dist_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
    public_use_file = web_dist_dir / "use.txt"
    if public_use_file.exists():
        app.mount("/static", StaticFiles(directory=web_dist_dir), name="static")

@app.get("/")
def root():
    index_path = web_dist_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Cyber Attack Detection API", "docs": "/docs"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_error": model_error,
        "preprocessor_loaded": preprocessor is not None,
        "preprocessor_error": preprocessor_error,
        "model": model_info if model_loaded else None,
    }

@app.get("/admin/runtime")
def admin_runtime(x_admin_token: str | None = Header(default=None, alias="X-Admin-Token")):
    require_admin_token(x_admin_token)
    return {
        "model_loaded": model_loaded,
        "model_error": model_error,
        "preprocessor_loaded": preprocessor is not None,
        "preprocessor_error": preprocessor_error,
        "knowledge_graph_loaded": knowledge_graph is not None,
        "model": model_info if model_loaded else None,
        "artifacts": artifact_status(),
    }

@app.post("/admin/reload", response_model=AdminReloadResponse)
def admin_reload(x_admin_token: str | None = Header(default=None, alias="X-Admin-Token")):
    require_admin_token(x_admin_token)
    load_model()
    load_knowledge_graph()
    load_preprocessor()
    return {
        "reloaded": True,
        "model_loaded": model_loaded,
        "preprocessor_loaded": preprocessor is not None,
        "knowledge_graph_loaded": knowledge_graph is not None,
        "model_error": model_error,
        "preprocessor_error": preprocessor_error,
    }

def download_file(path: Path, filename: str, media_type: str) -> FileResponse:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not found")
    return FileResponse(path, media_type=media_type, filename=filename)

@app.get("/artifacts/report")
def download_project_report():
    return download_file(script_dir / "outputs" / "project_report.md", "project_report.md", "text/markdown; charset=utf-8")

@app.get("/artifacts/runbook")
def download_demo_runbook():
    return download_file(script_dir / "outputs" / "demo_runbook.md", "demo_runbook.md", "text/markdown; charset=utf-8")

@app.get("/artifacts/manifest.json")
def download_manifest_file():
    return download_file(script_dir / "artifacts" / "model_manifest.json", "model_manifest.json", "application/json")

@app.get("/artifacts/data-profile.json")
def download_data_profile_file():
    return download_file(script_dir / "outputs" / "data_profile.json", "data_profile.json", "application/json")

@app.get("/artifacts/openapi.json")
def download_openapi_file():
    return download_file(script_dir / "outputs" / "openapi.json", "openapi.json", "application/json")

@app.get("/artifacts/acceptance-checklist")
def download_acceptance_checklist():
    return download_file(script_dir / "outputs" / "acceptance_checklist.md", "acceptance_checklist.md", "text/markdown; charset=utf-8")

@app.get("/artifacts/acceptance-checklist.json")
def download_acceptance_checklist_json():
    return download_file(script_dir / "outputs" / "acceptance_checklist.json", "acceptance_checklist.json", "application/json")

@app.get("/artifacts/completion-audit")
def download_completion_audit():
    return download_file(script_dir / "outputs" / "completion_audit.md", "completion_audit.md", "text/markdown; charset=utf-8")

@app.get("/artifacts/completion-audit.json")
def download_completion_audit_json():
    return download_file(script_dir / "outputs" / "completion_audit.json", "completion_audit.json", "application/json")

@app.get("/artifacts/release.zip")
def download_release_package():
    return download_file(script_dir / "release" / "cyberdd_release.zip", "cyberdd_release.zip", "application/zip")

@app.get("/artifacts/release-manifest.json")
def download_release_manifest():
    return download_file(script_dir / "release" / "release_manifest.json", "release_manifest.json", "application/json")

@app.get("/metadata")
def metadata():
    if not model_loaded:
        raise HTTPException(status_code=503, detail=model_error or "Model not loaded")

    manifest_path = script_dir / "artifacts" / "model_manifest.json"
    return {
        "input_dim": model_info["input_dim"],
        "num_classes": model_info["num_classes"],
        "class_names": model_info["class_names"],
        "architecture": model_info["architecture"],
        "checkpoint_path": model_info["checkpoint_path"],
        "best_val_acc": model_info.get("best_val_acc"),
        "preprocessor_loaded": preprocessor is not None,
        "feature_columns": preprocessor.feature_columns if preprocessor else None,
        "manifest_available": manifest_path.exists(),
    }

@app.get("/manifest")
def manifest():
    path = script_dir / "artifacts" / "model_manifest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model manifest not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/demo-samples")
def demo_samples():
    if not model_loaded:
        raise HTTPException(status_code=503, detail=model_error or "Model not loaded")

    return {
        "normal": load_demo_feature_sample("normal"),
        "attack": load_demo_feature_sample("attack"),
        "description": "样例来自 data/demo_traffic.csv，并已转换为模型实际使用的标准化输入空间。",
    }

@app.post("/demo/replay", response_model=DemoReplayResponse)
def demo_replay(max_rows: int = 16):
    if max_rows < 1 or max_rows > 200:
        raise HTTPException(status_code=422, detail="max_rows must be between 1 and 200")

    demo_path = script_dir / "data" / "demo_traffic.csv"
    if not demo_path.exists():
        raise HTTPException(status_code=404, detail="data/demo_traffic.csv not found")

    payload = analyze_csv_request(
        CsvDetectionRequest(
            csv_text=demo_path.read_text(encoding="utf-8"),
            label_column="Label",
            max_rows=max_rows,
        ),
        source="demo_replay",
    )
    return {
        "source_file": str(demo_path.relative_to(script_dir)),
        **payload,
    }

@app.get("/metrics")
def metrics():
    payload: dict[str, Any] = {}
    files = {
        "test_results": script_dir / "outputs" / "test_results.json",
    }
    for key, path in files.items():
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                payload[key] = json.load(f)
        else:
            payload[key] = None

    history_candidates = [
        script_dir / "checkpoints" / "training_history.json",
        script_dir / "outputs" / "training_history.json",
    ]
    payload["training_history"] = None
    for path in history_candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                payload["training_history"] = json.load(f)
            break
    return payload

@app.get("/dataset/summary", response_model=DatasetSummaryResponse)
def dataset_summary():
    path = script_dir / "outputs" / "dataset_summary.json"
    if not path.exists():
        manifest_path = script_dir / "artifacts" / "model_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_payload = json.load(f)
            summary = manifest_payload.get("dataset_summary")
            if summary:
                return summary
        raise HTTPException(status_code=404, detail="Dataset summary not found")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/dataset/profile", response_model=DatasetProfileResponse)
def dataset_profile():
    path = script_dir / "outputs" / "data_profile.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Data profile not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/events/export.csv")
def export_events(limit: int = 500):
    if limit < 1 or limit > 5000:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 5000")

    rows = event_store.list_recent(limit=limit)
    output = StringIO()
    fieldnames = [
        "id",
        "timestamp",
        "source",
        "row_index",
        "prediction",
        "confidence",
        "attack_type",
        "risk_level",
        "normal_probability",
        "attack_probability",
        "attack_chain_cn",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        probabilities = row.get("probabilities") or {}
        writer.writerow(
            {
                "id": row.get("id", ""),
                "timestamp": row.get("timestamp", ""),
                "source": row.get("source", ""),
                "row_index": row.get("row_index", ""),
                "prediction": row.get("prediction", ""),
                "confidence": row.get("confidence", ""),
                "attack_type": row.get("attack_type", ""),
                "risk_level": row.get("risk_level", ""),
                "normal_probability": probabilities.get("Normal", ""),
                "attack_probability": probabilities.get("Attack", ""),
                "attack_chain_cn": ";".join(row.get("attack_chain_cn") or []),
            }
        )

    return Response(
        content=output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="detection_events.csv"'},
    )

def analyze_csv_request(request: CsvDetectionRequest, source: str = "csv") -> dict[str, Any]:
    if not model_loaded or model is None:
        raise HTTPException(status_code=503, detail=model_error or "Model not loaded")
    if not request.csv_text.strip():
        raise HTTPException(status_code=422, detail="CSV content cannot be empty")
    if request.max_rows < 1 or request.max_rows > 1000:
        raise HTTPException(status_code=422, detail="max_rows must be between 1 and 1000")

    try:
        df = pd.read_csv(StringIO(request.csv_text), low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid CSV content: {e}") from e

    if df.empty:
        raise HTTPException(status_code=422, detail="CSV contains no rows")

    expected_dim = int(model_info["input_dim"])
    if preprocessor is not None:
        try:
            transformed = preprocessor.transform_frame(df.head(request.max_rows))
        except ValueError:
            # Uploaded CSV may use arbitrary numeric headers. Fall back to positional columns.
            transformed = None
        else:
            used_columns = preprocessor.feature_columns
    else:
        transformed = None

    if transformed is None:
        excluded_columns = set(DEFAULT_EXCLUDED_COLUMNS)
        if request.label_column:
            excluded_columns.add(request.label_column)

        feature_candidates = [col for col in df.columns if col not in excluded_columns]
        numeric_df = df[feature_candidates].apply(pd.to_numeric, errors="coerce")
        numeric_df = numeric_df.dropna(axis=1, how="all")

        if numeric_df.shape[1] < expected_dim:
            raise HTTPException(
                status_code=422,
                detail=f"CSV needs at least {expected_dim} numeric feature columns, got {numeric_df.shape[1]}",
            )

        used_columns = list(numeric_df.columns[:expected_dim])
        raw_features = numeric_df[used_columns].head(request.max_rows).fillna(0.0).to_numpy(dtype="float32")
        if preprocessor is not None and len(preprocessor.feature_columns) == expected_dim:
            transformed = preprocessor.transform_array(raw_features)
        else:
            transformed = raw_features

    results = []
    summary_counter: Counter[str] = Counter()
    for row_index, row in enumerate(transformed):
        prediction = predict_features(row.astype(float).tolist(), feature_names=used_columns)
        record_detection_event(prediction, source=source, row_index=row_index)
        summary_counter[prediction["prediction"]] += 1
        results.append(
            {
                "row_index": row_index,
                **prediction,
            }
        )

    return {
        "total_rows": int(len(df)),
        "processed_rows": int(len(results)),
        "used_columns": used_columns,
        "summary": dict(summary_counter),
        "results": results,
    }

def csv_detection_response_to_csv(payload: dict[str, Any]) -> str:
    output = StringIO()
    fieldnames = [
        "row_index",
        "prediction",
        "confidence",
        "risk_level",
        "attack_type",
        "normal_probability",
        "attack_probability",
        "top_feature",
        "top_feature_value",
        "top_feature_score",
        "attack_chain_cn",
        "recommendation",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in payload["results"]:
        probabilities = row.get("probabilities") or {}
        contributions = row.get("feature_contributions") or []
        top_feature = contributions[0] if contributions else {}
        writer.writerow(
            {
                "row_index": row.get("row_index", ""),
                "prediction": row.get("prediction", ""),
                "confidence": row.get("confidence", ""),
                "risk_level": row.get("risk_level", ""),
                "attack_type": row.get("attack_type", ""),
                "normal_probability": probabilities.get("Normal", ""),
                "attack_probability": probabilities.get("Attack", ""),
                "top_feature": top_feature.get("feature", ""),
                "top_feature_value": top_feature.get("value", ""),
                "top_feature_score": top_feature.get("score", ""),
                "attack_chain_cn": ";".join(row.get("attack_chain_cn") or []),
                "recommendation": row.get("recommendation", ""),
            }
        )
    return output.getvalue()

@app.get("/events", response_model=EventListResponse)
def list_events(limit: int = 50):
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500")
    return {"events": event_store.list_recent(limit=limit)}

@app.get("/events/summary", response_model=EventSummaryResponse)
def event_summary(limit: int = 500):
    if limit < 1 or limit > 5000:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 5000")
    return event_store.summary(limit=limit)

@app.delete("/events")
def clear_events():
    event_store.clear()
    return {"cleared": True}

@app.get("/knowledge-graph")
def knowledge_graph_summary():
    if knowledge_graph is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not loaded")

    data = knowledge_graph.to_dict()
    tactics = [
        {
            "id": entity["id"],
            "name": entity["name"],
            "name_cn": entity["name_cn"],
        }
        for entity in data["entities"].values()
        if entity["type"] == "tactic"
    ]
    return {
        "entity_count": len(data["entities"]),
        "relation_count": len(data["relations"]),
        "tactics": tactics,
    }

@app.post("/explain")
def explain(request: ExplainRequest):
    return explain_attack(request.attack_type, request.confidence)

@app.post("/predict", response_model=DetectionResponse)
def predict(request: DetectionRequest):
    if not model_loaded or model is None:
        raise HTTPException(status_code=503, detail=model_error or "Model not loaded")
    
    try:
        result = predict_features(request.features)
        record_detection_event(result, source="single")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=list[DetectionResponse])
def predict_batch(requests: list[DetectionRequest]):
    if not model_loaded or model is None:
        raise HTTPException(status_code=503, detail=model_error or "Model not loaded")
    if not requests:
        raise HTTPException(status_code=422, detail="Batch request cannot be empty")

    try:
        results = []
        for index, request in enumerate(requests):
            result = predict_features(request.features)
            record_detection_event(result, source="batch", row_index=index)
            results.append(result)
        return results
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.post("/predict/csv", response_model=CsvDetectionResponse)
def predict_csv(request: CsvDetectionRequest):
    return analyze_csv_request(request, source="csv")

@app.post("/predict/csv/export")
def export_csv_predictions(request: CsvDetectionRequest):
    payload = analyze_csv_request(request, source="csv_export")
    return Response(
        content=csv_detection_response_to_csv(payload),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="prediction_results.csv"'},
    )

@app.post("/predict/upload", response_model=CsvDetectionResponse)
async def predict_upload(
    file: UploadFile = File(...),
    label_column: str | None = Form(default=None),
    max_rows: int = Form(default=200),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Only CSV files are supported")

    content = await file.read()
    try:
        csv_text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        csv_text = content.decode("gbk")

    return predict_csv(
        CsvDetectionRequest(
            csv_text=csv_text,
            label_column=label_column,
            max_rows=max_rows,
        )
    )

@app.post("/predict/upload/export")
async def export_upload_predictions(
    file: UploadFile = File(...),
    label_column: str | None = Form(default=None),
    max_rows: int = Form(default=200),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Only CSV files are supported")

    content = await file.read()
    try:
        csv_text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        csv_text = content.decode("gbk")

    payload = analyze_csv_request(
        CsvDetectionRequest(
            csv_text=csv_text,
            label_column=label_column,
            max_rows=max_rows,
        ),
        source="upload_export",
    )
    return Response(
        content=csv_detection_response_to_csv(payload),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="prediction_results.csv"'},
    )

if __name__ == "__main__":
    import uvicorn
    print(f"Working directory: {os.getcwd()}")
    print(f"Script directory: {script_dir}")
    print("Starting API server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
