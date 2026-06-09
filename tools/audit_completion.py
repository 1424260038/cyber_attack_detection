"""Audit whether CyberDD satisfies the full demo-system objective."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_json(path: str) -> dict:
    return json.loads((PROJECT_DIR / path).read_text(encoding="utf-8"))


def file_contains(path: str, needles: list[str]) -> bool:
    text = (PROJECT_DIR / path).read_text(encoding="utf-8")
    return all(needle in text for needle in needles)


def exists(path: str) -> bool:
    full_path = PROJECT_DIR / path
    return full_path.exists() and full_path.stat().st_size > 0


def openapi_has(paths: list[str]) -> bool:
    schema = load_json("outputs/openapi.json")
    return all(path in schema.get("paths", {}) for path in paths)


def release_contains(paths: list[str]) -> bool:
    manifest = load_json("release/release_manifest.json")
    included = set(manifest.get("included_files", []))
    return all(path in included for path in paths)


def metric_gate() -> bool:
    metrics = load_json("outputs/test_results.json")
    required = ["accuracy", "precision", "recall", "f1_score", "auc_roc", "average_precision"]
    return all(float(metrics.get(key, 0)) >= 0.95 for key in required)


def build_audit() -> dict:
    checks = [
        {
            "category": "数据闭环",
            "requirement": "具备可复现实验数据、标准化数据、数据摘要和质量画像",
            "passed": all(
                exists(path)
                for path in [
                    "data/demo_traffic.csv",
                    "data/prepared_traffic.csv",
                    "outputs/dataset_summary.json",
                    "outputs/data_profile.json",
                ]
            ),
            "evidence": "data/*.csv, outputs/dataset_summary.json, outputs/data_profile.json",
        },
        {
            "category": "模型闭环",
            "requirement": "具备模型检查点、预处理器、TorchScript 导出和模型清单",
            "passed": all(
                exists(path)
                for path in [
                    "checkpoints/best_model.pth",
                    "artifacts/preprocessor.json",
                    "artifacts/model.pt",
                    "artifacts/model_manifest.json",
                ]
            ),
            "evidence": "checkpoints/best_model.pth, artifacts/*",
        },
        {
            "category": "评估闭环",
            "requirement": "核心评估指标达到演示验收阈值 0.95",
            "passed": metric_gate(),
            "evidence": "outputs/test_results.json",
        },
        {
            "category": "API 功能",
            "requirement": "API 覆盖推理、批量检测、上传、回放、事件、解释、数据画像、运维和交付物下载",
            "passed": openapi_has(
                [
                    "/predict",
                    "/predict/csv",
                    "/predict/upload",
                    "/predict/csv/export",
                    "/predict/upload/export",
                    "/demo/replay",
                    "/events",
                    "/events/summary",
                    "/events/export.csv",
                    "/explain",
                    "/dataset/profile",
                    "/admin/runtime",
                    "/admin/reload",
                    "/artifacts/release.zip",
                ]
            ),
            "evidence": "outputs/openapi.json",
        },
        {
            "category": "前端控制台",
            "requirement": "前端覆盖单条检测、CSV 检测、回放、事件、指标、数据质量、知识图谱、运维和交付物下载",
            "passed": file_contains(
                "web/src/App.tsx",
                [
                    "单条流量检测",
                    "CSV 批量检测",
                    "回放演示流量",
                    "运行时检测统计",
                    "模型评估指标",
                    "数据集与知识图谱",
                    "系统运维状态",
                    "交付物下载",
                ],
            )
            and exists("web/dist/index.html"),
            "evidence": "web/src/App.tsx, web/dist/index.html",
        },
        {
            "category": "知识图谱解释",
            "requirement": "检测结果能映射到 ATT&CK 攻击链和处置建议",
            "passed": file_contains("kg/knowledge_graph.py", ["generate_attack_chain", "attack_chain_cn"])
            and file_contains("api.py", ["explain_attack", "recommendation"]),
            "evidence": "kg/knowledge_graph.py, api.py",
        },
        {
            "category": "运行审计",
            "requirement": "检测事件可记录、汇总、清空和导出",
            "passed": file_contains("utils/event_store.py", ["append", "summary", "clear"])
            and openapi_has(["/events", "/events/summary", "/events/export.csv"]),
            "evidence": "utils/event_store.py, outputs/openapi.json",
        },
        {
            "category": "部署交付",
            "requirement": "具备 Docker/Compose、一键流水线、发布包和发布清单",
            "passed": all(
                exists(path)
                for path in [
                    "Dockerfile",
                    "docker-compose.yml",
                    "tools/build_demo_system.py",
                    "release/cyberdd_release.zip",
                    "release/release_manifest.json",
                ]
            ),
            "evidence": "Dockerfile, docker-compose.yml, tools/build_demo_system.py, release/*",
        },
        {
            "category": "验收材料",
            "requirement": "具备项目报告、答辩手册、OpenAPI、验收清单和完成度审计",
            "passed": all(
                exists(path)
                for path in [
                    "outputs/project_report.md",
                    "outputs/demo_runbook.md",
                    "outputs/openapi.json",
                    "outputs/acceptance_checklist.md",
                ]
            ),
            "evidence": "outputs/project_report.md, outputs/demo_runbook.md, outputs/openapi.json, outputs/acceptance_checklist.md",
        },
        {
            "category": "发布包内容",
            "requirement": "发布包包含源码、模型、前端构建、报告、手册、接口规范和验收材料",
            "passed": release_contains(
                [
                    "api.py",
                    "artifacts\\model.pt",
                    "checkpoints\\best_model.pth",
                    "web\\dist\\index.html",
                    "outputs\\project_report.md",
                    "outputs\\demo_runbook.md",
                    "outputs\\openapi.json",
                    "outputs\\acceptance_checklist.md",
                ]
            ),
            "evidence": "release/release_manifest.json",
        },
        {
            "category": "质量门禁",
            "requirement": "完整质量检查脚本覆盖单测、自检、HTTP 冒烟测试、前端类型检查、ESLint 和构建",
            "passed": file_contains(
                "tools/run_all_checks.py",
                ["unittest", "system_check.py", "smoke_test_service.py", "tsc", "eslint", "vite"],
            )
            and file_contains("tests/test_api.py", ["test_demo_replay_records_events", "test_artifact_downloads"]),
            "evidence": "tools/run_all_checks.py, tests/test_api.py",
        },
    ]

    passed_count = sum(1 for check in checks if check["passed"])
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total": len(checks),
        "passed": passed_count,
        "failed": len(checks) - passed_count,
        "status": "complete" if passed_count == len(checks) else "incomplete",
        "checks": checks,
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# CyberDD 完成度审计",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        f"审计结论：{payload['status']}",
        "",
        f"通过项：{payload['passed']}/{payload['total']}",
        "",
        "| 类别 | 要求 | 状态 | 证据 |",
        "|------|------|------|------|",
    ]
    for check in payload["checks"]:
        status = "通过" if check["passed"] else "未通过"
        lines.append(f"| {check['category']} | {check['requirement']} | {status} | {check['evidence']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CyberDD completion against the full-system objective.")
    parser.add_argument("--json-output", default="outputs/completion_audit.json")
    parser.add_argument("--md-output", default="outputs/completion_audit.md")
    args = parser.parse_args()

    payload = build_audit()
    json_path = PROJECT_DIR / args.json_output
    md_path = PROJECT_DIR / args.md_output
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")

    print(f"Completion audit: {payload['status']} ({payload['passed']}/{payload['total']}) -> {md_path}")
    return 0 if payload["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
