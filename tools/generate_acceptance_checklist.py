"""Generate acceptance checklist artifacts for CyberDD."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]

CHECK_ITEMS = [
    ("数据准备", "data/demo_traffic.csv", "演示数据集"),
    ("数据准备", "data/prepared_traffic.csv", "标准化训练数据"),
    ("数据质量", "outputs/dataset_summary.json", "数据集摘要"),
    ("数据质量", "outputs/data_profile.json", "数据质量画像"),
    ("模型工件", "checkpoints/best_model.pth", "最佳模型检查点"),
    ("模型工件", "artifacts/preprocessor.json", "预处理器"),
    ("模型工件", "artifacts/model.pt", "TorchScript 部署模型"),
    ("模型工件", "artifacts/model_manifest.json", "模型清单"),
    ("评估结果", "outputs/test_results.json", "测试指标"),
    ("API 文档", "outputs/openapi.json", "OpenAPI 接口规范"),
    ("前端", "web/dist/index.html", "前端生产构建"),
    ("部署", "Dockerfile", "Docker 镜像定义"),
    ("部署", "docker-compose.yml", "Docker Compose 编排"),
    ("报告材料", "outputs/project_report.md", "项目报告"),
    ("报告材料", "outputs/demo_runbook.md", "答辩演示手册"),
    ("交付包", "release/cyberdd_release.zip", "完整发布包"),
    ("交付包", "release/release_manifest.json", "发布包清单"),
]


def file_info(relative_path: str) -> dict:
    path = PROJECT_DIR / relative_path
    return {
        "path": relative_path,
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() else 0,
    }


def build_checklist() -> dict:
    items = []
    for category, path, description in CHECK_ITEMS:
        info = file_info(path)
        items.append(
            {
                "category": category,
                "description": description,
                **info,
                "status": "passed" if info["exists"] and info["size"] > 0 else "missing",
            }
        )

    passed = sum(1 for item in items if item["status"] == "passed")
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total": len(items),
        "passed": passed,
        "failed": len(items) - passed,
        "items": items,
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# CyberDD 项目验收清单",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        f"验收结果：{payload['passed']}/{payload['total']} 项通过",
        "",
        "| 类别 | 验收项 | 文件 | 状态 | 大小 |",
        "|------|--------|------|------|------|",
    ]
    for item in payload["items"]:
        status = "通过" if item["status"] == "passed" else "缺失"
        lines.append(
            f"| {item['category']} | {item['description']} | `{item['path']}` | {status} | {item['size']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CyberDD acceptance checklist.")
    parser.add_argument("--json-output", default="outputs/acceptance_checklist.json")
    parser.add_argument("--md-output", default="outputs/acceptance_checklist.md")
    args = parser.parse_args()

    payload = build_checklist()
    json_path = PROJECT_DIR / args.json_output
    md_path = PROJECT_DIR / args.md_output
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")

    if payload["failed"]:
        print(f"Acceptance checklist has {payload['failed']} missing items -> {md_path}")
        return 1

    print(f"Acceptance checklist passed -> {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
