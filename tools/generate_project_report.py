"""Generate a concise Markdown report for the CyberDD project."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from data.preprocessing import TabularPreprocessor
from utils.model_loader import load_checkpoint_model


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    output_path = PROJECT_DIR / "outputs" / "project_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _, model_info = load_checkpoint_model(PROJECT_DIR / "checkpoints" / "best_model.pth")
    metrics = load_json(PROJECT_DIR / "outputs" / "test_results.json") or {}
    history = load_json(PROJECT_DIR / "checkpoints" / "training_history.json") or {}
    dataset_summary = load_json(PROJECT_DIR / "outputs" / "dataset_summary.json") or {}
    data_profile = load_json(PROJECT_DIR / "outputs" / "data_profile.json") or {}
    manifest = load_json(PROJECT_DIR / "artifacts" / "model_manifest.json") or {}
    preprocessor = TabularPreprocessor.load(PROJECT_DIR / "artifacts" / "preprocessor.json")
    manifest_model = manifest.get("model", {})

    report = f"""# CyberDD 系统报告

生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. 系统概述

CyberDD 是面向大学生创新创业训练计划的网络攻击行为识别系统，当前版本已形成“数据准备 -> 预处理 -> 模型训练 -> 评估 -> API 推理 -> 前端控制台 -> 攻击链解释 -> 部署导出”的完整闭环。

## 2. 模型与工件

| 项目 | 当前值 |
|------|--------|
| 模型结构 | {model_info.get("architecture")} |
| 输入维度 | {model_info.get("input_dim")} |
| 分类类别数 | {model_info.get("num_classes")} |
| 最佳验证准确率 | {model_info.get("best_val_acc")} |
| 检查点 | checkpoints/best_model.pth |
| 预处理工件 | artifacts/preprocessor.json |
| TorchScript | artifacts/model.pt |
| 特征列数量 | {len(preprocessor.feature_columns)} |
| 模型 SHA256 | {str(manifest_model.get("checkpoint_sha256", "N/A"))[:16]}... |

## 3. 数据集摘要

| 项目 | 当前值 |
|------|--------|
| 数据来源 | {dataset_summary.get("input", "N/A")} |
| 标准化输出 | {dataset_summary.get("output", "N/A")} |
| 样本行数 | {dataset_summary.get("rows", "N/A")} |
| 特征列数 | {dataset_summary.get("feature_columns", "N/A")} |
| 标签分布 | {dataset_summary.get("label_distribution", "N/A")} |
| 数据质量分 | {data_profile.get("quality_score", "N/A")} |
| 缺失/非法值比例 | {data_profile.get("missing_rate", "N/A")} |
| 类别失衡比 | {data_profile.get("imbalance_ratio", "N/A")} |
| 质量预警 | {data_profile.get("warnings", []) or "无"} |

## 4. 测试指标

| 指标 | 数值 |
|------|------|
| Accuracy | {metrics.get("accuracy", "N/A")} |
| Precision | {metrics.get("precision", "N/A")} |
| Recall | {metrics.get("recall", "N/A")} |
| F1-Score | {metrics.get("f1_score", "N/A")} |
| AUC-ROC | {metrics.get("auc_roc", "N/A")} |
| Average Precision | {metrics.get("average_precision", "N/A")} |

训练轮数记录：{len(history.get("train_loss", []))}

## 5. API 能力

| 接口 | 功能 |
|------|------|
| GET / | 托管前端控制台或返回 API 信息 |
| GET /health | 服务、模型和预处理器状态 |
| GET /metadata | 模型结构、输入维度、类别、特征列 |
| GET /manifest | 模型、预处理器、指标和数据摘要清单 |
| GET /demo-samples | 正常/攻击样例 |
| POST /demo/replay | 回放演示流量并写入运行时事件 |
| POST /predict | 单条检测 |
| POST /predict/batch | JSON 批量检测 |
| POST /predict/csv | CSV 文本检测 |
| POST /predict/csv/export | CSV 文本检测并导出结果 |
| POST /predict/upload | CSV 文件上传检测 |
| POST /predict/upload/export | CSV 文件上传检测并导出结果 |
| GET /metrics | 训练与测试指标 |
| GET /dataset/summary | 数据集来源、样本数、特征列和标签分布 |
| GET /knowledge-graph | ATT&CK 知识图谱摘要 |
| POST /explain | 攻击链解释 |
| GET /events | 查询近期检测事件 |
| GET /events/summary | 查询运行时检测统计 |
| GET /events/export.csv | 导出近期检测事件 CSV |
| DELETE /events | 清空演示检测事件 |
| GET /admin/runtime | 查看运行时模型、预处理器和工件状态 |
| POST /admin/reload | 重新加载模型、预处理器和知识图谱 |
| GET /artifacts/report | 下载项目报告 |
| GET /artifacts/runbook | 下载答辩演示手册 |
| GET /artifacts/manifest.json | 下载模型清单 |
| GET /artifacts/data-profile.json | 下载数据质量画像 |
| GET /artifacts/openapi.json | 下载 OpenAPI 接口规范 |
| GET /artifacts/acceptance-checklist | 下载 Markdown 验收清单 |
| GET /artifacts/acceptance-checklist.json | 下载 JSON 验收清单 |
| GET /artifacts/completion-audit | 下载系统完成度审计 |
| GET /artifacts/completion-audit.json | 下载系统完成度审计 JSON |
| GET /artifacts/release.zip | 下载完整项目发布包 |
| GET /artifacts/release-manifest.json | 下载发布包清单 |

## 6. 演示流程

1. 运行 `python tools/build_demo_system.py` 刷新演示数据、工件、报告和质量检查。
2. 运行 `python api.py` 启动后端和前端静态托管。
3. 打开 `outputs/demo_runbook.md` 作为现场演示提纲。
4. 打开 `http://localhost:8000`，依次演示样例检测、Top 特征贡献、演示流量回放、CSV 文件上传检测、批量结果导出、数据集概览、评估指标、知识图谱解释。
5. 在“运行时检测统计”和“近期告警记录”中展示检测审计日志，并导出事件 CSV。
6. 在“系统运维状态”和“交付物下载”中展示运行时工件、执行重载并下载报告/手册/发布包。

## 7. 真实数据接入流程

将 CICIDS2017、NSL-KDD 或其他网络流量 CSV 放入任意目录后，执行：

```bash
python tools/prepare_dataset.py --input 原始CSV或目录 --output data/prepared_traffic.csv --summary outputs/dataset_summary.json
python tools/profile_dataset.py --input data/prepared_traffic.csv --output outputs/data_profile.json
python tools/fit_preprocessor.py --input data/prepared_traffic.csv --output artifacts/preprocessor.json --input-dim 64
python main.py --mode train --model cnn_lstm --data ./data --epochs 100 --output_dir checkpoints
python main.py --mode test --model cnn_lstm --checkpoint checkpoints/best_model.pth --data ./data --output_dir outputs
python tools/export_model.py --checkpoint checkpoints/best_model.pth --output artifacts/model.pt
python tools/generate_model_manifest.py
python tools/export_openapi.py
python tools/generate_acceptance_checklist.py
python tools/audit_completion.py
python tools/package_release.py
```

## 8. 后续扩展

- 接入 CICIDS2017 / NSL-KDD 等真实公开数据集。
- 将多模态模型从研究代码切换为生产推理模型。
- 增加 Docker 镜像与离线安装包。
"""

    output_path.write_text(report, encoding="utf-8")
    print(f"Generated report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
