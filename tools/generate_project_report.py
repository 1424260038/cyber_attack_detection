"""Generate the Markdown project report for CyberDD."""

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
    dataset_summary = load_json(PROJECT_DIR / "outputs" / "dataset_summary.json") or {}
    data_profile = load_json(PROJECT_DIR / "outputs" / "data_profile.json") or {}
    manifest = load_json(PROJECT_DIR / "artifacts" / "model_manifest.json") or {}
    preprocessor = TabularPreprocessor.load(PROJECT_DIR / "artifacts" / "preprocessor.json")
    manifest_model = manifest.get("model", {})
    label_distribution = dataset_summary.get("label_distribution", {})
    labels = "、".join(label_distribution.keys()) if isinstance(label_distribution, dict) else "N/A"

    report = f"""# CyberDD 项目报告

生成时间：{datetime.now().strftime("%Y-%m-%d")}

## 1. 项目概述

本项目全称为《打击新型网络犯罪中基于多模态深度学习驱动的网络攻击行为识别研究》，面向大学生创新训练项目和公安网络安全实战需求，研究如何利用多模态深度学习技术识别 APT 攻击、勒索软件、钓鱼欺诈、DDoS、恶意软件传播等新型网络犯罪行为。

项目当前已形成“演示数据 -> 特征预处理 -> 模型训练与评估 -> API 推理 -> Web 控制台 -> 攻击链解释 -> 报告与发布包”的完整工程闭环，可用于大创阶段性展示、结题答辩和后续真实数据接入扩展。

## 2. 立项背景与意义

新型网络犯罪具有隐蔽性强、变异快、跨协议层联动和攻击链长等特点。传统基于规则、签名或单一数据源的检测方法难以全面捕捉攻击行为的复杂性，容易产生误报和漏报，也难以还原攻击过程。

多模态深度学习能够融合网络流量、协议交互、日志行为、时序模式和攻击知识，从多维度提取攻击特征，提升未知攻击识别能力。对公安机关而言，该技术可辅助提升网络犯罪发现、研判、溯源和处置效率，为维护网络空间安全和社会稳定提供技术支撑。

## 3. 研究内容

### 3.1 多模态网络流量识别

项目提取 IP、端口、协议、包量比、持续时间、字节熵、TLS 握手参数、载荷分片模式等特征，构建加密流量和协议混淆场景下的网络行为表示，用于识别正常流量与攻击流量。

### 3.2 多模态网络攻击识别

项目通过时序建模和图结构关联，分析攻击行为在时间维度和网络结构中的传播过程，覆盖 DDoS、Probe、R2L、U2R、Malware、Phishing、APT 等攻击类别。

### 3.3 攻击链解释

项目基于 MITRE ATT&CK 框架构建攻击知识图谱，将检测结果映射到攻击战术、技术、风险等级、处置建议和攻击链阶段，增强模型输出的可解释性。

## 4. 技术路线

| 阶段 | 内容 |
| --- | --- |
| 数据准备 | 生成或接入网络流量 CSV，完成清洗、标准化、数据摘要和质量画像 |
| 特征工程 | 提取时序特征、统计特征、会话交互特征和攻击行为标签 |
| 模型训练 | 构建 CNN-LSTM、GNN/GAT、多模态融合等模型结构，训练攻击识别模型 |
| 模型评估 | 计算 Accuracy、Precision、Recall、F1、AUC 等指标 |
| API 服务 | 提供单条检测、批量检测、CSV 上传、事件审计、工件下载等接口 |
| 前端演示 | 提供 Web 控制台展示检测结果、指标、事件、知识图谱和交付物 |
| 知识图谱 | 将攻击类别映射到 ATT&CK 技战术，实现攻击链解释 |

## 5. 模型与工件

| 项目 | 当前值 |
| --- | --- |
| 模型结构 | {model_info.get("architecture")} |
| 输入维度 | {model_info.get("input_dim")} |
| 分类类别数 | {model_info.get("num_classes")} |
| 最佳验证准确率 | {model_info.get("best_val_acc")} |
| 检查点 | `checkpoints/best_model.pth` |
| 预处理器 | `artifacts/preprocessor.json` |
| TorchScript | `artifacts/model.pt` |
| 模型清单 | `artifacts/model_manifest.json` |
| 模型 SHA256 | {str(manifest_model.get("checkpoint_sha256", "N/A"))[:16]}... |

说明：当前模型指标基于内置演示数据，用于验证工程闭环和答辩演示。真实环境应用前应接入公开数据集或实战脱敏数据重新训练和评估。

## 6. 数据集摘要

| 项目 | 当前值 |
| --- | --- |
| 数据来源 | `{dataset_summary.get("input", "N/A")}` |
| 标准化输出 | `{dataset_summary.get("output", "N/A")}` |
| 样本行数 | {dataset_summary.get("rows", "N/A")} |
| 特征列数 | {len(preprocessor.feature_columns)} |
| 覆盖类别 | {labels} |
| 数据质量分 | {data_profile.get("quality_score", "N/A")} |
| 缺失/非法值比例 | {data_profile.get("missing_rate", "N/A")} |
| 类别失衡比 | {data_profile.get("imbalance_ratio", "N/A")} |

## 7. 测试指标

| 指标 | 数值 |
| --- | --- |
| Accuracy | {metrics.get("accuracy", "N/A")} |
| Precision | {metrics.get("precision", "N/A")} |
| Recall | {metrics.get("recall", "N/A")} |
| F1-Score | {metrics.get("f1_score", "N/A")} |
| AUC-ROC | {metrics.get("auc_roc", "N/A")} |
| Average Precision | {metrics.get("average_precision", "N/A")} |

## 8. 系统能力

- 单条网络流量特征检测。
- JSON 批量检测。
- CSV 文本检测与文件上传检测。
- 检测结果导出。
- 演示流量回放。
- 运行时事件记录、统计、清空和导出。
- 数据集摘要与质量画像展示。
- ATT&CK 知识图谱摘要和攻击链解释。
- 模型、预处理器、报告、OpenAPI、验收材料和发布包下载。

## 9. 演示流程

1. 运行 `python tools/build_demo_system.py` 刷新演示数据、工件、报告和质量检查。
2. 运行 `python api.py` 启动后端服务。
3. 运行 `cd web && pnpm run dev` 启动前端控制台。
4. 打开 `http://localhost:5173`，依次演示正常样例、攻击样例、CSV 批量检测、攻击链解释、事件回放和交付物下载。
5. 使用 `outputs/demo_runbook.md` 作为现场答辩提纲。

## 10. 项目成果

| 成果 | 文件 |
| --- | --- |
| 训练模型 | `checkpoints/best_model.pth` |
| 部署模型 | `artifacts/model.pt` |
| 预处理器 | `artifacts/preprocessor.json` |
| 模型清单 | `artifacts/model_manifest.json` |
| 项目报告 | `outputs/project_report.md` |
| 答辩手册 | `outputs/demo_runbook.md` |
| OpenAPI 规范 | `outputs/openapi.json` |
| 验收清单 | `outputs/acceptance_checklist.md` |
| 完成度审计 | `outputs/completion_audit.md` |
| 发布包 | `release/cyberdd_release.zip` |

## 11. 后续扩展

- 接入 CICIDS2017、NSL-KDD 或公安实战脱敏数据。
- 将日志文本、流量可视化图像和会话图结构纳入统一多模态模型。
- 增加混淆矩阵、分类别召回率、误报分析和阈值校准。
- 扩展攻击知识图谱，实现更细粒度的攻击链溯源。
- 建立 GitHub Actions 自动测试、构建和发布流程。
"""

    output_path.write_text(report, encoding="utf-8")
    print(f"Generated report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
