# CyberDD 项目验收清单

生成时间：2026-06-09T13:49:57

验收结果：17/17 项通过

| 类别 | 验收项 | 文件 | 状态 | 大小 |
|------|--------|------|------|------|
| 数据准备 | 演示数据集 | `data/demo_traffic.csv` | 通过 | 564960 |
| 数据准备 | 标准化训练数据 | `data/prepared_traffic.csv` | 通过 | 564960 |
| 数据质量 | 数据集摘要 | `outputs/dataset_summary.json` | 通过 | 358 |
| 数据质量 | 数据质量画像 | `outputs/data_profile.json` | 通过 | 1043 |
| 模型工件 | 最佳模型检查点 | `checkpoints/best_model.pth` | 通过 | 615157 |
| 模型工件 | 预处理器 | `artifacts/preprocessor.json` | 通过 | 4010 |
| 模型工件 | TorchScript 部署模型 | `artifacts/model.pt` | 通过 | 215757 |
| 模型工件 | 模型清单 | `artifacts/model_manifest.json` | 通过 | 6040 |
| 评估结果 | 测试指标 | `outputs/test_results.json` | 通过 | 2186 |
| API 文档 | OpenAPI 接口规范 | `outputs/openapi.json` | 通过 | 38588 |
| 前端 | 前端生产构建 | `web/dist/index.html` | 通过 | 350 |
| 部署 | Docker 镜像定义 | `Dockerfile` | 通过 | 339 |
| 部署 | Docker Compose 编排 | `docker-compose.yml` | 通过 | 279 |
| 报告材料 | 项目报告 | `outputs/project_report.md` | 通过 | 5778 |
| 报告材料 | 答辩演示手册 | `outputs/demo_runbook.md` | 通过 | 4824 |
| 交付包 | 完整发布包 | `release/cyberdd_release.zip` | 通过 | 1417047 |
| 交付包 | 发布包清单 | `release/release_manifest.json` | 通过 | 2957 |
