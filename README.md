# CyberDD 网络攻击检测系统

CyberDD 是一个面向大学生创新创业训练计划的网络攻击检测与演示系统。项目围绕“网络流量数据 -> 特征预处理 -> 模型训练与评估 -> API 推理 -> Web 可视化 -> 攻击链解释 -> 交付物生成”形成完整闭环，适合用于课程展示、大创结题答辩和后续科研原型扩展。

项目仓库：[https://github.com/1424260038/cyber_attack_detection](https://github.com/1424260038/cyber_attack_detection)

## 项目亮点

- 端到端闭环：覆盖数据生成、数据质量画像、训练、测试、模型导出、后端服务、前端控制台和发布打包。
- 可演示性强：内置 960 条平衡演示流量样本，覆盖 8 类攻击/正常流量，支持单条检测、批量检测、CSV 上传、事件回放和结果导出。
- 模型可部署：提供 PyTorch checkpoint、预处理器配置、TorchScript 模型和模型清单，便于本地部署与复现实验。
- 可解释输出：检测结果包含风险等级、置信度、Top 特征贡献、ATT&CK 攻击链解释和处置建议。
- 工程化交付：提供 FastAPI 后端、React 前端、Docker/Compose、一键构建脚本、OpenAPI 文档、验收清单和发布包生成工具。

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 后端服务 | Python, FastAPI, Uvicorn, Pydantic |
| 模型训练 | PyTorch, NumPy, Pandas, scikit-learn |
| 前端控制台 | React, TypeScript, Vite, Tailwind CSS |
| 工程交付 | Docker, Docker Compose, Git, GitHub |
| 质量检查 | unittest, compileall, HTTP smoke test, TypeScript, ESLint, Vite build |

## 当前能力

| 项目 | 当前状态 |
| --- | --- |
| 演示数据 | `data/demo_traffic.csv`，960 行，64 维特征 |
| 分类类别 | normal, dos, probe, r2l, u2r, malware, phishing, apt |
| 最佳模型 | `checkpoints/best_model.pth` |
| 预处理器 | `artifacts/preprocessor.json` |
| TorchScript | `artifacts/model.pt` |
| 测试指标 | Accuracy / Precision / Recall / F1 均为 1.0（基于内置演示数据） |
| 验收结果 | `outputs/acceptance_checklist.md`，17/17 通过 |
| 完成度审计 | `outputs/completion_audit.md`，11/11 通过 |

说明：当前指标来自项目内置的确定性演示数据，适合展示系统闭环和工程能力。若用于真实网络环境，应接入 CICIDS2017、NSL-KDD 或自采流量重新训练和评估。

## 目录结构

```text
cyber_attack_detection/
├── api.py                       # FastAPI 后端服务
├── main.py                      # 模型训练、测试、知识图谱评估入口
├── run.cmd                      # Windows 一键菜单
├── system_check.py              # 系统自检
├── configs/                     # 默认配置
├── data/                        # 数据加载、预处理、演示数据
├── models/                      # 分类模型与多模态模型代码
├── training/                    # 训练器与数据增强
├── evaluation/                  # 评估器
├── kg/                          # 知识图谱与 ATT&CK 映射
├── utils/                       # 配置、日志、事件存储、模型加载
├── tools/                       # 数据生成、构建、导出、打包和质量检查工具
├── tests/                       # API 与核心功能单元测试
├── web/                         # React + Vite 前端控制台
├── artifacts/                   # 预处理器、TorchScript、模型清单
├── checkpoints/                 # 模型 checkpoint
└── outputs/                     # 报告、OpenAPI、验收清单、审计结果
```

## 快速运行

### 方式一：Windows 菜单启动

双击或在命令行运行：

```bat
run.cmd
```

常用选项：

| 选项 | 功能 |
| --- | --- |
| 3 | 启动 API 后端 |
| 4 | 启动 Web 前端 |
| 6 | 快速 Demo 测试 |
| 7 | 完整质量检查 |
| 8 | 一键刷新演示系统 |
| 11 | 生成项目报告 |
| 12 | 生成发布包 |

演示时建议先打开一个终端选择 `3` 启动后端，再打开另一个终端选择 `4` 启动前端。

### 方式二：命令行启动

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

刷新演示数据、模型工件、报告和验收材料：

```bash
python tools/build_demo_system.py
```

启动后端：

```bash
python api.py
```

后端地址：

- API 首页：[http://localhost:8000](http://localhost:8000)
- Swagger 文档：[http://localhost:8000/docs](http://localhost:8000/docs)

启动前端：

```bash
cd web
pnpm install
pnpm run dev
```

前端地址：[http://localhost:5173](http://localhost:5173)

## 核心 API

| 接口 | 功能 |
| --- | --- |
| `GET /health` | 服务、模型和预处理器状态 |
| `GET /metadata` | 模型结构、输入维度、类别和特征列 |
| `GET /manifest` | 模型、预处理器、指标和数据摘要清单 |
| `GET /demo-samples` | 获取正常/攻击演示样本 |
| `POST /demo/replay` | 回放演示流量并记录事件 |
| `POST /predict` | 单条特征检测 |
| `POST /predict/batch` | JSON 批量检测 |
| `POST /predict/csv` | CSV 文本检测 |
| `POST /predict/upload` | CSV 文件上传检测 |
| `GET /metrics` | 训练与测试指标 |
| `GET /dataset/summary` | 数据集摘要 |
| `GET /dataset/profile` | 数据质量画像 |
| `GET /knowledge-graph` | ATT&CK 知识图谱摘要 |
| `POST /explain` | 攻击链解释与处置建议 |
| `GET /events` | 近期检测事件 |
| `GET /events/export.csv` | 导出检测事件 |
| `GET /artifacts/report` | 下载项目报告 |
| `GET /artifacts/openapi.json` | 下载 OpenAPI 规范 |

管理员接口可通过环境变量启用令牌保护：

```bash
set CYBERDD_ADMIN_TOKEN=your-token
```

## 前端演示功能

- 系统运行状态：展示模型、预处理器、工件和服务状态。
- 单条检测：输入 64 维特征，输出预测类别、风险等级、置信度和解释。
- 批量检测：支持 JSON 批量检测和 CSV 文本检测。
- 文件上传：上传 CSV 文件并导出检测结果。
- 演示回放：自动回放内置流量样本，生成运行时告警事件。
- 事件审计：查看、统计、清空和导出近期检测事件。
- 数据画像：展示数据集规模、类别分布和数据质量评分。
- 知识图谱：展示攻击类型、战术技术映射和处置建议。
- 交付物下载：下载报告、OpenAPI、验收清单、完成度审计和发布包。

## 常用命令

训练模型：

```bash
python main.py --mode train --model cnn_lstm --data ./data --epochs 100 --device cpu --output_dir checkpoints
```

测试模型：

```bash
python main.py --mode test --model cnn_lstm --checkpoint checkpoints/best_model.pth --device cpu --output_dir outputs
```

导出 TorchScript：

```bash
python tools/export_model.py --checkpoint checkpoints/best_model.pth --output artifacts/model.pt
```

生成项目报告：

```bash
python tools/generate_model_manifest.py
python tools/generate_project_report.py
```

运行完整质量检查：

```bash
python tools/run_all_checks.py
```

如需跳过前端检查：

```bash
python tools/run_all_checks.py --skip-frontend
```

## 接入真实数据

将真实网络流量 CSV 放入项目目录后，可按以下流程重新生成数据、训练和交付物：

```bash
python tools/prepare_dataset.py --input path/to/raw.csv --output data/prepared_traffic.csv --summary outputs/dataset_summary.json
python tools/profile_dataset.py --input data/prepared_traffic.csv --output outputs/data_profile.json
python tools/fit_preprocessor.py --input data/prepared_traffic.csv --output artifacts/preprocessor.json --input-dim 64
python main.py --mode train --model cnn_lstm --data ./data --epochs 100 --device cpu --output_dir checkpoints
python main.py --mode test --model cnn_lstm --checkpoint checkpoints/best_model.pth --data ./data --device cpu --output_dir outputs
python tools/export_model.py --checkpoint checkpoints/best_model.pth --output artifacts/model.pt
python tools/generate_model_manifest.py
python tools/export_openapi.py
python tools/generate_acceptance_checklist.py
python tools/audit_completion.py
python tools/package_release.py
```

真实数据接入时建议补充：

- 明确每一列特征含义和标签字段。
- 划分训练集、验证集和测试集，避免数据泄漏。
- 记录类别分布、缺失值、异常值和重复样本。
- 使用混淆矩阵、ROC-AUC、PR-AUC 和分类别召回率评估模型。

## Docker 部署

构建并启动服务：

```bash
docker compose up --build
```

停止服务：

```bash
docker compose down
```

## 质量与交付物

项目已内置自动化检查脚本：

```bash
python tools/run_all_checks.py
```

检查内容包括：

- Python 编译检查
- 单元测试
- 系统自检
- HTTP 冒烟测试
- 前端 TypeScript 检查
- 前端 ESLint 检查
- Vite 生产构建

主要交付物：

| 文件 | 说明 |
| --- | --- |
| `outputs/project_report.md` | 项目报告 |
| `outputs/demo_runbook.md` | 答辩演示手册 |
| `outputs/openapi.json` | OpenAPI 接口规范 |
| `outputs/acceptance_checklist.md` | 验收清单 |
| `outputs/completion_audit.md` | 完成度审计 |
| `artifacts/model_manifest.json` | 模型与数据清单 |
| `release/cyberdd_release.zip` | 发布包，运行打包脚本后生成 |

## 后续优化方向

- 接入 CICIDS2017、NSL-KDD 或自采真实流量数据，建立更可信的泛化评估。
- 增加混淆矩阵、模型校准、阈值策略和误报分析。
- 将知识图谱扩展为可编辑的攻击链知识库。
- 增加用户登录、角色权限和操作审计。
- 增加 CI/CD 流水线，在 GitHub Actions 中自动运行测试和前端构建。

## 维护信息

- 项目维护者：Serendipity
- 邮箱：1424260038@qq.com
- 仓库地址：[https://github.com/1424260038/cyber_attack_detection](https://github.com/1424260038/cyber_attack_detection)
