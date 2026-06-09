# CyberDD 网络攻击行为识别系统

本项目对应大学生创新训练项目《打击新型网络犯罪中基于多模态深度学习驱动的网络攻击行为识别研究》。系统围绕新型网络犯罪场景，构建“数据采集与预处理 -> 多模态特征建模 -> 攻击行为识别 -> 攻击链解释 -> 可视化演示 -> 交付物生成”的工程闭环，为公安机关打击 APT 攻击、勒索软件、钓鱼欺诈等新型网络犯罪提供智能化技术支撑。

项目仓库：[https://github.com/1424260038/cyber_attack_detection](https://github.com/1424260038/cyber_attack_detection)

## 项目背景

随着互联网和数字化业务快速发展，APT 攻击、勒索软件、钓鱼欺诈、加密流量攻击和多阶段渗透行为不断增多。这类攻击往往具有隐蔽性、变异性和跨协议层联动特征，传统基于规则、签名或单一数据源的检测方法难以及时识别未知攻击，容易出现误报率高、漏报率高、攻击链无法还原等问题。

从公安实战角度看，网络攻击行为识别不仅要判断“是否攻击”，还要回答“攻击属于哪类行为”“攻击链如何演化”“应该如何处置”。因此，本项目将多模态深度学习、网络流量特征工程、时序建模、知识图谱和 MITRE ATT&CK 技战术映射结合起来，形成面向新型网络犯罪打击场景的可解释检测原型。

## 研究目标

- 构建多模态深度学习驱动的网络攻击行为识别模型，提升对未知攻击、加密流量和复杂攻击行为的识别能力。
- 融合网络流量、协议交互、时序行为和攻击知识，突破传统单点检测无法关联跨协议层攻击行为的问题。
- 建立攻击行为知识图谱，将模型检测结果映射到 MITRE ATT&CK 框架，实现攻击链还原、威胁解释和处置建议输出。
- 形成可运行、可演示、可验收的工程系统，支撑大创答辩、课程展示和后续科研扩展。

## 研究内容

### 多模态网络流量识别

项目面向网络流量元数据、协议交互特征、载荷统计特征和时序行为特征进行融合建模。通过提取 IP、端口、协议、包量比、持续时间、字节熵、TLS 握手参数、载荷分片模式等特征，构建抗扰动的网络行为表示，用于识别 Tor/VPN、加密流量和协议混淆场景中的异常行为。

### 多模态网络攻击识别

项目关注 APT、勒索软件、DDoS、钓鱼欺诈、恶意软件传播等攻击行为的分类识别。通过时序建模和图结构关联，追踪攻击行为随时间变化的趋势以及在网络结构中的传播路径，解决传统检测方法难以关联多阶段攻击的问题。

### 攻击链解释与知识图谱

项目构建攻击行为本体模型，将检测结果映射到 MITRE ATT&CK 技战术框架，输出攻击链阶段、攻击类型、风险等级、关键特征贡献和处置建议。该能力用于支撑 C&C 命令与控制、恶意软件传播路径、漏洞利用到数据加密等攻击过程的可视化解释。

## 技术路线

1. 数据采集与特征工程：整合网络流量、系统日志、行为数据等多源信息，完成清洗、归一化、特征提取和统一表征。
2. 时空特征建模：采用滑动窗口机制提取流量强度时序特征，结合会话图特征和交互模式建立网络行为基线。
3. 多模态模型构建：使用 CNN-LSTM、GNN/GAT、多模态融合等模型结构处理时序特征、空间特征和图结构特征。
4. 模型训练与评估：通过训练集、验证集和测试集评估准确率、精确率、召回率、F1、AUC 等指标，并进行可靠性测试。
5. 知识图谱应用：基于 MITRE ATT&CK 构建攻击知识图谱，实现攻击链还原、威胁溯源和处置建议生成。
6. 系统集成交付：提供 FastAPI 后端、React 前端、Docker 部署、OpenAPI 文档、验收清单和发布包。

## 创新点

- 攻击行为时空特征工程：将 IP、端口、协议、双向包量比、持续时间、字节熵、TLS 参数和载荷分片模式等特征统一建模，形成抗扰动的加密流量特征空间。
- 多模态动态攻击识别：结合图神经网络与 LSTM 时序建模，捕捉攻击行为在时间维度和网络结构维度上的演化模式，提高对未知攻击的检测能力。
- 攻击行为知识图谱：建立网络攻击本体模型，将模型输出映射到 MITRE ATT&CK 框架，实现攻击技战术解释、攻击链路径还原和处置建议输出。
- 工程化演示闭环：不仅提供算法代码，还提供可运行 Web 控制台、API 服务、演示数据、质量检查、答辩手册和发布包，便于大创验收展示。

## 当前实现

| 模块 | 当前状态 |
| --- | --- |
| 演示数据 | `data/demo_traffic.csv`，960 行，64 维特征，覆盖 8 类流量 |
| 模型工件 | `checkpoints/best_model.pth`、`artifacts/model.pt`、`artifacts/preprocessor.json` |
| 后端服务 | FastAPI，支持单条检测、批量检测、CSV 上传、事件审计和工件下载 |
| 前端控制台 | React + Vite，支持检测演示、指标展示、知识图谱摘要、事件回放和交付物下载 |
| 知识图谱 | 支持攻击类型、ATT&CK 战术技术、攻击链解释和处置建议 |
| 质量检查 | 单元测试、系统自检、HTTP 冒烟测试、TypeScript、ESLint、Vite build |
| 验收材料 | 项目报告、答辩手册、OpenAPI、验收清单、完成度审计 |

说明：当前指标和演示效果基于项目内置确定性演示数据，适合大创答辩和系统闭环展示。若用于真实环境，应接入 CICIDS2017、NSL-KDD 或公安实战脱敏流量重新训练、验证和评估。

## 项目结构

```text
cyber_attack_detection/
├── api.py                  # FastAPI 后端服务
├── main.py                 # 训练、测试、知识图谱评估入口
├── 启动器.cmd              # Windows 双击启动入口，调用 run.cmd
├── run.cmd                 # Windows 一键启动菜单
├── system_check.py         # 系统自检入口
├── configs/                # 配置文件
├── data/                   # 数据加载、预处理、演示数据
├── models/                 # CNN-LSTM、GAT、多模态融合等模型代码
├── training/               # 训练器与数据增强
├── evaluation/             # 模型评估
├── kg/                     # 攻击知识图谱与 ATT&CK 映射
├── utils/                  # 配置、日志、事件存储、模型加载
├── tools/                  # 数据生成、报告、打包、质量检查工具
├── tests/                  # 单元测试
├── web/                    # React 前端控制台
├── artifacts/              # 预处理器、TorchScript、模型清单
├── checkpoints/            # 最佳模型 checkpoint
└── outputs/                # 自动生成的报告、OpenAPI、验收清单、审计结果
```

## 文档说明

仓库主文档只维护 `README.md`。`outputs/project_report.md`、`outputs/demo_runbook.md`、`outputs/acceptance_checklist.md` 和 `outputs/completion_audit.md` 是脚本自动生成的答辩/验收交付物，供 WebUI 下载和发布包打包使用，不再当作重复的人工说明文档维护。前端模板自带的 `web/README.md` 已删除，避免文档入口混乱。

## 快速运行

### Windows 菜单方式

```bat
启动器.cmd
```

也可以直接运行主菜单脚本：

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

答辩演示建议：先打开一个终端选择 `3` 启动后端，再打开另一个终端选择 `4` 启动前端。

### 命令行方式

安装依赖：

```bash
pip install -r requirements.txt
cd web
pnpm install
cd ..
```

刷新演示系统：

```bash
python tools/build_demo_system.py
```

启动后端：

```bash
python api.py
```

启动前端：

```bash
cd web
pnpm run dev
```

访问地址：

- 前端控制台：[http://localhost:5173](http://localhost:5173)
- 后端 API：[http://localhost:8000](http://localhost:8000)
- Swagger 文档：[http://localhost:8000/docs](http://localhost:8000/docs)

## 演示流程

1. 系统状态展示：查看模型、预处理器、知识图谱和数据工件是否加载。
2. 样本检测演示：加载正常样例和攻击样例，展示预测类别、置信度、风险等级和处置建议。
3. 批量检测演示：粘贴 CSV 或上传 CSV 文件，查看批量识别结果并导出。
4. 攻击链解释：展示 ATT&CK 技战术映射、攻击链阶段和关键特征贡献。
5. 事件审计：回放演示流量，查看近期告警、风险统计和事件导出。
6. 交付物下载：下载项目报告、答辩手册、OpenAPI、验收清单和完成度审计。

## 核心命令

训练模型：

```bash
python main.py --mode train --model cnn_lstm --data ./data --epochs 100 --device cpu --output_dir checkpoints
```

测试模型：

```bash
python main.py --mode test --model cnn_lstm --checkpoint checkpoints/best_model.pth --device cpu --output_dir outputs
```

导出部署模型：

```bash
python tools/export_model.py --checkpoint checkpoints/best_model.pth --output artifacts/model.pt
```

运行完整质量检查：

```bash
python tools/run_all_checks.py
```

跳过前端检查：

```bash
python tools/run_all_checks.py --skip-frontend
```

## 真实数据接入

接入真实网络流量 CSV 后，可按以下流程重新生成数据、训练模型和交付物：

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

真实数据实验应重点补充标签字段说明、训练/验证/测试划分、类别分布、缺失值处理、数据泄漏检查、混淆矩阵和分类别召回率分析。

## 交付物

| 文件 | 说明 |
| --- | --- |
| `outputs/project_report.md` | 自动生成的项目报告 |
| `outputs/demo_runbook.md` | 自动生成的答辩演示手册 |
| `outputs/openapi.json` | OpenAPI 接口规范 |
| `outputs/acceptance_checklist.md` | 自动生成的验收清单 |
| `outputs/completion_audit.md` | 自动生成的完成度审计 |
| `artifacts/model_manifest.json` | 模型、数据和评估清单 |
| `release/cyberdd_release.zip` | 发布包，运行打包脚本后生成 |

## 项目成员与分工

| 成员 | 分工 |
| --- | --- |
| 董益辰 | 算法研究、模型搭建 |
| 王卓 | 实验测试 |
| 张婧晗 | 数据收集 |
| 赵展浩 | 数据收集 |
| 周宇杰 | 论文撰写 |
| 张志强 | 指导教师 |

## 研究进度安排

| 阶段 | 内容 |
| --- | --- |
| 第 1-2 周 | 组建团队、文献调研、明确研究目标和开题方案 |
| 第 3-6 周 | 多源数据收集、数据清洗、归一化、样本标注和多模态对齐 |
| 第 7-12 周 | 多模态深度学习模型设计、训练和优化 |
| 第 13-14 周 | 模型性能评估、可靠性测试和复杂场景验证 |
| 第 15-16 周 | 整理代码、模型、实验结果、技术文档和论文初稿 |
| 第 17-18 周 | 准备验收材料、答辩 PPT、模拟答辩和最终优化 |

## 后续优化方向

- 接入真实公开数据集或公安实战脱敏数据，建立更可信的泛化评估。
- 加强多模态融合能力，将日志文本、流量图像和会话图特征纳入统一模型。
- 增加混淆矩阵、误报分析、阈值校准和可解释性报告。
- 扩展攻击知识图谱，使攻击链路径还原和威胁溯源更细粒度。
- 增加 CI/CD，在 GitHub Actions 中自动运行测试、前端构建和发布包生成。

## 维护信息

- 项目维护者：Serendipity
- 项目负责人：董益辰
- 指导教师：张志强
- 仓库地址：[https://github.com/1424260038/cyber_attack_detection](https://github.com/1424260038/cyber_attack_detection)
