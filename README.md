# 🛡️ CyberDD - 网络攻击检测系统

<div align="center">
**基于多模态深度学习的网络攻击行为识别系统**

---

## 📋 项目简介

### 背景

随着网络技术的快速发展，网络攻击手段日趋多样化和智能化。传统的基于规则或签名的检测方法已难以应对新型网络攻击。本项目实现了一个基于**多模态深度学习**的网络攻击检测系统，通过整合网络流量的时序特征、统计特征和图结构特征，实现对各类网络攻击行为的高效识别。

### 主要特性

- 🔬 **多模态特征融合** - 整合时序、统计和图特征
- 🧠 **多种深度学习模型** - GAT、CNN-LSTM、多模态融合
- 📊 **高精度检测** - 验证准确率可达 99%+
- 🔄 **知识图谱集成** - 基于 MITRE ATT&CK 框架
- 🚀 **API 服务** - RESTful 接口便于集成部署
- 🧪 **批量检测** - 支持单条特征、批量 JSON 和 CSV 文本检测
- 📤 **结果导出** - 支持导出批量检测结果和运行时事件日志
- 🧭 **攻击链解释** - 将检测结果映射到 MITRE ATT&CK 战术与技术
- 🔎 **特征贡献解释** - 返回触发判断的 Top 特征偏离度
- 🖥️ **可视化控制台** - 前端展示模型状态、评估指标、批量检测和知识图谱摘要
- 📱 **轻量化模型** - 支持知识蒸馏压缩

### 应用场景

- 企业网络安全监控
- 入侵检测系统 (IDS)
- 安全运营中心 (SOC)
- 威胁情报分析
- 科研教学实验

---

## 🏗️ 项目架构

```
cyber_attack_detection/
├── 📂 web/                      # 前端Web界面
│   ├── src/                    # React源代码
│   ├── dist/                   # 构建产物
│   ├── package.json            # 前端依赖
│   ├── vite.config.ts          # Vite配置
│   └── index.html              # 入口文件
│
├── 📂 data/                      # 数据处理模块
│   ├── data_loader.py           # 数据加载器
│   └── feature_extractor.py     # 特征提取器
│
├── 📂 models/                   # 深度学习模型
│   ├── cnn_lstm_model.py       # CNN-LSTM 混合模型
│   ├── gat_model.py            # 图注意力网络 (GAT)
│   ├── multimodal_fusion.py     # 多模态融合模块
│   └── lightweight_model.py    # 轻量化模型
│
├── 📂 training/                 # 训练模块
│   ├── trainer.py               # 训练器
│   └── data_augmentation.py    # 数据增强
│
├── 📂 evaluation/               # 评估模块
│   └── evaluator.py            # 模型评估
│
├── 📂 kg/                       # 知识图谱模块
│   └── knowledge_graph.py       # ATT&CK 知识图谱
│
├── 📂 utils/                    # 工具函数
│   ├── config.py                # 配置管理
│   └── logger.py                # 日志工具
│
├── 📂 configs/                  # 配置文件
│   └── default_config.yaml     # 默认配置
│
├── 📂 checkpoints/             # 模型权重
│   ├── best_model.pth          # 最佳模型
│   └── last_model.pth          # 最新模型
│
├── 📂 outputs/                 # 输出目录
│   └── training_history.json   # 训练历史
│
├── main.py                     # 主程序入口
├── api.py                      # API 服务
├── run.cmd                     # Windows启动菜单
├── test_demo.py                # 测试脚本
├── requirements.txt            # 依赖清单
└── README.md                   # 项目文档
```

---

## 🧩 技术栈

| 类别 | 技术 |
|------|------|
| 编程语言 | Python 3.8+ |
| 深度学习 | PyTorch 2.0+ |
| 数据处理 | NumPy, Pandas, Scikit-learn |
| Web 框架 | FastAPI, Uvicorn |
| 配置管理 | PyYAML |
| 可视化 | Matplotlib, Seaborn |
| 知识图谱 | NetworkX, Neo4j (可选) |

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-repo/cyber_attack_detection.git
cd cyber_attack_detection
```

### 2. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

如需 API 服务，额外安装：

```bash
pip install fastapi uvicorn pydantic
```

### 4. 准备数据

将数据集放入 `data/` 目录，支持格式：

- **CSV 文件** - 每一行一条网络流量记录
- **必需列**：`Label` 或 `label`（攻击标签）
- **特征列**：数值型特征

#### 数据格式示例

```csv
packet_count,byte_count,duration,packet_rate,byte_rate,mean_packet_size,std_packet_size,min_packet_size,max_packet_size,Label
120,15420,0.5,240,30840,128.5,45.2,64,512,normal
85,10240,0.3,283,34133,120.4,38.7,60,512,dos
...
```

项目内置了可复现实验数据生成脚本，用于答辩演示或无真实数据集时快速跑通全流程：

```bash
python tools/generate_demo_dataset.py --output data/demo_traffic.csv --samples-per-class 120 --input-dim 64
python tools/profile_dataset.py --input data/demo_traffic.csv --output outputs/data_profile.json
python tools/fit_preprocessor.py --input data/demo_traffic.csv --output artifacts/preprocessor.json --input-dim 64
```

接入真实公开数据集或自采 CSV 时，先运行标准化准备脚本。该脚本会自动识别常见标签列，兼容 CICIDS2017、NSL-KDD 常见攻击标签，并输出统一标签：

```bash
python tools/prepare_dataset.py --input 原始CSV文件或目录 --output data/prepared_traffic.csv --summary outputs/dataset_summary.json
python tools/profile_dataset.py --input data/prepared_traffic.csv --output outputs/data_profile.json
python tools/fit_preprocessor.py --input data/prepared_traffic.csv --output artifacts/preprocessor.json --input-dim 64
```

`outputs/data_profile.json` 会记录数据质量分、缺失/非法值比例、类别失衡比、低方差特征和质量预警，可直接用于中期检查、结题报告或答辩说明。

#### 支持的攻击类型

| 标签 | 类别 ID | 说明 |
|------|---------|------|
| normal / benign | 0 | 正常流量 |
| dos / ddos | 1 | 拒绝服务攻击 |
| probe / scan | 2 | 侦察扫描 |
| r2l | 3 | 远程到本地攻击 |
| u2r | 4 | 用户到根攻击 |
| malware / ransomware | 5 | 恶意软件 |
| phishing | 6 | 钓鱼攻击 |
| apt | 7 | 高级持续性威胁 |

### 5. 训练模型

```bash
# 使用 CNN-LSTM 模型（推荐）
python main.py --mode train --model cnn_lstm --data ./data --epochs 100 --output_dir checkpoints

# 使用 GAT 模型（图神经网络）
python main.py --mode train --model gat --epochs 100

# 使用多模态融合模型
python main.py --mode train --model multimodal --epochs 100

# 使用轻量化模型（知识蒸馏）
python main.py --mode train --model lightweight --distill --checkpoint checkpoints/best_model.pth
```

### 6. 测试模型

```bash
python main.py --mode test --model cnn_lstm --checkpoint checkpoints/best_model.pth
```

导出可部署模型：

```bash
python tools/export_model.py --checkpoint checkpoints/best_model.pth --output artifacts/model.pt
python tools/generate_model_manifest.py
```

### 7. 启动 API 服务

```bash
python api.py
```

服务启动后访问：http://localhost:8000/docs

> ⚠️ **注意**：默认绑定 `0.0.0.0`（可局域网访问）。如只需本机访问，可改为 `127.0.0.1`

如果已经执行过 `cd web && pnpm exec vite build`，后端会直接托管 `web/dist`，浏览器打开 `http://localhost:8000` 即可访问前端控制台；开发阶段仍可单独运行 Vite。

常用接口：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 服务与模型加载状态 |
| `/metadata` | GET | 当前模型结构、输入维度、类别数 |
| `/manifest` | GET | 模型、预处理器、数据集和指标清单 |
| `/demo-samples` | GET | 正常/攻击演示样例 |
| `/demo/replay` | POST | 回放演示流量并写入运行时事件 |
| `/predict` | POST | 单条特征检测 |
| `/predict/batch` | POST | JSON 批量检测 |
| `/predict/csv` | POST | CSV 文本批量检测 |
| `/predict/csv/export` | POST | CSV 文本批量检测并导出结果 |
| `/predict/upload` | POST | CSV 文件上传检测 |
| `/predict/upload/export` | POST | CSV 文件上传检测并导出结果 |
| `/metrics` | GET | 读取训练/测试指标 |
| `/dataset/summary` | GET | 读取数据集来源、样本数、特征列和标签分布 |
| `/dataset/profile` | GET | 读取数据质量画像和质量预警 |
| `/knowledge-graph` | GET | 知识图谱实体、关系和战术摘要 |
| `/explain` | POST | 根据攻击类型生成 ATT&CK 攻击链解释 |
| `/events` | GET | 查询近期检测事件 |
| `/events/summary` | GET | 查询运行时检测统计 |
| `/events/export.csv` | GET | 导出近期检测事件 CSV |
| `/events` | DELETE | 清空检测事件日志 |
| `/admin/runtime` | GET | 查看运行时模型、预处理器和工件状态 |
| `/admin/reload` | POST | 重新加载模型、预处理器和知识图谱 |
| `/artifacts/report` | GET | 下载项目报告 |
| `/artifacts/runbook` | GET | 下载答辩演示手册 |
| `/artifacts/manifest.json` | GET | 下载模型清单 |
| `/artifacts/data-profile.json` | GET | 下载数据质量画像 |
| `/artifacts/openapi.json` | GET | 下载 OpenAPI 接口规范 |
| `/artifacts/acceptance-checklist` | GET | 下载 Markdown 验收清单 |
| `/artifacts/acceptance-checklist.json` | GET | 下载 JSON 验收清单 |
| `/artifacts/completion-audit` | GET | 下载系统完成度审计 |
| `/artifacts/completion-audit.json` | GET | 下载系统完成度审计 JSON |
| `/artifacts/release.zip` | GET | 下载完整项目发布包 |
| `/artifacts/release-manifest.json` | GET | 下载发布包清单 |

> `/admin/*` 接口用于本地演示和内网部署。设置环境变量 `CYBERDD_ADMIN_TOKEN` 后，需要在请求头中提供 `X-Admin-Token`；正式上线时还应限制访问来源。

### 8. 启动 Web 前端

```bash
cd web
npm install
npm run dev
```

启动后访问：http://localhost:5173

> ⚠️ **提示**：使用前端前请确保 API 服务已启动

### 9. 使用 Windows 启动菜单

双击 `run.cmd`，选择对应选项即可启动：
- 选项 3：启动 API 服务
- 选项 4：启动 Web 前端
- 选项 7：完整质量检查
- 选项 8：一键构建/刷新演示系统
- 选项 9：生成演示数据与预处理工件
- 选项 10：导出 TorchScript 部署模型
- 选项 11：生成项目报告 `outputs/project_report.md`
- 选项 12：生成项目发布包 `release/cyberdd_release.zip`

也可以直接运行：

```bash
python tools/build_demo_system.py
python tools/export_openapi.py
python tools/generate_acceptance_checklist.py
python tools/audit_completion.py
python tools/package_release.py
python tools/run_all_checks.py
python tools/smoke_test_service.py
```

完整质量检查会验证 Python 编译、单元测试、模型、预处理器、TorchScript、API 路由、CSV 上传、演示流量回放、事件审计、数据集摘要、真实 HTTP 服务冒烟测试、前端 TypeScript、ESLint 和前端构建。只想快速检查后端时可运行 `python system_check.py`。

答辩演示步骤见 `outputs/demo_runbook.md`。

---

## 📖 详细使用指南

### 命令行参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | - | train | 运行模式：train/test/serve/export/evaluate_kg |
| `--model` | - | cnn_lstm | 模型类型：gat/cnn_lstm/multimodal/lightweight |
| `--config` | - | configs/default_config.yaml | 配置文件路径 |
| `--checkpoint` | - | None | 模型检查点路径 |
| `--data` | - | ./data/CICIDS2017 | 数据集路径 |
| `--device` | - | cpu | 计算设备：cuda/cpu |
| `--epochs` | - | 100 | 训练轮数 |
| `--batch_size` | - | 64 | 批次大小 |
| `--lr` | - | 0.001 | 学习率 |
| `--output_dir` | - | ./outputs | 输出目录 |
| `--log_level` | - | INFO | 日志级别 |
| `--distill` | - | False | 启用知识蒸馏 |
| `--kg_eval` | - | False | 评估知识图谱 |

### Python API 使用

#### 基本训练流程

```python
import torch
from data.data_loader import create_dataloaders
from models.cnn_lstm_model import CNNLSTMClassifier
from training.trainer import Trainer
from evaluation.evaluator import Evaluator

# 1. 创建数据加载器
train_loader, val_loader, test_loader = create_dataloaders(
    data_path='./data',
    batch_size=64,
    num_workers=4
)

# 2. 创建模型
model = CNNLSTMClassifier(
    input_dim=64,              # 输入特征维度
    cnn_hidden_channels=[128, 256, 128],  # CNN 隐藏层
    lstm_hidden_size=256,      # LSTM 隐藏层维度
    lstm_num_layers=2,         # LSTM 层数
    num_classes=8,            # 分类类别数
    dropout=0.3               # Dropout 比率
)

# 3. 创建训练器
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda',             # 或 'cpu'
    checkpoint_dir='./checkpoints'
)

# 4. 训练模型
history = trainer.train(epochs=100)

# 5. 加载最佳模型并评估
trainer.load_checkpoint('checkpoints/best_model.pth')
evaluator = Evaluator(model, device='cuda')
metrics = evaluator.evaluate(test_loader, num_classes=8)
evaluator.print_results(metrics)
```

#### 加载已有模型进行推理

```python
import torch
from models.cnn_lstm_model import CNNLSTMClassifier

# 创建模型架构
model = CNNLSTMClassifier(
    input_dim=64,
    num_classes=8
)

# 加载预训练权重
checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

# 推理
with torch.no_grad():
    # 输入: [batch_size, features_dim]
    input_tensor = torch.randn(1, 64)
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()
    probabilities = torch.softmax(output, dim=1)

print(f"预测类别: {prediction}")
print(f"各类别概率: {probabilities}")
```

#### 使用 API 服务

```python
import requests

# 查看当前加载模型需要的输入维度
metadata = requests.get("http://localhost:8000/metadata").json()
input_dim = metadata["input_dim"]

# 单条预测
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "features": [0.1] * input_dim
    }
)
result = response.json()
print(result)
# 输出: {"prediction": "Normal", "confidence": 0.98, "probabilities": {"Normal": 0.98, "Attack": 0.02}}

# 批量预测
response = requests.post(
    "http://localhost:8000/predict/batch",
    json=[
        {"features": [0.1] * input_dim},
        {"features": [0.2] * input_dim}
    ]
)
print(response.json())
```

当前 API 会自动识别 `checkpoints/best_model.pth` 或 `outputs/best_model.pth` 的模型结构和输入维度。前端演示页也会读取 `/metadata`，因此不要再手动固定为 78 维；以接口返回的 `input_dim` 为准。

---

## 🔧 配置说明

配置文件：`configs/default_config.yaml`

### 数据配置

```yaml
data:
  dataset_path: "./data/CICIDS2017"
  train_ratio: 0.7      # 训练集比例
  val_ratio: 0.15       # 验证集比例
  test_ratio: 0.15      # 测试集比例
  
  features:
    temporal:           # 时序特征
      - packet_count
      - byte_count
      - duration
      - packet_rate
      - byte_rate
    statistical:       # 统计特征
      - mean_packet_size
      - std_packet_size
      - min_packet_size
      - max_packet_size
    graph:             # 图特征
      - degree
      - centrality
      - clustering_coeff
      - pagerank
  
  augmentation:
    enabled: true
    methods:
      - noise_injection
      - random_masking
      - mixup
    augmentation_ratio: 0.3
```

### 模型配置

```yaml
model:
  # CNN-LSTM 配置
  cnn_lstm:
    cnn:
      in_channels: 64
      hidden_channels: [128, 256, 128]
      kernel_sizes: [3, 5, 7]
      dropout: 0.3
    lstm:
      input_size: 64
      hidden_size: 256
      num_layers: 2
      dropout: 0.3
      bidirectional: true
  
  # GAT 配置
  gat:
    in_channels: 128
    hidden_channels: 256
    out_channels: 128
    num_heads: 8
    dropout: 0.3
    layers: 3
  
  # 融合配置
  fusion:
    method: "attention"
    hidden_dim: 512
    output_dim: 256
```

### 训练配置

```yaml
training:
  optimizer:
    type: "AdamW"
    lr: 0.001
    weight_decay: 0.0001
    betas: [0.9, 0.999]
  
  scheduler:
    type: "CosineAnnealingWarmRestarts"
    T_0: 10
    T_mult: 2
    eta_min: 0.00001
  
  epochs: 100
  batch_size: 64
  gradient_clip: 1.0
  
  early_stopping:
    patience: 15
    min_delta: 0.001
  
  device: "cpu"
  num_workers: 4
```

---

## 📊 模型说明

### CNN-LSTM 模型

卷积神经网络与长短期记忆网络的混合架构，适合处理时序型网络流量数据。

**结构特点：**
- 多尺度卷积核（3, 5, 7）提取不同粒度的特征
- 双向 LSTM 捕捉时序依赖关系
- 注意力机制聚焦关键特征

**适用场景：** 大多数网络流量检测任务

### GAT 模型

图注意力网络（Graph Attention Network），适合处理具有拓扑结构的网络数据。

**结构特点：**
- 自注意力机制学习节点重要性
- 多头注意力增强表达能力
- 适合处理网络拓扑关系

**适用场景：** 需要考虑网络拓扑的检测任务

### 多模态融合模型

整合多种特征模态的融合模型。

**结构特点：**
- 独立编码器处理不同模态
- 注意力机制融合特征
- 跨模态对比学习

**适用场景：** 复杂网络环境下的综合检测

### 轻量化模型

通过知识蒸馏压缩的轻量化模型。

**结构特点：**
- 知识蒸馏压缩
- 模型剪枝
- 量化推理

**适用场景：** 边缘设备部署、实时检测

---

## 🐳 部署指南

### 本地部署

```bash
# 开发环境
python api.py

# 生产环境（推荐使用 gunicorn）
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app --bind 0.0.0.0:8000
```

### Docker 部署

项目已提供 `Dockerfile` 和 `docker-compose.yml`。

```bash
# 构建镜像
docker build -t cyberdd:latest .

# 运行容器
docker run -d -p 8000:8000 --name cyberdd cyberdd:latest
```

### Docker Compose 部署

```bash
docker-compose up -d
```

### 生成项目报告

```bash
python tools/generate_project_report.py
```

报告输出到 `outputs/project_report.md`，可作为答辩或中期检查材料附录。

### Nginx 反向代理

```nginx
# /etc/nginx/conf.d/cyberdd.conf
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 超时设置
        proxy_connect_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

---

## 🧪 评估指标

系统支持以下评估指标：

| 指标 | 说明 |
|------|------|
| Accuracy | 准确率 |
| Precision | 精确率 |
| Recall | 召回率 |
| F1-Score | F1 分数 |
| AUC-ROC | ROC 曲线下面积 |
| AP | 平均精度 |

评估输出包括：
- 混淆矩阵
- 分类报告
- ROC 曲线
- PR 曲线

---

## 📦 项目文件说明

### 核心模块

| 文件 | 说明 |
|------|------|
| `main.py` | 主程序入口，支持训练/测试/评估 |
| `api.py` | RESTful API 服务 |
| `test_demo.py` | 快速测试脚本 |
| `requirements.txt` | Python 依赖 |

### 数据模块

| 文件 | 说明 |
|------|------|
| `data/data_loader.py` | 数据集加载与预处理 |
| `data/feature_extractor.py` | 特征提取器 |

### 模型模块

| 文件 | 说明 |
|------|------|
| `models/cnn_lstm_model.py` | CNN-LSTM 模型 |
| `models/gat_model.py` | GAT 图神经网络 |
| `models/multimodal_fusion.py` | 多模态融合 |
| `models/lightweight_model.py` | 轻量化模型 |

### 训练模块

| 文件 | 说明 |
|------|------|
| `training/trainer.py` | 模型训练器 |
| `training/data_augmentation.py` | 数据增强 |

### 评估模块

| 文件 | 说明 |
|------|------|
| `evaluation/evaluator.py` | 模型评估 |

### 知识图谱模块

| 文件 | 说明 |
|------|------|
| `kg/knowledge_graph.py` | ATT&CK 知识图谱 |

---

## ⚠️ 常见问题

### Q1: 模型加载失败？

**A:** 确保：
1. `checkpoints/best_model.pth` 文件存在
2. Python 版本 >= 3.8
3. PyTorch 版本 >= 2.0

### Q2: 内存不足？

**A:** 
- 减小 `batch_size`
- 使用 CPU 模式：`--device cpu`
- 减少数据增强比例

### Q3: 输入维度不匹配？

**A:** 检查特征维度，确保与训练时一致。默认 64 维，可修改 `configs/default_config.yaml` 或模型初始化参数。

### Q4: 如何提高准确率？

**A:**
1. 增加训练轮数
2. 调整学习率
3. 使用多模态模型
4. 增加数据增强
5. 使用更大的验证集

### Q5: 如何导出模型？

```python
# 导出为 TorchScript
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")

# 导出为 ONNX
torch.onnx.export(model, torch.randn(1, 64), "model.onnx")
```



---

## 📧 联系方式

- 项目维护者：[Serendipity]
- 邮箱：[1424260038@qq.com]
- 项目主页：https://github.com/your-repo/cyber_attack_detection

---

## 🙏 致谢

- [CICIDS2017](https://www.unb.ca/cic/research/datasets/cicids2017.html) 数据集
- [MITRE ATT&CK](https://attack.mitre.org/) 框架
- 开源社区贡献者
