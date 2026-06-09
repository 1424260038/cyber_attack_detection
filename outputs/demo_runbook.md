# CyberDD 答辩演示运行手册

## 1. 开场说明

本项目名称为《打击新型网络犯罪中基于多模态深度学习驱动的网络攻击行为识别研究》。项目针对 APT 攻击、勒索软件、钓鱼欺诈、加密流量攻击等新型网络犯罪隐蔽性强、变异快、攻击链长的问题，构建网络攻击行为识别与解释系统。

答辩时建议先说明三点：

- 为什么做：传统规则检测和单一数据源检测难以应对未知攻击和跨阶段攻击链。
- 做了什么：系统融合流量特征、时序行为、攻击知识图谱和 Web 可视化，形成可运行原型。
- 有什么价值：可辅助公安网络安全实战中的攻击发现、风险研判、攻击链解释和证据展示。

## 2. 演示前刷新

刷新演示数据、标准化数据、数据质量画像、预处理器、TorchScript、模型清单、项目报告和完整质量检查：

```bash
python tools/build_demo_system.py
```

如果需要重新训练演示模型：

```bash
python tools/build_demo_system.py --train --epochs 16
```

单独验证真实 HTTP 服务：

```bash
python tools/smoke_test_service.py
```

## 3. 启动系统

启动后端：

```bash
python api.py
```

启动前端：

```bash
cd web
pnpm run dev
```

浏览器访问：

```text
http://localhost:5173
```

API 文档：

```text
http://localhost:8000/docs
```

也可以直接运行：

```bat
run.cmd
```

建议先选择 `3` 启动 API，再开第二个窗口选择 `4` 启动 Web 前端。

## 4. 现场演示顺序

1. 展示首页系统概览：说明模型结构、输入维度、测试 F1、知识图谱、预处理工件、事件数量、数据样本和数据质量。
2. 点击“加载正常样例”并执行检测：说明系统能识别正常流量并给出处置建议。
3. 点击“加载攻击样例”并执行检测：说明攻击概率、风险等级、攻击类型、攻击链和关键特征贡献。
4. 展示攻击链解释：强调检测结果可映射到 MITRE ATT&CK，便于解释攻击者的策略、技术和流程。
5. 点击“生成示例 CSV”并执行批量检测：展示批量识别和结果导出能力。
6. 点击“回放演示流量”：观察运行时检测统计和近期告警记录增长。
7. 展示数据集与知识图谱区域：说明样本类别分布、数据质量、缺失率和知识图谱战术标签。
8. 展示“系统运维状态”和“交付物下载”：下载项目报告、答辩手册、OpenAPI、验收清单和完成度审计。

## 5. 答辩讲解要点

### 研究背景

新型网络犯罪已经从单点攻击转向多阶段、隐蔽化、自动化和跨协议层攻击。APT、勒索软件、钓鱼欺诈和加密流量攻击会造成个人隐私泄露、企业资产损失和社会安全风险。传统检测方法依赖规则或单模态数据，难以适应未知攻击模式。

### 技术路线

项目采用“数据预处理 -> 时空特征工程 -> 模型训练 -> 攻击检测 -> 知识图谱解释 -> Web 演示”的路线。当前系统使用 64 维流量特征完成攻击识别，并通过知识图谱将检测结果转化为攻击链解释。

### 创新点

- 攻击行为时空特征工程：融合会话图、流量统计、时序行为和加密流量相关特征。
- 多模态动态攻击识别：结合深度学习模型和攻击行为演化规律识别新型攻击。
- 攻击行为知识图谱：将结果映射到 ATT&CK 技战术框架，输出攻击链和处置建议。
- 工程化闭环：系统具备后端 API、Web 控制台、质量检查、报告生成和发布包。

### 当前限制

当前数据主要用于演示系统闭环，指标不代表真实网络环境泛化效果。下一步应接入 CICIDS2017、NSL-KDD 或公安实战脱敏数据，完善多模态数据集、分类别召回率、混淆矩阵和误报分析。

## 6. 发布包与验收材料

生成发布包：

```bash
python tools/package_release.py
```

生成验收证据：

```bash
python tools/export_openapi.py
python tools/generate_acceptance_checklist.py
python tools/audit_completion.py
```

关键文件：

| 文件 | 用途 |
| --- | --- |
| `outputs/project_report.md` | 项目报告 |
| `outputs/demo_runbook.md` | 答辩演示手册 |
| `outputs/openapi.json` | API 文档 |
| `outputs/acceptance_checklist.md` | 验收清单 |
| `outputs/completion_audit.md` | 完成度审计 |
| `release/cyberdd_release.zip` | 发布包 |

## 7. 管理接口

查看运行时状态：

```bash
curl http://localhost:8000/admin/runtime
```

重载模型、预处理器和知识图谱：

```bash
curl -X POST http://localhost:8000/admin/reload
```

正式部署时建议设置管理令牌：

```bash
set CYBERDD_ADMIN_TOKEN=your-token
curl -H "X-Admin-Token: your-token" http://localhost:8000/admin/runtime
```
