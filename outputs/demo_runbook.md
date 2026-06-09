# CyberDD 答辩演示运行手册

## 1. 演示前刷新

```bash
python tools/build_demo_system.py
```

默认会刷新演示数据、标准化数据、数据质量画像、预处理器、TorchScript、模型清单、项目报告和完整质量检查。若需要重新训练模型：

```bash
python tools/build_demo_system.py --train --epochs 16
```

如需单独验证真实 HTTP 服务：

```bash
python tools/smoke_test_service.py
```

## 2. 启动系统

```bash
python api.py
```

浏览器访问：

```text
http://localhost:8000
```

API 文档：

```text
http://localhost:8000/docs
```

## 3. 现场演示顺序

1. 查看首页状态卡片：模型、输入维度、测试 F1、数据质量、检测事件。
2. 点击“加载正常样例”并执行检测，说明正常概率和处置建议。
3. 点击“加载攻击样例”并执行检测，说明攻击概率、风险等级、攻击链和 Top 特征贡献。
4. 点击“回放演示流量”，观察运行时事件和近期告警记录自动增长。
5. 在 CSV 批量检测中点击“生成示例 CSV”，执行批量检测并导出结果。
6. 展示数据集与知识图谱区域：样本数、类别分布、质量分、缺失率、类别失衡比。
7. 展示运行时检测统计与近期告警记录，并导出事件 CSV。
8. 展示“系统运维状态”和“交付物下载”，执行运行时重载并下载项目报告/答辩手册/发布包。
9. 打开 `outputs/project_report.md`、`outputs/openapi.json`、`outputs/acceptance_checklist.md` 和 `outputs/completion_audit.md` 作为系统验收佐证。

## 4. 发布包

```bash
python tools/package_release.py
```

验收证据生成：

```bash
python tools/export_openapi.py
python tools/generate_acceptance_checklist.py
python tools/audit_completion.py
```

发布包位置：

```text
release/cyberdd_release.zip
```

发布清单位置：

```text
release/release_manifest.json
```

## 5. 管理接口

```bash
curl http://localhost:8000/admin/runtime
curl -X POST http://localhost:8000/admin/reload
```

`/admin/reload` 可在替换模型或预处理器后重新加载运行时状态。设置 `CYBERDD_ADMIN_TOKEN` 后，请在请求头中携带 `X-Admin-Token`。

```bash
set CYBERDD_ADMIN_TOKEN=your-token
curl -H "X-Admin-Token: your-token" http://localhost:8000/admin/runtime
```

正式部署时建议放在内网或增加鉴权。
