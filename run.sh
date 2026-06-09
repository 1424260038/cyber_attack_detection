#!/bin/bash
# -*- coding: utf-8 -*-
"""
CyberDD 启动脚本
Cyber Attack Detection System - Launcher
"""

# 设置颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}  CyberDD 网络攻击检测系统${NC}"
echo -e "${GREEN}  Cyber Attack Detection System${NC}"
echo -e "${GREEN}=======================================${NC}"
echo ""
echo -e "${BLUE}请选择运行模式:${NC}"
echo "  [1] 训练模型 (CNN-LSTM)"
echo "  [2] 测试模型"
echo "  [3] 启动 API 服务"
echo "  [4] 评估知识图谱"
echo "  [5] 快速测试 Demo"
echo "  [0] 退出"
echo ""

read -p "请输入选项 (0-5): " choice

case $choice in
    1)
        echo ""
        echo -e "${YELLOW}开始训练模型...${NC}"
        echo "======================================="
        python3 main.py --mode train --model cnn_lstm --epochs 100 --device cpu
        echo ""
        echo -e "${GREEN}训练完成！模型保存在 checkpoints/ 目录${NC}"
        ;;
    2)
        echo ""
        echo -e "${YELLOW}开始测试模型...${NC}"
        echo "======================================="
        python3 main.py --mode test --model cnn_lstm --checkpoint checkpoints/best_model.pth --device cpu
        ;;
    3)
        echo ""
        echo -e "${YELLOW}启动 API 服务...${NC}"
        echo "======================================="
        echo -e "服务地址: ${GREEN}http://localhost:8000${NC}"
        echo -e "API 文档: ${GREEN}http://localhost:8000/docs${NC}"
        echo ""
        echo "按 Ctrl+C 停止服务"
        echo ""
        python3 api.py
        ;;
    4)
        echo ""
        echo -e "${YELLOW}评估知识图谱...${NC}"
        echo "======================================="
        python3 main.py --mode evaluate_kg --device cpu
        ;;
    5)
        echo ""
        echo -e "${YELLOW}运行快速测试...${NC}"
        echo "======================================="
        python3 test_demo.py
        ;;
    0)
        echo "退出"
        exit 0
        ;;
    *)
        echo -e "${RED}无效选项${NC}"
        exit 1
        ;;
esac

echo ""
read -p "按回车键退出..."
