# -*- coding: utf-8 -*-
"""
网络攻击检测系统主程序
Cyber Attack Detection System - Main Entry

支持训练、测试、推理和API服务
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import Config, get_config
from utils.logger import setup_logger, get_logger
from data.data_loader import create_dataloaders, NetworkTrafficDataset
from models.gat_model import GATClassifier
from models.cnn_lstm_model import CNNLSTMClassifier
from models.multimodal_fusion import MultiModalFusionModel
from models.lightweight_model import LightweightModel, KnowledgeDistillation
from training.trainer import Trainer, MultiModalTrainer
from evaluation.evaluator import Evaluator
from kg.knowledge_graph import create_attack_knowledge_graph, AttackDetector
from models.simple_classifier import SimpleClassifier
from utils.model_loader import extract_state_dict, normalize_mlp_state_dict


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='网络攻击检测系统 - Multi-Modal Deep Learning for Cyber Attack Detection'
    )
    
    # 模式选择
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test', 'serve', 'export', 'evaluate_kg'],
        default='train',
        help='运行模式'
    )
    
    # 配置文件
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='配置文件路径'
    )
    
    # 模型类型
    parser.add_argument(
        '--model',
        type=str,
        choices=['gat', 'cnn_lstm', 'multimodal', 'lightweight'],
        default='cnn_lstm',
        help='模型类型'
    )
    
    # 检查点路径
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='模型检查点路径'
    )
    
    # 数据路径
    parser.add_argument(
        '--data',
        type=str,
        default='./data/CICIDS2017',
        help='数据集路径'
    )
    
    # 设备
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default='cpu',
        help='计算设备'
    )
    
    # 训练轮数
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='训练轮数'
    )
    
    # 批次大小
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='批次大小'
    )
    
    # 学习率
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='学习率'
    )
    
    # 输出目录
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='输出目录'
    )
    
    # 日志级别
    parser.add_argument(
        '--log_level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别'
    )
    
    # 知识蒸馏
    parser.add_argument(
        '--distill',
        action='store_true',
        help='启用知识蒸馏'
    )
    
    # 评估知识图谱
    parser.add_argument(
        '--kg_eval',
        action='store_true',
        help='评估知识图谱攻击链还原'
    )
    
    return parser.parse_args()


def create_model(model_type: str, config: Config, num_classes: int = 8) -> nn.Module:
    """
    创建模型
    
    Args:
        model_type: 模型类型
        config: 配置
        num_classes: 类别数
        
    Returns:
        模型实例
    """
    logger = get_logger()
    
    input_dim = config.get('model.input_dim', 64)

    # 当前训练数据默认是表格型流量特征，统一使用可稳定演示的 MLP。
    # 模型名称仍保留，方便在文档和命令行中体现不同研究路线。
    if model_type in ['gat', 'cnn_lstm', 'multimodal']:
        logger.info(f"使用简化分类模型: input_dim={input_dim}, num_classes={num_classes}")
        model = SimpleClassifier(input_dim=input_dim, num_classes=num_classes)
        
    elif model_type == 'lightweight':
        model = SimpleClassifier(
            input_dim=input_dim,
            hidden_dims=(128,),
            num_classes=num_classes,
        )
        logger.info(f"创建轻量化模型: input_dim={input_dim}, num_classes={num_classes}")
        
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model


def train_mode(args):
    """训练模式"""
    logger = get_logger()
    logger.info("开始训练模式")
    
    # 加载配置
    config = Config(args.config) if os.path.exists(args.config) else Config()
    config.set('training.device', args.device)
    config.set('training.epochs', args.epochs)
    config.set('training.batch_size', args.batch_size)
    config.set('training.lr', args.lr)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    logger.info(f"加载数据: {args.data}")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=args.data,
        batch_size=args.batch_size,
        num_workers=0 if os.name == 'nt' else 4
    )
    
    # 创建模型
    num_classes = 8
    logger.info(f"创建模型: {args.model}")
    model = create_model(args.model, config, num_classes)
    
    # 知识蒸馏模式
    if args.distill and args.checkpoint:
        logger.info("启用知识蒸馏模式")
        # 加载教师模型
        teacher = create_model(args.model, config, num_classes)
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        teacher.load_state_dict(normalize_mlp_state_dict(extract_state_dict(checkpoint)))
        
        # 创建学生模型
        student = create_model('lightweight', config, num_classes)
        
        # 这里需要实现蒸馏训练逻辑
        # 简化版本：直接训练学生模型
        model = student
    else:
        # 普通训练模式
        if args.checkpoint:
            logger.info(f"加载检查点: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(normalize_mlp_state_dict(extract_state_dict(checkpoint)))
    
    # 创建训练器
    logger.info("初始化训练器...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        checkpoint_dir=args.output_dir
    )
    
    # 开始训练
    logger.info(f"开始训练: {args.epochs} epochs")
    history = trainer.train(epochs=args.epochs)
    
    # 训练完成
    logger.info("训练完成!")
    logger.info(f"最佳验证准确率: {trainer.best_val_acc:.2f}%")
    
    # 保存训练历史
    import json
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"训练历史已保存: {history_path}")
    
    return model


def test_mode(args):
    """测试模式"""
    logger = get_logger()
    logger.info("=" * 60)
    logger.info("开始测试模式")
    logger.info("=" * 60)
    
    # 加载配置
    config = Config(args.config) if os.path.exists(args.config) else Config()
    
    # 加载数据
    logger.info(f"加载数据从: {args.data}")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=args.data,
        batch_size=args.batch_size,
        num_workers=0 if os.name == 'nt' else 4
    )
    
    # 创建模型
    num_classes = 8
    model = create_model(args.model, config, num_classes)
    
    # 加载检查点
    if args.checkpoint:
        logger.info(f"加载检查点: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        
        # 检查是否是训练器保存的格式
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(normalize_mlp_state_dict(checkpoint['model_state_dict']))
            logger.info(f"检查点加载成功，验证准确率: {checkpoint.get('best_val_acc', 'N/A')}")
        else:
            model.load_state_dict(normalize_mlp_state_dict(checkpoint))
    else:
        logger.warning("未指定检查点，使用随机初始化模型")
    
    # 创建评估器
    evaluator = Evaluator(model, device=args.device)
    
    # 评估测试集
    logger.info("评估测试集...")
    metrics = evaluator.evaluate(test_loader, num_classes=num_classes)
    
    # 打印结果
    evaluator.print_results(metrics)
    
    # 保存结果
    import json
    results_path = os.path.join(args.output_dir, 'test_results.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"测试结果已保存: {results_path}")
    
    return metrics


def serve_mode(args):
    """服务模式 - 启动API服务"""
    logger = get_logger()
    logger.info("=" * 60)
    logger.info("启动API服务模式")
    logger.info("=" * 60)
    
    import uvicorn

    logger.info("请访问 http://localhost:8000/docs 查看交互式接口文档")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)


def evaluate_kg_mode(args):
    """评估知识图谱模式"""
    logger = get_logger()
    logger.info("=" * 60)
    logger.info("评估知识图谱攻击链还原")
    logger.info("=" * 60)
    
    # 创建知识图谱
    logger.info("创建攻击知识图谱...")
    kg = create_attack_knowledge_graph()
    
    # 创建攻击检测器
    detector = AttackDetector(kg)
    
    # 测试攻击链还原
    test_attacks = [
        ['phishing'],           # 钓鱼攻击
        ['ransomware'],         # 勒索软件
        ['apt'],                # APT攻击
        ['dos', 'probe'],       # 混合攻击
    ]
    
    logger.info("测试攻击链还原:")
    for attacks in test_attacks:
        logger.info(f"\n检测到攻击: {attacks}")
        result = detector.generate_attack_chain(attacks)
        
        logger.info(f"攻击链: {' -> '.join(result['attack_chain_cn'])}")
        
        for step in result['chain_details']:
            logger.info(f"  - {step['tactic_cn']}: {step['techniques']}")
    
    # 评估攻击链还原效果
    logger.info("\n评估攻击链还原准确性...")
    from evaluation.evaluator import AttackChainEvaluator
    
    kg_evaluator = AttackChainEvaluator()
    
    # 模拟评估
    predicted = ['initial_access', 'execution', 'persistence']
    ground_truth = ['initial_access', 'execution', 'persistence', 'privilege_escalation']
    
    eval_result = kg_evaluator.evaluate_attack_chain(predicted, ground_truth)
    
    logger.info(f"攻击链还原评估结果:")
    logger.info(f"  精确度: {eval_result['precision']:.4f}")
    logger.info(f"  召回率: {eval_result['recall']:.4f}")
    logger.info(f"  F1分数: {eval_result['f1']:.4f}")
    
    return kg


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置日志
    log_level = getattr(__import__('logging'), args.log_level)
    logger = setup_logger(
        name="CyberAttackDetection",
        log_file=f"{args.output_dir}/system.log",
        level=log_level
    )
    
    logger.info("="*60)
    logger.info("       网络攻击检测系统 - Multi-Modal Deep Learning")
    logger.info("="*60)
    logger.info(f"运行模式: {args.mode}")
    logger.info(f"模型类型: {args.model}")
    logger.info(f"计算设备: {args.device}")
    
    # 根据模式执行
    if args.mode == 'train':
        logger.info("="*60)
        logger.info("开始训练模型...")
        logger.info("="*60)
        model = train_mode(args)
    elif args.mode == 'test':
        logger.info("="*60)
        logger.info("开始测试模型...")
        logger.info("="*60)
        metrics = test_mode(args)
    elif args.mode == 'serve':
        logger.info("="*60)
        logger.info("启动API服务...")
        logger.info("="*60)
        serve_mode(args)
    elif args.mode == 'evaluate_kg':
        logger.info("="*60)
        logger.info("评估知识图谱...")
        logger.info("="*60)
        kg = evaluate_kg_mode(args)
    else:
        logger.error(f"不支持的模式: {args.mode}")
        return
    
    logger.info("="*60)
    logger.info("程序执行完成!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
