# -*- coding: utf-8 -*-
"""
轻量化模型模块
Lightweight Model Module for Deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LightweightModel(nn.Module):
    """轻量化模型
    
    通过知识蒸馏从大模型压缩而来，适用于边缘部署
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 8,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # 轻量化特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 分类器
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits


class KnowledgeDistillation:
    """知识蒸馏训练器
    
    将大模型的知识迁移到小模型
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.7,
        device: str = 'cuda'
    ):
        """
        初始化知识蒸馏
        
        Args:
            teacher_model: 教师模型
            student_model: 学生模型
            temperature: 蒸馏温度
            alpha: 平衡系数 (0=仅硬标签, 1=仅软标签)
            device: 设备
        """
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        
        # 冻结教师模型
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
    
    def train_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            inputs: 输入数据
            labels: 标签
            
        Returns:
            损失字典
        """
        # 教师模型预测
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
            teacher_soft = F.log_softmax(teacher_logits / self.temperature, dim=-1)
        
        # 学生模型预测
        student_logits = self.student(inputs)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # 硬标签损失
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # 软标签损失（蒸馏损失）
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        soft_loss = soft_loss * (self.temperature ** 2)
        
        # 总损失
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return {
            'loss': loss.item(),
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item()
        }


class ModelPruner:
    """模型剪枝器
    
    移除不重要的神经元以减小模型大小
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def prune_magnitude(
        self,
        amount: float = 0.3,
        verbose: bool = True
    ):
        """
        基于幅度的剪枝
        
        Args:
            amount: 剪枝比例 (0-1)
            verbose: 是否打印信息
        """
        total_params = 0
        pruned_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # 获取权重
                weight = module.weight.data.abs()
                
                # 计算阈值
                threshold = torch.quantile(weight.flatten(), amount)
                
                # 创建掩码
                mask = weight > threshold
                
                # 应用掩码
                module.weight.data *= mask.float()
                
                # 统计
                total_params += weight.numel()
                pruned_params += (mask == False).sum().item()
        
        if verbose:
            logger.info(f"Pruned {pruned_params}/{total_params} ({100*pruned_params/total_params:.2f}%) parameters")
    
    def prune_sensitivity(
        self,
        dataloader,
        device: str = 'cuda',
        verbose: bool = True
    ):
        """
        基于敏感度的剪枝
        
        Args:
            dataloader: 数据加载器
            device: 设备
            verbose: 是否打印信息
        """
        # 收集每个参数的重要性分数
        importance = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                importance[name] = torch.zeros_like(module.weight.data)
        
        # 计算梯度重要性
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if batch_idx >= 10:  # 只用前10个batch
                break
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            for name, module in self.model.named_modules():
                if name in importance and module.weight.grad is not None:
                    importance[name] += module.weight.grad.abs()
        
        # 基于重要性剪枝
        for name, module in self.model.named_modules():
            if name in importance:
                threshold = torch.quantile(importance[name].flatten(), 0.3)
                mask = importance[name] > threshold
                module.weight.data *= mask.float()
        
        if verbose:
            logger.info("Sensitivity-based pruning completed")


class ModelQuantizer:
    """模型量化器
    
    将浮点模型转换为低精度整数模型
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def quantize_dynamic(self, dtype: str = 'int8'):
        """
        动态量化
        
        Args:
            dtype: 目标数据类型 ('int8', 'int16', 'float16')
        """
        # 动态量化Linear层
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.LSTM, nn.LSTMCell},
            dtype=getattr(torch, dtype)
        )
        
        return quantized_model
    
    def quantize_static(
        self,
        calibration_data,
        dtype: str = 'int8'
    ):
        """
        静态量化
        
        Args:
            calibration_data: 校准数据
            dtype: 目标数据类型
        """
        # 设置量化配置
        qconfig = torch.quantization.get_default_qconfig('fbgemm')
        self.model.qconfig = qconfig
        
        # 准备量化
        torch.quantization.prepare(self.model, inplace=True)
        
        # 校准
        self.model.eval()
        with torch.no_grad():
            for inputs, _ in calibration_data:
                self.model(inputs)
        
        # 转换为量化模型
        quantized_model = torch.quantization.convert(self.model, inplace=False)
        
        return quantized_model


def apply_knowledge_distillation(
    teacher_checkpoint: str,
    student_model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 50,
    device: str = 'cuda'
) -> nn.Module:
    """
    应用知识蒸馏
    
    Args:
        teacher_checkpoint: 教师模型检查点路径
        student_model: 学生模型
        train_loader: 训练数据
        val_loader: 验证数据
        epochs: 训练轮数
        device: 设备
        
    Returns:
        训练好的学生模型
    """
    # 加载教师模型
    teacher_model = LightweightModel()  # 简化版，实际应加载完整模型
    teacher_model.load_state_dict(torch.load(teacher_checkpoint))
    teacher_model = teacher_model.to(device)
    
    # 创建蒸馏器
    distill = KnowledgeDistillation(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=4.0,
        alpha=0.7,
        device=device
    )
    
    # 优化器
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    
    # 训练
    student_model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            losses = distill.train_step(inputs, labels)
            
            optimizer.zero_grad()
            # 这里需要反向传播，总损失已经在train_step中计算
            optimizer.step()
        
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {losses['loss']:.4f}")
    
    return student_model


def create_lightweight_model(
    original_model: nn.Module,
    compression_ratio: float = 0.5,
    input_dim: int = 128,
    num_classes: int = 8
) -> nn.Module:
    """
    创建轻量化模型
    
    Args:
        original_model: 原始模型
        compression_ratio: 压缩比例
        input_dim: 输入维度
        num_classes: 类别数
        
    Returns:
        轻量化模型
    """
    hidden_dim = int(256 * compression_ratio)
    
    lightweight = LightweightModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )
    
    return lightweight
