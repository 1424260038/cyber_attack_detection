# -*- coding: utf-8 -*-
"""
训练器模块
Trainer Module
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Trainer:
    """模型训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        log_interval: int = 10
    ):
        """
        初始化训练器
        
        Args:
            model: 待训练模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            device: 设备
            checkpoint_dir: 检查点保存目录
            log_interval: 日志打印间隔
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        
        # 移动模型到设备
        self.model = self.model.to(device)
        
        # 损失函数
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        
        # 优化器
        if optimizer is None:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=0.001,
                weight_decay=0.0001
            )
        else:
            self.optimizer = optimizer
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=0.00001
        )
        
        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            # 调试：打印形状
            if batch_idx == 0 and self.current_epoch == 0:
                print(f"DEBUG: inputs.shape={inputs.shape}, labels.shape={labels.shape}")
            
            # 确保输入是2D的 [batch, features]
            if inputs.dim() == 3:
                # 如果是3D [batch, seq, features]，取平均
                inputs = inputs.mean(dim=1)
            elif inputs.dim() == 2:
                pass  # 已经是2D，直接使用
            else:
                # 如果是1D，reshape
                inputs = inputs.view(inputs.size(0), -1)
            
            # 移动数据到设备
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            if (batch_idx + 1) % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(self.val_loader, desc="Validating"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        epochs: int,
        early_stopping_patience: int = 15
    ) -> Dict:
        """
        训练模型
        
        Args:
            epochs: 训练轮数
            early_stopping_patience: 早停耐心值
            
        Returns:
            训练历史
        """
        patience_counter = 0
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # 打印信息
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.6f}"
            )
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_model.pth')
                patience_counter = 0
                logger.info(f"Best model saved! Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # 定期保存
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        # 保存最终模型
        self.save_checkpoint('last_model.pth')
        
        logger.info(f"Training completed! Best Val Acc: {self.best_val_acc:.2f}%")
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint.get('history', self.history)
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """预测"""
        self.model.eval()
        
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            _, predicted = outputs.max(1)
        
        return predicted


class MultiModalTrainer(Trainer):
    """多模态训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints'
    ):
        super().__init__(model, train_loader, val_loader, None, None, device, checkpoint_dir)
        
        # 多任务损失
        self.attack_criterion = nn.CrossEntropyLoss()
        self.contrastive_criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self) -> Tuple[float, float]:
        """训练一个epoch（多模态版本）"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # 多模态输入处理
            # 假设输入数据包含多个模态特征
            # 这里简化处理：直接使用原始特征
            graph_features = inputs[:, :64]  # 简化的图特征
            temporal_features = inputs[:, 64:]  # 简化的时序特征
            
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(
                graph_features=graph_features,
                temporal_features=temporal_features.unsqueeze(1) if temporal_features.dim() == 2 else temporal_features
            )
            
            # 损失
            loss = self.attack_criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
