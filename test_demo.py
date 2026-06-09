# -*- coding: utf-8 -*-
"""
快速测试脚本
"""

import torch
import torch.nn as nn
from data.data_loader import create_dataloaders
from training.trainer import Trainer
from evaluation.evaluator import Evaluator

# 简单模型
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=64, num_classes=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.net(x)

print("="*50)
print("模型训练")
print("="*50)

# 创建数据
train_loader, val_loader, test_loader = create_dataloaders(
    data_path='./data',
    batch_size=32,
    num_workers=0
)

# 创建模型
model = SimpleClassifier(input_dim=64, num_classes=8)

# 训练
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cpu',
    checkpoint_dir='./checkpoints'
)

print("开始训练...")
history = trainer.train(epochs=3)

print()
print("="*50)
print("模型测试")
print("="*50)

# 加载最佳模型
trainer.load_checkpoint('checkpoints/best_model.pth')

# 评估
evaluator = Evaluator(model, device='cpu')

# 获取实际类别数
num_actual_classes = len(torch.unique(torch.cat([labels for _, labels in test_loader])))
print(f"实际类别数: {num_actual_classes}")

metrics = evaluator.evaluate(test_loader, num_classes=num_actual_classes)

# 打印结果
evaluator.print_results(metrics)

print()
print("="*50)
print("测试完成!")
print("="*50)
