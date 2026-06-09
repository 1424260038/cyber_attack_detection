# -*- coding: utf-8 -*-
"""
数据加载器模块
Data Loader Module
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import logging

from utils.attack_taxonomy import LABEL_ALIASES, label_to_id

logger = logging.getLogger(__name__)


class NetworkTrafficDataset(Dataset):
    """网络流量数据集"""
    
    # 攻击类型标签
    ATTACK_LABELS = {key: label_to_id(key) for key in LABEL_ALIASES}
    
    def __init__(
        self,
        data_path: str,
        mode: str = 'train',
        transform: Optional[Any] = None,
        max_samples: Optional[int] = None
    ):
        """
        初始化数据集
        
        Args:
            data_path: 数据路径
            mode: 数据集模式 train/val/test
            transform: 数据转换
            max_samples: 最大样本数
        """
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.max_samples = max_samples
        
        # 加载数据
        self.data, self.labels, self.metadata = self._load_data()
        
        logger.info(f"Loaded {len(self.data)} samples for {mode} mode")
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """加载数据"""
        data = []
        labels = []
        metadata = []
        
        # 支持多种数据格式
        data_dir = Path(self.data_path)
        
        if data_dir.exists():
            # 尝试加载CSV文件
            csv_files = list(data_dir.glob("*.csv"))
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file, low_memory=False)
                    
                    # 处理标签列
                    if 'Label' in df.columns:
                        label_col = 'Label'
                    elif 'label' in df.columns:
                        label_col = 'label'
                    else:
                        continue
                    
                    # 提取特征和标签
                    feature_cols = [c for c in df.columns if c not in ['Label', 'label', 'Flow ID', 'Src IP', 'Dst IP']]
                    
                    for idx, row in df.iterrows():
                        if pd.notna(row[feature_cols]).all():
                            features = row[feature_cols].values.astype(np.float32)
                            label = self._parse_label(row[label_col])
                            
                            data.append(features)
                            labels.append(label)
                            metadata.append({
                                'src_ip': row.get('Src IP', ''),
                                'dst_ip': row.get('Dst IP', ''),
                                'src_port': row.get('Src Port', 0),
                                'dst_port': row.get('Dst Port', 0),
                            })
                    
                    logger.info(f"Loaded {len(df)} samples from {csv_file.name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load {csv_file}: {e}")
        
        # 转换为numpy数组
        if len(data) > 0:
            data = np.array(data)
            labels = np.array(labels)
            
            # 处理缺失值和无穷值
            data = np.nan_to_num(data, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # 标准化
            data = self._normalize(data)
            
            # 限制样本数
            if self.max_samples and len(data) > self.max_samples:
                indices = np.random.choice(len(data), self.max_samples, replace=False)
                data = data[indices]
                labels = labels[indices]
                metadata = [metadata[i] for i in indices]
        else:
            # 如果没有数据，生成示例数据
            logger.warning("No data found, generating synthetic data for demonstration")
            data, labels, metadata = self._generate_synthetic_data()
        
        return data, labels, metadata
    
    def _parse_label(self, label) -> int:
        """解析标签"""
        return label_to_id(label)
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """标准化数据"""
        # Z-score标准化
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std = np.where(std == 0, 1, std)  # 避免除零
        
        return (data - mean) / std
    
    def _generate_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """生成合成数据用于演示"""
        np.random.seed(42)
        
        num_samples = 1000
        num_features = 64  # 与模型输入维度匹配
        
        # 生成正常流量
        normal_data = np.random.randn(num_samples // 2, num_features) * 0.5
        normal_labels = np.zeros(num_samples // 2, dtype=np.int64)
        
        # 生成攻击流量
        attack_data = np.random.randn(num_samples // 2, num_features) * 1.5 + 2
        attack_labels = np.ones(num_samples // 2, dtype=np.int64)
        
        # 合并
        data = np.vstack([normal_data, attack_data])
        labels = np.concatenate([normal_labels, attack_labels])
        
        # 打乱顺序
        shuffle_idx = np.random.permutation(len(data))
        data = data[shuffle_idx]
        labels = labels[shuffle_idx]
        
        # 生成元数据
        metadata = [
            {'src_ip': f'192.168.1.{i % 255}', 'dst_ip': f'10.0.0.{i % 255}', 
             'src_port': np.random.randint(1024, 65535), 'dst_port': 80}
            for i in range(len(data))
        ]
        
        return data, labels, metadata
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return torch.FloatTensor(self.data[idx]), self.labels[idx]


def create_dataloaders(
    data_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_samples: Optional[int] = None,
    pin_memory: Optional[bool] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建数据加载器
    
    Args:
        data_path: 数据路径
        batch_size: 批次大小
        num_workers: 工作进程数
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # 创建完整数据集
    full_dataset = NetworkTrafficDataset(
        data_path=data_path,
        mode='full',
        max_samples=max_samples
    )
    
    # 划分数据集
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info(f"Data loaders created: train={train_size}, val={val_size}, test={test_size}")
    
    return train_loader, val_loader, test_loader


class GraphDataset(Dataset):
    """图数据结构集（用于GNN）"""
    
    def __init__(self, edge_list: List[Tuple], node_features: np.ndarray, labels: np.ndarray):
        """
        初始化图数据集
        
        Args:
            edge_list: 边列表 [(src, dst), ...]
            node_features: 节点特征 [num_nodes, feature_dim]
            labels: 图标签
        """
        self.edge_list = edge_list
        self.node_features = node_features
        self.labels = labels
    
    def __len__(self):
        return len(self.edge_list)
    
    def __getitem__(self, idx):
        return self.node_features[idx], self.labels[idx]


def collate_graph_batch(batch):
    """图数据批处理"""
    node_features, labels = zip(*batch)
    return torch.FloatTensor(node_features), torch.LongTensor(labels)
