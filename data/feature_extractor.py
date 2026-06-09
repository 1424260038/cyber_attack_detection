# -*- coding: utf-8 -*-
"""
特征提取模块
Feature Extraction Module
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """特征提取器基类"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract(self, data) -> np.ndarray:
        raise NotImplementedError
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names


class TemporalFeatureExtractor(FeatureExtractor):
    """时序特征提取器"""
    
    def __init__(self):
        super().__init__()
        self.feature_names = [
            'packet_count', 'byte_count', 'duration',
            'packet_rate', 'byte_rate', 'avg_inter_arrival_time',
            'std_inter_arrival_time', 'min_inter_arrival_time',
            'max_inter_arrival_time', 'packet_size_mean',
            'packet_size_std', 'packet_size_min', 'packet_size_max'
        ]
    
    def extract(self, packet_data: np.ndarray) -> np.ndarray:
        """
        提取时序特征
        
        Args:
            packet_data: 数据包数据 [num_packets, packet_size]
            
        Returns:
            特征向量
        """
        if len(packet_data) == 0:
            return np.zeros(len(self.feature_names))
        
        features = []
        
        # 基本统计
        packet_count = len(packet_data)
        byte_count = np.sum(packet_data)
        features.append(packet_count)
        features.append(byte_count)
        
        # 假设duration为数据包数量的某个比例
        duration = packet_count * 0.001  # 简化的duration估计
        features.append(duration)
        
        # 速率特征
        packet_rate = packet_count / max(duration, 1e-6)
        byte_rate = byte_count / max(duration, 1e-6)
        features.append(packet_rate)
        features.append(byte_rate)
        
        # 包间隔时间特征（假设均匀分布）
        if packet_count > 1:
            inter_arrival = np.diff(packet_data) / byte_rate
            inter_arrival = np.abs(inter_arrival)
            features.append(np.mean(inter_arrival))
            features.append(np.std(inter_arrival))
            features.append(np.min(inter_arrival))
            features.append(np.max(inter_arrival))
        else:
            features.extend([0, 0, 0, 0])
        
        # 包大小特征
        features.append(np.mean(packet_data))
        features.append(np.std(packet_data))
        features.append(np.min(packet_data))
        features.append(np.max(packet_data))
        
        return np.array(features, dtype=np.float32)


class StatisticalFeatureExtractor(FeatureExtractor):
    """统计特征提取器"""
    
    def __init__(self):
        super().__init__()
        self.feature_names = [
            'mean', 'std', 'min', 'max', 'median',
            'q25', 'q75', 'iqr', 'skewness', 'kurtosis',
            'entropy', 'energy', 'rms'
        ]
    
    def extract(self, data: np.ndarray) -> np.ndarray:
        """提取统计特征"""
        if len(data) == 0:
            return np.zeros(len(self.feature_names))
        
        features = []
        
        # 基本统计
        features.append(np.mean(data))
        features.append(np.std(data))
        features.append(np.min(data))
        features.append(np.max(data))
        features.append(np.median(data))
        
        # 分位数
        q25 = np.percentile(data, 25)
        q75 = np.percentile(data, 75)
        features.append(q25)
        features.append(q75)
        features.append(q75 - q25)  # IQR
        
        # 高阶统计
        if len(data) > 2:
            # 偏度
            skewness = self._skewness(data)
            features.append(skewness)
            
            # 峰度
            kurtosis = self._kurtosis(data)
            features.append(kurtosis)
        else:
            features.extend([0, 0])
        
        # 熵
        entropy = self._entropy(data)
        features.append(entropy)
        
        # 能量
        energy = np.sum(data ** 2)
        features.append(energy)
        
        # 均方根
        rms = np.sqrt(np.mean(data ** 2))
        features.append(rms)
        
        return np.array(features, dtype=np.float32)
    
    def _skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _entropy(self, data: np.ndarray) -> float:
        """计算熵"""
        hist, _ = np.histogram(data, bins=50)
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist + 1e-10))


class GraphFeatureExtractor(FeatureExtractor):
    """图特征提取器"""
    
    def __init__(self):
        super().__init__()
        self.feature_names = [
            'degree', 'in_degree', 'out_degree',
            'centrality', 'betweenness', 'closeness',
            'pagerank', 'clustering_coeff', 'eigenvector'
        ]
    
    def extract_from_edges(
        self,
        edges: List[Tuple],
        node_id: int,
        all_nodes: Optional[List[int]] = None
    ) -> np.ndarray:
        """从边列表提取节点特征
        
        Args:
            edges: 边列表
            node_id: 目标节点ID
            all_nodes: 所有节点列表
            
        Returns:
            特征向量
        """
        if not edges:
            return np.zeros(len(self.feature_names))
        
        # 构建邻接关系
        in_neighbors = set()
        out_neighbors = set()
        
        for src, dst in edges:
            if src == node_id:
                out_neighbors.add(dst)
            if dst == node_id:
                in_neighbors.add(src)
        
        features = []
        
        # 度
        degree = len(in_neighbors) + len(out_neighbors)
        features.append(degree)
        features.append(len(in_neighbors))
        features.append(len(out_neighbors))
        
        # 中心性（简化版）
        all_degrees = {}
        for src, dst in edges:
            all_degrees[src] = all_degrees.get(src, 0) + 1
            all_degrees[dst] = all_degrees.get(dst, 0) + 1
        
        centrality = degree / max(len(all_degrees), 1)
        features.append(centrality)
        
        # 介数中心性（简化）
        betweenness = 0
        for src, dst in edges:
            if (src < node_id < dst) or (dst < node_id < src):
                betweenness += 1
        features.append(betweenness / max(len(edges), 1))
        
        # 接近中心性（简化）
        features.append(1.0 / (centrality + 1))
        
        # PageRank（简化）
        features.append(degree / max(sum(all_degrees.values()), 1))
        
        # 聚类系数
        neighbors = list(in_neighbors | out_neighbors)
        if len(neighbors) > 1:
            cluster_edges = 0
            for n1 in neighbors:
                for n2 in neighbors:
                    if n1 != n2 and (n1, n2) in edges or (n2, n1) in edges:
                        cluster_edges += 1
            clustering = cluster_edges / (len(neighbors) * (len(neighbors) - 1))
        else:
            clustering = 0
        features.append(clustering)
        
        # 特征向量中心性（简化）
        features.append(centrality ** 2)
        
        return np.array(features, dtype=np.float32)
    
    def extract_from_adj_matrix(self, adj_matrix: np.ndarray) -> np.ndarray:
        """从邻接矩阵提取图特征
        
        Args:
            adj_matrix: 邻接矩阵
            
        Returns:
            特征矩阵
        """
        n = adj_matrix.shape[0]
        
        # 计算度
        degrees = np.sum(adj_matrix, axis=1)
        
        # 计算特征向量中心性（简化：使用度的归一化）
        eigenvector = degrees / (np.sum(degrees) + 1e-10)
        
        # 计算聚类系数
        clustering = np.zeros(n)
        for i in range(n):
            neighbors = np.where(adj_matrix[i] > 0)[0]
            if len(neighbors) > 1:
                subgraph = adj_matrix[np.ix_(neighbors, neighbors)]
                clustering[i] = np.sum(subgraph) / (len(neighbors) * (len(neighbors) - 1))
        
        features = np.stack([
            degrees,
            degrees / (np.max(degrees) + 1e-10),  # 归一化度
            eigenvector,
            clustering,
            np.zeros(n)  # 预留
        ], axis=1)
        
        return features


class MultiModalFeatureExtractor:
    """多模态特征提取器"""
    
    def __init__(self):
        self.temporal_extractor = TemporalFeatureExtractor()
        self.statistical_extractor = StatisticalFeatureExtractor()
        self.graph_extractor = GraphFeatureExtractor()
    
    def extract_temporal(self, data: np.ndarray) -> np.ndarray:
        """提取时序特征"""
        return self.temporal_extractor.extract(data)
    
    def extract_statistical(self, data: np.ndarray) -> np.ndarray:
        """提取统计特征"""
        return self.statistical_extractor.extract(data)
    
    def extract_graph(self, edges: List[Tuple], node_id: int) -> np.ndarray:
        """提取图特征"""
        return self.graph_extractor.extract_from_edges(edges, node_id)
    
    def extract_all(
        self,
        temporal_data: Optional[np.ndarray] = None,
        statistical_data: Optional[np.ndarray] = None,
        edges: Optional[List[Tuple]] = None,
        node_id: int = 0
    ) -> Dict[str, np.ndarray]:
        """提取所有模态特征
        
        Returns:
            包含各模态特征的字典
        """
        features = {}
        
        if temporal_data is not None:
            features['temporal'] = self.extract_temporal(temporal_data)
        
        if statistical_data is not None:
            features['statistical'] = self.extract_statistical(statistical_data)
        
        if edges is not None:
            features['graph'] = self.extract_graph(edges, node_id)
        
        return features
    
    def get_feature_dim(self) -> Dict[str, int]:
        """获取各模态特征维度"""
        return {
            'temporal': len(self.temporal_extractor.feature_names),
            'statistical': len(self.statistical_extractor.feature_names),
            'graph': len(self.graph_extractor.feature_names)
        }


class FlowFeatureExtractor(nn.Module):
    """流特征提取神经网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
