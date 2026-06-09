# -*- coding: utf-8 -*-
"""
图注意力网络模型
Graph Attention Network Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class GraphAttentionLayer(nn.Module):
    """图注意力层
    
    实现基于注意力机制的图卷积操作，能够学习不同邻居节点的重要性
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 8,
        dropout: float = 0.3,
        alpha: float = 0.2
    ):
        """
        初始化图注意力层
        
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            num_heads: 注意力头数
            dropout: Dropout率
            alpha: LeakyReLU负斜率
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        
        assert self.head_dim * num_heads == out_features, "out_features must be divisible by num_heads"
        
        # 线性变换
        self.W = nn.Linear(in_features, num_heads * self.head_dim, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, num_heads, 2 * self.head_dim))
        
        # 输出变换
        self.O = nn.Linear(num_heads * self.head_dim, out_features)
        
        # 正则化
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(alpha)
        
        # 初始化
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.normal_(self.att)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, in_features]
            edge_index: 边索引 [2, num_edges]
            
        Returns:
            更新后的节点特征 [num_nodes, out_features]
        """
        N = x.size(0)
        
        # 线性变换
        h = self.W(x).view(N, self.num_heads, self.head_dim)  # [N, num_heads, head_dim]
        
        # 计算注意力系数
        # 边的源节点和目标节点
        src, dst = edge_index
        
        # 拼接源节点和目标节点的特征
        h_src = h[src]  # [num_edges, num_heads, head_dim]
        h_dst = h[dst]  # [num_edges, num_heads, head_dim]
        
        # 计算注意力分数
        att_input = torch.cat([h_src, h_dst], dim=-1)  # [num_edges, num_heads, 2*head_dim]
        e = (att_input * self.att).sum(dim=-1)  # [num_edges, num_heads]
        e = self.leaky_relu(e)
        
        # 稀疏softmax
        attention = self._sp_softmax(e, edge_index, N)
        attention = self.dropout(attention)
        
        # 特征聚合
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        out.index_add_(0, src, attention.unsqueeze(-1) * h_dst)
        
        # 残差连接和输出变换
        out = out.view(N, -1)
        out = self.O(out)
        
        return out
    
    def _sp_softmax(self, e: torch.Tensor, edge_index: torch.Tensor, N: int) -> torch.Tensor:
        """稀疏softmax"""
        src, dst = edge_index
        
        # 找出每行的最大值并减去（数值稳定）
        e_max = e.max(dim=1, keepdim=True)[0]
        e = e - e_max
        
        # 计算exp
        exp_e = torch.exp(e)
        
        # 按目标节点分组求和
        deg = torch.zeros(N, device=e.device)
        deg.scatter_add_(0, dst, exp_e.sum(dim=1))
        
        # 归一化
        attention = exp_e / (deg[dst] + 1e-16)
        
        return attention


class GATClassifier(nn.Module):
    """基于GAT的网络攻击分类器
    
    使用多层图注意力网络进行网络流量分类
    """
    
    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 256,
        out_channels: int = 128,
        num_heads: int = 8,
        num_classes: int = 8,
        dropout: float = 0.3,
        num_layers: int = 3
    ):
        """
        初始化GAT分类器
        
        Args:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出特征维度
            num_heads: 注意力头数
            num_classes: 分类类别数
            dropout: Dropout率
            num_layers: 图注意力层数
        """
        super().__init__()
        
        self.num_layers = num_layers
        
        # 输入投影
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # 图注意力层
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(
                hidden_channels,
                hidden_channels,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # BatchNorm层
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            
        Returns:
            分类 logits [num_nodes, num_classes]
        """
        # 输入投影
        x = self.input_proj(x)
        x = F.relu(x)
        
        # 多层图注意力
        for i, (gat_layer, bn_layer) in enumerate(zip(self.gat_layers, self.bn_layers)):
            x = gat_layer(x, edge_index)
            
            # BatchNorm (需要reshape)
            x = x.view(-1, x.size(-1))
            x = bn_layer(x)
            x = x.view(-1, x.size(0), x.size(1))
            
            if i < self.num_layers - 1:
                x = F.relu(x)
            
            x = self.dropout(x)
        
        # 输出投影
        x = self.output_proj(x)
        
        # 分类
        logits = self.classifier(x)
        
        return logits
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """获取节点嵌入"""
        with torch.no_grad():
            x = self.input_proj(x)
            x = F.relu(x)
            
            for i, (gat_layer, bn_layer) in enumerate(zip(self.gat_layers, self.bn_layers)):
                x = gat_layer(x, edge_index)
                x = x.view(-1, x.size(-1))
                x = bn_layer(x)
                x = x.view(-1, x.size(0), x.size(1))
                
                if i < self.num_layers - 1:
                    x = F.relu(x)
            
            x = self.output_proj(x)
        
        return x


class SpatialTemporalGAT(nn.Module):
    """时空图注意力网络
    
    结合空间注意力和时间注意力，用于捕捉网络攻击的时空模式
    """
    
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int = 256,
        out_channels: int = 128,
        num_heads: int = 8,
        num_timesteps: int = 10,
        dropout: float = 0.3
    ):
        """
        初始化时空GAT
        
        Args:
            num_nodes: 图中节点数
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出特征维度
            num_heads: 注意力头数
            num_timesteps: 时间步数
            dropout: Dropout率
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_timesteps = num_timesteps
        
        # 空间注意力
        self.spatial_gat = GraphAttentionLayer(
            in_channels,
            hidden_channels,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 时间注意力
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=num_nodes * hidden_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 时间卷积
        self.temporal_conv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1
        )
        
        # 输出层
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 时空特征 [batch, timesteps, num_nodes, features]
            edge_index: 边索引 [2, num_edges]
            
        Returns:
            输出特征 [batch, num_nodes, out_channels]
        """
        batch_size, timesteps, num_nodes, features = x.shape
        
        # 空间注意力
        x_spatial = x.view(-1, num_nodes, features)  # [batch*timesteps, num_nodes, features]
        
        # 对每个时间步应用GAT
        x_out = []
        for t in range(timesteps):
            h = self.spatial_gat(x_spatial[:, t, :], edge_index)
            x_out.append(h)
        
        x_spatial = torch.stack(x_out, dim=1)  # [batch*timesteps, num_nodes, hidden]
        
        # 时间建模
        x_spatial = x_spatial.view(batch_size, timesteps, -1)  # [batch, timesteps, num_nodes*hidden]
        
        # 时间注意力
        x_temp, _ = self.temporal_attention(x_spatial, x_spatial, x_spatial)
        x_temp = x_temp + x_spatial  # 残差连接
        
        # 输出
        x_temp = x_temp.mean(dim=1)  # [batch, num_nodes*hidden]
        output = self.output_proj(x_temp)
        
        return output
