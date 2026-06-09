# -*- coding: utf-8 -*-
"""
多模态融合模块
Multi-Modal Fusion Module - 多模态融合模型

整合图神经网络、CNN-LSTM和文本特征，实现多模态网络攻击检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MultiModalFusionModel(nn.Module):
    """多模态融合模型
    
    整合图神经网络、CNN-LSTM和文本特征，实现多模态网络攻击检测
    """
    
    def __init__(
        self,
        # 图神经网络配置
        gat_config: Dict,
        # CNN-LSTM配置
        cnn_lstm_config: Dict,
        # 融合配置
        fusion_config: Dict,
        num_classes: int = 8,
        dropout: float = 0.3
    ):
        """
        初始化多模态融合模型
        
        Args:
            gat_config: GAT模型配置
            cnn_lstm_config: CNN-LSTM模型配置
            fusion_config: 融合层配置
            num_classes: 分类类别数
            dropout: Dropout率
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # 特征投影层（将不同模态的特征投影到相同维度）
        self.graph_projection = nn.Linear(
            gat_config.get('out_channels', 128),
            fusion_config.get('hidden_dim', 512)
        )
        
        self.temporal_projection = nn.Linear(
            cnn_lstm_config.get('lstm', {}).get('hidden_size', 256) * 2,  # *2 for bidirectional
            fusion_config.get('hidden_dim', 512)
        )
        
        self.text_projection = nn.Linear(256, fusion_config.get('hidden_dim', 512))
        
        # 模态特定的编码器
        self.modal_encoders = nn.ModuleDict({
            'graph': nn.Sequential(
                nn.Linear(fusion_config.get('hidden_dim', 512), fusion_config.get('hidden_dim', 512)),
                nn.LayerNorm(fusion_config.get('hidden_dim', 512)),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            'temporal': nn.Sequential(
                nn.Linear(fusion_config.get('hidden_dim', 512), fusion_config.get('hidden_dim', 512)),
                nn.LayerNorm(fusion_config.get('hidden_dim', 512)),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            'text': nn.Sequential(
                nn.Linear(fusion_config.get('hidden_dim', 512), fusion_config.get('hidden_dim', 512)),
                nn.LayerNorm(fusion_config.get('hidden_dim', 512)),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        })
        
        # 跨模态注意力
        self.cross_attention = CrossModalAttention(
            embed_dim=fusion_config.get('hidden_dim', 512),
            num_heads=8,
            dropout=dropout
        )
        
        # 融合方法
        fusion_method = fusion_config.get('method', 'attention')
        
        if fusion_method == 'attention':
            self.fusion_layer = AttentionFusion(
                hidden_dim=fusion_config.get('hidden_dim', 512),
                num_modals=3
            )
        elif fusion_method == 'concat':
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_config.get('hidden_dim', 512) * 3, fusion_config.get('hidden_dim', 512)),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.fusion_layer = nn.Linear(fusion_config.get('hidden_dim', 512), fusion_config.get('hidden_dim', 512))
        
        # 输出分类器
        self.classifier = nn.Sequential(
            nn.Linear(fusion_config.get('hidden_dim', 512), fusion_config.get('hidden_dim', 512) // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_config.get('hidden_dim', 512) // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        graph_features: Optional[torch.Tensor] = None,
        temporal_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            graph_features: 图特征 [batch, graph_dim]
            temporal_features: 时序特征 [batch, seq_len, temporal_dim]
            text_features: 文本特征 [batch, text_dim]
            edge_index: 边索引（用于GAT）
            
        Returns:
            分类 logits [batch, num_classes]
        """
        modal_features = {}
        
        # 处理图特征
        if graph_features is not None:
            graph_emb = self.graph_projection(graph_features)
            graph_emb = self.modal_encoders['graph'](graph_emb)
            modal_features['graph'] = graph_emb
        
        # 处理时序特征
        if temporal_features is not None:
            if temporal_features.dim() == 2:
                temporal_features = temporal_features.unsqueeze(1)
            temporal_emb = self.temporal_projection(temporal_features)
            temporal_emb = self.modal_encoders['temporal'](temporal_emb)
            modal_features['temporal'] = temporal_emb
        
        # 处理文本特征
        if text_features is not None:
            text_emb = self.text_projection(text_features)
            text_emb = self.modal_encoders['text'](text_emb)
            modal_features['text'] = text_emb
        
        # 跨模态注意力
        if len(modal_features) > 1:
            fused = self.cross_attention(list(modal_features.values()))
        elif len(modal_features) == 1:
            fused = list(modal_features.values())[0]
        else:
            # 如果没有输入，返回零向量
            batch_size = graph_features.shape[0] if graph_features is not None else 1
            fused = torch.zeros(batch_size, self.classifier[0].in_features, device=graph_features.device if graph_features is not None else 'cpu')
        
        # 融合
        if isinstance(self.fusion_layer, AttentionFusion):
            output = self.fusion_layer(list(modal_features.values()))
        else:
            # 简单拼接
            concat_features = torch.cat(list(modal_features.values()), dim=-1)
            output = self.fusion_layer(concat_features)
        
        # 分类
        logits = self.classifier(output)
        
        return logits
    
    def get_modal_embeddings(
        self,
        graph_features: Optional[torch.Tensor] = None,
        temporal_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """获取各模态的嵌入"""
        embeddings = {}
        
        if graph_features is not None:
            graph_emb = self.graph_projection(graph_features)
            embeddings['graph'] = self.modal_encoders['graph'](graph_emb)
        
        if temporal_features is not None:
            if temporal_features.dim() == 2:
                temporal_features = temporal_features.unsqueeze(1)
            temporal_emb = self.temporal_projection(temporal_features)
            embeddings['temporal'] = self.modal_encoders['temporal'](temporal_emb)
        
        if text_features is not None:
            text_emb = self.text_projection(text_features)
            embeddings['text'] = self.modal_encoders['text'](text_emb)
        
        return embeddings


class CrossModalAttention(nn.Module):
    """跨模态注意力模块
    
    允许不同模态之间相互注意，捕捉模态间的相关性
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.3):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, modal_features: list) -> torch.Tensor:
        """
        Args:
            modal_features: 各模态特征列表 [num_modals, batch, embed_dim]
            
        Returns:
            融合后的特征 [batch, embed_dim]
        """
        # Stack modalities
        stacked = torch.stack(modal_features, dim=1)  # [batch, num_modals, embed_dim]
        
        # 自注意力
        attended, _ = self.attention(stacked, stacked, stacked)
        
        # 残差连接和LayerNorm
        output = self.norm(stacked + attended)
        
        # 平均池化
        output = output.mean(dim=1)
        
        return output


class AttentionFusion(nn.Module):
    """注意力融合模块
    
    使用可学习的权重对各模态特征进行加权融合
    """
    
    def __init__(self, hidden_dim: int, num_modals: int = 3):
        super().__init__()
        
        self.attention_weights = nn.Parameter(torch.ones(num_modals))
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, modal_features: list) -> torch.Tensor:
        """
        Args:
            modal_features: 各模态特征列表
            
        Returns:
            加权融合后的特征
        """
        # 计算注意力权重
        weights = self.softmax(self.attention_weights)
        
        # 加权求和
        fused = torch.zeros_like(modal_features[0])
        for i, feat in enumerate(modal_features):
            fused += weights[i] * feat
        
        return fused


class ContrastiveLearning(nn.Module):
    """跨模态对比学习模块
    
    用于在特征空间中拉近相关样本，推开不相关样本
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        temperature: float = 0.1,
        queue_size: int = 4096
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.queue_size = queue_size
        
        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # 队列（用于对比学习）
        self.register_buffer('queue', torch.randn(queue_size, embedding_dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
    
    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: 嵌入向量 [batch, embedding_dim]
            
        Returns:
            (投影后的嵌入, 对比损失)
        """
        # 投影
        z = self.projector(embeddings)
        z = F.normalize(z, dim=1)
        
        # 对比损失
        loss = self._contrastive_loss(z)
        
        # 更新队列
        self._dequeue_and_enqueue(z)
        
        return z, loss
    
    def _contrastive_loss(self, z: torch.Tensor) -> torch.Tensor:
        """计算对比损失"""
        batch_size = z.shape[0]
        
        # 计算相似度矩阵
        sim = torch.matmul(z, z.T) / self.temperature
        
        # 创建掩码（对角线为正样本）
        mask = torch.eye(batch_size, device=z.device)
        
        # 计算损失
        exp_sim = torch.exp(sim)
        exp_sim = exp_sim * (1 - mask)  # 排除对角线
        
        pos_sim = (sim * mask).sum(dim=-1)
        loss = -pos_sim + torch.log(exp_sim.sum(dim=-1) + 1e-8)
        
        return loss.mean()
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, z: torch.Tensor):
        """更新队列"""
        batch_size = z.shape[0]
        ptr = int(self.queue_ptr)
        
        # 替换队列中的样本
        self.queue[ptr:ptr + batch_size] = z
        ptr = (ptr + batch_size) % self.queue_size
        
        self.queue_ptr[0] = ptr
