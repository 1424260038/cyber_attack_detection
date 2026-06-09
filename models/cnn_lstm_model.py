# -*- coding: utf-8 -*-
"""
CNN-LSTM 混合模型
CNN-LSTM Hybrid Model for Network Attack Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TemporalConvBlock(nn.Module):
    """时序卷积块

    使用一维卷积捕捉时序数据中的局部模式
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.3
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class CNNLSTMClassifier(nn.Module):
    """CNN-LSTM 混合分类器

    结合CNN的局部特征提取能力和LSTM的时序建模能力
    适用于网络流量的时序模式分析
    """

    def __init__(
        self,
        input_dim: int = 64,
        cnn_hidden_channels: Tuple[int, ...] = (128, 256, 128),
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        num_classes: int = 8,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        初始化CNN-LSTM分类器

        Args:
            input_dim: 输入特征维度
            cnn_hidden_channels: CNN隐藏层通道数
            kernel_sizes: 卷积核大小列表
            lstm_hidden_size: LSTM隐藏层维度
            lstm_num_layers: LSTM层数
            num_classes: 分类类别数
            dropout: Dropout率
            bidirectional: 是否使用双向LSTM
        """
        super().__init__()

        self.input_dim = input_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # CNN特征提取
        self.cnn_branches = nn.ModuleList()
        for out_channels in cnn_hidden_channels:
            branch = nn.ModuleList([
                TemporalConvBlock(
                    in_channels=input_dim if i == 0 else cnn_hidden_channels[i-1],
                    out_channels=out_channels,
                    kernel_size=kernel_sizes[i],
                    dropout=dropout
                )
                for i in range(len(cnn_hidden_channels))
            ])
            self.cnn_branches.append(branch)

        # 简化版CNN - 第一层把input_dim转成128
        self.conv1 = TemporalConvBlock(input_dim, 128, kernel_size=3, dropout=dropout)
        self.conv2 = TemporalConvBlock(128, 256, kernel_size=5, dropout=dropout)
        self.conv3 = TemporalConvBlock(256, 128, kernel_size=7, dropout=dropout)

        # 保存input_dim供后续使用
        self.input_dim = input_dim

        # 自适应池化
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)

        # CNN输出通道数
        cnn_out_channels = cnn_hidden_channels[-1]  # 128

        # 添加一个投影层，将CNN输出投影到适合LSTM的维度
        self.lstm_projection = nn.Linear(cnn_out_channels, cnn_out_channels)

        # LSTM时序建模
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden_size * self.num_directions, lstm_hidden_size),
            nn.Tanh(),
            nn.Linear(lstm_hidden_size, 1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * self.num_directions, lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, seq_len, input_dim] 或 [batch, input_dim]
            
        Returns:
            分类 logits [batch, num_classes]
        """
        # 如果是2D输入 [batch, features]，转换为3D [batch, 1, features]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.shape
        
        # 如果序列太短，进行填充到至少16个时间步（避免多层卷积后序列变短）
        min_seq_len = 16
        if seq_len < min_seq_len:
            # 用复制的方式填充
            repeat_times = (min_seq_len + seq_len - 1) // seq_len
            x = x.repeat(1, repeat_times, 1)[:, :min_seq_len, :]
            seq_len = min_seq_len

        # 将输入转换为CNN需要的格式 [batch, channels, seq_len]
        x_cnn = x.transpose(1, 2)

        # CNN特征提取
        x = self.conv1(x_cnn)
        x = self.conv2(x)
        x = self.conv3(x)

        # 池化
        x = self.adaptive_pool(x).squeeze(-1)  # [batch, 128]

        # 为LSTM准备时序数据 - 创建8个时间窗口
        num_windows = 8
        x_for_lstm = x  # [batch, 128]

        # 复制特征创建8个时间步
        windows = [x_for_lstm for _ in range(num_windows)]
        x_seq = torch.stack(windows, dim=1)  # [batch, num_windows, 128]

        # 通过LSTM投影层
        x_seq = self.lstm_projection(x_seq)

        # LSTM时序建模
        lstm_out, (h_n, _) = self.lstm(x_seq)

        # 注意力机制
        attn_weights = self.attention(lstm_out)  # [batch, num_windows, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # [batch, hidden*directions]

        # 分类
        logits = self.classifier(context)

        return logits

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """获取特征嵌入"""
        with torch.no_grad():
            # 如果是2D输入，转换为3D
            if x.dim() == 2:
                x = x.unsqueeze(1)
            
            batch_size, seq_len, _ = x.shape
            
            # 如果序列太短，进行填充到至少16个时间步
            min_seq_len = 16
            if seq_len < min_seq_len:
                repeat_times = (min_seq_len + seq_len - 1) // seq_len
                x = x.repeat(1, repeat_times, 1)[:, :min_seq_len, :]
                seq_len = min_seq_len

            x_cnn = x.transpose(1, 2)
            x = self.conv1(x_cnn)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.adaptive_pool(x).squeeze(-1)

            num_windows = 8
            x_for_lstm = x  # [batch, 128]

            # 复制特征创建8个时间步
            windows = [x_for_lstm for _ in range(num_windows)]
            x_seq = torch.stack(windows, dim=1)

            lstm_out, _ = self.lstm(x_seq)
            attn_weights = self.attention(lstm_out)
            attn_weights = F.softmax(attn_weights, dim=1)
            context = torch.sum(lstm_out * attn_weights, dim=1)

        return context


class AttentionLSTM(nn.Module):
    """带注意力机制的LSTM模型

    专门用于处理变长序列的时序特征
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 8,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * self.num_directions,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 输入 [batch, seq_len, input_size]
            mask: 注意力掩码 [batch, seq_len]
        """
        # LSTM
        lstm_out, _ = self.lstm(x)

        # 注意力
        if mask is not None:
            # 转换为注意力掩码格式
            attn_mask = mask == 0
        else:
            attn_mask = None

        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, key_padding_mask=attn_mask)

        # 取最后一帧
        context = attn_out.mean(dim=1)

        # 分类
        logits = self.classifier(context)

        return logits


class ResNet1D(nn.Module):
    """一维ResNet用于时序特征提取

    借鉴ResNet的残差连接思想，用于深度时序特征提取
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
