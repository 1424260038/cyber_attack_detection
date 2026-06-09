# -*- coding: utf-8 -*-
"""
配置管理模块
Configuration Management Module
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """数据配置"""
    dataset_path: str = "./data/CICIDS2017"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass 
class ModelConfig:
    """模型配置"""
    gat: Dict[str, Any] = field(default_factory=lambda: {
        "in_channels": 128,
        "hidden_channels": 256, 
        "out_channels": 128,
        "num_heads": 8,
        "dropout": 0.3,
        "layers": 3
    })
    cnn_lstm: Dict[str, Any] = field(default_factory=lambda: {
        "cnn": {"in_channels": 64, "hidden_channels": [128, 256, 128]},
        "lstm": {"input_size": 128, "hidden_size": 256, "num_layers": 2}
    })


@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 100
    batch_size: int = 64
    lr: float = 0.001
    device: str = "cuda"
    seed: int = 42


class Config:
    """配置管理类"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = {}
        if config_path and os.path.exists(config_path):
            self.load(config_path)
    
    def load(self, config_path: str):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def save(self, config_path: str):
        """保存配置文件"""
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """设置配置项"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    @property
    def data(self) -> Dict:
        return self.config.get('data', {})
    
    @property
    def model(self) -> Dict:
        return self.config.get('model', {})
    
    @property
    def training(self) -> Dict:
        return self.config.get('training', {})
    
    @property
    def evaluation(self) -> Dict:
        return self.config.get('evaluation', {})
    
    @property
    def knowledge_graph(self) -> Dict:
        return self.config.get('knowledge_graph', {})
    
    @property
    def lightweight(self) -> Dict:
        return self.config.get('lightweight', {})
    
    @property
    def system(self) -> Dict:
        return self.config.get('system', {})


# 全局配置实例
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def set_config(config: Config):
    """设置全局配置"""
    global _config
    _config = config
