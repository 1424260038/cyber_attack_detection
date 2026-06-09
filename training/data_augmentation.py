# -*- coding: utf-8 -*-
"""
数据增强模块
Data Augmentation Module - 对抗样本生成

使用GAN等方法进行数据增强，提升模型对隐蔽攻击的鲁棒性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class NoiseInjector:
    """噪声注入数据增强
    
    向训练数据中添加随机噪声以增强模型鲁棒性
    """
    
    def __init__(self, noise_level: float = 0.1):
        """
        初始化噪声注入器
        
        Args:
            noise_level: 噪声强度
        """
        self.noise_level = noise_level
    
    def inject(self, x: torch.Tensor, noise_type: str = 'gaussian') -> torch.Tensor:
        """
        注入噪声
        
        Args:
            x: 输入数据
            noise_type: 噪声类型 ('gaussian', 'uniform', 'salt_pepper')
            
        Returns:
            增强后的数据
        """
        if noise_type == 'gaussian':
            noise = torch.randn_like(x) * self.noise_level
        elif noise_type == 'uniform':
            noise = (torch.rand_like(x) - 0.5) * 2 * self.noise_level
        elif noise_type == 'salt_pepper':
            noise = self._salt_pepper_noise(x)
        else:
            noise = 0
        
        return x + noise
    
    def _salt_pepper_noise(self, x: torch.Tensor) -> torch.Tensor:
        """盐椒噪声"""
        noise = torch.zeros_like(x)
        
        # 随机选择添加噪声的位置
        salt_mask = torch.rand_like(x) < (self.noise_level / 2)
        pepper_mask = torch.rand_like(x) < (self.noise_level / 2)
        
        noise[salt_mask] = x.max()
        noise[pepper_mask] = x.min()
        
        return noise


class RandomMasking:
    """随机掩码数据增强
    
    随机遮挡部分输入特征，模拟数据缺失情况
    """
    
    def __init__(self, mask_ratio: float = 0.3):
        """
        初始化随机掩码
        
        Args:
            mask_ratio: 掩码比例
        """
        self.mask_ratio = mask_ratio
    
    def mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用随机掩码
        
        Args:
            x: 输入数据 [batch, seq_len, features]
            
        Returns:
            (掩码后的数据, 掩码)
        """
        mask = torch.rand_like(x[:, :, 0]) > self.mask_ratio
        mask = mask.unsqueeze(-1).expand_as(x)
        
        x_masked = x.clone()
        x_masked[~mask] = 0
        
        return x_masked, mask


class MixupAugmentation:
    """Mixup数据增强
    
    混合两个样本的特征和标签
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        初始化Mixup
        
        Args:
            alpha: Beta分布参数
        """
        self.alpha = alpha
    
    def mixup(
        self,
        x1: torch.Tensor,
        y1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
        y2: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mixup增强
        
        Args:
            x1, y1: 第一个样本
            x2, y2: 第二个样本（如果为None，则随机选择）
            
        Returns:
            (混合后的特征, 混合后的标签, lambda值)
        """
        if x2 is None or y2 is None:
            # 随机打乱
            indices = torch.randperm(x1.size(0))
            x2 = x1[indices]
            y2 = y1[indices]
        
        # 生成混合比例
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 混合
        x_mixed = lam * x1 + (1 - lam) * x2
        y_mixed = lam * y1 + (1 - lam) * y2
        
        return x_mixed, y_mixed, lam


class FGSM:
    """快速梯度符号方法 (Fast Gradient Sign Method)
    
    对抗样本生成方法
    """
    
    def __init__(self, epsilon: float = 0.01):
        """
        初始化FGSM
        
        Args:
            epsilon: 扰动大小
        """
        self.epsilon = epsilon
    
    def generate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        生成对抗样本
        
        Args:
            model: 模型
            x: 输入
            y: 标签
            loss_fn: 损失函数
            
        Returns:
            对抗样本
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        # 设置为梯度模式
        x_adv = x.clone().detach().requires_grad_(True)
        
        # 前向传播
        outputs = model(x_adv)
        
        # 计算损失
        loss = loss_fn(outputs, y)
        
        # 反向传播
        model.zero_grad()
        loss.backward()
        
        # 生成对抗扰动
        with torch.no_grad():
            gradient = x_adv.grad.sign()
            x_adv = x_adv + self.epsilon * gradient
        
        # 裁剪到有效范围
        x_adv = torch.clamp(x_adv, x - self.epsilon, x + self.epsilon)
        
        return x_adv.detach()


class PGD:
    """投影梯度下降 (Projected Gradient Descent)
    
    迭代式对抗样本生成
    """
    
    def __init__(
        self,
        epsilon: float = 0.01,
        alpha: float = 0.001,
        num_iter: int = 10
    ):
        """
        初始化PGD
        
        Args:
            epsilon: 最大扰动
            alpha: 步长
            num_iter: 迭代次数
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
    
    def generate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """生成对抗样本"""
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        # 初始化
        x_adv = x.clone().detach()
        
        # 随机初始化
        x_adv = x_adv + torch.randn_like(x_adv) * self.epsilon
        x_adv = torch.clamp(x_adv, x - self.epsilon, x + self.epsilon)
        
        # 迭代
        for i in range(self.num_iter):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            
            outputs = model(x_adv)
            loss = loss_fn(outputs, y)
            
            model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                x_adv = x_adv + self.alpha * x_adv.grad.sign()
                x_adv = torch.clamp(x_adv, x - self.epsilon, x + self.epsilon)
                x_adv = torch.clamp(x_adv, 0, 1)  # 假设输入在[0,1]范围
        
        return x_adv.detach()


class WGANGenerator(nn.Module):
    """WGAN生成器
    
    用于生成对抗性网络流量样本
    """
    
    def __init__(self, latent_dim: int = 100, output_dim: int = 128):
        """
        初始化WGAN生成器
        
        Args:
            latent_dim: 潜在空间维度
            output_dim: 输出维度
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # 生成器网络
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.net(z)
    
    def generate(self, num_samples: int, device: str = 'cuda') -> torch.Tensor:
        """生成样本"""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.generate(z)
    
    def generate_attack_samples(
        self,
        num_samples: int,
        attack_type: str,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        生成特定类型的攻击样本
        
        Args:
            num_samples: 样本数量
            attack_type: 攻击类型
            device: 设备
            
        Returns:
            生成的样本
        """
        # 为不同攻击类型使用不同的潜在向量
        np.random.seed(hash(attack_type) % (2**32))
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # 恢复随机种子
        np.random.seed(None)
        
        return self.forward(z)


class DataAugmentor:
    """数据增强器
    
    整合多种数据增强方法
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        初始化数据增强器
        
        Args:
            config: 配置字典
        """
        config = config or {}
        
        self.noise_injector = NoiseInjector(
            noise_level=config.get('noise_level', 0.1)
        )
        self.random_masker = RandomMasking(
            mask_ratio=config.get('mask_ratio', 0.3)
        )
        self.mixup = MixupAugmentation(
            alpha=config.get('mixup_alpha', 0.2)
        )
        self.fgsm = FGSM(
            epsilon=config.get('fgsm_epsilon', 0.01)
        )
        self.pgd = PGD(
            epsilon=config.get('pgd_epsilon', 0.01),
            alpha=config.get('pgd_alpha', 0.001),
            num_iter=config.get('pgd_iter', 10)
        )
        
        self.augmentation_prob = config.get('augmentation_prob', 0.5)
    
    def augment(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        method: str = 'all'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        数据增强
        
        Args:
            x: 输入数据
            y: 标签
            method: 增强方法 ('noise', 'mask', 'mixup', 'fgsm', 'pgd', 'all')
            
        Returns:
            增强后的数据
        """
        if method == 'noise':
            x_aug = self.noise_injector.inject(x)
        elif method == 'mask':
            x_aug, _ = self.random_masker.mask(x)
        elif method == 'mixup':
            x_aug, y_aug, _ = self.mixup.mixup(x, y)
            return x_aug, y_aug
        elif method == 'fgsm':
            # FGSM需要模型，在训练中使用
            return x, y
        elif method == 'pgd':
            return x, y
        elif method == 'all':
            # 随机选择一种方法
            r = np.random.rand()
            if r < 0.25:
                x_aug = self.noise_injector.inject(x)
            elif r < 0.5:
                x_aug, _ = self.random_masker.mask(x)
            else:
                x_aug = x
        else:
            x_aug = x
        
        return x_aug, y
    
    def generate_adversarial(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        method: str = 'fgsm'
    ) -> torch.Tensor:
        """
        生成对抗样本
        
        Args:
            model: 模型
            x: 输入
            y: 标签
            method: 对抗生成方法
            
        Returns:
            对抗样本
        """
        if method == 'fgsm':
            return self.fgsm.generate(model, x, y)
        elif method == 'pgd':
            return self.pgd.generate(model, x, y)
        else:
            return x


def augment_training_data(
    dataloader,
    model: nn.Module,
    device: str = 'cuda',
    methods: list = ['noise', 'mixup']
) -> list:
    """
    增强训练数据
    
    Args:
        dataloader: 数据加载器
        model: 模型（用于生成对抗样本）
        device: 设备
        methods: 增强方法列表
        
    Returns:
        增强后的数据
    """
    augmentor = DataAugmentor()
    
    augmented_data = []
    
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        
        for method in methods:
            x_aug, y_aug = augmentor.augment(x, y, method)
            augmented_data.append((x_aug, y_aug))
            
            # 生成对抗样本
            if method in ['fgsm', 'pgd']:
                x_adv = augmentor.generate_adversarial(model, x, y, method)
                augmented_data.append((x_adv, y))
    
    return augmented_data
