# -*- coding: utf-8 -*-
"""
网络攻击检测器 - 开箱即用的检测模块
Cyber Attack Detector - Ready-to-use Detection Module

使用方法:
    from detector import AttackDetector
    
    detector = AttackDetector(model_path='checkpoints/best_model.pth')
    result = detector.predict(features)
"""

import torch
import numpy as np
from typing import List, Dict, Union, Optional
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from models.cnn_lstm_model import CNNLSTMClassifier


class AttackDetector:
    """
    网络攻击检测器
    
    支持单条检测、批量检测、实时流式检测
    """
    
    # 攻击类型映射
    ATTACK_TYPES = {
        0: "Normal",
        1: "DoS/DDoS",
        2: "Probe/Scan",
        3: "R2L",
        4: "U2R",
        5: "Malware/Ransomware",
        6: "Phishing",
        7: "APT"
    }
    
    # 二分类结果
    BINARY_RESULT = {
        0: "Normal",
        1: "Attack"
    }
    
    def __init__(
        self,
        model_path: str = "checkpoints/best_model.pth",
        input_dim: int = 64,
        num_classes: int = 8,
        device: str = "cpu",
        use_binary: bool = True
    ):
        """
        初始化检测器
        
        Args:
            model_path: 模型文件路径
            input_dim: 输入特征维度
            num_classes: 分类类别数
            device: 计算设备 (cpu/cuda)
            use_binary: 是否使用二分类 (Normal/Attack)
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.device = device
        self.use_binary = use_binary
        
        # 创建模型
        self.model = CNNLSTMClassifier(
            input_dim=input_dim,
            cnn_hidden_channels=[128, 256, 128],
            lstm_hidden_size=256,
            lstm_num_layers=2,
            num_classes=num_classes,
            dropout=0.3
        )
        
        # 加载权重
        self._load_model(model_path)
        
        self.model.eval()
        print(f"✅ 检测器加载成功! 设备: {device}")
    
    def _load_model(self, model_path: str):
        """加载模型权重"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 处理不同格式的 checkpoint
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict, strict=False)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
    
    def predict(
        self,
        features: Union[List[float], np.ndarray, torch.Tensor],
        return_probs: bool = False
    ) -> Dict:
        """
        单条数据预测
        
        Args:
            features: 特征向量 (64维)
            return_probs: 是否返回各类别概率
            
        Returns:
            dict: 预测结果
        """
        # 转换为 tensor
        if isinstance(features, list):
            features = np.array(features)
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        if isinstance(features, torch.Tensor):
            features = features.clone().detach()
        
        # 确保维度正确
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            output = self.model(features)
            probs = torch.softmax(output, dim=1)[0]
            pred_class = torch.argmax(probs).item()
        
        # 构建结果
        if self.use_binary:
            # 二分类: 0=Normal, 1+=Attack
            is_attack = 1 if pred_class > 0 else 0
            result = {
                "prediction": self.BINARY_RESULT[is_attack],
                "confidence": probs[pred_class].item() if pred_class <= 1 else probs[1:].max().item(),
                "is_attack": is_attack == 1
            }
        else:
            # 多分类
            result = {
                "prediction": self.ATTACK_TYPES[pred_class],
                "confidence": probs[pred_class].item(),
                "attack_type_id": pred_class
            }
        
        if return_probs:
            if self.use_binary:
                result["probabilities"] = {
                    "Normal": probs[0].item(),
                    "Attack": probs[1:].max().item()
                }
            else:
                result["probabilities"] = {
                    self.ATTACK_TYPES[i]: probs[i].item() 
                    for i in range(self.num_classes)
                }
        
        return result
    
    def predict_batch(
        self,
        features_batch: Union[List[List[float]], np.ndarray],
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        批量预测
        
        Args:
            features_batch: 批量特征 [batch_size, 64]
            threshold: 攻击判定阈值
            
        Returns:
            list: 预测结果列表
        """
        if isinstance(features_batch, list):
            features_batch = np.array(features_batch)
        
        features = torch.tensor(features_batch, dtype=torch.float32)
        
        results = []
        with torch.no_grad():
            outputs = self.model(features)
            probs = torch.softmax(outputs, dim=1)
            pred_classes = torch.argmax(probs, dim=1)
            
            for i in range(len(features_batch)):
                pred_class = pred_classes[i].item()
                prob = probs[i]
                
                if self.use_binary:
                    is_attack = 1 if pred_class > 0 else 0
                    confidence = prob[pred_class].item() if pred_class <= 1 else prob[1:].max().item()
                    results.append({
                        "prediction": self.BINARY_RESULT[is_attack],
                        "confidence": confidence,
                        "is_attack": is_attack == 1
                    })
                else:
                    results.append({
                        "prediction": self.ATTACK_TYPES[pred_class],
                        "confidence": prob[pred_class].item(),
                        "attack_type_id": pred_class
                    })
        
        return results
    
    def detect_stream(
        self,
        features_generator,
        callback=None,
        alert_threshold: float = 0.8
    ):
        """
        流式检测 - 适合实时数据流
        
        Args:
            features_generator: 特征生成器/迭代器
            callback: 检测到攻击时的回调函数
            alert_threshold: 报警阈值
            
        Yields:
            dict: 每次检测的结果
        """
        for features in features_generator:
            result = self.predict(features)
            
            # 如果检测到攻击且置信度高于阈值
            if result.get("is_attack", False) and result["confidence"] >= alert_threshold:
                result["alert"] = True
                
                if callback:
                    callback(result)
            else:
                result["alert"] = False
            
            yield result


# ============================================================
# 便捷函数
# ============================================================

def quick_detect(
    features: List[float],
    model_path: str = "checkpoints/best_model.pth"
) -> Dict:
    """
    快速单条检测
    
    Args:
        features: 64维特征向量
        model_path: 模型路径
        
    Returns:
        dict: 预测结果
    """
    detector = AttackDetector(model_path=model_path)
    return detector.predict(features)


def detect_csv(
    input_file: str,
    output_file: str,
    model_path: str = "checkpoints/best_model.pth",
    feature_cols: Optional[List[str]] = None,
    label_col: Optional[str] = None
):
    """
    批量检测 CSV 文件
    
    Args:
        input_file: 输入 CSV 文件路径
        output_file: 输出 CSV 文件路径
        model_path: 模型路径
        feature_cols: 特征列名列表（默认使用除标签列外的所有列）
        label_col: 标签列名（如果存在会被排除）
    """
    import pandas as pd
    
    # 读取数据
    df = pd.read_csv(input_file)
    
    # 确定特征列
    if feature_cols:
        features = df[feature_cols].values
    elif label_col and label_col in df.columns:
        features = df.drop(columns=[label_col]).values
    else:
        features = df.values
    
    # 检测
    detector = AttackDetector(model_path=model_path)
    results = detector.predict_batch(features)
    
    # 添加结果
    df["Prediction"] = [r["prediction"] for r in results]
    df["Confidence"] = [r["confidence"] for r in results]
    df["Is_Attack"] = [r.get("is_attack", False) for r in results]
    
    # 保存
    df.to_csv(output_file, index=False)
    print(f"✅ 检测完成! 结果已保存到 {output_file}")
    print(f"   总样本数: {len(df)}")
    print(f"   攻击样本: {df['Is_Attack'].sum()}")
    print(f"   正常样本: {(~df['Is_Attack']).sum()}")


# ============================================================
# 示例用法
# ============================================================

if __name__ == "__main__":
    # 示例 1: 单条检测
    print("=" * 50)
    print("示例 1: 单条检测")
    print("=" * 50)
    
    detector = AttackDetector(model_path="checkpoints/best_model.pth")
    
    # 模拟 64 维特征
    test_features = np.random.randn(64).tolist()
    result = detector.predict(test_features, return_probs=True)
    
    print(f"预测结果: {result['prediction']}")
    print(f"置信度: {result['confidence']:.4f}")
    print(f"详细概率: {result['probabilities']}")
    
    # 示例 2: 批量检测
    print("\n" + "=" * 50)
    print("示例 2: 批量检测")
    print("=" * 50)
    
    batch_features = np.random.randn(10, 64)
    results = detector.predict_batch(batch_features)
    
    for i, r in enumerate(results):
        print(f"样本 {i+1}: {r['prediction']} (置信度: {r['confidence']:.4f})")
    
    # 示例 3: CSV 文件检测
    # print("\n" + "=" * 50)
    # print("示例 3: CSV 文件检测")
    # print("=" * 50)
    # detect_csv("input.csv", "output.csv")
