# -*- coding: utf-8 -*-
"""
评估模块
Evaluation Module
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    average_precision_score
)
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """模型评估器"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        初始化评估器
        
        Args:
            model: 待评估模型
            device: 设备
        """
        self.model = model
        self.device = device
        self.model = model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        num_classes: int = 8,
        return_predictions: bool = False
    ) -> Dict:
        """
        评估模型
        
        Args:
            dataloader: 数据加载器
            num_classes: 类别数
            return_predictions: 是否返回预测结果
            
        Returns:
            评估指标字典
        """
        all_preds = []
        all_labels = []
        all_probs = []
        
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            outputs = self.model(inputs)
            
            # 获取预测和概率
            probs = torch.softmax(outputs, dim=-1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # 计算指标
        metrics = self._calculate_metrics(
            all_labels,
            all_preds,
            all_probs,
            num_classes
        )
        
        if return_predictions:
            metrics['predictions'] = all_preds
            metrics['labels'] = all_labels
            metrics['probabilities'] = all_probs
        
        return metrics
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
        num_classes: int
    ) -> Dict:
        """计算评估指标"""
        if y_probs.ndim == 2:
            num_classes = y_probs.shape[1]
        all_class_labels = list(range(num_classes))
        observed_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        metric_labels = [label for label in observed_labels if 0 <= label < num_classes]
        if not metric_labels:
            metric_labels = all_class_labels
        
        # 基础指标
        accuracy = accuracy_score(y_true, y_pred)
        
        # 多分类指标（使用macro平均）
        precision = precision_score(
            y_true,
            y_pred,
            labels=metric_labels,
            average='macro',
            zero_division=0,
        )
        recall = recall_score(
            y_true,
            y_pred,
            labels=metric_labels,
            average='macro',
            zero_division=0,
        )
        f1 = f1_score(
            y_true,
            y_pred,
            labels=metric_labels,
            average='macro',
            zero_division=0,
        )
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=all_class_labels)
        
        # 分类报告
        report = classification_report(
            y_true, y_pred,
            labels=metric_labels,
            target_names=[f'Class_{i}' for i in metric_labels],
            zero_division=0,
            output_dict=True
        )
        
        # 计算AUC-ROC（多分类）
        auc_roc = self._calculate_auc_roc(y_true, y_probs, num_classes)
        
        # 计算平均精度 - 修复多分类问题
        try:
            # One-hot 编码
            y_true_onehot = np.zeros((len(y_true), num_classes))
            y_true_onehot[np.arange(len(y_true)), y_true] = 1
            ap_scores = []
            for i in range(num_classes):
                if len(np.unique(y_true_onehot[:, i])) > 1:
                    ap_scores.append(average_precision_score(y_true_onehot[:, i], y_probs[:, i]))
            ap = float(np.mean(ap_scores)) if ap_scores else 0.0
        except Exception:
            ap = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'average_precision': ap,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
    
    def _calculate_auc_roc(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        num_classes: int
    ) -> float:
        """计算多分类AUC-ROC"""
        try:
            # One-hot编码
            y_true_onehot = np.zeros((len(y_true), num_classes))
            y_true_onehot[np.arange(len(y_true)), y_true] = 1
            
            # 计算每个类别的AUC
            aucs = []
            for i in range(num_classes):
                if len(np.unique(y_true_onehot[:, i])) > 1:
                    fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_probs[:, i])
                    auc_val = auc(fpr, tpr)
                    aucs.append(auc_val)
            
            return np.mean(aucs) if aucs else 0.0
        except Exception as e:
            logger.warning(f"Failed to calculate AUC-ROC: {e}")
            return 0.0
    
    def print_results(self, metrics: Dict):
        """打印评估结果"""
        logger.info("=" * 50)
        logger.info("Evaluation Results")
        logger.info("=" * 50)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
        logger.info(f"Avg Precision: {metrics['average_precision']:.4f}")
        logger.info("=" * 50)
        logger.info("Classification Report:")
        for key, value in metrics['classification_report'].items():
            if isinstance(value, dict):
                logger.info(f"  {key}: {value}")
        logger.info("=" * 50)


class AttackChainEvaluator:
    """攻击链还原评估器
    
    专门评估基于知识图谱的攻击链还原效果
    """
    
    def __init__(self):
        self.attack_names = [
            'normal', 'dos', 'probe', 'r2l', 'u2r', 
            'malware', 'ransomware', 'apt'
        ]
    
    def evaluate_attack_chain(
        self,
        predicted_chain: list,
        ground_truth_chain: list
    ) -> Dict:
        """
        评估攻击链还原效果
        
        Args:
            predicted_chain: 预测的攻击链
            ground_truth_chain: 真实的攻击链
            
        Returns:
            评估指标
        """
        if not predicted_chain or not ground_truth_chain:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # 计算序列相似度
        predicted_set = set(predicted_chain)
        ground_truth_set = set(ground_truth_chain)
        
        # 交集
        intersection = predicted_set & ground_truth_set
        
        # 精确度：预测正确的比例
        precision = len(intersection) / len(predicted_set) if predicted_set else 0
        
        # 召回率：真实攻击链被还原的比例
        recall = len(intersection) / len(ground_truth_set) if ground_truth_set else 0
        
        # F1分数
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 序列准确率（完全匹配）
        accuracy = 1.0 if predicted_chain == ground_truth_chain else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predicted_chain': predicted_chain,
            'ground_truth_chain': ground_truth_chain,
            'common_steps': list(intersection)
        }
    
    def evaluate_knowledge_graph(
        self,
        predicted_entities: list,
        predicted_relations: list,
        ground_truth_entities: list,
        ground_truth_relations: list
    ) -> Dict:
        """评估知识图谱构建效果"""
        
        # 实体评估
        entity_precision = len(set(predicted_entities) & set(ground_truth_entities)) / len(set(predicted_entities)) if predicted_entities else 0
        entity_recall = len(set(predicted_entities) & set(ground_truth_entities)) / len(set(ground_truth_entities)) if ground_truth_entities else 0
        
        # 关系评估
        relation_precision = len(set(predicted_relations) & set(ground_truth_relations)) / len(set(predicted_relations)) if predicted_relations else 0
        relation_recall = len(set(predicted_relations) & set(ground_truth_relations)) / len(set(ground_truth_relations)) if ground_truth_relations else 0
        
        return {
            'entity_precision': entity_precision,
            'entity_recall': entity_recall,
            'relation_precision': relation_precision,
            'relation_recall': relation_recall
        }


def print_confusion_matrix(cm: np.ndarray, class_names: list):
    """打印混淆矩阵"""
    logger.info("Confusion Matrix:")
    logger.info("-" * 50)
    
    # 打印表头
    header = "True\\Pred".ljust(12)
    for name in class_names:
        header += name[:8].center(10)
    logger.info(header)
    logger.info("-" * 50)
    
    # 打印每一行
    for i, name in enumerate(class_names):
        row = name[:8].ljust(12)
        for j in range(min(len(class_names), cm.shape[1])):
            row += str(cm[i, j]).center(10)
        logger.info(row)
    
    logger.info("-" * 50)
