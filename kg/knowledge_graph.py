# -*- coding: utf-8 -*-
"""
知识图谱模块
Knowledge Graph Module - 基于MITRE ATT&CK的网络安全知识图谱

用于构建攻击知识图谱、实现攻击链还原和推理
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class MITREATTACKFramework:
    """MITRE ATT&CK 框架
    
    封装MITRE ATT&CK框架的战术、技术和关系
    """
    
    # 攻击战术（ Tactics）
    TACTICS = [
        "reconnaissance",           # 侦察
        "resource_development",     # 资源开发
        "initial_access",           # 初始访问
        "execution",                 # 执行
        "persistence",              # 持久化
        "privilege_escalation",     # 权限提升
        "defense_evasion",          # 防御规避
        "credential_access",        # 凭证访问
        "discovery",                # Discovery
        "lateral_movement",         # 横向移动
        "collection",                # 收集
        "command_and_control",       # 命令与控制
        "exfiltration",              # 数据泄露
        "impact"                    # 影响
    ]
    
    # 战术中文名称映射
    TACTICS_CN = {
        "reconnaissance": "侦察",
        "resource_development": "资源开发",
        "initial_access": "初始访问",
        "execution": "执行",
        "persistence": "持久化",
        "privilege_escalation": "权限提升",
        "defense_evasion": "防御规避",
        "credential_access": "凭证访问",
        "discovery": "发现",
        "lateral_movement": "横向移动",
        "collection": "收集",
        "command_and_control": "命令与控制",
        "exfiltration": "数据泄露",
        "impact": "影响"
    }
    
    # 常见攻击技术（简化版）
    TECHNIQUES = {
        # 初始访问
        "initial_access": [
            "T1566 Phishing",           # 网络钓鱼
            "T1190 Exploit Public-Facing Application",  # 利用面向公众的应用
            "T1133 External Remote Services",  # 外部远程服务
            "T1200 Hardware Additions",  # 硬件添加
            "T1078 Valid Accounts",     # 有效账户
        ],
        # 执行
        "execution": [
            "T1059 Command and Scripting Interpreter",  # 命令和脚本解释器
            "T1204 User Execution",     # 用户执行
            "T1203 Exploitation for Execution",  # 利用执行
        ],
        # 持久化
        "persistence": [
            "T1053 Scheduled Task/Job",  # 计划任务
            "T1547 Boot or Logon Autostart Execution",  # 启动或登录自启动执行
            "T1136 Create Account",      # 创建账户
            "T1552 Credential Dumping",  # 凭证转储
        ],
        # 权限提升
        "privilege_escalation": [
            "T1068 Exploitation for Privilege Escalation",  # 利用权限提升
            "T1548 Abuse Elevation Control Mechanism",  # 滥用提升控制机制
        ],
        # 防御规避
        "defense_evasion": [
            "T1070 Indicator Removal",  # 移除指标
            "T1036 Masquerading",        # 伪装
            "T1027 Obfuscated Files or Information",  # 混淆文件或信息
        ],
        # 凭证访问
        "credential_access": [
            "T1552 Credential Dumping",  # 凭证转储
            "T1110 Brute Force",         # 暴力破解
            "T1555 Credentials from Password Stores",  # 密码存储凭证
        ],
        # 发现
        "discovery": [
            "T1595 Active Scanning",  # 主动扫描
            "T1087 Account Discovery",  # 账户发现
            "T1082 System Information Discovery",  # 系统信息发现
            "T1083 File and Directory Discovery",  # 文件和目录发现
        ],
        # 横向移动
        "lateral_movement": [
            "T1021 Remote Services",   # 远程服务
            "T1080 Taint Shared Content",  # 污染共享内容
        ],
        # 收集
        "collection": [
            "T1560 Archive Collected Data",  # 归档收集的数据
            "T1123 Audio Capture",        # 音频捕获
            "T1119 Automated Collection", # 自动收集
        ],
        # 命令与控制
        "command_and_control": [
            "T1071 Application Layer Protocol",  # 应用层协议
            "T1573 Encrypted Channel",      # 加密通道
            "T1105 Ingress Tool Transfer",  # 入口工具传输
        ],
        # 数据泄露
        "exfiltration": [
            "T1041 Exfiltration Over C2 Channel",  # 通过C2通道泄露
            "T1048 Exfiltration Over Alternative Protocol",  # 通过替代协议泄露
        ],
        # 影响
        "impact": [
            "T1498 Network Denial of Service",  # 网络拒绝服务
            "T1486 Data Encrypted for Impact",  # 为影响加密数据
            "T1489 Service Stop",         # 服务停止
            "T1529 System Shutdown/Reboot",  # 系统关闭/重启
        ]
    }
    
    # 攻击链顺序（Kill Chain）
    KILL_CHAIN_ORDER = [
        "reconnaissance",
        "resource_development", 
        "initial_access",
        "execution",
        "persistence",
        "privilege_escalation",
        "defense_evasion",
        "credential_access",
        "discovery",
        "lateral_movement",
        "collection",
        "command_and_control",
        "exfiltration",
        "impact"
    ]
    
    @classmethod
    def get_tactic_name(cls, tactic: str, cn: bool = True) -> str:
        """获取战术名称"""
        if cn:
            return cls.TACTICS_CN.get(tactic, tactic)
        return tactic
    
    @classmethod
    def get_techniques(cls, tactic: str) -> List[str]:
        """获取指定战术下的所有技术"""
        return cls.TECHNIQUES.get(tactic, [])


class AttackEntity:
    """攻击实体类
    
    表示知识图谱中的一个实体（攻击技术、攻击组织、恶意软件等）
    """
    
    def __init__(
        self,
        entity_id: str,
        entity_type: str,
        name: str,
        name_cn: str = "",
        properties: Optional[Dict] = None
    ):
        """
        初始化攻击实体
        
        Args:
            entity_id: 实体ID
            entity_type: 实体类型 (technique, tactic, malware, actor, etc.)
            name: 实体名称
            name_cn: 中文名称
            properties: 其他属性
        """
        self.id = entity_id
        self.type = entity_type
        self.name = name
        self.name_cn = name_cn
        self.properties = properties or {}
        self.embeddings: Optional[np.ndarray] = None
    
    def __repr__(self):
        return f"AttackEntity(id={self.id}, type={self.type}, name={self.name})"


class AttackRelation:
    """攻击关系类
    
    表示知识图谱中实体之间的关系
    """
    
    def __init__(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: Optional[Dict] = None
    ):
        """
        初始化攻击关系
        
        Args:
            source_id: 源实体ID
            target_id: 目标实体ID
            relation_type: 关系类型
            properties: 关系属性
        """
        self.source_id = source_id
        self.target_id = target_id
        self.type = relation_type
        self.properties = properties or {}


class KnowledgeGraph:
    """网络安全知识图谱
    
    基于MITRE ATT&CK框架构建的知识图谱，用于攻击链还原和推理
    """
    
    def __init__(self):
        """初始化知识图谱"""
        self.entities: Dict[str, AttackEntity] = {}
        self.relations: List[AttackRelation] = []
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        self.relation_types: Dict[str, Set[str]] = defaultdict(set)
        
        # 初始化ATT&CK框架
        self._init_attck_framework()
    
    def _init_attck_framework(self):
        """初始化ATT&CK框架"""
        logger.info("正在初始化MITRE ATT&CK框架...")
        
        # 添加战术节点
        for tactic in MITREATTACKFramework.TACTICS:
            entity = AttackEntity(
                entity_id=f"tactic_{tactic}",
                entity_type="tactic",
                name=tactic,
                name_cn=MITREATTACKFramework.get_tactic_name(tactic, cn=True)
            )
            self.add_entity(entity)
        
        # 添加技术节点并建立战术-技术关系
        for tactic, techniques in MITREATTACKFramework.TECHNIQUES.items():
            tactic_id = f"tactic_{tactic}"
            
            for tech in techniques:
                tech_id = f"tech_{tech.split()[0]}"
                
                # 如果技术节点不存在，则添加
                if tech_id not in self.entities:
                    tech_name = tech.split(None, 1)[1] if ' ' in tech else tech
                    entity = AttackEntity(
                        entity_id=tech_id,
                        entity_type="technique",
                        name=tech,
                        name_cn=tech_name
                    )
                    self.add_entity(entity)
                
                # 建立战术-技术关系
                self.add_relation(
                    AttackRelation(
                        source_id=tactic_id,
                        target_id=tech_id,
                        relation_type="has_technique"
                    )
                )
        
        logger.info(f"知识图谱初始化完成: {len(self.entities)} 个实体, {len(self.relations)} 个关系")
    
    def add_entity(self, entity: AttackEntity):
        """添加实体"""
        self.entities[entity.id] = entity
    
    def add_relation(self, relation: AttackRelation):
        """添加关系"""
        self.relations.append(relation)
        self.adjacency[relation.source_id].add(relation.target_id)
        self.relation_types[relation.type].add(relation.source_id)
    
    def get_entity(self, entity_id: str) -> Optional[AttackEntity]:
        """获取实体"""
        return self.entities.get(entity_id)
    
    def get_neighbors(self, entity_id: str) -> List[AttackEntity]:
        """获取邻居实体"""
        neighbor_ids = self.adjacency.get(entity_id, set())
        return [self.entities[nid] for nid in neighbor_ids if nid in self.entities]
    
    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5
    ) -> List[List[str]]:
        """查找路径
        
        查找两个实体之间的所有路径
        
        Args:
            start_id: 起始实体ID
            end_id: 目标实体ID
            max_depth: 最大深度
            
        Returns:
            路径列表
        """
        paths = []
        
        def dfs(current: str, target: str, path: List[str], visited: Set[str]):
            if current == target:
                paths.append(path.copy())
                return
            
            if len(path) >= max_depth:
                return
            
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, target, path, visited)
                    path.pop()
                    visited.remove(neighbor)
        
        dfs(start_id, end_id, [start_id], {start_id})
        return paths
    
    def get_attack_chain(self, technique_ids: List[str]) -> List[str]:
        """还原攻击链
        
        根据检测到的技术，还原完整的攻击链
        
        Args:
            technique_ids: 检测到的技术ID列表
            
        Returns:
            攻击链（按战术顺序排列）
        """
        attack_chain = []
        
        # 按Kill Chain顺序排列
        for tactic in MITREATTACKFramework.KILL_CHAIN_ORDER:
            tactic_id = f"tactic_{tactic}"
            
            # 检查该战术下的技术是否被检测到
            for tech_id in technique_ids:
                if tech_id in self.adjacency[tactic_id]:
                    attack_chain.append(tactic)
                    break
        
        return attack_chain
    
    def to_dict(self) -> Dict:
        """导出为字典"""
        return {
            "entities": {
                eid: {
                    "id": e.id,
                    "type": e.type,
                    "name": e.name,
                    "name_cn": e.name_cn,
                    "properties": e.properties
                }
                for eid, e in self.entities.items()
            },
            "relations": [
                {
                    "source": r.source_id,
                    "target": r.target_id,
                    "type": r.type,
                    "properties": r.properties
                }
                for r in self.relations
            ]
        }


class AttackDetector:
    """攻击检测器
    
    将模型检测结果映射到知识图谱
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        """
        初始化攻击检测器
        
        Args:
            knowledge_graph: 知识图谱
        """
        self.kg = knowledge_graph
        
        # 攻击类型到技术的映射
        self.attack_to_techniques = {
            "normal": [],
            "dos": ["T1498 Network Denial of Service"],
            "ddos": ["T1498 Network Denial of Service"],
            "probe": ["T1595 Active Scanning"],
            "scan": ["T1595 Active Scanning"],
            "r2l": ["T1078 Valid Accounts"],
            "u2r": ["T1068 Exploitation for Privilege Escalation"],
            "malware": ["T1059 Command and Scripting Interpreter"],
            "ransomware": ["T1486 Data Encrypted for Impact"],
            "phishing": ["T1566 Phishing"],
            "apt": ["T1071 Application Layer Protocol"]
        }
    
    def map_detection_to_ttp(
        self,
        attack_type: str,
        confidence: float = 0.8
    ) -> List[Tuple[str, str, float]]:
        """
        将检测结果映射到MITRE ATT&CK战术和技术
        
        Args:
            attack_type: 攻击类型
            confidence: 置信度
            
        Returns:
            [(技术ID, 技术名称, 置信度), ...]
        """
        techniques = self.attack_to_techniques.get(attack_type, [])
        
        ttps = []
        for tech in techniques:
            tech_id = f"tech_{tech.split()[0]}"
            ttps.append((tech_id, tech, confidence))
        
        return ttps
    
    def generate_attack_chain(
        self,
        detected_attacks: List[str]
    ) -> Dict:
        """
        生成攻击链
        
        Args:
            detected_attacks: 检测到的攻击列表
            
        Returns:
            攻击链信息
        """
        technique_ids = []
        
        for attack_type in detected_attacks:
            ttps = self.map_detection_to_ttp(attack_type)
            for tech_id, _, _ in ttps:
                if tech_id not in technique_ids:
                    technique_ids.append(tech_id)
        
        # 还原攻击链
        chain = self.kg.get_attack_chain(technique_ids)
        
        # 构建详细信息
        chain_details = []
        for tactic in chain:
            tactic_cn = MITREATTACKFramework.get_tactic_name(tactic, cn=True)
            tactic_id = f"tactic_{tactic}"
            
            # 获取该战术下的技术
            techniques = []
            for neighbor in self.kg.get_neighbors(tactic_id):
                if neighbor.type == "technique":
                    techniques.append({
                        "id": neighbor.id,
                        "name": neighbor.name,
                        "name_cn": neighbor.name_cn
                    })
            
            chain_details.append({
                "tactic": tactic,
                "tactic_cn": tactic_cn,
                "techniques": techniques
            })
        
        return {
            "attack_chain": chain,
            "attack_chain_cn": [MITREATTACKFramework.get_tactic_name(t, cn=True) for t in chain],
            "chain_details": chain_details,
            "detected_techniques": technique_ids
        }


def create_attack_knowledge_graph() -> KnowledgeGraph:
    """创建攻击知识图谱"""
    return KnowledgeGraph()
