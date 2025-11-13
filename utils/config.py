from typing import Dict
import yaml


def load_config(path: str) -> Dict:
    """加载YAML配置文件。"""
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg