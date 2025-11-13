import os
from typing import Dict
import torch


def save_checkpoint(state: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str) -> Dict:
    return torch.load(path, map_location='cpu')