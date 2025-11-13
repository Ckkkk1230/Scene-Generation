from typing import List
import os
import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image


def save_image_grid(tensors: List[torch.Tensor], path: str, nrow: int = 4) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = vutils.make_grid(torch.cat(tensors, dim=0), nrow=nrow, normalize=True, value_range=(-1, 1))
    ndarr = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(ndarr).save(path)


def save_image(tensor: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = (tensor.clamp(-1, 1) + 1) / 2.0
    ndarr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(ndarr).save(path)