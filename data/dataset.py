from typing import Tuple, Optional
import os
import random
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from utils.depth import compute_depth


class SceneDataset(Dataset):
    """
    条件式场景生成数据集。
    - 若labels目录存在PNG标签图，则以其作为条件输入；
    - 否则使用图像的Canny边缘作为条件输入（自动生成）。
    图像标准化至[-1,1]，条件输入标准化至[0,1]后再映射到[-1,1]以适配网络。
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: Optional[str],
        image_size: int = 256,
        random_crop: bool = True,
        random_flip: bool = True,
        color_jitter: bool = False,
        condition_mode: str = "auto",
    ) -> None:
        self.images_dir = images_dir
        self.labels_dir = labels_dir if (labels_dir and os.path.isdir(labels_dir)) else None
        self.image_size = image_size
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.color_jitter = color_jitter
        # 条件模式：auto|label|edge|depth_cv|depth_midas
        self.condition_mode = (condition_mode or "auto").lower()

        self.images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        if len(self.images) == 0:
            raise FileNotFoundError(f"图像目录中无文件: {images_dir}")

        # 预处理pipelines
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.resize = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC)
        self.jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05) if color_jitter else None

    def __len__(self) -> int:
        return len(self.images)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert('RGB')
        return img

    def _load_condition(self, img_path: str) -> Image.Image:
        base = os.path.splitext(os.path.basename(img_path))[0]
        # 1) 显式标签
        if self.condition_mode in ("label", "auto") and self.labels_dir:
            for ext in [".png", ".jpg"]:
                p = os.path.join(self.labels_dir, base + ext)
                if os.path.isfile(p):
                    return Image.open(p).convert('L')

        # 2) 深度（CV/MiDaS）
        if self.condition_mode in ("depth_cv", "depth", "depth_midas"):
            mode = "midas" if self.condition_mode == "depth_midas" else "cv"
            return compute_depth(img_path, mode=mode)

        # 3) 边缘（默认回退）
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img, 100, 200)
        return Image.fromarray(edges)

    def _augment(self, img: Image.Image, cond: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if self.random_flip and random.random() < 0.5:
            img = T.functional.hflip(img)
            cond = T.functional.hflip(cond)
        # 统一resize
        img = self.resize(img)
        cond = self.resize(cond)
        if self.jitter is not None:
            img = self.jitter(img)
        return img, cond

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        img = self._load_image(img_path)
        cond = self._load_condition(img_path)
        img, cond = self._augment(img, cond)

        img_t = self.normalize(self.to_tensor(img))  # [-1,1]
        cond_t = self.to_tensor(cond)  # [0,1] 单通道
        cond_t = cond_t * 2.0 - 1.0  # 映射到[-1,1]
        return cond_t, img_t