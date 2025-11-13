import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image

import torch


def compute_depth_cv(img_path: str) -> Image.Image:
    """
    
    步骤：双边滤波 -> Sobel梯度幅值 -> 高斯平滑 -> 归一化并反相（边缘更亮、视为更近）。

    返回单通道PIL图像（L，0-255）。
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 保边降噪
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # Sobel梯度
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)

    # 平滑并归一化到0-255
    mag = cv2.GaussianBlur(mag, (5, 5), 0)
    mag_norm = cv2.normalize(mag, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # 反相：边缘更亮 -> 近似更近，便于网络学习结构到纹理
    depth = 255 - mag_norm
    depth_u8 = depth.astype(np.uint8)
    return Image.fromarray(depth_u8, mode='L')


def _load_midas_model(model_name: str = "DPT_Small"):
    """
    通过torch.hub加载MiDaS模型（如不可用则抛出异常）。
    DPT模型依赖timm；如果环境中缺失timm或网络不可用，将抛出异常，调用方需处理回退。
    """
    try:
        import timm  # noqa: F401
    except Exception as e:
        raise RuntimeError("缺少timm依赖，无法加载MiDaS DPT模型") from e

    # 可能会触发在线下载权重；受网络环境影响
    midas = torch.hub.load("intel-isl/MiDaS", model_name)
    midas.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if "DPT" in model_name:
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    return midas, transform, device


def compute_depth_midas(img_path: str, model_name: str = "DPT_Small") -> Image.Image:
    """
    使用MiDaS单目深度估计生成深度图。返回单通道PIL图像（L，0-255）。
    如加载失败（依赖/网络），应由上层回退到compute_depth_cv。
    """
    midas, transform, device = _load_midas_model(model_name)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    input_batch = transform(img_rgb).to(device)
    with torch.no_grad():
        pred = midas(input_batch)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=img_rgb.shape[:2], mode="bicubic", align_corners=False
        ).squeeze()

    depth_np = pred.cpu().numpy()
    depth_norm = cv2.normalize(depth_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    depth_u8 = depth_norm.astype(np.uint8)
    return Image.fromarray(depth_u8, mode='L')


def compute_depth(img_path: str, mode: str = "cv") -> Image.Image:
    """
    统一入口：根据mode选择深度估计实现。
    - "cv": 纯OpenCV梯度近似（无需网络/新依赖）。
    - "midas": MiDaS单目深度（依赖timm，可能在线下载模型权重）。
    当"midas"失败时，自动回退到"cv"。
    """
    mode = (mode or "cv").lower()
    if mode == "cv":
        return compute_depth_cv(img_path)
    if mode == "midas":
        try:
            return compute_depth_midas(img_path)
        except Exception:
            # 回退
            return compute_depth_cv(img_path)
    # 未知模式回退
    return compute_depth_cv(img_path)