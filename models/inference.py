from typing import Optional, Dict
import os
import glob
import torch
from PIL import Image

from utils.logger import get_logger
from utils.visualize import save_image
from utils.checkpoint import load_checkpoint
from data.dataset import SceneDataset
from models.generator import UNetGenerator


def _load_generator(cfg: Dict, ckpt_path: Optional[str]) -> torch.nn.Module:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g_args = cfg['model']['generator']
    netG = UNetGenerator(
        in_channels=g_args['in_channels'],
        out_channels=g_args['out_channels'],
        base_channels=g_args['base_channels'],
    )
    if ckpt_path and os.path.isfile(ckpt_path):
        state = load_checkpoint(ckpt_path)
        netG.load_state_dict(state['netG'])
    netG.eval().to(device)
    return netG


def run_inference(cfg: Dict, input_dir: Optional[str] = None, single_image: Optional[str] = None) -> None:
    logger = get_logger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dir = cfg['paths']['checkpoints']
    # 选择最新检查点
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'epoch_*.pt')))
    ckpt_path = ckpts[-1] if ckpts else None
    netG = _load_generator(cfg, ckpt_path)

    # 输入来源：
    # 1) single_image：单样本条件输入文件路径（图像或标签PNG）
    # 2) input_dir：批量条件目录（labels或图像目录）
    out_dir = cfg['paths']['outputs']
    os.makedirs(out_dir, exist_ok=True)

    def _process_one(path: str):
        # 使用SceneDataset的增强与标准化逻辑进行一致处理
        ds = SceneDataset(
            images_dir=os.path.dirname(path),
            labels_dir=None,
            image_size=cfg['dataset']['image_size'],
            random_crop=False,
            random_flip=False,
            color_jitter=False,
            condition_mode=cfg['dataset'].get('condition_mode', 'auto'),
        )
        # 通过索引找到该文件位置
        fname = os.path.basename(path)
        idx = ds.images.index(fname)
        cond_t, _ = ds[idx]
        cond_t = cond_t.unsqueeze(0).to(device)
        with torch.no_grad():
            fake = netG(cond_t)[0].cpu()
        save_image(fake, os.path.join(out_dir, f"infer_{os.path.splitext(fname)[0]}.png"))

    if single_image:
        logger.info(f"单样本推理: {single_image}")
        _process_one(single_image)
        return

    if input_dir:
        logger.info(f"批量推理目录: {input_dir}")
        files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        for f in files:
            _process_one(f)
        logger.info(f"已生成{len(files)}个结果至: {out_dir}")
        return

    # 默认：对验证集的labels或图像进行批量推理
    paths = cfg['paths']
    data_root = paths['data_root']
    val_labels = os.path.join(data_root, paths['labels_val']) if paths.get('labels_val') else None
    val_images = os.path.join(data_root, paths['images_val'])
    src_dir = val_labels if (val_labels and os.path.isdir(val_labels)) else val_images
    logger.info(f"默认批量推理目录: {src_dir}")
    files = sorted([os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    for f in files[:50]:  # 限制数量避免占用过多时间
        _process_one(f)
    logger.info(f"已生成{min(len(files),50)}个结果至: {out_dir}")