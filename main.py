import argparse
import os
from typing import Optional

from utils.config import load_config
from utils.logger import get_logger
from scripts.download_dataset import download_and_prepare
from models.trainer import Trainer
from models.inference import run_inference


def ensure_dirs(cfg):
    paths = cfg["paths"]
    for key in ["checkpoints", "outputs", "data_root"]:
        os.makedirs(paths[key], exist_ok=True)


def cmd_download(cfg):
    logger = get_logger()
    ensure_dirs(cfg)
    download_cfg = cfg.get("download", {})
    if not download_cfg.get("enabled", True):
        logger.info("下载配置已禁用，跳过下载。")
        return
    download_and_prepare(cfg)


def cmd_train(cfg):
    logger = get_logger()
    ensure_dirs(cfg)
    trainer = Trainer(cfg)
    logger.info("开始训练...")
    trainer.train()


def cmd_infer(cfg, input_dir: Optional[str] = None, single_image: Optional[str] = None):
    ensure_dirs(cfg)
    run_inference(cfg, input_dir=input_dir, single_image=single_image)


def parse_args():
    parser = argparse.ArgumentParser(description="Scene Generation Pipeline")
    parser.add_argument("action", choices=["download", "train", "infer"], help="选择运行的动作")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--input_dir", default=None, help="推理时的条件输入目录（labels或图像目录）")
    parser.add_argument("--single_image", default=None, help="单样本条件输入文件路径")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.action == "download":
        cmd_download(cfg)
    elif args.action == "train":
        cmd_train(cfg)
    elif args.action == "infer":
        cmd_infer(cfg, input_dir=args.input_dir, single_image=args.single_image)


if __name__ == "__main__":
    main()