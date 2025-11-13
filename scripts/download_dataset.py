import os
import zipfile
import tarfile
from typing import Dict
import requests
from tqdm import tqdm

from utils.logger import get_logger


def _download_file(url: str, dest_path: str) -> None:
    logger = get_logger()
    logger.info(f"下载: {url} -> {dest_path}")
    # 某些环境可能出现证书校验问题，关闭verify以保证可下载
    with requests.get(url, stream=True, timeout=60, verify=False) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(dest_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def _extract(archive_path: str, extract_to: str) -> None:
    logger = get_logger()
    logger.info(f"解压: {archive_path} -> {extract_to}")
    os.makedirs(extract_to, exist_ok=True)
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as z:
            z.extractall(extract_to)
    elif archive_path.endswith('.tar') or archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:*') as t:
            t.extractall(extract_to)
    else:
        raise ValueError(f"不支持的压缩格式: {archive_path}")


def _validate_structure(cfg: Dict) -> None:
    """校验数据目录结构，如果labels不存在则提示将自动使用边缘条件。"""
    logger = get_logger()
    paths = cfg["paths"]
    data_root = paths["data_root"]
    images_train = os.path.join(data_root, paths["images_train"])
    images_val = os.path.join(data_root, paths["images_val"])
    labels_train = os.path.join(data_root, paths["labels_train"]) if paths.get("labels_train") else None
    labels_val = os.path.join(data_root, paths["labels_val"]) if paths.get("labels_val") else None

    for p in [images_train, images_val]:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"缺少目录: {p}")
        else:
            logger.info(f"已检测到图像目录: {p}")

    for p in [labels_train, labels_val]:
        if p is None:
            continue
        if not os.path.isdir(p):
            logger.warning(f"未检测到标签目录: {p}，将使用自动边缘图作为条件输入。")
        else:
            logger.info(f"已检测到标签目录: {p}")


def download_and_prepare(cfg: Dict) -> None:
    logger = get_logger()
    paths = cfg["paths"]
    data_root = paths["data_root"]
    urls = cfg.get("download", {}).get("urls", {})
    os.makedirs(data_root, exist_ok=True)

    # 下载COCO 2017图像
    train_zip = urls.get("images_train_zip")
    val_zip = urls.get("images_val_zip")
    ann_zip = urls.get("annotations_zip")

    if train_zip:
        train_zip_path = os.path.join(data_root, os.path.basename(train_zip))
        if not os.path.isfile(train_zip_path):
            _download_file(train_zip, train_zip_path)
        _extract(train_zip_path, os.path.join(data_root, 'images'))

    if val_zip:
        val_zip_path = os.path.join(data_root, os.path.basename(val_zip))
        if not os.path.isfile(val_zip_path):
            _download_file(val_zip, val_zip_path)
        _extract(val_zip_path, os.path.join(data_root, 'images'))

    # 注释（多边形/分割）可选下载
    if ann_zip:
        ann_zip_path = os.path.join(data_root, os.path.basename(ann_zip))
        if not os.path.isfile(ann_zip_path):
            _download_file(ann_zip, ann_zip_path)
        _extract(ann_zip_path, os.path.join(data_root, 'annotations'))

    # 若用户提供labels(语义分割PNG)，应放置于paths.labels_*目录
    # 校验结构
    _validate_structure(cfg)
    logger.info("数据集准备完成。")