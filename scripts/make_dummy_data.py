"""
快速生成少量假数据用于冒烟测试：
- 写入到 config.yaml 指定的 images/train2017 与 images/val2017
- 无标签目录时，训练会自动用Canny边缘作为条件
"""
import os
import random
from typing import Tuple
from PIL import Image, ImageDraw
import yaml


def load_cfg(path: str = "config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def rand_color() -> Tuple[int, int, int]:
    return tuple(random.randint(0, 255) for _ in range(3))


def make_image(size: int = 256) -> Image.Image:
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # 画一些随机几何图形，增加边缘与纹理
    for _ in range(10):
        x1, y1 = random.randint(0, size//2), random.randint(0, size//2)
        x2, y2 = random.randint(x1+10, size), random.randint(y1+10, size)
        if random.random() < 0.5:
            draw.rectangle([x1, y1, x2, y2], outline=rand_color(), width=random.randint(1, 4))
        else:
            draw.ellipse([x1, y1, x2, y2], outline=rand_color(), width=random.randint(1, 4))
    for _ in range(2000):  # 噪点
        x, y = random.randint(0, size-1), random.randint(0, size-1)
        img.putpixel((x, y), rand_color())
    return img


def main():
    cfg = load_cfg()
    paths = cfg["paths"]
    root = paths["data_root"]
    train_dir = os.path.join(root, paths["images_train"])
    val_dir = os.path.join(root, paths["images_val"])
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    size = cfg["dataset"]["image_size"]

    # 生成训练与验证图像
    for i in range(32):
        img = make_image(size)
        img.save(os.path.join(train_dir, f"dummy_{i:03d}.png"))
    for i in range(8):
        img = make_image(size)
        img.save(os.path.join(val_dir, f"dummy_{i:03d}.png"))
    print(f"写入训练集: {train_dir}")
    print(f"写入验证集: {val_dir}")


if __name__ == "__main__":
    main()