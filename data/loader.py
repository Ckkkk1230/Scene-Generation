from typing import Dict, Tuple
import os
from torch.utils.data import DataLoader
from data.dataset import SceneDataset


def create_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    paths = cfg["paths"]
    data_root = paths["data_root"]
    train_images = os.path.join(data_root, paths["images_train"])
    val_images = os.path.join(data_root, paths["images_val"])
    train_labels = os.path.join(data_root, paths["labels_train"]) if paths.get("labels_train") else None
    val_labels = os.path.join(data_root, paths["labels_val"]) if paths.get("labels_val") else None

    ds_args = cfg["dataset"]
    train_ds = SceneDataset(
        images_dir=train_images,
        labels_dir=train_labels,
        image_size=ds_args["image_size"],
        random_crop=ds_args["augment"]["random_crop"],
        random_flip=ds_args["augment"]["random_flip"],
        color_jitter=ds_args["augment"]["color_jitter"],
    )
    val_ds = SceneDataset(
        images_dir=val_images,
        labels_dir=val_labels,
        image_size=ds_args["image_size"],
        random_crop=False,
        random_flip=False,
        color_jitter=False,
    )

    train = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"], drop_last=True)
    val = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"], drop_last=False)
    return train, val