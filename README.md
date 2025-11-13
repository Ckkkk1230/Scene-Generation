# Complete Scene Generation Solution (Open Source Dataset)

This project implements conditional scene generation based on open-source datasets (default: COCO 2017), demonstrating an edge-to-image pix2pix pipeline. The code features modular design, including data download, preprocessing, model training, inference, and visualization, accompanied by `config.yaml` configuration file with type annotations and concise docstrings.

## Overview
- **Dataset Preparation**: Automatically downloads and extracts COCO 2017 dataset (annotations optional), with directory structure validation.
- **Data Preprocessing**:
  - Implements `torch.utils.data.Dataset` supporting label PNGs as conditional input; automatically generates Canny edges when labels are unavailable.
  - Data augmentation and normalization (Resize, Flip, Normalize).
- **Model & Training**:
  - Generator: U-Net (pix2pix style).
  - Discriminator: PatchGAN.
  - Loss: Adversarial loss + L1 reconstruction loss, with checkpoint saving and validation visualization.
- **Inference Deployment**:
  - Supports batch and single-sample inference, saving results to `outputs/` directory.
  - Allows specifying input directory or single image path.

## Project Structure
.

├── config.yaml

├── requirements.txt

├── main.py

├── scripts/

│   └── download_dataset.py

├── data/

│   ├── dataset.py

│   └── loader.py

├── models/

│   ├── generator.py

│   ├── discriminator.py

│   ├── trainer.py

│   └── inference.py

└── utils/

├── config.py

├── logger.py

├── checkpoint.py

└── visualize.py

## Environment Setup

pip install -r requirements.txt

## Configuration
Edit `config.yaml`:
- `paths.data_root`: Root data directory (default: `./data/coco2017`).
- `paths.images_train/val`: Training/validation image directories (automatically extracted to `images/train2017`, `images/val2017`).
- `paths.labels_train/val`: Optional label PNG directories; uses auto-generated edges if missing.
- `dataset.image_size`: Input size (default: 256).
- `dataset.cond_channels`: Number of conditional channels (1 for edges, expandable for one-hot semantic maps).
- `train.*`: Training hyperparameters.

## Usage

1. **Download and Prepare Dataset** (default: COCO 2017):

python main.py download --config config.yaml

2. **Start Training**:

python main.py train --config config.yaml

Saves checkpoints every `save_every` epochs and generates validation samples in `outputs/` directory.

3. **Run Inference**:
- Single sample:

python main.py infer --config config.yaml --single_image path/to/your/cond.png

- Batch processing:

python main.py infer --config config.yaml --input_dir path/to/your/conds

If unspecified, performs batch inference on validation directory (prioritizes labels, then images).

## Adaptation for Cityscapes/COCO-Stuff
- **Cityscapes**: Requires registration/download; place label PNGs in `paths.labels_*` directories for conditional generation.
- **COCO-Stuff**: Provides pixel-level semantic labels; download and place in `labels/train2017`, `labels/val2017`, ensuring filename consistency with images (extensions may differ).
- For multi-class one-hot semantic conditions: Increase `dataset.cond_channels` and modify `SceneDataset` to convert labels to one-hot format matching model input.

## Design Trade-offs
- For reliability and generality, uses edge maps as fallback when labels are unavailable; real semantic PNGs yield better results.
- Generator/discriminator follow classic pix2pix architecture, facilitating reproducibility and extension to other conditions (depth, boundaries, layout, etc.).

## License & Data Sources
- COCO dataset source: https://cocodataset.org/
- For academic/R&D demonstration only. Specific data licenses apply as per dataset official terms.