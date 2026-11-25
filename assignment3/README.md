# Assignment 3 — Multimodal Depth Completion on KITTI (M2 Draft)

**Group RTX5090** · CO3133 Semester-261 · Instructor: Dr. Le Thanh Sach

- Web: [Assignment 3](https://phsuuuuuu.github.io/CO3133_sem261_groupRTX5090/assignment3/assignment3.html) · [M2 Draft](https://phsuuuuuu.github.io/CO3133_sem261_groupRTX5090/assignment3/m2.html) · Proposal: [Assignment3_M1_RTX5090.pdf](../docs/reports/Assignment3_M1_RTX5090.pdf) · AI log: [AI_USAGE.md](../AI_USAGE.md)

## Environment
Python 3.12, PyTorch `2.11.0+cu130`, NVIDIA RTX 5070 Ti (local M2) / Colab A100 (M1, M3). See [requirements.txt](requirements.txt).

```powershell
cd assignment3
python -m venv .venv; ./.venv/Scripts/activate.ps1
pip install -r requirements.txt
```

## Data
KITTI Depth Completion + raw RGB (`image_02`), cropped to `352x1216`, split by drive (no overlap): **8,237 train / 1,023 val / 1,000 test**.
Prepared via `src/colab/kitti_download_colab.ipynb` and `src/colab/kitti_prepare_dataset_colab.ipynb` into `data/processed/{train,val,test}/{rgb,lidar,gt}/`.

## Train & Evaluate
Shared defaults in `configs/base.yaml` (ResNet-34 ImageNet-pretrained encoder + U-Net decoder, AdamW, cosine LR, AMP, masked RMSE loss, seed `666`).

```powershell
python -m src.train --config configs/m2_lidar.yaml
python -m src.evaluate --ckpt checkpoints/m2_lidar_base_352x1216_s42/best.pt --split test
```

- `m1_rgb` and `m3_split` (20 epochs, `batch_size=40`) were trained on Colab via `src/colab/training_3.ipynb`.
- Checkpoints (`.pt`) are excluded from Git (stored on Google Drive); `best.pt` = lowest `val_rmse`, logs in `checkpoints/<run>/log.csv`, test metrics in `results/`.

## Preliminary Results (M2 Draft, Validation `best.pt`)
| Model | Epochs (Best) | Val RMSE (m) | Val MAE (m) | Test RMSE / MAE (m) |
|---|---:|---:|---:|---:|
| M1 RGB-only | 20 (15) | 2.833 | 1.152 | Pending |
| M2 LiDAR-only | 16 (10) | **0.927** | **0.308** | 1.982 / 0.494 |
| M3 Early fusion (split stem) | 20 (15) | 1.204 | 0.447 | Pending |
