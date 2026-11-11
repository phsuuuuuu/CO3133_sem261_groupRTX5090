# Assignment 2 — U-Net vs. pretrained DeepLabV3-ResNet50 on Cityscapes

**Group RTX5090** · CO3133 Semester-261 · Instructor: Dr. Le Thanh Sach

- Web page: https://phsuuuuuu.github.io/CO3133_sem261_groupRTX5090/assignment2/assignment2.html
- Report: [Assignment2_M3_RTX5090.pdf](../docs/reports/Assignment2_M3_RTX5090.pdf) · Slides: [Assignment2_Slides_RTX5090.pdf](../docs/reports/Assignment2_Slides_RTX5090.pdf)
- Video: https://youtu.be/TAB6kAfnlxA · AI usage: [AI_USAGE.md](../AI_USAGE.md)

## Environment
Windows 11, Python 3.12, PyTorch 2.11 + torchvision 0.26 (CUDA 13.0). Dependency versions are in `requirements.txt`.
Hardware: RTX 5070 Ti (U-Net, WCE, CE+Dice) and Colab RTX PRO 6000 (DeepLabV3 frozen / partial / full, batch size 16, see `colab.ipynb`).

```powershell
cd assignment2
python -m venv .venv; ./.venv/Scripts/activate.ps1
pip install -r requirements.txt
```

## Data
Download `gtFine_trainvaltest.zip` and `leftImg8bit_trainvaltest.zip` from https://www.cityscapes-dataset.com/downloads/ and extract them into `dataset/`.
The split is by city (**seed 666**, no randomness): 2,481 train / 494 val / 500 test images, saved in `splits/*.txt`.

```powershell
python scripts/prepare_data.py     # raw IDs -> 19 trainIds
python scripts/pre_resize.py       # resize to 1024x512
python scripts/create_splits.py    # city split
python -m src.eda                  # EDA figures
```

## Train and evaluate
Each run is one YAML file in `configs/` (seed 666, AdamW, AMP).

```powershell
python -m src.train --config configs/unet_final.yaml                  # U-Net baseline (40 epochs)
python -m src.train --config configs/deeplabv3.yaml                   # DeepLabV3, frozen backbone
python -m src.train --config configs/resnet_layer3_4_finetune.yaml    # DeepLabV3, partial fine-tune
python -m src.train --config configs/resnet50_full_deeplabv3.yaml     # DeepLabV3, full fine-tune
python -m src.train --config configs/dlv3_wce.yaml                    # partial + weighted CE
python -m src.train --config configs/dlv3_cedice.yaml                 # partial + CE + Dice (final model)

python -m src.evaluate --config configs/dlv3_cedice.yaml --checkpoint checkpoints/dlv3_cedice_1024x512_s666/best.pt --split test
```

Error analysis, cost and demo:

```powershell
python scripts/m3_error_qual.py    # error analysis + qualitative figures
python scripts/benchmark_cost.py   # params, FLOPs, latency
python demo/demo.py                # segment images in demo/input/
```

Per-epoch logs go to `checkpoints/<run>/log.csv` and test metrics to `checkpoints/<run>/metrics_test.json`.

## Checkpoints
The `.pt` files are not committed (they are too large). To recreate them, run the training commands above. Each run saves:

- `best.pt`: the epoch with the best **validation mIoU** (this is the checkpoint used for the test results)
- `last.pt`: the last epoch

Note: the three DeepLabV3 runs (frozen / partial / full) were trained on Colab with batch size 16, while the YAML files use batch size 8 (local GPU).

## Main results (test set, 500 images)
| Model | mIoU (%) | Dice (%) | Pixel acc (%) |
|---|---|---|---|
| U-Net, CE | 49.35 | 60.86 | 91.65 |
| DeepLabV3 frozen, CE | 57.01 | 70.21 | 91.55 |
| DeepLabV3 full, CE | 61.10 | 73.83 | 93.05 |
| DeepLabV3 partial, CE | 66.53 | 78.48 | 94.07 |
| DeepLabV3 partial, weighted CE | 67.21 | 79.10 | 93.90 |
| DeepLabV3 partial, CE + Dice (final) | 68.09 | 79.81 | 93.92 |
