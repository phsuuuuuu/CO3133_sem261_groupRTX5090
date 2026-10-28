# Assignment 2 — Deep Learning on Large-Scale Data and Specialized Tasks
**Topic:** Semantic Segmentation on the Cityscapes Dataset  
**Course:** CO3133 — Deep Learning and Its Applications (Semester-261)  
**Instructor:** Dr. Le Thanh Sach  
**Group:** RTX5090 (Dang Thanh Phu · Cao Duc Thanh)  
**Milestone:** M2 — Draft Progress Checkpoint  

---

## 1. Project Overview & Deliverables
* **Task:** Semantic segmentation of urban driving scenes across 19 evaluation classes (with ignore index `255`).
* **Online Presentation:** [Assignment 2 GitHub Pages](https://phsuuuuuu.github.io/CO3133_sem261_groupRTX5090/assignment2/assignment2.html)
* **Milestone 2 Draft Page:** [M2 Draft Report & Findings](https://phsuuuuuu.github.io/CO3133_sem261_groupRTX5090/assignment2/m2.html)
* **AI Disclosure:** [AI_USAGE.md](../AI_USAGE.md)

---

## 2. Hardware & Software Environment
* **GPU Accelerator:** NVIDIA GeForce RTX 5070 Ti (16 GB GDDR7 VRAM, Blackwell Architecture)
* **OS:** Windows 11 (PowerShell)
* **Python Environment:** Python 3.12
* **Core Libraries:** PyTorch 2.13.0 + cu126, torchvision 0.28.0, numpy, Pillow, PyYAML, tqdm, cityscapesscripts, torchmetrics

---

## 3. Environment Setup & Data Preparation

### Virtual Environment Setup
```powershell
# Create virtual environment
python -m venv .venv

# Activate environment
./.venv/Scripts/activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset Preparation
1. Download Cityscapes dataset from [official portal](https://www.cityscapes-dataset.com/downloads/):
   * `gtFine_trainvaltest.zip` (241 MB)
   * `leftImg8bit_trainvaltest.zip` (11 GB)
2. Extract the archives into `assignment2/dataset/` such that:
   * `assignment2/dataset/gtFine_trainvaltest/gtFine/`
   * `assignment2/dataset/leftImg8bit_trainvaltest/leftImg8bit/`
3. Generate city-level partitions to prevent spatial leakage:
```powershell
python scripts/prepare_data.py
python scripts/create_splits.py
```
* **Train Set:** 2,481 images from 14 cities
* **Validation Set:** 494 images from 4 held-out cities (Erfurt, Jena, Tubingen, Zurich)
* **Test Set:** 500 images from 3 cities of the official validation set (Frankfurt, Lindau, Munster)

---

## 4. Exploratory Data Analysis (EDA)
Generate all exploratory analysis plots (region sizes, mask quality, co-occurrence, and split distribution):
```powershell
python -m src.eda
```
Generated figures are saved in `docs/figures/` and `assignment2/docs/figures/`:
* `a2_eda_region_size.png`: Region and instance size distribution per class.
* `a2_eda_mask_quality.png`: Spatial ignore distribution and boundary mask quality.
* `a2_eda_cooccurrence.png`: Scene complexity and conditional class co-occurrence matrix.
* `a2_eda_split_distribution.png`: Valid pixel percentage distribution across Train, Val, and Test splits.
* `a2_batch.png`: Preprocessed batch visual verification.

---

## 5. Model Training & Evaluation

### A. Baseline U-Net (From Scratch)
* **Architecture:** 4 downsampling stages (DoubleConv + MaxPool2d), bottleneck (512 channels), 4 upsampling stages (ConvTranspose2d) with skip connections.
* **Config:** `configs/unet_base.yaml` (Batch size: 8, Epochs: 10, Seed: 666).
```powershell
python -m src.train --config configs/unet_base.yaml
```

### B. Pretrained DeepLabV3-ResNet50 (Fine-Tuning in Progress)
* **Architecture:** ResNet-50 pre-trained on ImageNet-1K (frozen backbone) + Atrous Spatial Pyramid Pooling (ASPP) with dilation rates (6, 12, 18) + 19-class linear classifier.
* **Config:** `configs/deeplabv3.yaml` (Batch size: 8, Epochs: 10, Seed: 666, `drop_last: true`).
```powershell
python -m src.train --config configs/deeplabv3.yaml
```

### C. Validation & Evaluation
Evaluate any saved checkpoint on validation or test sets and export JSON metrics:
```powershell
# Evaluate Baseline U-Net
python -m src.evaluate --config configs/unet_base.yaml --checkpoint checkpoints/unet_base_1024x512_s42/best.pt --split val

# Evaluate Pretrained DeepLabV3
python -m src.evaluate --config configs/deeplabv3.yaml --checkpoint checkpoints/deeplabv3/exp_1/best.pt --split val --save_json checkpoints/deeplabv3/exp_1/metrics_val.json
```

---

## 6. Preliminary Comparison (M2 Milestone Results)

| Model | Pretrained Weights | Train Loss (Ep 10) | Val mIoU (Best) | Val Mean Dice (Best) |
| :--- | :--- | :---: | :---: | :---: |
| **U-Net Baseline** | None (Kaiming Scratch) | 0.3580 | **31.34%** | **38.20%** |
| **DeepLabV3-ResNet50** | ImageNet-1K (`IMAGENET1K_V1`) | 0.2262 | **56.57%** | **69.95%** |
| **Improvement** | &mdash; | *-0.1318* | **+25.23%** *(+80.5% rel)* | **+31.75%** *(+83.1% rel)* |

---

## 7. Next Steps for M3 (Final Milestone)
1. **Controlled Experiments:** Ablation on backbone unfreezing (freeze vs. partial vs. full fine-tuning), and impact of data augmentations.
2. **Error Analysis:** Detailed per-class breakdown (inspecting small classes like Traffic Lights vs. dominant classes like Road).
3. **Compute-Cost Analysis:** Inference latency (FPS), peak VRAM consumption, and FLOPs.
4. **Final Deliverables:** Final Report, presentation slides, and YouTube presentation video.
