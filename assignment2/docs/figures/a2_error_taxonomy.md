# Quantitative Error Analysis & Taxonomy (Task A3.5)

**Project:** Semantic Segmentation on Cityscapes (Course CO3133 - Group RTX5090)  
**Evaluated Model:** DeepLabV3-ResNet50 (Partial Fine-Tune layer 3+4)  
**Evaluation Set:** Official Validation Set / Local Test Split (500 images)  
**Generated Files:** `docs/figures/a2_confusion_matrix.png`, `docs/figures/a2_boundary_error.png`

---

## 1. Top-10 Pairwise Semantic Confusions

The table below details the top-10 most frequent off-diagonal confusion pairs $(i ightarrow j)$ sorted by absolute misclassified pixel counts:

| Rank | Confusion Pair (True $\rightarrow$ Pred) | Misclassified Pixels | Error Rate (% True Class) | Representative Test Sample |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `building` → `vegetation` | 2,375,055 | 4.73% | `lindau/lindau_000014_000019` |
| 2 | `road` → `sidewalk` | 1,925,980 | 2.23% | `lindau/lindau_000036_000019` |
| 3 | `pole` → `building` | 689,793 | 20.34% | `lindau/lindau_000050_000019` |
| 4 | `sidewalk` → `road` | 624,089 | 5.04% | `frankfurt/frankfurt_000001_017459` |
| 5 | `pole` → `vegetation` | 538,871 | 15.89% | `frankfurt/frankfurt_000001_041664` |
| 6 | `fence` → `building` | 473,612 | 25.21% | `frankfurt/frankfurt_000001_012738` |
| 7 | `vegetation` → `building` | 387,629 | 0.98% | `frankfurt/frankfurt_000001_021406` |
| 8 | `road` → `car` | 324,648 | 0.38% | `frankfurt/frankfurt_000001_060906` |
| 9 | `fence` → `wall` | 290,877 | 15.48% | `lindau/lindau_000011_000019` |
| 10 | `wall` → `building` | 280,276 | 16.70% | `lindau/lindau_000045_000019` |

---

## 2. Four-Group Error Taxonomy Table

| Error Group | Description & Scope | Quantitative Frequency / Rate | Concrete Examples | Hypothesized Root Cause | Actionable Improvement Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Group 1: Semantic & Geometric Confusion** | Visual and contextual resemblance between functionally adjacent semantic classes (`road` $\leftrightarrow$ `sidewalk`, `wall` $\leftrightarrow$ `fence`, `rider` $\leftrightarrow$ `person`, `truck` $\leftrightarrow$ `bus`). | **38.4% of total misclassified pixels.** Highest pair: `terrain` $\rightarrow$ `vegetation` (4.73%) and `sidewalk` $\rightarrow$ `road` (5.04%). | `frankfurt_000001_031266` (rider/person), `lindau_000018_000019` (wall/fence), `lindau_000047_000019` (road/terrain). | Shared surface texture, identical material reflectance (asphalt vs. stone), and lack of 3D curb height clues in 2D monocular RGB images. | (1) Boundary-aware edge supervision (e.g., Active Boundary Loss). (2) Enforcing spatial topological adjacency priors. |
| **Group 2: Boundary Localization Errors** | Pixel errors localized within a narrow band of $k=3$ pixels along object transitions. | **Error rate on boundary band: 32.54%** vs. **4.01% in interior** (8.1x higher density). **54.8% of all errors** occur on boundary transitions. | `frankfurt_000001_061763` (boundary contour of parked cars), `munster_000008_000019` (sidewalk curb). | Spatial downsampling (stride 8/16 in deep backbones) blurs high-frequency edge gradients; bilinear interpolation during upsampling causes interpolation blur. | (1) Test-Time Augmentation (TTA) multi-scale inference. (2) PointRend refinement module for boundary pixel selection. |
| **Group 3: Small & Slender Structure Attenuation** | Thin objects and distant instances with area $< 32\times 32$ pixels (`pole`, `traffic light`, `traffic sign`). | **Small objects recall: 45.6%** vs. Large objects recall: 95.2%. Slender classes mean recall: **63.2%** (Pole IoU: 39.7%, Sign IoU: 60.7%). | `frankfurt_000001_061763` (broken utility poles), `frankfurt_000001_031266` (distant traffic lights). | High dilation rates (atrous rate 12, 24) skip slender 1-2 pixel structures; successive pooling discards fine details early. | (1) High-resolution skip connections (HRNet or U-Net feature fusion). (2) Multi-scale patch-based training. |
| **Group 4: Extreme Class Imbalance & Rare Omission** | Failure or false negatives on classes occupying $< 1\%$ total pixels (`motorcycle`, `rider`, `train`). | **10-rare classes mIoU: 37.02%** compared to **89.97%** for top-3 frequent classes (`road`, `building`, `vegetation`). Gap: **53.0% mIoU**. | `munster_000058_000019` (bicycle cluster), `lindau_000047_000019` (rural unpaved road). | Cross-Entropy loss is dominated by frequent background classes (75.6% of pixels), causing optimization gradients to suppress rare object categories. | (1) Class-weighted Cross-Entropy ($w_c = 1 + \log(\frac{\max N}{N_c})$). (2) Combined CE + Soft Dice loss (tested in Experiment E2). |

---

## 3. Per-Class Quantitative Benchmark

| Class ID | Class Name | Frequency Tier | Precision (%) | Recall (%) | IoU (%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | `road` | Frequent (>10%) | 99.03% | 97.11% | 96.19% |
| 1 | `sidewalk` | Moderate (1-10%) | 80.00% | 89.71% | 73.27% |
| 2 | `building` | Frequent (>10%) | 94.54% | 91.28% | 86.71% |
| 3 | `wall` | Rare (<1%) | 44.49% | 54.84% | 32.56% |
| 4 | `fence` | Rare (<1%) | 44.76% | 43.59% | 28.34% |
| 5 | `pole` | Moderate (1-10%) | 72.56% | 46.70% | 39.69% |
| 6 | `traffic light` | Rare (<1%) | 59.16% | 68.38% | 46.45% |
| 7 | `traffic sign` | Rare (<1%) | 76.66% | 74.52% | 60.74% |
| 8 | `vegetation` | Frequent (>10%) | 90.36% | 95.92% | 87.01% |
| 9 | `terrain` | Moderate (1-10%) | 73.29% | 62.08% | 50.63% |
| 10 | `sky` | Moderate (1-10%) | 95.11% | 96.57% | 92.01% |
| 11 | `person` | Moderate (1-10%) | 80.12% | 81.56% | 67.83% |
| 12 | `rider` | Rare (<1%) | 54.54% | 56.92% | 38.60% |
| 13 | `car` | Moderate (1-10%) | 92.34% | 96.35% | 89.22% |
| 14 | `truck` | Rare (<1%) | 67.47% | 32.64% | 28.20% |
| 15 | `bus` | Rare (<1%) | 66.99% | 43.38% | 35.74% |
| 16 | `train` | Rare (<1%) | 30.82% | 50.72% | 23.72% |
| 17 | `motorcycle` | Rare (<1%) | 65.52% | 15.02% | 13.92% |
| 18 | `bicycle` | Rare (<1%) | 69.15% | 85.55% | 61.92% |

---
*Generated automatically by `src/error_analysis.py` on 2026-11-09.*
