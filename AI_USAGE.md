# AI Usage Log - Group RTX5090 - CO3133, Semester-261

## Course Website Setup

### Entry CW-1

- **Tool:** ChatGPT
- **Used by:** Dang Thanh Phu
- **Task:** Week 3 — Setting up the course website (HTML/CSS, navigation links, GitHub Pages, disclosure wording)
- **Prompt summary:** Asked to create a simple CO3133 website for Group RTX5090 with a homepage and three assignment report pages, preserving existing repository content.
- **AI contribution:** Suggested HTML/CSS page structure, navigation links, GitHub Pages setup steps, and disclosure wording.
- **Student verification:** Adjusted group details, repository URLs, and page paths; checked all local links in the browser. Cross-checked with the course handbook, repository files, and official GitHub Pages documentation.
- **Affected files/sections:** `index.html`; `assignment1.html`; `assignment2.html`; `assignment3.html`; `AI_USAGE.md`
- **Responsible member:** Dang Thanh Phu

---

## Assignment 1

### Entry A1-1

- **Tool:** ChatGPT (OpenAI, GPT-5.6 Sol)
- **Used by:** Dang Thanh Phu
- **Task:** A1 M1 — Data preparation, EDA, and training pipeline
- **Prompt summary:** Asked about DataLoader function prototypes, required libraries and syntax, using `argparse` in `train.py`, saving per-epoch training/validation results (loss, accuracy, macro-F1) to CSV, explaining/fixing PyTorch function syntax, and an example of plotting.
- **AI contribution:** Explained function prototypes, library usage, and syntax; gave examples for the CLI argument parser, CSV logging, and plotting.
- **Student verification:** Read and understood the explanations before use; wrote and adapted the code personally. Tested locally by checking dataset sizes, preprocessing output, EDA figures, training execution, checkpoints, and saved CSV files. Compared with PyTorch documentation.
- **Affected files/sections:** `src/train.py` (direct); `src/data.py`, `src/eda.py` (indirect)
- **Responsible member:** Dang Thanh Phu

### Entry A1-2

- **Tool:** Gemini 3.8 Flash
- **Used by:** Cao Duc Thanh
- **Task:** A1 M1 — Plotting results from CSV
- **Prompt summary:** Asked how to use matplotlib to plot loss, F1 score, and accuracy from `linear_history.csv` and `mlp_history.csv` to compare the Linear and MLP models. Log: https://share.gemini.google/bx1k8Hgs9Epo
- **AI contribution:** Taught matplotlib usage and syntax with example plotting code.
- **Student verification:** Wrote the code, studied each function, and updated parameters and paths to match the assignment. Ran the script and checked that `training_curves.png` showed the expected graph.
- **Affected files/sections:** `src/plot_curves.py`
- **Responsible member:** Cao Duc Thanh

### Entry A1-3

- **Tool:** Gemini 3.8 Flash
- **Used by:** Cao Duc Thanh
- **Task:** A1 M1 — Editing `assignment1.html` for the M1 draft
- **Prompt summary:** Asked how to edit and add details to the HTML page of the A1 M1 draft (no prior HTML experience). Log: https://share.gemini.google/TgX9DyJ9Nqcz
- **AI contribution:** Explained how to edit the HTML page.
- **Student verification:** Edited the file to match the assignment and checked the deployed page at https://phsuuuuuu.github.io/CO3133_sem261_groupRTX5090/assignment1/assignment1.html
- **Affected files/sections:** `docs/assignment1/assignment1.html`
- **Responsible member:** Cao Duc Thanh

### Entry A1-4

- **Tool:** Gemini 3.8 Flash
- **Used by:** Cao Duc Thanh
- **Task:** A1 M2 — Fixing LSTM errors
- **Prompt summary:** Asked to explain and fix runtime errors in an LSTM implemented following the D2L book. Log: https://share.gemini.google/wVjtB3LHu1b6
- **AI contribution:** Explained the errors and suggested fixes for the PyTorch code.
- **Student verification:** Followed the explanations, fixed the errors, and discussed alternatives with the AI. Ran the code successfully and confirmed accuracy reached ~90% in `results/figures/training_curves.png`.
- **Affected files/sections:** `src/models/lstm.py`
- **Responsible member:** Cao Duc Thanh

### Entry A1-5

- **Tool:** Gemini 3.8 Flash
- **Used by:** Cao Duc Thanh
- **Task:** A1 M2 — Implementing the Transformer
- **Prompt summary:** Asked how to implement the Transformer architecture in code. Log: https://share.gemini.google/X2LdCo2AsJpd
- **AI contribution:** Explained how to implement the Transformer and discussed design options for the model.
- **Student verification:** Implemented the model following the AI's explanation and the D2L book. Ran the code successfully and confirmed accuracy reached ~80% with ViT in `results/figures/training_curves.png`.
- **Affected files/sections:** `src/models/transformer.py`
- **Responsible member:** Cao Duc Thanh

### Entry A1-6

- **Tool:** Gemini 3.8 Flash
- **Used by:** Cao Duc Thanh
- **Task:** A1 M2 — Writing the report
- **Prompt summary:** Asked to help organize ideas, suggest content covering the handbook requirements, and check the report. Log: https://share.gemini.google/phX1ifjJFugp
- **AI contribution:** Organized ideas, suggested content, and reviewed the draft.
- **Student verification:** Wrote the report based on the suggestions, then checked it against the Assignment 1 files.
- **Affected files/sections:** Assignment 1 Report
- **Responsible member:** Cao Duc Thanh

### Entry A1-7

- **Tool:** GPT-6
- **Used by:** Dang Thanh Phu
- **Task:** A1 M2 — Writing the report
- **Prompt summary:**
  - *LSTM architecture and parameters:* How the LSTM processes Fashion-MNIST images as 28 rows of 28 features; the parameter-count formula for hidden dimension 2,048; why PyTorch has separate input-to-hidden and hidden-to-hidden biases.
  - *Transformer architecture:* Splitting 28 × 28 images into 49 patches, 32-dim embeddings, classification token, positional embeddings, Q/K/V projections, four attention heads, residual connections, feed-forward layers, tensor dimensions, and the 28,202 trainable parameters.
  - *Training behavior and checkpoint selection:* Interpreting training/validation curves, overfitting/underfitting, and why the best validation macro-F1 checkpoint may differ from the lowest validation loss epoch.
  - *Results and limitations:* Interpreting accuracy, macro-F1, training/inference time, and parameter counts of the five architectures; the Transformer improvement from 10 to 30 epochs; limitations of comparing models under unequal parameter and compute budgets.
- **AI contribution:** Explained LSTM/Transformer concepts, PyTorch LSTM biases, data normalization, training procedures, and evaluation metrics; helped structure and refine technical explanations in the report.
- **Student verification:** Wrote the report from the explanations, then compared all statements with the recorded experimental results and Assignment 1 files to make sure the data was correct.
- **Affected files/sections:** Report Sec. 2.1, 2.5–2.8, 3.1–3.2, 3.6
- **Responsible member:** Dang Thanh Phu

### Entry A1-8

- **Tool:** GPT-6
- **Used by:** Dang Thanh Phu
- **Task:** A1 M2 — Website organization and report integration
- **Prompt summary:** Asked to restructure the Assignment 1 and Assignment 3 folders/pages following Assignment 2's format, add the A1 M2 report to the website, and leave a placeholder for the YouTube presentation link.
- **AI contribution:** Reorganized the pages using Assignment 2's layout, integrated the M2 report viewer, and added presentation link placeholders.
- **Student verification:** Used the existing Assignment 2 page as the layout reference; kept A1 draft content in a separate M2 page and linked the M2 viewer to the existing report PDF. Checked local link destinations, milestone cards, page titles, placeholders, and formatting against files in `docs/assignment1`, `docs/assignment2`, `docs/assignment3`, and `docs/reports/Assignment1_M3_RTX5090.pdf`.
- **Affected files/sections:** Assignment 1 and Assignment 3 landing pages; A1 M2 draft page; A1 M3 report viewer
- **Responsible member:** Dang Thanh Phu

---

## Assignment 2

### Entry A2-1

- **Tool:** Gemini 3.8 Flash
- **Used by:** Cao Duc Thanh
- **Task:** A2 M1 — Understanding the requirements and finding a dataset
- **Prompt summary:** Asked to explain the Assignment 2 specification and suggest suitable datasets. Log: https://gemini.google.com/app/6c45defefd0a4e2a
- **AI contribution:** Explained the specification and suggested candidate datasets.
- **Student verification:** Looked up each suggested dataset online:
  - https://www.kaggle.com/datasets/bardiaardakanian/voc0712
  - https://www.kaggle.com/datasets/sovitrath/voc-2012-segmentation-data
  - https://www.cityscapes-dataset.com/
  - https://www.kaggle.com/datasets/tanlikesmath/the-oxfordiiit-pet-dataset
  - https://www.kaggle.com/datasets/pengcw1/market-1501
  - https://www.kaggle.com/datasets/whurobin/dukemtmcreid
  - https://www.kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset
- **Affected files/sections:** Assignment 2 dataset selection
- **Responsible member:** Cao Duc Thanh

### Entry A2-2

- **Tool:** Gemini 3.8 Flash
- **Used by:** Cao Duc Thanh
- **Task:** A2 M1 — Research and proposal writing
- **Prompt summary:** Asked about the datasets, pretrained and from-scratch models, the training process, how to write a proposal, and whether the proposal met the specification. Log: https://share.gemini.google/DGWCyHXayaPi
- **AI contribution:** Provided background on datasets, models, and training; gave proposal-writing guidance and reviewed the proposal against the specification.
- **Student verification:** Checked the information against:
  - https://arxiv.org/pdf/2105.15203
  - https://www.cityscapes-dataset.com/
  - https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
  - https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/
  - https://wiki.cloudfactory.com/docs/mp-wiki/model-architectures/deeplabv3
- **Affected files/sections:** Assignment 2 Proposal
- **Responsible member:** Cao Duc Thanh

### Entry A2-3

- **Tool:** Gemini 3.8 Flash
- **Used by:** Cao Duc Thanh
- **Task:** A2 M2 — How to implement the data.py
- **Prompt summary:** Asked about how to implement src/data.py and explain what the file did
- **AI contribution:** Give the sample code and explain what each line of code means and how it connects to the problem
- **Student verification:** implemnt the code and try to understand every line of code, run it with the model U-net
- **Affected files/sections:** `assignment2/src/data.py`
- **Responsible member:** Cao Duc Thanh

### Entry A2-3

- **Tool:** Gemini 3.8 Flash
- **Used by:** Cao Duc Thanh
- **Task:** A2 M2 — Debugging a DeepLabV3 batch-size crash and optimizing the DataLoader
- **Prompt summary:** Asked to fix `ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 256, 1, 1])` in DeepLabV3 training on Cityscapes.
- **AI contribution:** Explained why single-sample batches break BatchNorm in ASPPPooling; suggested `drop_last=True` and a batch-size safeguard.
- **Student verification:** Read the PyTorch `nn.BatchNorm2d` docs and the torchvision DeepLabV3 implementation; confirmed 2,481 training samples with batch size 8 leave a final batch of 1. Set `drop_last=True` in `src/data.py`, added a single-sample batch skip in `src/train.py`, and ran the full 10 epochs on an RTX 5070 Ti without errors.
- **Affected files/sections:** `assignment2/src/data.py`; `assignment2/src/train.py`; `assignment2/configs/deeplabv3.yaml`
- **Responsible member:** Cao Duc Thanh

### Entry A2-4

- **Tool:** Gemini 3.8 Flash
- **Used by:** Cao Duc Thanh
- **Task:** A2 M2 — Learn and implement the U-net
- **Prompt summary:** Ask about how U-net works and how each layer contributes to the whole model, and how to implement U-net using pytorch
- **AI contribution:** Give the explaination of U-net and produce the sample code for U-net
- **Student verification:** Learn and try to understand the model. Then implement the U-net and compare the code with the theory.
- **Affected files/sections:** `src/models/unet.py`
- **Responsible member:** Cao Duc Thanh

### Entry A2-5

- **Tool:** Gemini 3.8 Flash
- **Used by:** Cao Duc Thanh
- **Task:** A2 M2 — Preliminary metrics comparison and M2 draft webpage scaffolding
- **Prompt summary:** Asked to plot comparative training curves (Loss, mIoU, Dice) from the U-Net and DeepLabV3 `log.csv` files, and to format the M2 draft webpage with EDA findings, model architectures, and preliminary result tables.
- **AI contribution:** Produced plotting code for the comparison curves and scaffolded the M2 draft page following Handbook Sec. 7.5 and Sec. 22.
- **Student verification:** Confirmed all numbers (U-Net best mIoU 31.34%, DeepLabV3 best mIoU 56.57%) against `assignment2/checkpoints/unet_base_1024x512_s42/log.csv` and `assignment2/checkpoints/deeplabv3/exp_1/log.csv`; checked against `handbook-ene.pdf` Sec. 7.5. Verified image paths, responsive layout, and navigation on GitHub Pages.
- **Affected files/sections:** `docs/assignment2/m2.html`; `docs/assignment2/assignment2.html`; `docs/figures/a2_training_curves.png`; `assignment2/README.md`
- **Responsible member:** Cao Duc Thanh

---

## Assignment 3

### Entry A3-1

- **Declaration:** The group declares that no generative AI tool was used in this assignment to date (Course Project Handbook Section 5.4).
