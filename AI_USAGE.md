# AI Usage Log - Group RTX5090 - CO3133, Semester-261

## Course Website Setup

### Entry CW-1

- **Tool name / version:** ChatGPT
- **Member who used it:** Dang Thanh Phu
- **Time / development stage:** Week 3 — Course website setup
- **Purpose:** Assistance with HTML/CSS, navigation links, GitHub Pages setup, and disclosure wording.
- **Affected assignment section(s):** `index.html`, `assignment1.html`, `assignment2.html`, `assignment3.html`, and `AI_USAGE.md`
- **Representative prompt / prompt-log link:** “Create a simple CO3133 website for Group RTX5090 with a homepage and three assignment report pages, preserving existing repository content.”
- **How AI output was edited and verified:** Group details, repository URLs, and page paths were adjusted. Local links were checked and verified in browser.
- **Sources used for verification:** Course handbook, repository files, and official GitHub Pages documentation.
- **Member responsible for final verification:** Dang Thanh Phu

---

## Assignment 1

### Entry A1-1

- **Tool name / version:** ChatGPT (OpenAI, GPT-5.6 Sol)
- **Member who used it:** Dang Thanh Phu
- **Time / development stage:** Assignment 1 M1 — Data preparation, EDA, and training pipeline
- **Purpose:** Asked about function prototypes for DataLoader, library needed and syntax, writing CLI argument parser in `train.py`, and saving training/validation epoch results into CSV.
- **Affected assignment section(s):** Directly: `src/train.py`; Indirectly: `src/data.py`, `src/eda.py`
- **Representative prompt / prompt-log link:**
  - "What function should I create for the DataLoader?"
  - "How do I use `argparse` in `train.py`?"
  - "How do I save training results from each epoch into a CSV file?"
  - "What does this PyTorch function do?"
  - "Can you explain/fix the syntax of this function?"
  - "How do I save the training and validation results from each epoch, such as loss, accuracy, and macro-F1, into a CSV file?"
  - "guide me an example about how to plot something"
- **How AI output was edited and verified:** Read and understood explanations and syntax before using. Wrote and adapted code myself and tested locally. Checked dataset sizes, preprocessing output, EDA figures, training execution, checkpoints, and saved CSV files to ensure correct operation.
- **Sources used for verification:** PyTorch documentation and local test runs.
- **Member responsible for final verification:** Dang Thanh Phu

### Entry A1-2

- **Tool name / version:** Gemini 3.8 Flash
- **Member who used it:** Cao Duc Thanh
- **Time / development stage:** Assignment 1 M1 — Plot the result from csv
- **Purpose:** Our team needed to plot a graph from linear_history.csv and mlp_history.csv to compare the loss, f1 score, and accuracy between MLP and Linear Model. I didn't know how to use matplotlib and its syntaxes so I needed gemini teach me how to use the library.
- **Affected assignment section(s):** `src/plot_curves.py`
- **Representative prompt / prompt-log link:** https://share.gemini.google/bx1k8Hgs9Epo
- **How AI output was edited and verified:** I coded and tried to understand the function and update some parameters and paths to match with out assignments. I then ran the file and see that the graph is like what I need.
- **Sources used for verification:** `training_curves.png`
- **Member responsible for final verification:** Cao Duc Thanh

### Entry A1-3

- **Tool name / version:** Gemini 3.8 Flash
- **Member who used it:** Cao Duc Thanh
- **Time / development stage:** Assignment 1 M1 — Edit the assignment1.html
- **Purpose:** I wanted to edit and add some details about the draft M1 of assignment 1. I haven't coded HTML before so I needed gemini teach me how to edit it.
- **Affected assignment section(s):** `docs/assignment1/assignment1.html`
- **Representative prompt / prompt-log link:** https://share.gemini.google/TgX9DyJ9Nqcz
- **How AI output was edited and verified:** I edited that file to match with our assignments. I go to github.io to check.
- **Sources used for verification:** https://phsuuuuuu.github.io/CO3133_sem261_groupRTX5090/assignment1/assignment1.html
- **Member responsible for final verification:** Cao Duc Thanh

### Entry A1-4

- **Tool name / version:** Gemini 3.8 Flash
- **Member who used it:** Cao Duc Thanh
- **Time / development stage:** Assignment 1 M2 — Fixing LSTM errors
- **Purpose:** I have read the D2L book and coded LSTM like it, but when I ran the code it had some errors. Therefore I need gemini to understand the error and fix it because I were not used to the pytorch library.
- **Affected assignment section(s):** `src/models/lstm.py`
- **Representative prompt / prompt-log link:** https://share.gemini.google/wVjtB3LHu1b6
- **How AI output was edited and verified:** I read the instructions of gemini and try to fix the errors, I discussed with AI to find the best way to write my code. I finally ran the code successfully and saw that the accuracy went up to 90%.
- **Sources used for verification:** `results/figures/training_curves.png`
- **Member responsible for final verification:** Cao Duc Thanh

### Entry A1-5

- **Tool name / version:** Gemini 3.8 Flash
- **Member who used it:** Cao Duc Thanh
- **Time / development stage:** Assignment 1 M2 — Implement Transformer
- **Purpose:** I have learned transformer architecture but I didn't know how to implement it. Therefore I needed to learn how to implement the code.
- **Affected assignment section(s):** `src/models/transformer.py`
- **Representative prompt / prompt-log link:** https://share.gemini.google/X2LdCo2AsJpd
- **How AI output was edited and verified:** I read the instructions of gemini and read the D2L book to implement transformer, I discussed with AI to find the best way to build my model. I finally ran the code successfully and saw that the accuracy went up to 80% with ViT.
- **Sources used for verification:** `results/figures/training_curves.png` and D2L book
- **Member responsible for final verification:** Cao Duc Thanh

### Entry A1-6

- **Tool name / version:** Gemini 3.8 Flash
- **Member who used it:** Cao Duc Thanh
- **Time / development stage:** Assignment 1 M2 — Write report
- **Purpose:** I don't have much idea to write all the the requirement in handbook. Therefore I used Gemini to organize the idea, suggest some ideas for me to write, and check my report.
- **Affected assignment section(s):** Assignment 1 Report
- **Representative prompt / prompt-log link:** https://share.gemini.google/phX1ifjJFugp
- **How AI output was edited and verified:** I read the output of gemini and try to write the report, then I used gemini to check if I write right.
- **Sources used for verification:** The files of assignment 1
- **Member responsible for final verification:** Cao Duc Thanh

### Entry A1-7

- **Tool name / version:** GPT-6
- **Member who used it:** Dang Thanh Phu
- **Time / development stage:** Assignment 1 M2 — Write report
- **Purpose:** Consulted ChatGPT to better understand LSTM and Transformer architectures, including hidden states, attention mechanisms, tensor dimensions, and parameter calculations. Asked questions about PyTorch LSTM biases, data normalization, training procedures, and evaluation metrics. Additionally, used ChatGPT to clarify experimental results, discuss model limitations, and assist in structuring and refining technical explanations in the report.
- **Affected assignment section(s):** Report: Sections 2.1, 2.5–2.8, 3.1–3.2, and 3.6
- **Representative prompt / prompt-log link:**
  - *LSTM Architecture and Parameter Calculation:* Asked how the LSTM processes Fashion-MNIST images as sequences of 28 rows, each containing 28 features. Specifically questioned the parameter-count formula for a hidden dimension of 2,048 and why PyTorch includes separate input-to-hidden and hidden-to-hidden bias vectors.
  - *Transformer Architecture:* Asked for explanations of how 28 × 28 images are divided into 49 patches, projected into 32-dimensional embeddings, and combined with a classification token and positional embeddings. Also discussed Query, Key, Value projections, four attention heads, residual connections, feed-forward layers, tensor dimensions, and the calculation of 28,202 trainable parameters.
  - *Training Behavior and Checkpoint Selection:* Asked how to interpret training and validation curves, identify possible overfitting or underfitting, and explain why the checkpoint with the highest validation macro-F1 may differ from the epoch with the lowest validation loss.
  - *Experimental Results and Limitations:* Requested assistance interpreting the recorded accuracy, macro-F1, training time, inference time, and parameter counts of the five architectures. Discussed the Transformer improvement between 10 and 30 epochs, differences in model capacity, and limitations of comparing architectures under unequal parameter and computational budgets.
- **How AI output was edited and verified:** I read the output of GPT and try to write the report, then I used GPT to check if I write right and checked and compared it with the recorded experimental results to make sure the data is not wrong.
- **Sources used for verification:** The files of assignment 1
- **Member responsible for final verification:** Dang Thanh Phu

### Entry A1-8

- **Tool name / version:** GPT-6
- **Member who used it:** Dang Thanh Phu
- **Time / development stage:** Assignment 1 M3 — Website organization and report integration
- **Purpose:** Used ChatGPT to reorganize the Assignment 1 and Assignment 3 website pages following Assignment 2’s layout, integrate the Assignment 1 M3 report, and add placeholders for YouTube presentation links.
- **Affected assignment section(s):** Assignment 1 and Assignment 3 landing pages, Assignment 1 M2 draft page, and Assignment 1 M3 report viewer
- **Representative prompt / prompt-log link:**
  - *Page structure and formatting:* Asked ChatGPT to use Assignment 2’s format for the Assignment 1 and Assignment 3 folders and pages.
  - *Report integration:* Asked ChatGPT to add the Assignment 1 M3 report to the website.
  - *Presentation link:* Asked ChatGPT to leave a placeholder for the YouTube presentation link.
- **How AI output was edited and verified:** The existing Assignment 2 page was used as the layout reference. Assignment 1’s draft content was preserved in a separate M2 page, and the M3 viewer was linked to the existing report PDF. Automated checks verified local link destinations, milestone cards, page titles, presentation placeholders, and formatting.
- **Sources used for verification:** Existing website files in `docs/assignment1`, `docs/assignment2`, and `docs/assignment3`, and `docs/reports/Assignment1_M3_RTX5090.pdf`
- **Member responsible for final verification:** Dang Thanh Phu

---

## Assignment 2

### Entry A2-1

- **Tool name / version:** Gemini 3.8 Flash
- **Member who used it:** Cao Duc Thanh
- **Time / development stage:** Assignment 2 M1 — Understand the requirement and find the dataset
- **Purpose:** I need to understand the specs of assignment 2, and find a dataset to implement the assignment 2.
- **Affected assignment section(s):** Dataset of assignment 2
- **Representative prompt / prompt-log link:** https://gemini.google.com/app/6c45defefd0a4e2a
- **How AI output was edited and verified:** I search it in the internet to see each dataset.
- **Sources used for verification:**
  - https://www.kaggle.com/datasets/bardiaardakanian/voc0712
  - https://www.kaggle.com/datasets/sovitrath/voc-2012-segmentation-data
  - https://www.cityscapes-dataset.com/
  - https://www.kaggle.com/datasets/tanlikesmath/the-oxfordiiit-pet-dataset
  - https://www.kaggle.com/datasets/pengcw1/market-1501
  - https://www.kaggle.com/datasets/whurobin/dukemtmcreid
  - https://www.kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset
- **Member responsible for final verification:** Cao Duc Thanh

### Entry A2-2

- **Tool name / version:** Gemini 3.8 Flash
- **Member who used it:** Cao Duc Thanh
- **Time / development stage:** Assignment 2 M1 — Do research and write proposal for assignment 2
- **Purpose:** I need to do some researches about the datasets, the models (pretrained and scratch model), the training process, learn how to write a proposal, and check if my proposal meets requirements of the specs.
- **Affected assignment section(s):** Proposal of assignment 2
- **Representative prompt / prompt-log link:** https://share.gemini.google/DGWCyHXayaPi
- **How AI output was edited and verified:** I search it in the internet to see each dataset.
- **Sources used for verification:**
  - https://arxiv.org/pdf/2105.15203
  - https://www.cityscapes-dataset.com/
  - https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
  - https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/
  - https://wiki.cloudfactory.com/docs/mp-wiki/model-architectures/deeplabv3
- **Member responsible for final verification:** Cao Duc Thanh

### Entry A2-3

- **Tool name / version:** Gemini 3.8 Flash
- **Member who used it:** Cao Duc Thanh
- **Time / development stage:** Assignment 2 M2 — DeepLabV3 batch-size crash debugging and DataLoader optimization
- **Purpose:** Diagnose `ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 256, 1, 1])` occurring during DeepLabV3 training on Cityscapes, explain why single-sample batches cause BatchNorm failures in ASPPPooling, and configure DataLoader with `drop_last=True` and batch safeguards.
- **Affected assignment section(s):** `assignment2/src/data.py`, `assignment2/src/train.py`, `assignment2/configs/deeplabv3.yaml`
- **Representative prompt / prompt-log link:** "Fix ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 256, 1, 1]) in DeepLabV3 training on Cityscapes."
- **How AI output was edited and verified:** Read PyTorch BatchNorm2d documentation, verified that 2,481 training samples divided by batch size 8 leaves a remainder of 1 at the final batch. Configured `drop_last=True` in `src/data.py` and added single-sample batch skip safeguard in `src/train.py`. Ran full 10 epochs on RTX 5070 Ti successfully without errors.
- **Sources used for verification:** PyTorch documentation for `nn.BatchNorm2d` and torchvision DeepLabV3 implementation.
- **Member responsible for final verification:** Cao Duc Thanh

### Entry A2-4

- **Tool name / version:** Gemini 3.8 Flash
- **Member who used it:** Dang Thanh Phu
- **Time / development stage:** Assignment 2 M2 — Preliminary metrics comparison and M2 Draft website scaffolding
- **Purpose:** Assist in creating comparative training curve plots (Loss, mIoU, Dice) between Baseline U-Net and DeepLabV3-ResNet50, and scaffold the M2 Draft webpage (`docs/assignment2/m2.html`) according to Course Project Handbook Section 7.5 and Section 22.
- **Affected assignment section(s):** `docs/assignment2/m2.html`, `docs/assignment2/assignment2.html`, `docs/figures/a2_training_curves.png`, `assignment2/README.md`
- **Representative prompt / prompt-log link:** "Plot comparative training curves for Loss, mIoU, and Dice from U-Net and DeepLabV3 log.csv files, and format HTML for M2 draft webpage with EDA findings, model architectures, and quantitative preliminary tables."
- **How AI output was edited and verified:** Checked and confirmed all numerical figures (U-Net best mIoU 31.34%, DeepLabV3 best mIoU 56.57%) against the generated `log.csv` files. Verified local image paths, responsive layout, and navigation on GitHub Pages.
- **Sources used for verification:** `assignment2/checkpoints/unet_base_1024x512_s42/log.csv`, `assignment2/checkpoints/deeplabv3/exp_1/log.csv`, `handbook-ene.pdf` Section 7.5.
- **Member responsible for final verification:** Dang Thanh Phu

---

## Assignment 3

### Entry A3-1

- **Declaration:** The group declares that no generative AI tool was used in this assignment to date (Course Project Handbook Section 5.4).
