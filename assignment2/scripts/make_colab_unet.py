import json
from pathlib import Path

nb_path = Path("colab.ipynb")
nb = json.load(open(nb_path, encoding="utf-8"))

# 1. Update Title in cell 0
nb["cells"][0]["source"] = [
    "# Assignment 2 — Cityscapes Semantic Segmentation (U-Net Baseline & DeepLabV3)\n",
    "### U-Net (Scratch) · DeepLabV3 (Frozen, Partial, Full)\n",
    "Includes auto-tuning support for GCP G4 (NVIDIA RTX PRO 6000 Blackwell 96GB), A100, L4, T4, and V100.\n"
]

# 2. Update Configuration panel in cell 1
cell1_lines = nb["cells"][1]["source"]
new_c1 = []
for line in cell1_lines:
    if "TRAIN_MODEL_1_FROZEN = True" in line:
        new_c1.append("TRAIN_UNET = True  # @param {type:\"boolean\"}\n")
        new_c1.append("# @markdown --- Train U-Net Baseline (from scratch) ---\n")
    if "HEAD_LR = 0.0001" in line:
        new_c1.append("UNET_LR = 0.001  # @param {type:\"number\"}\n")
    new_c1.append(line)

nb["cells"][1]["source"] = new_c1

# 3. Update RUNS in cell 5
cell5_text = "".join(nb["cells"][5]["source"])
old_runs = """RUNS = {
    "model1": {"label": "Model 1 (Frozen Backbone)","model": "deeplabv3_resnet50","run": "deeplabv3/local"},
    "model2": {"label": "Model 2 (Partial Fine-Tuning)","model": "resnet_layer3_4_finetune", "run": "deeplabv3/resnet_layer3_4_finetune"},
    "model3": {"label": "Model 3 (Full Fine-Tuning)","model": "resnet50_full_deeplabv3","run": "deeplabv3/resnet50_full_deeplabv3"},
}"""

new_runs = """RUNS = {
    "unet": {"label": "U-Net (Scratch Baseline)", "model": "unet", "run": "unet_final_1024x512_s42"},
    "model1": {"label": "Model 1 (Frozen Backbone)", "model": "deeplabv3_resnet50", "run": "deeplabv3/local"},
    "model2": {"label": "Model 2 (Partial Fine-Tuning)", "model": "resnet_layer3_4_finetune", "run": "deeplabv3/resnet_layer3_4_finetune"},
    "model3": {"label": "Model 3 (Full Fine-Tuning)", "model": "resnet50_full_deeplabv3", "run": "deeplabv3/resnet50_full_deeplabv3"},
}"""

if old_runs in cell5_text:
    cell5_text = cell5_text.replace(old_runs, new_runs)
    nb["cells"][5]["source"] = [l + "\n" for l in cell5_text.split("\n")[:-1]] + [cell5_text.split("\n")[-1]]

# 4. Update GPU Detection & Config building in cell 15
cell15_text = "".join(nb["cells"][15]["source"])

old_gpu_detect = """    if "A100" in raw_device_name:
        detected_gpu_title, auto_batch_size, auto_workers = f"NVIDIA A100 ({vram_gb:.1f} GB)", 16, min(cpu_cores, 8)"""

new_gpu_detect = """    # G4 Instance: NVIDIA RTX PRO 6000 Blackwell (96 GB GDDR7) or RTX 6000 Ada (48 GB)
    if any(k in raw_device_name for k in ["6000", "BLACKWELL", "RTX PRO"]):
        detected_gpu_title, auto_batch_size, auto_workers = f"G4 - NVIDIA RTX PRO 6000 ({vram_gb:.1f} GB)", 32, min(cpu_cores, 8)
    elif "A100" in raw_device_name:
        detected_gpu_title, auto_batch_size, auto_workers = f"NVIDIA A100 ({vram_gb:.1f} GB)", 16, min(cpu_cores, 8)"""

if old_gpu_detect in cell15_text:
    cell15_text = cell15_text.replace(old_gpu_detect, new_gpu_detect)

old_cfgs = """CFGS = {
    "model1": {**BASE_CFG, "model": RUNS["model1"]["model"]},
    "model2": {**BASE_CFG, "model": RUNS["model2"]["model"], "backbone_lr": BACKBONE_LR_PARTIAL},
    "model3": {**BASE_CFG, "model": RUNS["model3"]["model"], "backbone_lr": BACKBONE_LR_FULL},
}"""

new_cfgs = """CFGS = {
    "unet": {**BASE_CFG, "model": RUNS["unet"]["model"], "lr": UNET_LR, "head_lr": UNET_LR, "base_channels": 32},
    "model1": {**BASE_CFG, "model": RUNS["model1"]["model"]},
    "model2": {**BASE_CFG, "model": RUNS["model2"]["model"], "backbone_lr": BACKBONE_LR_PARTIAL},
    "model3": {**BASE_CFG, "model": RUNS["model3"]["model"], "backbone_lr": BACKBONE_LR_FULL},
}"""

if old_cfgs in cell15_text:
    cell15_text = cell15_text.replace(old_cfgs, new_cfgs)

nb["cells"][15]["source"] = [l + "\n" for l in cell15_text.split("\n")[:-1]] + [cell15_text.split("\n")[-1]]

# 5. Insert U-Net Training Cell before Model 1
unet_train_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# @title U-Net Baseline: Train from Scratch\n",
        "if not ('TRAIN_UNET' in globals() and TRAIN_UNET):\n",
        "    print('  Skipped (TRAIN_UNET = False)')\n",
        "else:\n",
        "    print(f\"Training {RUNS['unet']['label']}...\")\n",
        "    train_model(CFGS['unet'])\n"
    ]
}

# Find index of Step 7 Markdown
step7_idx = None
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "markdown" and "Step 7" in "".join(c["source"]):
        step7_idx = i
        c["source"] = ["## Step 7: Train U-Net Baseline and DeepLabV3 models\n"]
        break

if step7_idx is not None:
    nb["cells"].insert(step7_idx + 1, unet_train_cell)

with open("colab_U_net.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Successfully created colab_U_net.ipynb!")
