"""Qualitative Visualization for Cityscapes Semantic Segmentation (Task A3.4).

Generates multi-column comparison grids:
[Input Image (RGB) | Ground Truth | U-Net Baseline | DeepLabV3 | Error Map]
- Error map color conventions:
    * RED   (#E61919): Prediction error (Pred != GT).
    * GRAY  (#8C8C8C): Ignored / void pixels (GT == 255).
    * DARK  (#232323): Correct prediction (Pred == GT).

Produces 4 figure sets covering curated evaluation categories:
1. a2_qual_good.png      : High-confidence / successful predictions.
2. a2_qual_hard.png      : Hard but correct (small objects, thin structures, rare classes).
3. a2_qual_fail.png      : Failure modes and semantic ambiguities.
4. a2_qual_unlabeled.png : Qualitative inference on 1,525 unlabeled test set images.
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import yaml

from src.common import get_logits
from src.models import build_model
from src.transforms import SegTransform

# Official 19 Cityscapes evaluation classes palette (RGB)
CITYSCAPES_PALETTE = np.array([
    [128,  64, 128],  # 0: road
    [244,  35, 232],  # 1: sidewalk
    [ 70,  70,  70],  # 2: building
    [102, 102, 156],  # 3: wall
    [190, 153, 153],  # 4: fence
    [153, 153, 153],  # 5: pole
    [250, 170,  30],  # 6: traffic light
    [220, 220,   0],  # 7: traffic sign
    [107, 142,  35],  # 8: vegetation
    [152, 251, 152],  # 9: terrain
    [ 70, 130, 180],  # 10: sky
    [220,  20,  60],  # 11: person
    [255,   0,   0],  # 12: rider
    [  0,   0, 142],  # 13: car
    [  0,   0,  70],  # 14: truck
    [  0,  60, 100],  # 15: bus
    [  0,  80, 100],  # 16: train
    [  0,   0, 230],  # 17: motorcycle
    [119,  11,  32],  # 18: bicycle
], dtype=np.uint8)


def colorize_mask(mask_2d, ignore_index=255):
    """Map class ID mask [H, W] to RGB color image using Cityscapes palette."""
    h, w = mask_2d.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    valid = (mask_2d >= 0) & (mask_2d < 19)
    color_img[valid] = CITYSCAPES_PALETTE[mask_2d[valid]]
    # Set ignored pixels (255) to black / dark background
    color_img[mask_2d == ignore_index] = [0, 0, 0]
    return color_img


def make_error_map(pred, gt, ignore_index=255):
    """
    Construct pixel-wise error map:
    - Wrong pixels    : Bright Red [230, 25, 25]
    - Ignored pixels  : Neutral Gray [140, 140, 140]
    - Correct pixels  : Dark Charcoal [35, 35, 35]
    """
    h, w = gt.shape
    err = np.full((h, w, 3), 35, dtype=np.uint8)          # Correct: dark
    ignore_mask = (gt == ignore_index)
    err[ignore_mask] = [140, 140, 140]                     # Ignore: gray
    err[(pred != gt) & (~ignore_mask)] = [230, 25, 25]      # Error: bright red
    return err


def unnormalize_image(img_tensor):
    """Convert ImageNet-normalized [3, H, W] tensor back to uint8 RGB array."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_np = img_tensor.cpu().numpy() * std + mean
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    return np.transpose(img_np, (1, 2, 0))


def load_model(cfg_path, ckpt_path, device):
    """Instantiate and load checkpoint weights into model."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def plot_grid(rows_data, col_titles, save_path, suptitle):
    """Plot and save multi-row, multi-column qualitative comparison grid."""
    num_rows = len(rows_data)
    num_cols = len(col_titles)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4.2, num_rows * 2.4), squeeze=False)
    fig.patch.set_facecolor("#ffffff")
    plt.suptitle(suptitle, fontsize=15, fontweight="bold", y=0.995)

    for r_idx, row in enumerate(rows_data):
        for c_idx in range(num_cols):
            ax = axes[r_idx, c_idx]
            ax.imshow(row[c_idx])
            ax.axis("off")
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=12, fontweight="bold", pad=8)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved grid: {save_path}")


# 4 curated samples per group from splits/test.txt (frankfurt, munster, lindau)
GOOD_SAMPLES = [
    ("munster/munster_000008_000019", "Munster 08 - Straight avenue, uniform daylight, clear boundaries"),
    ("munster/munster_000023_000019", "Munster 23 - Residential street, distinct parked vehicles and sidewalk"),
    ("munster/munster_000151_000019", "Munster 151 - Open intersection, prominent vegetation and sky"),
    ("frankfurt/frankfurt_000001_071781", "Frankfurt 71781 - Large architecture facades and canonical car sizes"),
]

HARD_SAMPLES = [
    ("frankfurt/frankfurt_000001_031266", "Frankfurt 31266 - Crowded intersection, distant pedestrians and cyclists"),
    ("frankfurt/frankfurt_000001_061763", "Frankfurt 61763 - Slender structures: poles, traffic signs, narrow walkway"),
    ("munster/munster_000058_000019", "Munster 58 - Dense cluster of parked bicycles occluding storefront"),
    ("frankfurt/frankfurt_000001_008688", "Frankfurt 8688 - Overhanging foliage casting deep shadows on asphalt"),
]

FAIL_SAMPLES = [
    ("lindau/lindau_000047_000019", "Lindau 47 - Winding rural incline with ambiguous road/terrain boundary"),
    ("lindau/lindau_000019_000019", "Lindau 19 - Unpaved countryside path with heavy surface glare"),
    ("lindau/lindau_000018_000019", "Lindau 18 - Moss-covered stone wall blended with overgrown fence/bushes"),
    ("lindau/lindau_000040_000019", "Lindau 40 - Severe direct backlight and dappled sun glare saturation"),
]

UNLABELED_SAMPLES = [
    "dataset/leftImg8bit_trainvaltest/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png",
    "dataset/leftImg8bit_trainvaltest/leftImg8bit/test/bielefeld/bielefeld_000000_000321_leftImg8bit.png",
    "dataset/leftImg8bit_trainvaltest/leftImg8bit/test/bonn/bonn_000000_000019_leftImg8bit.png",
    "dataset/leftImg8bit_trainvaltest/leftImg8bit/test/munich/munich_000000_000019_leftImg8bit.png",
]


def process_labeled_group(samples_meta, unet, dlv3, device, out_png, group_title, dataset_dir):
    """Process a group of labeled evaluation samples and generate a comparison grid."""
    transform = SegTransform(is_train=False)
    rows = []
    col_titles = ["Input Image (RGB)", "Ground Truth", "U-Net Baseline", "DeepLabV3", "Error Map (DeepLabV3)"]

    for stem, desc in samples_meta:
        city, stem_id = stem.split("/")
        img_path = dataset_dir / "leftImg8bit_trainvaltest/leftImg8bit/val" / city / f"{stem_id}_leftImg8bit.png"
        gt_path = dataset_dir / "gtFine_trainvaltest/gtFine/val" / city / f"{stem_id}_gtFine_trainIds.png"

        img_pil = Image.open(img_path).convert("RGB")
        gt_pil = Image.open(gt_path)

        img_tensor, mask_tensor = transform(img_pil, gt_pil)
        img_in = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_u = get_logits(unet(img_in)).argmax(dim=1).squeeze(0).cpu().numpy()
            pred_d = get_logits(dlv3(img_in)).argmax(dim=1).squeeze(0).cpu().numpy()

        rgb = unnormalize_image(img_tensor)
        gt_np = mask_tensor.numpy()

        gt_color = colorize_mask(gt_np)
        pred_u_color = colorize_mask(pred_u)
        pred_d_color = colorize_mask(pred_d)
        err_map = make_error_map(pred_d, gt_np)

        rows.append([rgb, gt_color, pred_u_color, pred_d_color, err_map])

    plot_grid(rows, col_titles, out_png, group_title)


def process_unlabeled_group(paths, unet, dlv3, device, out_png, group_title):
    """Process unlabeled test set images and visualize model prediction agreement."""
    transform = SegTransform(is_train=False)
    rows = []
    col_titles = ["Unlabeled Test Image", "U-Net Prediction", "DeepLabV3 Prediction", "Disagreement Map"]

    dummy_mask = Image.new("L", (1024, 512), 0)
    for p in paths:
        img_path = BASE_DIR / p
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor, _ = transform(img_pil, dummy_mask)
        img_in = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_u = get_logits(unet(img_in)).argmax(dim=1).squeeze(0).cpu().numpy()
            pred_d = get_logits(dlv3(img_in)).argmax(dim=1).squeeze(0).cpu().numpy()

        rgb = unnormalize_image(img_tensor)
        pred_u_color = colorize_mask(pred_u)
        pred_d_color = colorize_mask(pred_d)

        # Disagreement map: bright red where models disagree
        diff_map = np.full_like(rgb, 35)
        diff_map[pred_u != pred_d] = [230, 40, 40]

        rows.append([rgb, pred_u_color, pred_d_color, diff_map])

    plot_grid(rows, col_titles, out_png, group_title)


def main():
    parser = argparse.ArgumentParser(description="A3.4 Qualitative Visualization for Cityscapes")
    parser.add_argument("--unet_config", default="configs/unet_final.yaml")
    parser.add_argument("--unet_ckpt", default="checkpoints/unet_final_1024x512_s42/best.pt")
    parser.add_argument("--dlv3_config", default="configs/resnet_layer3_4_finetune.yaml")
    parser.add_argument("--dlv3_ckpt", default="checkpoints/deeplabv3/resnet_layer3_4_finetune/best.pt")
    parser.add_argument("--out_dir", default="docs/figures")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading models...")
    unet = load_model(BASE_DIR / args.unet_config, BASE_DIR / args.unet_ckpt, device)
    dlv3 = load_model(BASE_DIR / args.dlv3_config, BASE_DIR / args.dlv3_ckpt, device)

    out_dir = BASE_DIR / args.out_dir
    dataset_dir = BASE_DIR / "dataset"

    print("\n1. Generating Group 1: Good predictions (High Accuracy)...")
    process_labeled_group(
        GOOD_SAMPLES, unet, dlv3, device,
        out_dir / "a2_qual_good.png",
        "A2 Qualitative: Group 1 - Good Predictions (High Accuracy)",
        dataset_dir
    )

    print("\n2. Generating Group 2: Hard but correct (Small Objects & Complex Scenes)...")
    process_labeled_group(
        HARD_SAMPLES, unet, dlv3, device,
        out_dir / "a2_qual_hard.png",
        "A2 Qualitative: Group 2 - Hard but Correct (Small Objects & Complex Scenes)",
        dataset_dir
    )

    print("\n3. Generating Group 3: Failure modes (Challenging Scenarios & Ambiguities)...")
    process_labeled_group(
        FAIL_SAMPLES, unet, dlv3, device,
        out_dir / "a2_qual_fail.png",
        "A2 Qualitative: Group 3 - Failure Modes (Challenging Scenarios & Ambiguities)",
        dataset_dir
    )

    print("\n4. Generating Group 4: Unlabeled test set inference (Generalization)...")
    process_unlabeled_group(
        UNLABELED_SAMPLES, unet, dlv3, device,
        out_dir / "a2_qual_unlabeled.png",
        "A2 Qualitative: Group 4 - Unlabeled Test Set Inference (Generalization)"
    )

    print("\nAll qualitative visualization figures created successfully in:", out_dir)


if __name__ == "__main__":
    main()
