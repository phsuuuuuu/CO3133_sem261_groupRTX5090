import argparse
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
import seaborn as sns
from cityscapesscripts.helpers.labels import labels

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
SPLITS_DIR = BASE_DIR / "splits"
FIG_DIR = BASE_DIR / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EVAL_LABELS = sorted([l for l in labels if 0 <= l.trainId < 19], key=lambda x: x.trainId)
CLASS_NAMES = [l.name for l in EVAL_LABELS]
LABEL_TO_TRAIN_ID = {l.id: l.trainId for l in labels if 0 <= l.trainId < 19}
THING_LABEL_IDS = {l.id: l.trainId for l in EVAL_LABELS if l.hasInstances}


def load_split(split_name):
    split_file = SPLITS_DIR / f"{split_name}.txt"
    with open(split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def resolve_paths(sample_id, is_test_split=False):
    city, stem = sample_id.split("/")
    folder = "val" if is_test_split else "train"
    gt_dir = DATASET_DIR / "gtFine_trainvaltest" / "gtFine" / folder / city
    train_id_path = gt_dir / f"{stem}_gtFine_trainIds.png"
    instance_id_path = gt_dir / f"{stem}_gtFine_instanceIds.png"
    return train_id_path, instance_id_path


def run_region_size_analysis(train_samples, max_samples=500):
    print(f"Running region size analysis on {min(len(train_samples), max_samples)} samples...")
    class_areas = defaultdict(list)
    subset = train_samples[:max_samples]

    for sample in subset:
        train_path, inst_path = resolve_paths(sample, is_test_split=False)
        mask = np.array(Image.open(train_path))

        # Cityscapes instance encoding: instance_id = label_id * 1000 + instance_num
        if inst_path.exists():
            inst_map = np.array(Image.open(inst_path))
            u_ids, u_counts = np.unique(inst_map, return_counts=True)
            for uid, cnt in zip(u_ids, u_counts):
                if uid >= 1000:
                    lbl_id = uid // 1000
                    if lbl_id in THING_LABEL_IDS:
                        t_id = THING_LABEL_IDS[lbl_id]
                        if cnt >= 20:
                            class_areas[CLASS_NAMES[t_id]].append(cnt)

        # Connected component sizes for stuff and non-instance classes
        downscaled = mask[::2, ::2]
        for t_id, lbl in enumerate(EVAL_LABELS):
            if not lbl.hasInstances:
                bin_mask = (downscaled == t_id)
                if np.any(bin_mask):
                    labeled, num = ndimage.label(bin_mask)
                    sizes = ndimage.sum(bin_mask, labeled, range(1, num + 1)) * 4
                    for sz in sizes:
                        if sz >= 50:
                            class_areas[lbl.name].append(sz)

    plt.figure(figsize=(15, 6))
    plot_data = []
    plot_labels = []
    for name in CLASS_NAMES:
        vals = class_areas.get(name, [])
        if vals:
            plot_data.append(np.log10(vals))
            plot_labels.append(f"{name}\n(n={len(vals)})")

    sns.boxplot(data=plot_data, palette="Set3", showfliers=False)
    plt.xticks(range(len(plot_labels)), plot_labels, rotation=45, ha="right", fontsize=9)
    plt.ylabel(r"Area $\log_{10}(\text{pixels})$", fontsize=11)
    plt.title("A2.7.1 - Region and Instance Size Distribution per Class", fontsize=13, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = FIG_DIR / "a2_eda_region_size.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def run_mask_quality_analysis(train_samples):
    print(f"Running mask quality analysis on {len(train_samples)} training samples...")
    ignore_ratios = []
    spatial_acc = np.zeros((256, 512), dtype=np.float32)

    for sample in train_samples:
        train_path, _ = resolve_paths(sample, is_test_split=False)
        mask = np.array(Image.open(train_path).resize((512, 256), resample=Image.NEAREST))
        ign = (mask == 255)
        ignore_ratios.append(float(np.mean(ign) * 100.0))
        spatial_acc += ign

    spatial_prob = (spatial_acc / len(train_samples)) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    im = axes[0].imshow(spatial_prob, cmap="magma")
    axes[0].set_title("Spatial Ignore Probability Map (%)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Height (px)")
    fig.colorbar(im, ax=axes[0], label="Probability (%)")

    mean_ign = float(np.mean(ignore_ratios))
    std_ign = float(np.std(ignore_ratios))
    sns.histplot(ignore_ratios, bins=35, kde=True, ax=axes[1], color="firebrick")
    axes[1].axvline(mean_ign, color="navy", linestyle="--", label=f"Mean: {mean_ign:.2f}% (std: {std_ign:.2f}%)")
    axes[1].set_title("Per-Image Ignore Pixel Distribution", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Ignore Pixels (%)")
    axes[1].set_ylabel("Image Count")
    axes[1].legend()

    plt.suptitle("A2.7.2 - Mask Quality and Spatial Ignore Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = FIG_DIR / "a2_eda_mask_quality.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")
    print(f"Ignore stats: mean={mean_ign:.2f}%, std={std_ign:.2f}%, min={np.min(ignore_ratios):.2f}%, max={np.max(ignore_ratios):.2f}%")


def run_cooccurrence_analysis(train_samples):
    print(f"Running co-occurrence analysis on {len(train_samples)} training samples...")
    class_counts_per_img = []
    cooccur = np.zeros((19, 19), dtype=np.int32)
    single_counts = np.zeros(19, dtype=np.int32)

    for sample in train_samples:
        train_path, _ = resolve_paths(sample, is_test_split=False)
        mask = np.array(Image.open(train_path))
        present = np.unique(mask)
        present = present[(present >= 0) & (present < 19)]

        class_counts_per_img.append(len(present))
        for c in present:
            single_counts[c] += 1
            for d in present:
                cooccur[c, d] += 1

    # P(Y | X) = count(X and Y) / count(X)
    cond_prob = cooccur / np.maximum(single_counts[:, None], 1)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [1, 1.8]})

    sns.histplot(class_counts_per_img, bins=np.arange(1, 21) - 0.5, ax=axes[0], color="cadetblue", discrete=True)
    mean_cls = float(np.mean(class_counts_per_img))
    axes[0].axvline(mean_cls, color="darkred", linestyle="--", label=f"Mean: {mean_cls:.2f}")
    axes[0].set_title("Unique Classes per Scene", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Number of Unique Classes")
    axes[0].set_ylabel("Image Count")
    axes[0].legend()

    sns.heatmap(cond_prob, cmap="YlGnBu", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
    axes[1].set_title(r"Conditional Co-occurrence Probability $P(\text{Class } Y \mid \text{Class } X)$", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Class Y")
    axes[1].set_ylabel("Class X")

    plt.suptitle("A2.7.3 - Scene Complexity and Class Co-occurrence", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = FIG_DIR / "a2_eda_cooccurrence.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")
    print(f"Classes per image: mean={mean_cls:.2f}, min={np.min(class_counts_per_img)}, max={np.max(class_counts_per_img)}")


def run_split_distribution_analysis(splits):
    print("Running split distribution comparison across Train, Val, and Test...")
    pixel_pcts = {}

    for split_name, samples in splits.items():
        is_test = (split_name == "test")
        counts = np.zeros(256, dtype=np.int64)
        for s in samples:
            train_path, _ = resolve_paths(s, is_test_split=is_test)
            mask = np.array(Image.open(train_path))
            counts += np.bincount(mask.flatten(), minlength=256)

        valid_counts = counts[:19]
        total_valid = np.sum(valid_counts)
        pixel_pcts[split_name] = (valid_counts / total_valid) * 100.0

    plt.figure(figsize=(16, 6))
    x = np.arange(len(CLASS_NAMES))
    width = 0.28

    plt.bar(x - width, pixel_pcts["train"], width, label=f"Train (n={len(splits['train'])})", color="#4C72B0")
    plt.bar(x, pixel_pcts["val"], width, label=f"Val (n={len(splits['val'])})", color="#55A868")
    plt.bar(x + width, pixel_pcts["test"], width, label=f"Test (n={len(splits['test'])})", color="#C44E52")

    plt.yscale("log")
    plt.xticks(x, CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    plt.ylabel("Valid Pixels (%) [Log Scale]", fontsize=11)
    plt.title("A2.7.4 - Class Pixel Distribution Across Splits", fontsize=13, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = FIG_DIR / "a2_eda_split_distribution.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")

    print("\n--- Rare Class Presence Verification (Pixel %) ---")
    header = f"{'Class':<15} | {'Train (%)':<10} | {'Val (%)':<10} | {'Test (%)':<10}"
    print(header)
    print("-" * len(header))
    for idx in [17, 12, 16, 15, 14]:
        name = CLASS_NAMES[idx]
        print(f"{name:<15} | {pixel_pcts['train'][idx]:<10.4f} | {pixel_pcts['val'][idx]:<10.4f} | {pixel_pcts['test'][idx]:<10.4f}")


def main():
    parser = argparse.ArgumentParser(description="A2.7 Exploratory Data Analysis for Semantic Segmentation")
    parser.add_argument("--region-samples", type=int, default=500, help="Max samples for region size analysis")
    args = parser.parse_args()

    splits = {
        "train": load_split("train"),
        "val": load_split("val"),
        "test": load_split("test"),
    }

    run_region_size_analysis(splits["train"], max_samples=args.region_samples)
    run_mask_quality_analysis(splits["train"])
    run_cooccurrence_analysis(splits["train"])
    run_split_distribution_analysis(splits)
    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()
