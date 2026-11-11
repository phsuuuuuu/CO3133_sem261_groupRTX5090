"""M3: quantitative EDA numbers used in the report (pixel share, void ratio, region sizes, batch shape).

Region sizes are measured on masks resized to the network input resolution (1024x512, nearest),
using 8-connected components per class. Output: docs/figures/m3/eda_stats.json, a2_m3_batch.png
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
OUT_DIR = BASE_DIR / "docs" / "figures" / "m3"

CLASS_NAMES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
               "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
               "motorcycle", "bicycle"]
NC = 19


def mask_path(sample, folder):
    city, stem = sample.split("/")
    return BASE_DIR / "dataset" / "gtFine_trainvaltest" / "gtFine" / folder / city / f"{stem}_gtFine_trainIds.png"


def split_stats(samples, folder, region=False):
    counts = np.zeros(NC, dtype=np.int64)
    ignore = 0
    total = 0
    per_img_ignore = []
    classes_per_img = []
    areas = {c: [] for c in range(NC)}
    for s in samples:
        m = np.array(Image.open(mask_path(s, folder)))
        counts += np.bincount(m[m < NC].ravel(), minlength=NC)
        ign = int((m == 255).sum())
        ignore += ign
        total += m.size
        per_img_ignore.append(ign / m.size * 100)
        classes_per_img.append(int(len(np.unique(m[m < NC]))))
        if region:
            small = np.array(Image.fromarray(m).resize((1024, 512), Image.NEAREST))
            for c in np.unique(small[small < NC]):
                n, _, st, _ = cv2.connectedComponentsWithStats((small == c).astype(np.uint8), connectivity=8)
                areas[int(c)].extend(st[1:, cv2.CC_STAT_AREA].tolist())
    out = {
        "num_images": len(samples),
        "pixel_share_pct": (counts / counts.sum() * 100).tolist(),
        "void_pct_of_all_pixels": ignore / total * 100,
        "void_pct_per_image_mean": float(np.mean(per_img_ignore)),
        "void_pct_per_image_std": float(np.std(per_img_ignore)),
        "classes_per_image_mean": float(np.mean(classes_per_img)),
        "classes_per_image_min": int(np.min(classes_per_img)),
        "classes_per_image_max": int(np.max(classes_per_img)),
    }
    if region:
        reg = {}
        for c in range(NC):
            a = np.array(areas[c])
            reg[CLASS_NAMES[c]] = {
                "num_regions": int(a.size),
                "median_area_px": float(np.median(a)) if a.size else None,
                "pct_regions_lt_1024px": float((a < 1024).mean() * 100) if a.size else None,
            }
        out["regions_at_1024x512"] = reg
    return out


def batch_figure():
    import matplotlib.pyplot as plt
    import torch
    from src.data import get_loaders
    torch.manual_seed(666)
    train_loader, _, test_loader = get_loaders({"batch_size": 8, "num_workers": 0, "seed": 666})
    imgs, masks = next(iter(train_loader))
    timgs, tmasks = next(iter(test_loader))
    shapes = {
        "train_images": [list(imgs.shape), str(imgs.dtype)],
        "train_masks": [list(masks.shape), str(masks.dtype)],
        "eval_images": [list(timgs.shape), str(timgs.dtype)],
        "eval_masks": [list(tmasks.shape), str(tmasks.dtype)],
        "train_image_value_range": [float(imgs.min()), float(imgs.max())],
        "train_mask_values": sorted(int(v) for v in torch.unique(masks)),
    }
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i in range(4):
        rgb = np.clip((imgs[i].numpy() * std + mean).transpose(1, 2, 0), 0, 1)
        axes[0, i].imshow(rgb); axes[0, i].axis("off")
        mk = masks[i].numpy().astype(float); mk[mk == 255] = np.nan
        axes[1, i].imshow(mk, cmap="tab20", vmin=0, vmax=19, interpolation="nearest"); axes[1, i].axis("off")
    plt.suptitle(f"One training batch after preprocessing: images {tuple(imgs.shape)}, masks {tuple(masks.shape)} (void = white)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "a2_m3_batch.png", dpi=150)
    plt.close()
    return shapes


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = {s: [l.strip() for l in open(BASE_DIR / "splits" / f"{s}.txt") if l.strip()] for s in ("train", "val", "test")}
    res = {
        "train": split_stats(splits["train"], "train", region=True),
        "val": split_stats(splits["val"], "train"),
        "test": split_stats(splits["test"], "val"),
        "cities": {s: sorted({x.split("/")[0] for x in v}) for s, v in splits.items()},
    }
    share = np.array(res["train"]["pixel_share_pct"])
    order = np.argsort(share)
    res["train_summary"] = {
        "top3": [(CLASS_NAMES[i], float(share[i])) for i in order[::-1][:3]],
        "top3_sum_pct": float(share[order[::-1][:3]].sum()),
        "rare10": [(CLASS_NAMES[i], float(share[i])) for i in order[:10]],
        "rare10_sum_pct": float(share[order[:10]].sum()),
    }
    res["batch"] = batch_figure()
    with open(OUT_DIR / "eda_stats.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res["train_summary"], indent=1))
    for k in ("train", "val", "test"):
        r = res[k]
        print(k, r["num_images"], "void%%=%.2f per-img %.2f+-%.2f cls/img %.2f [%d,%d]" % (
            r["void_pct_of_all_pixels"], r["void_pct_per_image_mean"], r["void_pct_per_image_std"],
            r["classes_per_image_mean"], r["classes_per_image_min"], r["classes_per_image_max"]))
    print("pixel share train:", {CLASS_NAMES[i]: round(v, 2) for i, v in enumerate(res["train"]["pixel_share_pct"])})
    print("pixel share test:", {CLASS_NAMES[i]: round(v, 2) for i, v in enumerate(res["test"]["pixel_share_pct"])})
    print("regions:", json.dumps(res["train"]["regions_at_1024x512"]))
    print("cities:", res["cities"])
    print("batch:", res["batch"])


if __name__ == "__main__":
    main()
