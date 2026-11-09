import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from src.losses import compute_weight_class

NUM_CLASSES = 19
DATA_DIR = BASE_DIR / "dataset"
SPLIT_FILE = BASE_DIR / "splits" / "train.txt"   # only our train split (no val / test)
GT_DIR = DATA_DIR / "gtFine_trainvaltest" / "gtFine" / "train"
OUT_FILE = BASE_DIR / "configs" / "class_weights.json"


def main():
    with open(SPLIT_FILE, "r") as f:
        samples = [line.strip() for line in f if line.strip()]

    counts = np.zeros(256, dtype=np.int64)
    for i, item in enumerate(samples):
        city, stem_id = item.split("/")
        mask_path = GT_DIR / city / f"{stem_id}_gtFine_trainIds.png"
        mask = np.array(Image.open(mask_path))
        counts += np.bincount(mask.flatten(), minlength=256)
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(samples)}")

    class_counts = counts[:NUM_CLASSES]
    weights = compute_weight_class(torch.tensor(class_counts, dtype=torch.float64))

    result = {
        "formula": "w_c = 1 + log(max_count / count_c)",
        "split": str(SPLIT_FILE.relative_to(BASE_DIR)).replace("\\", "/"),
        "num_images": len(samples),
        "ignore_pixels": int(counts[255]),
        "counts": [int(c) for c in class_counts],
        "weights": [round(float(w), 6) for w in weights],
    }
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    for c in range(NUM_CLASSES):
        print(f"class {c:2d}: count={class_counts[c]:>12d}  weight={weights[c]:.4f}")
    print(f"Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
