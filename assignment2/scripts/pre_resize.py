import concurrent.futures
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

TARGET_W, TARGET_H = 1024, 512
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

def _resize_single_img(path_str):
    try:
        with Image.open(path_str) as im:
            if im.size == (TARGET_W, TARGET_H):
                return False
            im_resized = im.resize((TARGET_W, TARGET_H), Image.BILINEAR)
            im_resized.save(path_str)
        return True
    except Exception:
        return False

def _resize_single_mask(path_str):
    try:
        with Image.open(path_str) as im:
            if im.size == (TARGET_W, TARGET_H):
                return False
            im_resized = im.resize((TARGET_W, TARGET_H), Image.NEAREST)
            im_resized.save(path_str)
        return True
    except Exception:
        return False

def main():
    print(f"Checking dataset directory: {DATASET_DIR}")
    if not DATASET_DIR.exists():
        print(f"[ERROR] Dataset folder not found at {DATASET_DIR}")
        return

    all_imgs = [str(p) for p in DATASET_DIR.rglob("*_leftImg8bit.png")]
    all_masks = [str(p) for p in DATASET_DIR.rglob("*_gtFine_trainIds.png")]

    print(f"Found {len(all_imgs)} RGB images and {len(all_masks)} trainId masks.")
    if not all_imgs:
        print("[ERROR] No leftImg8bit images found!")
        return

    cpu_workers = min(os.cpu_count() or 4, 12)
    print(f"Resizing images to {TARGET_W}x{TARGET_H} using {cpu_workers} CPU workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_workers) as executor:
        resized_imgs = sum(tqdm(executor.map(_resize_single_img, all_imgs), total=len(all_imgs), desc="Resize Images"))
        resized_masks = sum(tqdm(executor.map(_resize_single_mask, all_masks), total=len(all_masks), desc="Resize Masks"))

    print(f"Completed! Resized {resized_imgs} images and {resized_masks} masks to {TARGET_W}x{TARGET_H}.")

if __name__ == "__main__":
    main()
