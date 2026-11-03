from cityscapesscripts.helpers.labels import labels
from pathlib import Path
import numpy as np
from PIL import Image

lut = np.full(256, 255, dtype=np.uint8)

for label in labels:
    if label.id >= 0 and 0 <= label.trainId < 19:
        lut[label.id] = label.trainId

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

file_list = list(DATASET_DIR.rglob("*_gtFine_labelIds.png"))

if not file_list:
    print(f"[ERROR] No label files (*_gtFine_labelIds.png) found in {DATASET_DIR}!")
    print("Please make sure Cityscapes gtFine_trainvaltest dataset is extracted/linked into the dataset directory.")
    exit(1)

total_counts = np.zeros(256, dtype=np.int64)

for label_path in file_list:
    mask_labelId = np.array(Image.open(label_path))
    mask_trainId = lut[mask_labelId]

    train_id_path = str(label_path).replace("labelIds", "trainIds")
    Image.fromarray(mask_trainId).save(train_id_path)

    if "train" in label_path.parts:
        total_counts += np.bincount(mask_trainId.flatten(), minlength=256)

total_pixels = np.sum(total_counts)
if total_pixels > 0:
    ignore_ratio = (total_counts[255] / total_pixels) * 100
    print(f"Ignore ratio: {ignore_ratio:.2f}%")
else:
    print("Mask generation finished (0 training pixels counted).")
