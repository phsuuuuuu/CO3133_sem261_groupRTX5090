from cityscapesscripts.helpers.labels import labels
from pathlib import Path
import numpy as np
import glob
from PIL import Image

lut = np.full(256, 255, dtype=np.uint8)

for label in labels:
    if label.id >= 0 and 0 <= label.trainId < 19:
        lut[label.id] = label.trainId

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

file_list = list(DATASET_DIR.rglob("*_gtFine_labelIds.png"))

total_counts = np.zeros(256, dtype=np.int64)

for label_path in file_list:
    mask_labelId = np.array(Image.open(label_path))
    mask_trainId = lut[mask_labelId]

    train_id_path = str(label_path).replace("labelIds", "trainIds")
    Image.fromarray(mask_trainId).save(train_id_path)

    if "train" in label_path.parts:
        total_counts += np.bincount(mask_trainId.flatten(), minlength=256)
    

ignore_ratio = (total_counts[255] / np.sum(total_counts)) * 100

print(f"Ignore ratio: {ignore_ratio:.2f}%")