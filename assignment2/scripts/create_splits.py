from pathlib import Path
import json
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_GT_DIR = DATASET_DIR / "gtFine_trainvaltest" / "gtFine" / "train"
VAL_GT_DIR = DATASET_DIR / "gtFine_trainvaltest" / "gtFine" / "val"
SPLITS_DIR = BASE_DIR / "splits"

VAL_CITIES = {"erfurt", "jena", "tubingen", "zurich"}

def get_city_samples(city_dir: Path):
    samples = []
    for f in sorted(city_dir.glob("*_gtFine_labelIds.png")):
        stem_id = f.name.replace("_gtFine_labelIds.png", "")
        samples.append(f"{city_dir.name}/{stem_id}")
    return samples

SPLITS_DIR.mkdir(parents=True, exist_ok=True)

train_samples = []
val_samples = []
train_cities = set()

all_train_city_dirs = [d for d in sorted(TRAIN_GT_DIR.iterdir()) if d.is_dir()]
for c_dir in all_train_city_dirs:
    city_name = c_dir.name
    samples = get_city_samples(c_dir)
    if city_name in VAL_CITIES:
        val_samples.extend(samples)
    else:
        train_cities.add(city_name)
        train_samples.extend(samples)
        
test_samples = []
test_cities = set()
all_val_city_dirs = [d for d in sorted(VAL_GT_DIR.iterdir()) if d.is_dir()]
for c_dir in all_val_city_dirs:
    test_cities.add(c_dir.name)
    test_samples.extend(get_city_samples(c_dir))
    
print("DATASET SPLIT STATISTICS")
print(f"Train set: {len(train_samples)} images from {len(train_cities)} cities:")
print(f"  {sorted(list(train_cities))}")
print(f"Val set:   {len(val_samples)} images from {len(VAL_CITIES)} cities:")
print(f"  {sorted(list(VAL_CITIES))}")
print(f"Test set:  {len(test_samples)} images from {len(test_cities)} cities (official val):")
print(f"  {sorted(list(test_cities))}")
print(f"Total train+val images from official train: {len(train_samples) + len(val_samples)} / 2975")

assert len(train_cities.intersection(VAL_CITIES)) == 0, "ERROR: City leakage between Train and Val!"
assert len(train_cities.intersection(test_cities)) == 0, "ERROR: City leakage between Train and Test!"
assert len(VAL_CITIES.intersection(test_cities)) == 0, "ERROR: City leakage between Val and Test!"

set_train_samples = set(train_samples)
set_val_samples = set(val_samples)
set_test_samples = set(test_samples)
assert len(set_train_samples.intersection(set_val_samples)) == 0, "ERROR: Duplicate samples between Train and Val!"
assert len(set_train_samples.intersection(set_test_samples)) == 0, "ERROR: Duplicate samples between Train and Test!"
assert len(set_val_samples.intersection(set_test_samples)) == 0, "ERROR: Duplicate samples between Val and Test!"

train_file = SPLITS_DIR / "train.txt"
val_file = SPLITS_DIR / "val.txt"
test_file = SPLITS_DIR / "test.txt"

with open(train_file, "w") as f:
    f.write("\n".join(train_samples) + "\n")
with open(val_file, "w") as f:
    f.write("\n".join(val_samples) + "\n")
with open(test_file, "w") as f:
    f.write("\n".join(test_samples) + "\n")
    
