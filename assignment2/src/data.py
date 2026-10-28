import random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.transforms import SegTransform

class CityscapesSeg(Dataset):
    def __init__(self, data_dir, split_file, is_train=True):
        self.data_dir = Path(data_dir)
        self.transform = SegTransform(is_train=is_train)
        
        with open(split_file, "r") as f:
            self.samples = [line.strip() for line in f if line.strip()]

        is_official_val = "test.txt" in str(split_file)
        split_folder = "val" if is_official_val else "train"

        self.img_dir = self.data_dir / "leftImg8bit_trainvaltest" / "leftImg8bit" / split_folder
        self.gt_dir = self.data_dir / "gtFine_trainvaltest" / "gtFine" / split_folder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]  
        city, stem_id = item.split("/")
        img_path = self.img_dir / city / f"{stem_id}_leftImg8bit.png"
        mask_path = self.gt_dir / city / f"{stem_id}_gtFine_trainIds.png"

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        return self.transform(img, mask)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_loaders(cfg):
    """
    Returns train_loader, val_loader, test_loader given cfg dictionary or object.
    Images: float32 [B, 3, H, W] ImageNet-normalized.
    Masks:  int64 [B, H, W] with values in 0..18 and 255.
    """
    if isinstance(cfg, dict):
        base_dir = Path(cfg.get("base_dir", Path(__file__).resolve().parent.parent))
        data_dir = Path(cfg.get("data_dir", base_dir / "dataset"))
        splits_dir = Path(cfg.get("splits_dir", base_dir / "splits"))
        batch_size = cfg.get("batch_size", 4)
        num_workers = cfg.get("num_workers", 2)
        pin_memory = cfg.get("pin_memory", torch.cuda.is_available())
        seed = cfg.get("seed", 42)
    else:
        base_dir = Path(getattr(cfg, "base_dir", Path(__file__).resolve().parent.parent))
        data_dir = Path(getattr(cfg, "data_dir", base_dir / "dataset"))
        splits_dir = Path(getattr(cfg, "splits_dir", base_dir / "splits"))
        batch_size = getattr(cfg, "batch_size", 4)
        num_workers = getattr(cfg, "num_workers", 2)
        pin_memory = getattr(cfg, "pin_memory", torch.cuda.is_available())
        seed = getattr(cfg, "seed", 42)

    g = torch.Generator()
    g.manual_seed(seed)

    train_ds = CityscapesSeg(data_dir, splits_dir / "train.txt", is_train=True)
    val_ds = CityscapesSeg(data_dir, splits_dir / "val.txt", is_train=False)
    test_ds = CityscapesSeg(data_dir, splits_dir / "test.txt", is_train=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=seed_worker, generator=g,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
