import random
import numpy as np
import torch 

def select_loader(loaders, split="val"):
    """Works with get_loaders(cfg) returning a dict or a tuple."""
    if isinstance(loaders, dict):
        if split not in loaders:
            raise KeyError(f"Loader for split {split!r} is unavailable")
        return loaders[split]
    positions = {"train": 0, "val": 1, "test": 2}
    idx = positions[split]
    if len(loaders) <= idx:
        raise ValueError(f"get_loaders(cfg) did not return a {split!r} loader")
    return loaders[idx]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_logits(output):
    return output["out"] if isinstance(output, dict) else output

def load_config(path):
    pass

def smoke_config(cfg):
    pass

def build_model(cfg, pretrained_backbone=True):
    pass

def resize_logits(output, masks):
    pass

def select_device(name='auto'):
    pass
