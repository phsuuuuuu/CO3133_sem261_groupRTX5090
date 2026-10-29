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

def get_logits(output):
     return output["out"] if isinstance(output, dict) else output

def load_config(path):
    # TODO: Resolve data/output paths from assignment2, independent of the shell directory.
    pass


def set_seed(seed):
    # TODO: Seed the libraries used by model initialization and dataset transforms.
    pass


def smoke_config(cfg):
    # TODO: Keep random-data checkpoints separate from actual Cityscapes experiments.
    pass


def get_loaders(cfg):
    # TODO: Synthetic data tests the engine only; it is never a Cityscapes substitute.
    pass


def build_model(cfg, pretrained_backbone=True):
    # TODO: The epoch functions accept other segmentation models, without implementing U-Net.
    pass


def resize_logits(output, masks):
    # TODO: Support torchvision dictionaries and tensor outputs such as a future U-Net.
    pass


def select_device(name='auto'):
    # TODO: AMP is enabled separately only when the selected device is CUDA.
    pass
