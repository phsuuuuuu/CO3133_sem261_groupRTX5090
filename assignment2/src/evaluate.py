"""Cityscapes validation/test evaluation for [B, 19, H, W] segmentation logits."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.data import get_loaders
from src.metrics import ConfusionMatrix
from src.models import build_model

NUM_CLASSES = 19
IGNORE_INDEX = 255


def load_checkpoint(path, model):
    """Evaluation restores model weights; no optimizer is needed."""
    path = Path(path)
    if path.is_dir():
        path = path / "last.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    return path


def get_logits(output):
    return output["out"] if isinstance(output, dict) else output


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


@torch.inference_mode()
def validate(model, loader, device, amp=False, max_batches=0, metric=None):
    if max_batches < 0:
        raise ValueError("max_batches must be non-negative; use 0 for all batches")
    model.eval()
    if metric is None:
        metric = ConfusionMatrix(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, device=device)
    metric.reset()
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    use_amp = amp and device.type == "cuda"
    weighted_loss = 0.0
    loss_samples = 0
    processed = 0

    for batch_idx, (images, masks) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, dtype=torch.long, non_blocking=True)
        batch_size = images.size(0)
        processed += batch_size

        if not (masks != IGNORE_INDEX).any().item():
            continue

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = get_logits(model(images))
            loss = criterion(logits, masks)

        weighted_loss += loss.item() * batch_size
        loss_samples += batch_size
        metric.update(logits, masks)

    if processed == 0:
        raise ValueError("Evaluation loader produced no batches")

    if loss_samples == 0:
        raise ValueError("No valid labeled pixels in evaluation data")

    scores = metric.compute()
    return {
        "val_loss": weighted_loss / loss_samples,
        "miou": scores["miou"],
        "mdice": scores["mdice"],
        "pixel_acc": scores["pixel_acc"],
        "per_class_iou": scores["iou"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=["val", "test"])
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = args.checkpoint or cfg.get("checkpoint") or cfg.get("resume")
    if not checkpoint_path:
        parser.error("check checkpoint path in yaml")
    split = args.split or cfg.get("split", "val")
    if split not in ("val", "test"):
        parser.error("split must be 'val' or 'test'")
    max_batches = args.max_batches if args.max_batches is not None else int(cfg.get("max_batches"))
    if max_batches < 0:
        parser.error("use 0 for all batches")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    checkpoint_path = load_checkpoint(checkpoint_path, model)
    print("Checkpoint:", checkpoint_path)

    loader = select_loader(get_loaders(cfg), split)
    result = validate(model, loader, device, amp=bool(cfg.get("amp", False)), max_batches=max_batches)
    print(f"Split: {split} | Device: {device}")
    print(f"mIoU: {result['miou']:.4f} | Dice: {result['mdice']:.4f} | PixelAcc: {result['pixel_acc']:.4f}")
    if max_batches:
        print(f"Limited evaluation: at most {max_batches} batches.")


if __name__ == "__main__":
    main()
