"""Cityscapes validation/test evaluation for [B, 19, H, W] segmentation logits."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.common import select_loader, get_logits
from src.cost import TrainingTimer
from src.data import get_loaders
from src.metrics import ConfusionMatrix
from src.models import build_model

NUM_CLASSES = 19
IGNORE_INDEX = 255


def load_checkpoint(path, model):
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    return checkpoint



@torch.inference_mode()
def validate(model, loader, device, amp=False, max_batches=0, metric=None):
    if max_batches < 0:
        raise ValueError("max_batches must be non-negative; use 0 for all batches")
    model.eval()
    if metric is None: metric = ConfusionMatrix(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, device=device)
    metric.reset()
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    use_amp = amp and device.type == "cuda"
    total_samples=0
    total_loss=0
    total_batches = min(len(loader), max_batches) if max_batches else len(loader)

    for i, (images, masks) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, dtype=torch.long, non_blocking=True)

        if not (masks != IGNORE_INDEX).any().item():
            continue

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = get_logits(model(images))
            loss = criterion(logits, masks)

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        metric.update(logits, masks)

        if (i + 1) % 50 == 0 or (i + 1) == total_batches:
            print(f"  Batch [{i + 1}/{total_batches}] - Eval_loss_current_batch: {loss.item():.4f}")

    if total_samples == 0:
        raise ValueError("No valid labeled pixels in evaluation data")

    scores = metric.compute()
    return {
        "avr_loss": total_loss / total_samples,
        "total_samples": total_samples,
        "miou": scores["miou"],
        "mdice": scores["mdice"],
        "pixel_acc": scores["pixel_acc"],
        "per_class_iou": scores["iou"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=["val", "test"])
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--save_json", default=None, help="Path to save metrics json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = args.checkpoint or cfg.get("checkpoint") or cfg.get("resume")
    if not checkpoint_path:
        parser.error("check checkpoint path in yaml")
    split = args.split or cfg.get("split", "val")
    if split not in ("val", "test"):
        parser.error("split must be 'val' or 'test'")
    max_batches = (
        args.max_batches
        if args.max_batches is not None
        else int(cfg.get("max_eval_batches", 0))
    )
    if max_batches < 0:
        parser.error("use 0 for all batches")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model)
    print("Checkpoint:", checkpoint_path)

    loader = select_loader(get_loaders(cfg), split)
    with TrainingTimer(device) as eval_time:
        result = validate(model, loader, device, amp=bool(cfg.get("amp", False)), max_batches=max_batches)
    record = {
        "epoch": checkpoint["epoch"],
        "split": split,
        "device_name": eval_time.device_name,
        "eval_seconds": eval_time.time,
        "eval_size": result["total_samples"],
        "eval_loss": result["avr_loss"],
        "eval_miou": result["miou"],
        "eval_dice": result["mdice"],
        "pixel_acc": result["pixel_acc"],
    }
    print(" | ".join(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}" for key, value in record.items()))
    if max_batches:
        print(f"Limited evaluation: at most {max_batches} batches.")

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({**result, **record}, f, indent=4)
        print("Metrics saved to:", save_path)


if __name__ == "__main__":
    main()
