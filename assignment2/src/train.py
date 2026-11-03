import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.common import set_seed, select_loader, get_logits
from src.cost import TrainingTimer
from src.data import get_loaders
from src.evaluate import validate
from src.metrics import ConfusionMatrix
from src.models import build_model


def train_one_epoch(model, loader, optimizer, device, max_batches=0, amp=False, scaler=None):
    model.train()
    backbone = getattr(model, "backbone", None)
    if backbone and not any(p.requires_grad for p in backbone.parameters()):
        backbone.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    total_samples = 0
    total_loss = 0
    use_amp = amp and device.type == "cuda"

    for i, (images, masks) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        if not (masks != 255).any().item():
            continue
        if images.size(0) <= 1:
            continue

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, dtype=torch.long, non_blocking=True)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = get_logits(model(images))
            loss = criterion(logits, masks)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        if (i + 1) % 50 == 0 or (i + 1) == max_batches:
            print(f"  Batch [{i + 1}/{max_batches}] - Avr_loss_current_batches: {loss.item():.4f}")

    if total_samples == 0:
        raise ValueError("No valid training samples were processed")
    return {"avr_loss": total_loss / total_samples, "total_samples": total_samples}


def run_directory(cfg):
    path = Path(cfg.get("checkpoint_dir"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_resume_checkpoint(path, model, optimizer):
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def save_checkpoint(path, epoch, model, optimizer, best_miou):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_miou": best_miou,
    }, path)


def main(cfg):
    # initialize
    out_dir = run_directory(cfg)
    loaders = get_loaders(cfg)
    train_loader = select_loader(loaders, "train")
    eval_loader = select_loader(loaders, "val")
    log_path = out_dir / "log.csv"
    epochs = int(cfg["epochs"])
    start_epochs = 1
    end_epochs = start_epochs + epochs
    best_miou = -1.0

    # basic
    max_train_batches = len(train_loader) if cfg.get("max_train_batches", 0) == 0 else min(len(train_loader), int(cfg["max_train_batches"]))
    max_eval_batches = len(eval_loader) if cfg.get("max_eval_batches", 0) == 0 else min(len(eval_loader), int(cfg["max_eval_batches"]))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = build_model(cfg).to(device)
    backbone_lr = cfg.get("backbone_lr")

    if backbone_lr is None:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(cfg["lr"]),
        )
    else:
        optimizer = torch.optim.AdamW([
            {
                "params": [
                    p for p in model.backbone.parameters()
                    if p.requires_grad
                ],
                "lr": float(backbone_lr),
            },
            {
                "params": [
                    p for p in model.classifier.parameters()
                    if p.requires_grad
                ],
                "lr": float(cfg["lr"]),
            },
        ])
    print("Device: ", device)

    use_amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if cfg.get("resume"):
        checkpoint = load_resume_checkpoint(cfg["resume"], model, optimizer)
        start_epochs = checkpoint["epoch"] + 1
        end_epochs = start_epochs + epochs
        best_miou = checkpoint["best_miou"]
        print(f"Resume at {start_epochs}")

    for run in range(start_epochs, end_epochs):
        print(f"Epochs {run} :")
        with TrainingTimer(device) as train_time:
            train_result = train_one_epoch(
                model, train_loader, optimizer, device,
                max_batches=max_train_batches, amp=use_amp, scaler=scaler
            )

        metric = ConfusionMatrix(device=device)
        with TrainingTimer(device) as eval_time:
            eval_result = validate(
                model, eval_loader, device,
                amp=use_amp, max_batches=max_eval_batches, metric=metric
            )

        improved = eval_result["miou"] > best_miou
        best_miou = max(best_miou, eval_result["miou"])
        save_checkpoint(out_dir / "last.pt", run, model, optimizer, best_miou)
        if improved:
            save_checkpoint(out_dir / "best.pt", run, model, optimizer, best_miou)

        new_file = not log_path.exists()
        record = {
            "epoch": run,
            "device_name": train_time.device_name,
            "train_seconds": train_time.time,
            "train_size": train_result["total_samples"],
            "train_avr_loss": train_result["avr_loss"],
            "eval_seconds": eval_time.time,
            "eval_size": eval_result["total_samples"],
            "eval_loss": eval_result["avr_loss"],
            "eval_miou": eval_result["miou"],
            "eval_dice": eval_result["mdice"],
            "eval_accuracy": eval_result["pixel_acc"],
            "eval_per_class_iou": eval_result["per_class_iou"],
        }
        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(record)

        print(" | ".join(
            f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
            for key, value in record.items()
            if key != "eval_per_class_iou"
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a segmentation model using a YAML configuration.")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    main(config)
