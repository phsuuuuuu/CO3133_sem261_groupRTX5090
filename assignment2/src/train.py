
"""Run: python -m src.train --config configs/deeplabv3.yaml"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.data import get_loaders
from src.evaluate import select_loader, validate
from src.metrics import ConfusionMatrix
from src.models.deeplabv3 import resnet_deeplabv3
MAX_TRAIN_BATCHES=621
MAX_VALIDATION_BATCHES=124 
MAX_TEST_BATCHES=125

def train_one_epoch(model, loader, optimizer, device, max_batches=0):
    model.train()
    model.backbone.eval()  

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    total_loss = 0
    total_samples = 0

    for i, (images, masks) in enumerate(loader):
        if max_batches and i >= max_batches:
            break

        images = images.to(device)
        masks = masks.to(device, dtype=torch.long)

        # Skip batches containing only ignored pixels
        if not (masks != 255).any().item():
            continue

        optimizer.zero_grad()
        logits = model.forward(images)["out"]
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return total_loss / total_samples


def run_directory(cfg, max_batches):
    if cfg.get("resume"):
        path = Path(cfg["resume"])
        return path if path.is_dir() else path.parent

    base = Path(cfg.get("checkpoint_dir", "checkpoints/deeplabv3"))

    if max_batches:
        base = base / "exp"

    path, suffix = base, 1
    while path.exists():
        path = base.parent / f"{base.name}_{suffix}"
        suffix += 1

    path.mkdir(parents=True)
    return path


def load_resume_checkpoint(path, model, optimizer):
    path = Path(path)
    if path.is_dir():
        path = path / "last.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def save_checkpoint(path, epoch, model, optimizer, best_miou):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_miou": best_miou,
    }, path)


def main(cfg):
    epochs = int(cfg["epochs"])
    max_batches = int(cfg["max_batches"])

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = resnet_deeplabv3(cfg).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["lr"])
    )

    start_epoch = 0
    best_miou = -1.0

    # Resume training
    if cfg.get("resume"):
        ckpt = load_resume_checkpoint(cfg["resume"], model, optimizer)
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt["best_miou"]
        epochs += start_epoch
        print("Resuming at epoch:", start_epoch + 1)

    out_dir = run_directory(cfg, max_batches)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Checkpoint and log directory:", out_dir)

    loaders = get_loaders(cfg)
    train_loader = select_loader(loaders, "train")
    val_loader = select_loader(loaders, "val")

    log_path = out_dir / "log.csv"

    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, max_batches
        )

        metric = ConfusionMatrix(num_classes=19, ignore_index=255, device=device)
        metrics = validate(
            model, val_loader, device,
            amp=False, max_batches=max_batches, metric=metric
        )

        miou = metrics["miou"]
        improved = miou > best_miou
        best_miou = max(best_miou, miou)

        save_checkpoint(
            out_dir / "last.pt", epoch, model, optimizer, best_miou
        )

        if improved:
            save_checkpoint(
                out_dir / "best.pt", epoch, model, optimizer, best_miou
            )

        new_file = not log_path.exists()
        with log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["epoch", "train_loss", "miou", "dice"])
            writer.writerow([
                epoch + 1, train_loss, miou, metrics["mdice"]
            ])

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"mIoU: {miou:.4f} | "
            f"Dice: {metrics['mdice']:.4f}"
        )

    print("Best mIoU:", best_miou)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/deeplabv3.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)

    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.resume:
        cfg["resume"] = args.resume
    if args.epochs is not None and args.epochs > 0:
        cfg["epochs"] = args.epochs
    if args.max_batches is not None and args.max_batches > 0:
        cfg["max_batches"] = args.max_batches

    main(cfg)
