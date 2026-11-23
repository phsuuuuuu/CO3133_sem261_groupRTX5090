import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from src.cost import TrainingTimer
from src.data_and_transform import Transform, KittiDepthDataset
from src.loss_and_metric import m1_loss, m2_loss, m3_loss, Metric
from src.models import m1_rgb, m2_lidar, m3_early

MODELS = {"m1_rgb": m1_rgb.build_model, "m2_lidar": m2_lidar.build_model, "m3_early": m3_early.build_model}
LOSSES = {"m1_rgb": m1_loss, "m2_lidar": m2_loss, "m3_early": m3_loss}
#1 Set up
def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)

#checkpoint:
def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_rmse, bad_epochs):
    #TODO pls make it fit with ur local, im doing colab ver prob dont use it
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_rmse": best_rmse,
        "bad_epochs": bad_epochs,
    }, path)
    
def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"] + 1, ckpt["best_rmse"], ckpt["bad_epochs"]    # next epoch to run

#2 build
def build_model(cfg):
    return MODELS[cfg["model"]["name"]](cfg)    

def build_loss(cfg):
    return LOSSES[cfg["model"]["name"]]

def build_optimizer(model,cfg):
    train = cfg["train"]
    lr  = train["lr"]

    # AdamW: encoder lr = 0.1 x decoder lr
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": lr * train["encoder_lr_scale"]},
        {"params": model.decoder.parameters(), "lr": lr},
    ], weight_decay=train["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train["epochs"])
    return optimizer, scheduler

def build_scaler(cfg, device):
    "With AMP, much of the model runs in float16 instead of float32. That's faster and uses less GPU memory but when scaler up  loss some detail so use  sacler"
    use_amp = cfg["train"]["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    return scaler, use_amp

def build_data(cfg, split):
    data_dir = Path(cfg["data"]["data_dir"])
    return KittiDepthDataset(data_dir / split, data_dir, split, Transform(), batch_size=cfg["data"]["batch_size"], seed=cfg["seed"])

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#3  one  epoch
def train_one_epoch(model, data, loss_fn, optimizer, scaler, device, use_amp, max_batches=0):
    model.train()
    total_loss, n = 0.0, 0
    for i, batch in enumerate(data.get_batch()):
        if max_batches and i >= max_batches:  break
        rgb = batch["rgb"].to(device)
        lidar = batch["lidar"].to(device)
        gt =  batch["gt"].to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out  = model(rgb, lidar)
            loss = loss_fn(out["d"], gt)
        scaler.scale(loss).backward()    # loss x big number -> backward
        scaler.step(optimizer)    # grads / big number -> optimizer step
        scaler.update()    # adjust the big number for next batch

        total_loss += loss.item()
        n +=  1
    return total_loss/max(n,1)

@torch.no_grad()
def validate(model, data, device, use_amp, max_depth, max_batches=0):
    model.eval()
    metric = Metric(max_depth=max_depth, by_range=False)
    for i, batch in enumerate(data.get_batch()):
        if max_batches and i >= max_batches: break
        rgb = batch["rgb"].to(device)    # (B, 3, H, W)
        lidar = batch["lidar"].to(device)    # (B, 2, H, W)
        gt = batch["gt"].to(device)    # (B, 1, H, W)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(rgb, lidar)
        metric.update(out["d"], gt)
    return metric.compute()   



def train_n_epochs(model, train_data, val_data, loss_fn, optimizer, scheduler, scaler, device, use_amp, cfg, run_dir, max_batches=0):
    train = cfg["train"]
    max_depth = cfg["data"]["max_depth"]
    last_path, best_path, log_path = run_dir / "last.pt", run_dir / "best.pt", run_dir / "log.csv"

    # resume if last.pt exists
    start_epoch, best_rmse, bad_epochs = 0, float("inf"), 0
    if last_path.exists():
        start_epoch, best_rmse, bad_epochs = load_checkpoint(last_path, model, optimizer, scheduler, scaler, device)
        print(f"resume from epoch {start_epoch}")

    for epoch in range(start_epoch, train["epochs"]):
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        lr = optimizer.param_groups[1]["lr"]    # decoder lr used this epoch

        with TrainingTimer(device) as timer:
            train_loss = train_one_epoch(model, train_data, loss_fn, optimizer, scaler, device, use_amp, max_batches)
            val = validate(model, val_data, device, use_amp, max_depth, max_batches)
        scheduler.step()
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0    # GB
        
        # log.csv
        new_file = not log_path.exists()
        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            if new_file: w.writerow(["epoch", "train_loss", "val_rmse", "val_mae", "lr", "time", "peak_mem"])
            w.writerow([epoch, train_loss, val["rmse"], val["mae"], lr, timer.time, peak_mem])
        print(f"epoch {epoch} | loss {train_loss:.4f} | val_rmse {val['rmse']:.4f} | val_mae {val['mae']:.4f} | {timer.time:.1f}s | {peak_mem:.2f}GB")

        # best.pt by val rmse + early stopping counter
        if val["rmse"] < best_rmse:
            best_rmse, bad_epochs = val["rmse"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_rmse": best_rmse}, best_path)
        else:
            bad_epochs += 1

        save_checkpoint(last_path, model, optimizer, scheduler, scaler, epoch, best_rmse, bad_epochs)

        if bad_epochs >= train["early_stopping_patience"]:
            print(f"early stop: no better val rmse for {bad_epochs} epochs")
            break

def setup():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    args = p.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # checkpoints/<model>_base_352x1216_s42/ + config.yaml
    run_dir = Path(cfg["output"]["checkpoints_dir"]) / f"{cfg['model']['name']}_{cfg['run_id']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    set_seed(cfg["seed"])    # 666 from base.yaml
    return cfg, device, run_dir, cfg["train"]["max_batches"]

if __name__ == "__main__":
    cfg, device, run_dir, max_batches = setup()
    train_data = build_data(cfg, "train")    # data_dir/train/, shuffled
    val_data = build_data(cfg, "val")    # data_dir/val/, different drives
    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)
    optimizer, scheduler = build_optimizer(model, cfg)
    scaler, use_amp = build_scaler(cfg, device)

    print(f"run {run_dir.name} | device {device} | amp {use_amp}")
    train_n_epochs(model, train_data, val_data, loss_fn, optimizer, scheduler, scaler, device, use_amp, cfg, run_dir, max_batches)
