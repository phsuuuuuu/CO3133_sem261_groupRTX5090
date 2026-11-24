import argparse, csv, importlib, json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch, yaml
from src.data_and_transform import KittiDepthDataset, Transform
from src.loss_and_metric import Metric


def load_cfg(ckpt, ckpt_path, cfg_path):
    if cfg_path:
        return yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    for p in (ckpt_path.parent / "config.yaml", Path("results") / ckpt_path.parent.name / "config.yaml"):
        if p.exists():
            return yaml.safe_load(p.read_text(encoding="utf-8"))
    if isinstance(ckpt, dict):
        for k in ("config", "cfg"):
            if isinstance(ckpt.get(k), dict):
                return ckpt[k]
    return yaml.safe_load((Path(__file__).resolve().parent.parent / "configs/base.yaml").read_text(encoding="utf-8"))


def load_weights(model, ckpt):
    # support both raw state_dict and wrapped checkpoints (e.g. {"model": ...})
    sd = next((ckpt[k] for k in ("model", "model_state_dict", "state_dict") if isinstance(ckpt, dict) and k in ckpt), ckpt)
    model.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()})


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def n_valid(d):
    # Metric.compute() currently returns "total_valid " with a trailing space
    return d.get("n_valid", d.get("total_valid ", d.get("total_valid", 0)))


@torch.inference_mode()
def evaluate_one(ckpt_path, cfg_path, split, device, save_preds=0, data_dir=None):
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = load_cfg(ckpt, ckpt_path, cfg_path)
    # train.py saves ckpts under <model>_<run_id>; prefer it so models don't overwrite each other
    run_id = ckpt_path.parent.name if ckpt_path.parent.name not in ("", ".", "checkpoints") else f"{cfg['model']['name']}_{cfg.get('run_id', 'run')}"
    max_depth = float(cfg.get("data", {}).get("max_depth", 80.0))
    use_amp = bool(cfg.get("train", {}).get("amp", True)) and device.type == "cuda"

    model = importlib.import_module(f"src.models.{cfg['model']['name']}").build_model(cfg).to(device).eval()
    load_weights(model, ckpt)

    root = Path(data_dir or cfg["data"]["data_dir"])
    dataset = KittiDepthDataset(root / split, root, split, Transform(), batch_size=cfg["data"]["batch_size"], seed=cfg.get("seed", 42))

    metric = Metric(max_depth=max_depth, by_range=True)
    preds, saved, n_images = [], 0, 0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for batch in dataset.get_batch():
        rgb = batch["rgb"].to(device, non_blocking=True)
        lidar = batch["lidar"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            pred = model(rgb, lidar)["d"].clamp(0.0, max_depth)
        metric.update(pred, gt)
        # save_preds: 0 = skip, -1 = save all, >0 = cap at N samples
        if save_preds != 0 and (save_preds < 0 or saved < save_preds):
            take = pred.size(0) if save_preds < 0 else min(pred.size(0), save_preds - saved)
            preds.append(pred[:take, 0].half().cpu().numpy())
            saved += take
        n_images += rgb.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0

    res_root = Path(cfg.get("output", {}).get("results_dir", "results"))
    out_dir = res_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if preds:
        np.save(out_dir / f"preds_{split}.npy", np.concatenate(preds, axis=0))

    res = metric.compute()
    # Metric(by_range=True) nests each depth-range dict directly in res
    pr = {k: {"rmse": v["rmse"], "mae": v["mae"], "n_valid": n_valid(v)} for k, v in res.items() if isinstance(v, dict)}
    summary = {
        "run_id": run_id,
        "model": cfg["model"]["name"],
        "split": split,
        "ckpt": str(ckpt_path),
        "commit": git_hash(),
        "rmse": res["rmse"],
        "mae": res["mae"],
        "n_valid": n_valid(res),
        "n_images": n_images,
        "time_s": dt,
        "ms_per_img": dt * 1000.0 / max(n_images, 1),
        "fps": n_images / max(dt, 1e-9),
    }
    (out_dir / f"metrics_{split}.json").write_text(json.dumps({**summary, "per_range": pr}, indent=2), encoding="utf-8")

    with (out_dir / "per_range.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["range", "rmse", "mae", "n_valid"])
        w.writeheader()
        for k, v in [*pr.items(), ("all", summary)]:
            w.writerow({"range": k, "rmse": v["rmse"], "mae": v["mae"], "n_valid": v["n_valid"]})

    sum_path = res_root / "summary.csv"
    write_hdr = not sum_path.exists() or sum_path.stat().st_size == 0
    with sum_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if write_hdr:
            w.writeheader()
        w.writerow(summary)

    print(f"[{run_id} | {split}] RMSE={res['rmse']:.4f} MAE={res['mae']:.4f} ({summary['ms_per_img']:.2f} ms/img, {summary['fps']:.1f} FPS)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--save-preds", type=int, nargs="?", const=-1, default=0)
    p.add_argument("--data-dir", default=None)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for c in args.ckpt:
        evaluate_one(c, args.config, args.split, device, args.save_preds, args.data_dir)


if __name__ == "__main__":
    main()
