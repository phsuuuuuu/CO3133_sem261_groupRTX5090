"""M3: summarise every training run from its log.csv / metrics_test.json and plot curves.

All numbers come from the files listed in RUNS; nothing is typed in by hand.
Output: docs/figures/m3/training_summary.json, docs/figures/m3/a2_m3_training_curves.png
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "docs" / "figures" / "m3"
RARE_10_IDS = [17, 12, 16, 6, 15, 14, 18, 3, 7, 4]  # 10 rarest classes by train pixel count (configs/class_weights.json)

RUNS = [
    ("unet", "U-Net (scratch, CE)", "configs/unet_final.yaml", "checkpoints/unet_final_1024x512_s42"),
    ("frozen", "DLV3 frozen backbone (CE)", "configs/deeplabv3.yaml", "checkpoints/deeplabv3/local"),
    ("partial", "DLV3 partial L3+L4 (CE)", "configs/resnet_layer3_4_finetune.yaml", "checkpoints/deeplabv3/resnet_layer3_4_finetune"),
    ("full", "DLV3 full fine-tune (CE)", "configs/resnet50_full_deeplabv3.yaml", "checkpoints/deeplabv3/resnet50_full_deeplabv3"),
    ("wce", "DLV3 partial L3+L4 (WCE)", "configs/dlv3_wce.yaml", "checkpoints/dlv3_wce_1024x512_s666"),
    ("cedice", "DLV3 partial L3+L4 (CE+Dice)", "configs/dlv3_cedice.yaml", "checkpoints/dlv3_cedice_1024x512_s666"),
]


def rare10(per_class):
    return sum(per_class[i] for i in RARE_10_IDS) / len(RARE_10_IDS)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for key, label, cfg, ckpt_dir in RUNS:
        rows = list(csv.DictReader(open(BASE_DIR / ckpt_dir / "log.csv", encoding="utf-8")))
        ep = [int(r["epoch"]) for r in rows]
        tr = [float(r["train_avr_loss"]) for r in rows]
        va = [float(r["eval_miou"]) * 100 for r in rows]
        best = max(rows, key=lambda r: float(r["eval_miou"]))
        best_pc = json.loads(best["eval_per_class_iou"])
        info = {
            "label": label,
            "config": cfg,
            "log": f"{ckpt_dir}/log.csv",
            "device": rows[0]["device_name"],
            "epochs_run": len(rows),
            "best_epoch": int(best["epoch"]),
            "best_val_miou": float(best["eval_miou"]) * 100,
            "best_val_dice": float(best["eval_dice"]) * 100,
            "best_val_rare10_iou": rare10(best_pc) * 100,
            "final_epoch_val_miou": va[-1],
            "final_train_loss": tr[-1],
            "total_train_seconds": sum(float(r["train_seconds"]) for r in rows),
            "total_eval_seconds": sum(float(r["eval_seconds"]) for r in rows),
            "mean_epoch_train_seconds": sum(float(r["train_seconds"]) for r in rows) / len(rows),
            "first_epoch_val_miou_ge_55": next((e for e, v in zip(ep, va) if v >= 55.0), None),
        }
        test_json = BASE_DIR / ckpt_dir / "metrics_test.json"
        if test_json.exists():
            t = json.load(open(test_json, encoding="utf-8"))
            info["test"] = {
                "file": f"{ckpt_dir}/metrics_test.json",
                "checkpoint_epoch": t.get("epoch"),
                "device": t.get("device_name"),
                "miou": t["miou"] * 100,
                "mdice": t["mdice"] * 100,
                "pixel_acc": t["pixel_acc"] * 100,
                "loss": t["avr_loss"],
                "rare10_iou": rare10(t["per_class_iou"]) * 100,
                "per_class_iou": [v * 100 for v in t["per_class_iou"]],
            }
        summary[key] = info
        ax1.plot(ep, tr, label=label)
        ax2.plot(ep, va, label=label)

    ax1.set_title("Training loss (each run uses its own loss)")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss"); ax1.grid(alpha=0.3)
    ax2.set_title("Validation mIoU (494 images, 4 held-out cities)")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("val mIoU (%)"); ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "a2_m3_training_curves.png", dpi=200)
    plt.close()

    with open(OUT_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    for k, v in summary.items():
        t = v.get("test", {})
        print(f"{k:8s} dev={v['device'][:22]:22s} ep={v['epochs_run']} best_ep={v['best_epoch']} "
              f"val={v['best_val_miou']:.2f} valRare={v['best_val_rare10_iou']:.2f} "
              f"test={t.get('miou', float('nan')):.2f} dice={t.get('mdice', float('nan')):.2f} "
              f"acc={t.get('pixel_acc', float('nan')):.2f} testRare={t.get('rare10_iou', float('nan')):.2f} "
              f"ckptEp={t.get('checkpoint_epoch')} train_s={v['total_train_seconds']:.0f} "
              f"ep_s={v['mean_epoch_train_seconds']:.1f} first55={v['first_epoch_val_miou_ge_55']}")


if __name__ == "__main__":
    main()
