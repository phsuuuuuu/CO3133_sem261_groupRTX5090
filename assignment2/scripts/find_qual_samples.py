import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
import yaml

from src.common import get_logits, select_loader
from src.data import get_loaders
from src.metrics import ConfusionMatrix
from src.models import build_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load U-Net
    unet_cfg = yaml.safe_load(open(BASE_DIR / "configs/unet_final.yaml"))
    unet = build_model(unet_cfg).to(device)
    unet.load_state_dict(torch.load(BASE_DIR / "checkpoints/unet_final_1024x512_s42/best.pt", map_location=device)["model"])
    unet.eval()

    # Load DeepLabV3
    dlv3_cfg = yaml.safe_load(open(BASE_DIR / "configs/resnet_layer3_4_finetune.yaml"))
    dlv3 = build_model(dlv3_cfg).to(device)
    dlv3.load_state_dict(torch.load(BASE_DIR / "checkpoints/deeplabv3/resnet_layer3_4_finetune/best.pt", map_location=device)["model"])
    dlv3.eval()

    loaders = get_loaders({"data_dir": BASE_DIR / "dataset", "splits_dir": BASE_DIR / "splits", "batch_size": 1, "num_workers": 2})
    test_loader = select_loader(loaders, "test")
    samples = test_loader.dataset.samples

    scores = []
    with torch.no_grad():
        for idx, (img, mask) in enumerate(test_loader):
            img = img.to(device)
            mask = mask.to(device)

            pred_u = get_logits(unet(img)).argmax(dim=1)
            pred_d = get_logits(dlv3(img)).argmax(dim=1)

            valid = (mask != 255)
            if not valid.any():
                continue

            m_u = ConfusionMatrix(num_classes=19, ignore_index=255, device=device)
            m_u.update(pred_u, mask)
            miou_u = float(m_u.compute()["miou"])

            m_d = ConfusionMatrix(num_classes=19, ignore_index=255, device=device)
            m_d.update(pred_d, mask)
            miou_d = float(m_d.compute()["miou"])

            scores.append({
                "idx": idx,
                "name": samples[idx],
                "miou_u": miou_u,
                "miou_d": miou_d,
                "diff": miou_d - miou_u
            })
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx+1}/500")

    import json
    with open(BASE_DIR / "scripts/scores_test.json", "w") as f:
        json.dump(scores, f, indent=2)

    good = sorted(scores, key=lambda x: -(x["miou_d"] + x["miou_u"]))
    print("\nTop Good (Highest combined mIoU):")
    for s in good[:6]:
        print(f"  idx: {s['idx']}, name: {s['name']}, U-Net: {s['miou_u']:.3f}, DLV3: {s['miou_d']:.3f}")

    hard = sorted([s for s in scores if s["miou_d"] > 0.65 and s["diff"] > 0.20], key=lambda x: -x["diff"])
    print("\nTop Hard (DeepLabV3 >> U-Net):")
    for s in hard[:6]:
        print(f"  idx: {s['idx']}, name: {s['name']}, U-Net: {s['miou_u']:.3f}, DLV3: {s['miou_d']:.3f}, diff: +{s['diff']:.3f}")

    fail = sorted([s for s in scores if s["miou_d"] < 0.45], key=lambda x: x["miou_d"])
    print("\nTop Failure (Difficult for both):")
    for s in fail[:6]:
        print(f"  idx: {s['idx']}, name: {s['name']}, U-Net: {s['miou_u']:.3f}, DLV3: {s['miou_d']:.3f}")


if __name__ == "__main__":
    main()
