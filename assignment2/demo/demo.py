"""DeepLabV3 demo: image in, segmentation drawing out.

Output = [input | segmentation | overlay], saved to demo/output/.

Usage (from the assignment2 folder):
    python demo/demo.py                               # every image in demo/input/
    python demo/demo.py demo/input/berlin_1.png       # one image
    python demo/demo.py my_photo.jpg --model m1       # pick a model: m1, m2, m3
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
from PIL import Image

from src.common import get_logits
from src.transforms import SegTransform
from src.visualize import colorize_mask, load_model, unnormalize_image

DEMO_DIR = BASE_DIR / "demo"
MODELS = {
    "m1": ("configs/deeplabv3.yaml", "checkpoints/deeplabv3/local/best.pt"),
    "m2": ("configs/resnet_layer3_4_finetune.yaml",
           "checkpoints/deeplabv3/resnet_layer3_4_finetune/best.pt"),
    "m3": ("configs/resnet50_full_deeplabv3.yaml",
           "checkpoints/deeplabv3/resnet50_full_deeplabv3/best.pt"),
}


def segment(model, img_path, device):
    img = Image.open(img_path).convert("RGB")
    x, _ = SegTransform(is_train=False)(img, Image.new("L", img.size, 255))
    with torch.inference_mode():
        pred = get_logits(model(x[None].to(device))).argmax(1)[0].cpu().numpy()
    rgb, seg = unnormalize_image(x), colorize_mask(pred)
    overlay = (0.5 * rgb + 0.5 * seg).astype(np.uint8)
    return np.concatenate([rgb, seg, overlay], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", help="image paths (default: all of demo/input/)")
    parser.add_argument("--model", choices=MODELS, default="m2",
                        help="m1 frozen, m2 partial L3+L4 (default), m3 full")
    parser.add_argument("--out_dir", default=str(DEMO_DIR / "output"))
    args = parser.parse_args()

    images = [Path(p) for p in args.images] or sorted(
        p for p in (DEMO_DIR / "input").iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        parser.error("no images found; pass a path or put images in demo/input/")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, ckpt = MODELS[args.model]
    print(f"Model: {args.model} ({ckpt}) on {device}")
    model = load_model(BASE_DIR / cfg, BASE_DIR / ckpt, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for img_path in images:
        out_path = out_dir / f"{img_path.stem}_{args.model}.png"
        Image.fromarray(segment(model, img_path, device)).save(out_path)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()
