"""M3: compute-cost benchmark (params, FLOPs, inference latency, peak VRAM).

Inference: batch 1, 3x512x1024 input, eval mode, AMP (fp16 autocast) like src/evaluate.py,
30 warm-up + 200 timed iterations with CUDA events.
Training step: batch 8, 3x512x512 crops, AMP + GradScaler, AdamW over trainable params (as src/train.py).
Output: docs/figures/m3/cost.json
"""
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torch.utils.flop_counter import FlopCounterMode

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from src.common import get_logits  # noqa: E402
from src.models import build_model  # noqa: E402

OUT = BASE_DIR / "docs" / "figures" / "m3" / "cost.json"
VARIANTS = [
    ("unet", {"model": "unet", "base_channels": 32}),
    ("frozen", {"model": "deeplabv3_resnet50"}),
    ("partial", {"model": "resnet_layer3_4_finetune"}),
    ("full", {"model": "resnet50_full_deeplabv3"}),
]


def latency(model, x, warmup=30, iters=200):
    times = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(warmup):
            get_logits(model(x))
        torch.cuda.synchronize()
        for _ in range(iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            get_logits(model(x))
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
    return times


def main():
    torch.manual_seed(666)
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    res = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "protocol": {
            "inference": "batch 1, input 3x512x1024, eval mode, fp16 autocast, 30 warm-up + 200 timed iters (CUDA events), argmax excluded",
            "train_step": "batch 8, input 3x512x512, fp16 autocast + GradScaler, AdamW on trainable params, CE loss, 3 warm-up + 10 timed steps",
            "flops": "torch.utils.flop_counter on 1x3x512x1024 forward (counts 2 FLOPs per MAC)",
        },
        "models": {},
    }
    for key, cfg in VARIANTS:
        model = build_model(cfg).to(dev).eval()
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        x = torch.randn(1, 3, 512, 1024, device=dev)
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(x)
        flops = fc.get_total_flops()

        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t = latency(model, x)
        infer_peak = torch.cuda.max_memory_allocated() / 2**20

        model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        crit = nn.CrossEntropyLoss(ignore_index=255)
        xb = torch.randn(8, 3, 512, 512, device=dev)
        yb = torch.randint(0, 19, (8, 512, 512), device=dev)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        step_times = []
        for i in range(13):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(get_logits(model(xb)), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            e.record(); torch.cuda.synchronize()
            if i >= 3:
                step_times.append(s.elapsed_time(e))
        train_peak = torch.cuda.max_memory_allocated() / 2**20

        res["models"][key] = {
            "total_params": total,
            "trainable_params": trainable,
            "fp32_weights_MB": total * 4 / 1e6,
            "gflops_forward_1x3x512x1024": flops / 1e9,
            "latency_ms_median": statistics.median(t),
            "latency_ms_mean": statistics.mean(t),
            "latency_ms_std": statistics.stdev(t),
            "fps_from_median": 1000.0 / statistics.median(t),
            "inference_peak_vram_MB": infer_peak,
            "train_step_ms_median": statistics.median(step_times),
            "train_step_peak_vram_MB": train_peak,
        }
        print(key, json.dumps(res["models"][key]))
        del model, opt
        torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("saved", OUT)


if __name__ == "__main__":
    main()
