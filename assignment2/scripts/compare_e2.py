import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# 19 Cityscapes evaluation classes
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

# 10 rarest classes sorted by pixel count ascending from train.txt
# [17, 12, 16, 6, 15, 14, 18, 3, 7, 4]
# (motorcycle, rider, train, traffic light, bus, truck, bicycle, wall, traffic sign, fence)
RARE_10_IDS = [17, 12, 16, 6, 15, 14, 18, 3, 7, 4]

RUNS = [
    ("Baseline CE (Partial Fine-Tune L3+4)", "checkpoints/deeplabv3/resnet_layer3_4_finetune/metrics_test.json"),
    ("E2: Weighted CE (1+log)", "checkpoints/dlv3_wce_1024x512_s666/metrics_test.json"),
    ("E2: Combined CE + Dice (λ=1.0)", "checkpoints/dlv3_cedice_1024x512_s666/metrics_test.json"),
]

def load_result(rel_path):
    p = BASE_DIR / rel_path
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    print("=" * 85)
    print("                    E2 ABLATION STUDY: LOSS FUNCTIONS")
    print("=" * 85)
    
    results = []
    for label, rel_path in RUNS:
        data = load_result(rel_path)
        if data is None:
            results.append((label, rel_path, None))
            continue
        
        miou = data.get("miou", data.get("eval_miou", 0.0)) * 100
        mdice = data.get("mdice", data.get("eval_dice", 0.0)) * 100
        acc = data.get("pixel_acc", 0.0) * 100
        loss = data.get("avr_loss", data.get("eval_loss", 0.0))
        per_class = data.get("per_class_iou", [])
        
        if per_class and len(per_class) == 19:
            rare_iou = [per_class[i] * 100 for i in RARE_10_IDS]
            rare_miou = sum(rare_iou) / len(rare_iou)
        else:
            rare_miou = 0.0

        results.append((label, rel_path, {
            "miou": miou,
            "rare_miou": rare_miou,
            "mdice": mdice,
            "acc": acc,
            "loss": loss,
            "per_class": per_class
        }))

    print(f"{'Method / Loss':<38} | {'mIoU (%)':<9} | {'Rare-10 mIoU':<12} | {'Dice (%)':<9} | {'Pixel Acc':<9}")
    print("-" * 85)
    
    baseline_stats = results[0][2]
    for label, rel_path, stats in results:
        if stats is None:
            print(f"{label:<38} | NOT FOUND (Run training & evaluation first)")
            continue
        
        delta_str = ""
        if baseline_stats and stats != baseline_stats:
            d_miou = stats["miou"] - baseline_stats["miou"]
            d_rare = stats["rare_miou"] - baseline_stats["rare_miou"]
            delta_str = f" [Δ all: {d_miou:+.2f}%, Δ rare: {d_rare:+.2f}%]"
            
        print(f"{label:<38} | {stats['miou']:>8.2f}% | {stats['rare_miou']:>11.2f}% | {stats['mdice']:>8.2f}% | {stats['acc']:>8.2f}%{delta_str}")

    print("=" * 85)
    print("Tiêu chí chấp nhận E2 (theo rubric):")
    print("  * Chấp nhận giả thuyết nếu mIoU nhóm 10 lớp hiếm tăng >= 3.00% mà mIoU chung không giảm quá 1.00%")
    print("=" * 85)

if __name__ == "__main__":
    main()
