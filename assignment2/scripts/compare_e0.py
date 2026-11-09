import json
from pathlib import Path

ASSIGNMENT_DIR = Path('D:/HCMUT/HK261/Deep_Learning/CO3133_sem261_groupRTX5090/assignment2')
models = [
    ('U-Net Baseline (Local RTX 5070 Ti)', 'checkpoints/unet_final_1024x512_s42/metrics_test.json'),
    ('DeepLabV3 (Frozen Backbone)', 'checkpoints/deeplabv3/local/metrics_test.json'),
    ('DeepLabV3 (Partial Fine-Tune layer 3+4)', 'checkpoints/deeplabv3/resnet_layer3_4_finetune/metrics_test.json'),
    ('DeepLabV3 (Full Fine-Tune)', 'checkpoints/deeplabv3/resnet50_full_deeplabv3/metrics_test.json'),
]

for label, rel in models:
    p = ASSIGNMENT_DIR / rel
    if p.exists():
        d = json.load(open(p, encoding='utf-8'))
        miou = d.get('miou', d.get('eval_miou', 0)) * 100
        dice = d.get('mdice', d.get('eval_dice', 0)) * 100
        acc = d.get('pixel_acc', 0) * 100
        loss = d.get('avr_loss', d.get('eval_loss', 0))
        print(f'{label} | Test mIoU: {miou:.2f}% | Mean Dice: {dice:.2f}% | Pixel Acc: {acc:.2f}% | Test Loss: {loss:.4f}')
