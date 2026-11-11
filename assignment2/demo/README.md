# DeepLabV3 demo: image in, segmentation out

- `input/`: images to segment. Add any street photo here.
- `output/`: saved results, each showing input | segmentation | overlay.

`berlin_*.png` come from the unlabeled Cityscapes test split, which the model
never saw. `munster.png` and `frankfurt.png` come from the evaluation set.

## Run

From the `assignment2` folder in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1

python demo/demo.py                              # all images in demo/input/
python demo/demo.py demo/input/berlin_1.png      # one image
python demo/demo.py my_photo.jpg --model m1      # choose model

start demo/output/berlin_1_m2.png
```

Outputs are named `<image>_<model>.png`.

| `--model` | Model | Checkpoint |
| --- | --- | --- |
| `m1` | Frozen backbone | `checkpoints/deeplabv3/local/best.pt` |
| `m2` (default) | Partial fine-tune L3+L4 | `checkpoints/deeplabv3/resnet_layer3_4_finetune/best.pt` |
| `m3` | Full fine-tune | `checkpoints/deeplabv3/resnet50_full_deeplabv3/best.pt` |




.\.venv\Scripts\Activate.ps1



python -m src.evaluate --config configs/resnet_layer3_4_finetune.yaml --checkpoint checkpoints/deeplabv3/resnet_layer3_4_finetune/best.pt --split test --max_batches 0 --save_json results/partial_metrics_test.json
