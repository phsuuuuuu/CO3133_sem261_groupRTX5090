import math
import sys
import os
import torch
import pytest

# Add assignment2 root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.metrics import ConfusionMatrix


def test_confusion_matrix_manual():
    """Verify confusion matrix, per-class IoU, Dice, mIoU, mDice, and pixel accuracy against manual calculation."""
    num_classes = 3
    ignore_index = 255

    # 1 sample with 5 pixels:
    # target: [0, 0, 1, 2, 255]
    # pred:   [0, 1, 1, 0, 2]
    #
    # Valid pixels:
    # idx 0: target 0, pred 0 -> TP_0
    # idx 1: target 0, pred 1 -> FN_0, FP_1
    # idx 2: target 1, pred 1 -> TP_1
    # idx 3: target 2, pred 0 -> FN_2, FP_0
    # idx 4: target 255 -> ignored!
    #
    # Confusion Matrix (rows: target, cols: pred):
    #       pred 0   pred 1   pred 2
    # tgt 0:   1        1        0
    # tgt 1:   0        1        0
    # tgt 2:   1        0        0
    #
    # Per-class metrics:
    # Class 0: TP=1, FP=1, FN=1 -> IoU = 1/(1+1+1) = 1/3 ~ 0.333333, Dice = 2*1/(2*1+1+1) = 2/4 = 0.5
    # Class 1: TP=1, FP=1, FN=0 -> IoU = 1/(1+1+0) = 1/2 = 0.5,      Dice = 2*1/(2*1+1+0) = 2/3 ~ 0.666667
    # Class 2: TP=0, FP=0, FN=1 -> IoU = 0/(0+0+1) = 0.0,            Dice = 0/(0+0+1) = 0.0
    #
    # Aggregated metrics:
    # mIoU = (1/3 + 1/2 + 0) / 3 = 0.833333 / 3 ~ 0.277778
    # mDice = (0.5 + 2/3 + 0) / 3 = 1.166667 / 3 ~ 0.388889
    # Pixel accuracy = sum(TP) / sum(Valid) = (1 + 1 + 0) / 4 = 2 / 4 = 0.5

    target = torch.tensor([[0, 0, 1, 2, 255]], dtype=torch.int64)
    pred = torch.tensor([[0, 1, 1, 0, 2]], dtype=torch.int64)

    cm = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
    cm.update(pred, target)
    res = cm.compute()

    expected_mat = torch.tensor([
        [1, 1, 0],
        [0, 1, 0],
        [1, 0, 0]
    ], dtype=torch.int64)

    assert (torch.tensor(res["confusion_matrix"]) == expected_mat).all()
    assert math.isclose(res["iou"][0], 1.0 / 3.0, rel_tol=1e-5)
    assert math.isclose(res["iou"][1], 0.5, rel_tol=1e-5)
    assert math.isclose(res["iou"][2], 0.0, rel_tol=1e-5)
    assert math.isclose(res["miou"], (1/3 + 0.5 + 0) / 3, rel_tol=1e-5)

    assert math.isclose(res["dice"][0], 0.5, rel_tol=1e-5)
    assert math.isclose(res["dice"][1], 2.0 / 3.0, rel_tol=1e-5)
    assert math.isclose(res["dice"][2], 0.0, rel_tol=1e-5)
    assert math.isclose(res["mdice"], (0.5 + 2/3 + 0) / 3, rel_tol=1e-5)

    assert math.isclose(res["pixel_acc"], 0.5, rel_tol=1e-5)


def test_nan_class_exclusion():
    """Verify classes absent in both GT and prediction yield NaN and are excluded from mean metrics."""
    num_classes = 3
    # Only class 0 is present; classes 1 and 2 are absent in both target and prediction
    target = torch.tensor([[0, 0, 255]], dtype=torch.int64)
    pred = torch.tensor([[0, 0, 0]], dtype=torch.int64)

    cm = ConfusionMatrix(num_classes=num_classes, ignore_index=255)
    cm.update(pred, target)
    res = cm.compute()

    # Class 0: TP=2, FP=0, FN=0 -> IoU = 1.0, Dice = 1.0
    # Classes 1 & 2: NaN
    assert math.isclose(res["iou"][0], 1.0, rel_tol=1e-5)
    assert math.isnan(res["iou"][1])
    assert math.isnan(res["iou"][2])

    assert math.isclose(res["miou"], 1.0, rel_tol=1e-5)
    assert math.isclose(res["mdice"], 1.0, rel_tol=1e-5)
    assert math.isclose(res["pixel_acc"], 1.0, rel_tol=1e-5)


def test_cross_check_with_torchmetrics():
    """Cross-check IoU, mIoU, and pixel accuracy against torchmetrics standard implementations."""
    try:
        from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
    except ImportError:
        pytest.skip("torchmetrics is not installed")

    num_classes = 19
    torch.manual_seed(42)
    B, H, W = 4, 32, 32
    target = torch.randint(0, num_classes + 1, (B, H, W))
    target[target == num_classes] = 255  # Inject ignored pixels
    pred = torch.randint(0, num_classes, (B, H, W))

    cm = ConfusionMatrix(num_classes=num_classes, ignore_index=255)
    cm.update(pred, target)
    res = cm.compute()

    tm_jaccard = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=255, average='none')
    tm_iou_per_class = tm_jaccard(pred, target)
    
    tm_jaccard_mean = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=255, average='macro')
    tm_miou = tm_jaccard_mean(pred, target).item()

    tm_acc = MulticlassAccuracy(num_classes=num_classes, ignore_index=255, average='micro')
    tm_pixel_acc = tm_acc(pred, target).item()

    # Validate per-class IoU against torchmetrics
    for c in range(num_classes):
        our_val = res["iou"][c]
        tm_val = tm_iou_per_class[c].item()
        if math.isnan(our_val):
            assert math.isnan(tm_val)
        else:
            assert math.isclose(our_val, tm_val, abs_tol=1e-4)

    assert math.isclose(res["miou"], tm_miou, abs_tol=1e-4)
    assert math.isclose(res["pixel_acc"], tm_pixel_acc, abs_tol=1e-4)


if __name__ == "__main__":
    test_confusion_matrix_manual()
    test_nan_class_exclusion()
    print("Manual test passed successfully!")
    try:
        test_cross_check_with_torchmetrics()
        print("torchmetrics cross-check passed successfully!")
    except Exception as e:
        print(f"torchmetrics test skipped or failed: {e}")
