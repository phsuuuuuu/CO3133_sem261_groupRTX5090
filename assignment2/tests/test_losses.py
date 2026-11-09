import math
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.losses import compute_weight_class, WeightedCELoss, SoftDiceLoss, CEDiceLoss


def test_weight_class():
    counts = torch.tensor([100.0, 10.0, 1.0])
    w = compute_weight_class(counts)
    expected = torch.tensor([1.0, 1 + math.log(10), 1 + math.log(100)])
    assert torch.allclose(w, expected, atol=1e-5)
    # rarer class -> larger weight, most frequent class -> weight 1
    assert w[0] < w[1] < w[2]


def test_weighted_ce_manual():
    # 1 sample, 2 classes, 3 pixels (1x3): target [0, 1, 255]
    logits = torch.tensor([[[[2.0, 0.0, 5.0]],
                            [[0.0, 1.0, -5.0]]]])     # [1,2,1,3]
    target = torch.tensor([[[0, 1, 255]]])           # [1,1,3]
    weight = torch.tensor([1.0, 3.0])

    loss = WeightedCELoss(weight)(logits, target)

    # pixel 0: -log softmax([2,0])[0]; pixel 1: -log softmax([0,1])[1]; pixel 2 ignored
    l0 = -math.log(math.exp(2) / (math.exp(2) + 1))
    l1 = -math.log(math.e / (1 + math.e))
    expected = (1.0 * l0 + 3.0 * l1) / (1.0 + 3.0)
    assert abs(loss.item() - expected) < 1e-5


def test_dice_perfect_and_ignore():
    target = torch.tensor([[[0, 1], [1, 255]]])      # [1,2,2]
    logits = torch.zeros(1, 2, 2, 2)
    logits[0, 0][target[0] == 0] = 100.0
    logits[0, 1][target[0] == 1] = 100.0
    # ignore pixel: very wrong prediction, must not affect the loss
    logits[0, 0, 1, 1] = 100.0

    loss = SoftDiceLoss(num_classes=2, smooth=0.0)(logits, target)
    assert abs(loss.item()) < 1e-5


def test_dice_manual():
    # 2 classes, 2 pixels, target [0, 1], uniform prediction p = 0.5 everywhere
    target = torch.tensor([[[0, 1]]])
    logits = torch.zeros(1, 2, 1, 2)
    loss = SoftDiceLoss(num_classes=2, smooth=0.0)(logits, target)
    # class c: inter = 0.5, sum p = 1, sum g = 1 -> dice = 2*0.5/2 = 0.5
    assert abs(loss.item() - 0.5) < 1e-5


def test_dice_all_ignore():
    target = torch.full((1, 2, 2), 255)
    logits = torch.randn(1, 3, 2, 2, requires_grad=True)
    loss = SoftDiceLoss(num_classes=3)(logits, target)
    loss.backward()
    assert loss.item() == 0.0


def test_ce_dice_combo():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    target = torch.randint(0, 4, (2, 8, 8))
    target[0, :2] = 255
    weight = torch.tensor([1.0, 2.0, 1.5, 3.0])
    lam = 0.5

    combo = CEDiceLoss(weight, num_classes=4, lam=lam)(logits, target)
    ce = WeightedCELoss(weight)(logits, target)
    dice = SoftDiceLoss(num_classes=4)(logits, target)
    assert torch.allclose(combo, ce + lam * dice)

    combo.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
