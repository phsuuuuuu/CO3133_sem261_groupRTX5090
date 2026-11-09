import json

import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-9

def compute_weight_class(tensor):
    # w_c = 1 + log(max_count / count_c)
    tensor = tensor.double()
    max_class = torch.max(tensor)
    tensor = (max_class + EPSILON) / torch.add(tensor, EPSILON)
    tensor = 1 + torch.log(tensor)
    return tensor.float()

def load_class_weights(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return torch.tensor(data["weights"], dtype=torch.float32)


class WeightedCELoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255):
        super().__init__()
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        weight = None if self.weight is None else self.weight.to(logits.dtype)
        return F.cross_entropy(logits, target, weight=weight, ignore_index=self.ignore_index)


class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes=19, ignore_index=255, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target):
        # logits [B,C,H,W], target [B,H,W]
        probs = torch.softmax(logits.float(), dim=1)
        valid = (target != self.ignore_index)                      # [B,H,W]
        target_clean = torch.where(valid, target, torch.zeros_like(target))
        one_hot = F.one_hot(target_clean, self.num_classes)        # [B,H,W,C]
        one_hot = one_hot.permute(0, 3, 1, 2).float()              # [B,C,H,W]

        valid = valid.unsqueeze(1).float()                         # [B,1,H,W]
        probs = probs * valid
        one_hot = one_hot * valid

        dims = (0, 2, 3)
        inter = torch.sum(probs * one_hot, dims)
        card = torch.sum(probs, dims) + torch.sum(one_hot, dims)
        dice = (2 * inter + self.smooth) / (card + self.smooth)

        # only average over classes present in GT or predicted
        present = torch.sum(one_hot, dims) > 0
        if present.sum() == 0:
            return logits.sum() * 0.0
        return 1 - dice[present].mean()


class CEDiceLoss(nn.Module):
    def __init__(self, weight=None, num_classes=19, ignore_index=255, lam=1.0, smooth=1.0):
        super().__init__()
        self.ce = WeightedCELoss(weight, ignore_index)
        self.dice = SoftDiceLoss(num_classes, ignore_index, smooth)
        self.lam = lam

    def forward(self, logits, target):
        return self.ce(logits, target) + self.lam * self.dice(logits, target)


def build_loss(cfg):
    name = cfg.get("loss", "ce")
    num_classes = int(cfg.get("num_classes", 19))
    ignore_index = int(cfg.get("ignore_index", 255))

    weight = None
    if cfg.get("class_weights"):
        weight = load_class_weights(cfg["class_weights"])

    if name == "ce":
        return WeightedCELoss(None, ignore_index)
    if name == "wce":
        return WeightedCELoss(weight, ignore_index)
    if name == "dice":
        return SoftDiceLoss(num_classes, ignore_index)
    if name == "cedice":
        return CEDiceLoss(weight, num_classes, ignore_index, lam=float(cfg.get("dice_lambda", 1.0)))
    raise ValueError(f"Unknown loss: {name}")
