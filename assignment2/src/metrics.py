import torch 
from typing import Dict, Any

class ConfusionMatrix:
    """Confusion matrix accumulated on GPU/CPU (ignoring ignore_index=255).
    According to project specification:
    - update(pred, target): accepts pred [B, H, W] (or logits [B, C, H, W]) and target [B, H, W]
    - compute() -> dict(iou[19], miou, dice[19], mdice, pixel_acc)
    """

    def __init__(self, num_classes: int = 19, ignore_index: int = 255, device: str = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device 
        self.total_mat = None

    def reset(self):
        self.total_mat = None

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        if pred.shape != target.shape:
            if pred.dim() == 4 and pred.shape[1] == self.num_classes:
                pred = torch.argmax(pred, dim=1)
            else:
                raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
            
        if self.total_mat is None:
            dev = self.device if self.device is not None else pred.device
            self.total_mat = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64, device=dev)

        pred = pred.to(self.total_mat.device)
        target = target.to(self.total_mat.device)

        mask = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        t = target[mask]
        p = pred[mask]

        indices = t * self.num_classes + p
        bins = torch.bincount(indices, minlength=self.num_classes ** 2)
        self.total_mat += bins.view(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, Any]:
        if self.total_mat is None:
            self.total_mat = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)

        mat = self.total_mat.float()
        tp = torch.diag(mat)
        fp = mat.sum(dim=0) - tp
        fn = mat.sum(dim=1) - tp

        union = tp + fp + fn
        iou = torch.full((self.num_classes,), float('nan'), dtype=torch.float32, device=mat.device)
        valid_iou = union > 0
        iou[valid_iou] = tp[valid_iou] / union[valid_iou]

        dice_denom = 2 * tp + fp + fn
        dice = torch.full((self.num_classes,), float('nan'), dtype=torch.float32, device=mat.device)
        valid_dice = dice_denom > 0
        dice[valid_dice] = (2 * tp[valid_dice]) / dice_denom[valid_dice]

        miou = torch.nanmean(iou).item()
        mdice = torch.nanmean(dice).item()

        total_valid = mat.sum().item()
        pixel_acc = (tp.sum().item() / total_valid) if total_valid > 0 else 0.0

        return {
            "iou": iou.tolist(),
            "miou": miou,
            "dice": dice.tolist(),
            "mdice": mdice,
            "pixel_acc": pixel_acc,
            "confusion_matrix": self.total_mat.cpu().numpy()
        }