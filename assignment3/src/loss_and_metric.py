import torch

def  _valid(pred, gt):
    mask = gt > 0
    p = torch.masked_select(pred,  mask)
    g = torch.masked_select(gt,  mask)
    #1D array  as output
    return p, g
###Loss  function
#rgb: mae
def m1_loss(pred,  gt):
    p, g = _valid(pred,  gt)
    return ((p-g)**2).mean().sqrt()

#depth: rmse
def m2_loss(pred,  gt):
    p, g = _valid(pred,  gt)
    return ((p-g)**2).mean().sqrt()

#early fusion: w1*mae + w2*rmse
def m3_loss(pred, gt,w1=1,w2=1):
    p, g = _valid(pred,  gt)
    return ((p-g)**2).mean().sqrt()

#custom fusion: #TODO Thanh does it bro

###Metric:
class Metric:
    def __init__(self, max_depth=80, by_range=True):
        self.max_depth = max_depth
        self.by_range = by_range
        self.ranges = [(0, 20), (20, 40), (40, max_depth)]
        self.reset()
    
    def reset(self):
        self.sum_sq = 0
        self.sum_abs = 0
        self.total_valid  = 0 
        self.total_pixels = 0
        self.ranged_metric = {r:[0.0,0.0,0] for r in self.ranges}

    @torch.no_grad()
    def update(self, pred, gt):
        pred = pred.float().clamp(0, self.max_depth)
        p, g = _valid(pred, gt.float())
        e = p-g

        self.sum_sq += (e**2).sum().item()
        self.sum_abs += (e).abs().sum().item()
        self.total_valid  += e.numel()
        self.total_pixels += pred.numel()

        if  self.by_range:
            for low,  high in self.ranges:
                in_range  = (g >= low) & ((g <= high) if high == self.max_depth else (g < high))
                range = torch.masked_select(e, in_range)
                self.ranged_metric[(low, high)][0] += (range**2).sum().item()
                self.ranged_metric[(low, high)][1] += (range).abs().sum().item()
                self.ranged_metric[(low, high)][2] += range.numel()

    def compute(self):
        if self.total_valid  == 0: raise ValueError("no valid pixels (gt > 0) - check the data")
        n = self.total_valid 
        out = {
            "rmse": (self.sum_sq / n) ** 0.5,                 # metres
            "mae": self.sum_abs / n,                          # metres
            "total_valid ": self.total_valid ,
            "%valid/total_pixel": self.total_valid/self.total_pixels
        }
        if self.by_range:
            for (lo, hi), (sq, ab, n_r) in self.ranged_metric.items():
                name = f"{lo}-{hi}"
                out[name] = {
                    "rmse": (sq / n_r) ** 0.5 if n_r else float("nan"),
                    "mae": ab / n_r if n_r else float("nan"),
                    "total_valid ": n_r,
                    "%valid/total_valid": n_r / self.total_valid, 
                }
        return out
    