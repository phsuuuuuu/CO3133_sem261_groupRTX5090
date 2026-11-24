"""Transform works on 1 sample; get_data stacks samples into batches."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet34_Weights

# ImageNet MEAN / STD for the pretrained encoder, shape (3, 1, 1) to fit a (3, H, W) rgb
_preset = ResNet34_Weights.IMAGENET1K_V1.transforms()
MEAN = np.array(_preset.mean, np.float32)[:, None, None]
STD = np.array(_preset.std, np.float32)[:, None, None]
MAX_DEPTH = 80.0   # team contract: lidar channel 0 = depth / 80


class Transform:
    """Decoded arrays of one sample -> tensors (C, H, W)."""

    def rgb_to_tensor(self, rgb):
        """(3, H, W) 0..1 -> (3, H, W) standardised."""
        rgb = (rgb - MEAN) / STD
        return torch.from_numpy(rgb.copy())

    def lidar_to_tensor(self, lid):
        """(H, W) metres -> (2, H, W): [depth / MAX_DEPTH, mask]."""
        depth = lid / MAX_DEPTH
        mask = (lid > 0).astype(np.float32)
        return torch.from_numpy(np.stack([depth, mask]).copy())

    def gt_to_tensor(self, gt):
        """(H, W) metres -> (1, H, W) metres."""
        return torch.from_numpy(gt[None].copy())

    def __call__(self, rgb, lid, gt):
        """Returns {'rgb': (3, H, W), 'lidar': (2, H, W), 'gt': (1, H, W)}."""
        return {"rgb": self.rgb_to_tensor(rgb), "lidar": self.lidar_to_tensor(lid), "gt": self.gt_to_tensor(gt)}


class KittiDepthDataset:
    """Split folder input_dir/{rgb,lidar,gt}/<name>.png -> batches."""

    def __init__(self,input_dir, output_dir, split, transform, batch_size=10, seed=666):
        """Store settings and list file names (nothing loaded)."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.split = split
        self.transform = transform
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.names = sorted(p.name for p in (self.input_dir/"rgb").glob("*.png"))

    def load(self, name):
        """Read 3 PNGs, raw values: rgb (H, W, 3) 0..255, lid / gt (H, W) metres * 256."""
        rgb = np.asarray(Image.open(self.input_dir/"rgb"/name).convert('RGB'), dtype=np.float32)
        lid = np.asarray(Image.open(self.input_dir/"lidar"/name), dtype=np.float32)
        gt = np.asarray(Image.open(self.input_dir/"gt"/name), dtype=np.float32)
        return rgb, lid, gt

    def process(self, rgb, lid, gt):
        """Decode to rgb 0..1 / metres, then transform."""
        rgb = rgb.astype(np.float32).transpose(2,0,1)/255.0
        lid = lid.astype(np.float32)/256.0
        gt = gt.astype(np.float32)/256.0
        return self.transform(rgb, lid, gt)

    def __len__(self):
        return len(self.names)

    def _load_one(self, name):
        return self.process(*self.load(name))

    def get_batch(self):
        """Yield batches until all samples are used (train is shuffled)."""
        order = np.arange(len(self.names))
        if self.split == "train": self.rng.shuffle(order)
        n, bs = len(order), self.batch_size
        with ThreadPoolExecutor(max_workers=8) as ex:
            pending = [ex.submit(self._load_one, self.names[i]) for i in order[:bs * 2]]
            next_i = bs * 2
            for start in range(0, n, bs):
                take = min(bs, n - start)
                samples = [f.result() for f in pending[:take]]
                del pending[:take]
                end_next = min(n, next_i + take)
                pending.extend(ex.submit(self._load_one, self.names[i]) for i in order[next_i:end_next])
                next_i = end_next
                yield {
                    "rgb" : torch.stack([s["rgb"] for s in samples]),
                    "lidar" : torch.stack([s["lidar"] for s in samples]),
                    "gt" : torch.stack([s["gt"] for s in samples])
                }


if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent / 'kitti_samples'
    kitti = KittiDepthDataset(root / 'test', root, 'test', Transform(), batch_size=2)
    print('samples:', len(kitti))
    for batch in kitti.get_data():
        print('batch:', batch['rgb'].shape, batch['lidar'].shape, batch['gt'].shape)
    print('end')
