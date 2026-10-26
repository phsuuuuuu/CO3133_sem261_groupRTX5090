import random
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms

class SegTransform:
    def __init__(self, is_train=True, base_size=(512, 1024), crop_size=(512, 512)):
        self.is_train = is_train
        self.base_size = base_size  
        self.crop_size = crop_size
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, img, mask):
        img = TF.resize(img, self.base_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.base_size, interpolation=TF.InterpolationMode.NEAREST)

        if self.is_train:
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=self.crop_size)
            img = TF.crop(img, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            img = self.color_jitter(img)

        img = TF.to_tensor(img)
        img = self.normalize(img)
        mask = torch.as_tensor(transforms.PILToTensor()(mask).squeeze(0), dtype=torch.int64)
        return img, mask
