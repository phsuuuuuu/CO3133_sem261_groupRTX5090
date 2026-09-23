from pathlib import Path # to  handle  paths safely
import numpy as np
import  torch
from sklearn.model_selection import train_test_split 
from torch.utils.data import DataLoader, Subset  #  to use  batches
from torchvision import transforms #prprocessing  data
from torchvision.datasets import FashionMNIST #import  dataset as req

"""CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]"""

"""my workflow Fashion-MNIST 70000  = 54000 train + 6000 validation + 10000 test
     ↓
Load raw training images
     ↓
Split into train + validation
     ↓
Calculate mean and std from TRAIN only
     ↓
ToTensor()
     ↓
Normalize()
     ↓
Create DataLoaders
     ↓
Model training"""

def create_split(
    dataset_size = 60000,
    validation_size = 6000,
    random_seed = 666,
    split_path = Path("results/split.npz")
):
    generator = torch.Generator().manual_seed(random_seed)
    indices = torch.randperm(dataset_size, generator=generator).numpy()
    val_indices = indices[:validation_size]
    train_indices = indices[validation_size:]

    #save  indices
    split_path = Path(split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        split_path,
        train_indices=train_indices,
        val_indices=val_indices
    )

    return train_indices, val_indices
def calculate_mean_std(images, train_indices):
    """
    Calculate normalization values using training images only.
    """
    train_images = images[train_indices].float() / 255.0

    mean = train_images.mean().item()
    std = train_images.std().item()

    return mean, std 
def build_loaders(
        batch_size = 200,
        validation_size = 6000,
        random_seed = 666,
        split_path = Path("results/split.npz"),
        data_dir = Path("data")
):
    """Create train, validation, and test DataLoaders."""
    #load
    raw_train = FashionMNIST(
        root=data_dir,
        train=True,
        download=True
    )

    #making  indexing train + vaalidation
    train_indices, val_indices = create_split( 
        dataset_size=len(raw_train),
        validation_size=validation_size,
        random_seed=random_seed,
        split_path=split_path
        )


    #cal  mean +  std  for normalization  // no valid also no test only train
    mean, std = calculate_mean_std(raw_train.data, train_indices)

    print(f"Training mean: {mean:.4f}")
    print(f"Training std:  {std:.4f}")

    #preprocessing  data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    full_train = FashionMNIST(
        root=data_dir,
        train=True,
        download=False,
        transform=transform
    )

    test_dataset = FashionMNIST(
        root=data_dir,
        train=False,
        download=False,
        transform=transform
    )

    #create  subset for train + validation
    train_dataset = Subset(full_train, train_indices)
    val_dataset = Subset(full_train, val_indices)

    #create  dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = build_loaders()
    
    print("Train:", len(train_loader.dataset))
    print("Validation:", len(val_loader.dataset))
    print("Test:", len(test_loader.dataset))

    images, labels = next(iter(train_loader))

    print("Images:", images.shape)
    print("Labels:", labels.shape)