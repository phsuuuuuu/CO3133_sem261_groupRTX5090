from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import build_loaders
from torchvision.datasets import FashionMNIST

CLASS_NAMES = [
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
]

figure_path = Path("results/figures")
figure_path.mkdir(parents=True, exist_ok=True)

def plot_distribution():
    dataset = FashionMNIST(
        root="data",
        train=True,
        download=False
    )
    labels = dataset.targets.numpy()
    counts = np.bincount(labels)

    plt.figure(figsize=(10, 6))#10x6
    plt.bar(CLASS_NAMES, counts)
    plt.title("Fashion-MNIST Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(figure_path / "class_distribution.png")
    plt.show()
    plt.close()

    for name, count in zip(CLASS_NAMES, counts):
        print(f"{name}: {count} images")

def plot_samples(num_per_class=5,
                random_seed=666):
    dataset = FashionMNIST(
        root="data",
        train=True,
        download=False
    )   

    rng = np.random.default_rng(random_seed)

    fig, axes = plt.subplots(
        num_per_class,
        10,
        figsize=(15, 2 * num_per_class),
        squeeze=False
    )

    for label in range(10):
        indices = np.where(dataset.targets.numpy() == label)[0] #all indices of the current label
        selected_indices = rng.choice(indices, size=num_per_class, replace=False)

        for row,  index in enumerate(selected_indices):
            axes[row, label].imshow(dataset.data[index], cmap="gray")
            axes[row, label].axis("off")
            if  row == 0:
                axes[row,label].set_title(CLASS_NAMES[label], fontsize=10)
    plt.suptitle(f"{num_per_class} Representative Images per Class", fontsize=10)
    plt.tight_layout()
    plt.savefig(figure_path / "representative_images.png")          
    plt.show()
    plt.close()            

def plot_processed_batch():
    """Inspect one batch produced by our DataLoader."""

    train_loader, _, _ = build_loaders()

    # Take one batch
    images, labels = next(iter(train_loader))

    print("\nProcessed batch:")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Image type:", images.dtype)

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(8, 5)
    )

    for i, ax in enumerate(axes.flat):

        ax.imshow(
            images[i, 0],
            cmap="gray"
        )

        ax.set_title(
            CLASS_NAMES[labels[i].item()]
        )

        ax.axis("off")

    plt.suptitle("Processed Training Batch")

    plt.tight_layout()

    plt.savefig(
        figure_path / "processed_batch.png"
    )

    plt.show()
    plt.close()

def plt_distribution_after_spilit():
    print(Path.cwd())
    dataset = FashionMNIST(root = "data", train = True, download = False)
    indexing = Path("results/split.npz")
    if indexing.exists():
        print(f"loading indexing from {indexing}")
        split = np.load(indexing)
        train_indices = split["train_indices"]
        label = [dataset[idx][1] for idx in train_indices]
        train_counts = np.bincount(label, minlength=9)        
        bars = plt.bar(CLASS_NAMES, train_counts)
        plt.bar_label(bars, padding=3)
        plt.xlabel("Class")
        plt.ylabel("Number of samples")
        plt.title("FashionMNIST Training Split Distribution")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        plt.savefig(figure_path / "after_indexing_plot_distribution.png")
        plt.close()
    
    
    
if __name__ == "__main__":

    plot_distribution()

    plot_samples(
        num_per_class=5
    )

    plot_processed_batch()

    plt_distribution_after_spilit()