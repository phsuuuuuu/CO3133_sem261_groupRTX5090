import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.data import build_loaders 
from src.models import linear, mlp, cnn, lstm, transformer

os.makedirs('results/figures', exist_ok=True)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_, _, test_loader = build_loaders(batch_size=200, random_seed=666)

linear = linear.LinearClassifier()
mlp = mlp.MLP()
cnn = cnn.CNN()
lstm = lstm.LSTM()
transformer = transformer.Transformer()

models = {
    'linear': linear,
    'mlp': mlp,
    'cnn': cnn,
    'lstm': lstm,
    'transformer': transformer
}

for name, model in models.items():
    checkpoint = torch.load(f'checkpoints/{name}_ep10/best.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

with torch.no_grad():
    for name, model in models.items():
        print(f"Processing the model...")
        all_preds = []
        all_targets = []
        all_images = []

        for images, targets in test_loader:
            images_dev = images.to(device)
            outputs = model(images_dev)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_images.append(images.cpu())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_images = torch.cat(all_images, dim=0)

        cm = confusion_matrix(all_targets, all_preds)
        fig, ax = plt.subplots(figsize=(9,9))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(cmap='Blues', xticks_rotation=45, ax=ax, colorbar=False)
        ax.set_title(f'Confusion Matrix - {name.upper()}', fontsize=14, pad=12)
        plt.tight_layout()
        plt.savefig(f'results/figures/confusion_matrix_{name}.png', dpi=300)
        plt.close()

        correct_idx = np.where(all_preds == all_targets)[0]
        incorrect_idx = np.where(all_preds != all_targets)[0]

        n_samples = 5
        sample_correct = correct_idx[:n_samples]
        sample_incorrect = incorrect_idx[:n_samples]

        fig, axes = plt.subplots(2, n_samples, figsize=(15, 6))

        for i, idx in enumerate(sample_correct):
            img = all_images[idx].squeeze() 
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(
                f"Pred: {CLASS_NAMES[all_preds[idx]]}\nTrue: {CLASS_NAMES[all_targets[idx]]}",
                fontsize=9, color='green'
            )
            axes[0, i].axis('off')

        for i, idx in enumerate(sample_incorrect):
            img = all_images[idx].squeeze()
            axes[1, i].imshow(img, cmap='gray')
            axes[1, i].set_title(
                f"Pred: {CLASS_NAMES[all_preds[idx]]}\nTrue: {CLASS_NAMES[all_targets[idx]]}",
                fontsize=9, color='red'
            )
            axes[1, i].axis('off')

        axes[0, 0].text(-0.25, 0.5, 'CORRECT', transform=axes[0, 0].transAxes,
                        fontsize=12, fontweight='bold', va='center', rotation=90, color='green')
        axes[1, 0].text(-0.25, 0.5, 'INCORRECT', transform=axes[1, 0].transAxes,
                        fontsize=12, fontweight='bold', va='center', rotation=90, color='red')

        plt.suptitle(f'Prediction Examples - {name.upper()}', fontsize=14, y=0.98)
        plt.tight_layout()
        plt.savefig(f'results/figures/prediction_examples_{name}.png', dpi=300)
        plt.close()

print("Complete! the graph is saved on results/figures/.")   