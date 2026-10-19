from pathlib import Path
import csv
import argparse
import time


import torch
import torch.nn as nn

from src.data import build_loaders
from src.metrics import calculate_metrics
from src.models.linear import LinearClassifier
from src.models.mlp import MLP
from src.models.cnn import CNN
from src.models.lstm import LSTM
from src.models.transformer import Transformer
from src.metrics import count_parameters, measure_time

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

models = ['linear', 'mlp', 'cnn', 'lstm', 'transformer']

def run_batches(
    model,
    loader,
    loss_fn,
    optimizer=None,
    max_batches=None
):
    """
    Run training or validation.

    optimizer given  -> training
    optimizer None   -> validation

    max_batches=1    -> only run 1 batch
    max_batches=None -> run whole dataset
    """

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_samples = 0

    all_labels = []
    all_predictions = []

    with torch.set_grad_enabled(training):

        for batch_index, (images, labels) in enumerate(loader):

            # Stop early for fast testing
            if max_batches is not None and batch_index >= max_batches:
                break

            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            # Forward pass
            logits = model(images)

            # Loss
            loss = loss_fn(logits, labels)

            # Backpropagation
            if training:
                loss.backward()
                optimizer.step()

            predictions = logits.argmax(dim=1)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_labels.extend(
                labels.cpu().tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

    average_loss = total_loss / total_samples

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    return average_loss, metrics

def create_model(model_name):
    if model_name == "linear":
        return LinearClassifier()
    if model_name == "mlp":
        return MLP()
    if model_name == "cnn":
        return CNN()
    if model_name == 'lstm':
        return LSTM()
    if model_name == 'transformer':
        return Transformer()
    raise ValueError(
        "Model must be 'linear', 'mlp', 'cnn', or 'lstm"
    )


def train_model(
    model_name,
    epochs=5,
    learning_rate=0.001,
    batch_size=200,
    random_seed=666,
    max_batches=None,
    checkpoint_dir=Path("checkpoints"),
    results_dir = Path("results")

):
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    print("Device:", device)
    # Data
    train_loader, val_loader, test_loader = build_loaders(
        batch_size=batch_size,
        random_seed=random_seed
    )

    model = create_model(model_name)
    model = model.to(device)

    # Loss
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

   # Create separate checkpoint directory for each experiment
    run_checkpoint_dir = checkpoint_dir / f"{model_name}_ep{epochs}"

    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Path for the best model
    checkpoint_path = run_checkpoint_dir / "best.pt"

    best_val_macro_f1 = float("-inf")

    history = []
    
    start_train = time.time()
    #MULTI epoch training loop
    for epoch in range(1, epochs + 1):
        # Start timing this epoch
        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_start = time.perf_counter() #calculate epoch time
        train_loss, train_metrics = run_batches(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            max_batches=max_batches
        )

        val_loss, val_metrics = run_batches(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            max_batches=max_batches
        )
        # Test after every epoch (NO training)
        # Evaluate test set after every epoch
        test_loss, test_metrics = run_batches(
            model=model,
            loader=test_loader,
            loss_fn=loss_fn,
            optimizer=None,
            max_batches=max_batches
        )
        #just to print the results of the epochs
        print(
            f"Train | "
            f"Loss: {train_loss:.4f} | "
            f"Accuracy: {train_metrics['accuracy']:.4f} | "
            f"F1: {train_metrics['macro_f1']:.4f}"
        )

        print(
            f"Val   | "
            f"Loss: {val_loss:.4f} | "
            f"Accuracy: {val_metrics['accuracy']:.4f} | "
            f"F1: {val_metrics['macro_f1']:.4f}"
        )

        history.append({
            "epoch": epoch,

            # Training results
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],

            # Validation results
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],

            # Testing results
            "test_loss": test_loss,
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"]
        })
        # Save checkpoint for EVERY epoch in order to compare training progress later like when its saturated
    
        epoch_checkpoint_path = (
            run_checkpoint_dir / f"epoch_{epoch:03d}.pt"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            },
            epoch_checkpoint_path
        )
        
        print(f"Epoch checkpoint saved: {epoch_checkpoint_path}")
         # Save best checkpoint
        val_macro_f1 = val_metrics["macro_f1"]
        if val_macro_f1 > best_val_macro_f1:

            best_val_macro_f1 = val_macro_f1 

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"]
                },
                checkpoint_path
            )

            print("Saved:", checkpoint_path)

        # Wait for GPU operations to finish
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Calculate this epoch's execution time
        epoch_time = time.perf_counter() - epoch_start

        epoch_minutes = int(epoch_time // 60)
        epoch_seconds = epoch_time % 60

        print(
            f"Epoch {epoch}/{epochs} time: "
            f"{epoch_minutes:02d}:{epoch_seconds:05.2f} "
            f"({epoch_time:.2f} seconds)"
        )

    # Save history
    # Create a separate results folder for each model
    model_results_dir = results_dir / model_name
    model_results_dir.mkdir(parents=True, exist_ok=True)

    # Save history for each experiment
    history_path = (
        model_results_dir / f"{model_name}_ep{epochs}_history.csv"
    )
    with history_path.open(
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys()
        )

        writer.writeheader()
        writer.writerows(history)

    training_time = time.time() - start_train

    print("\nTraining finished.")
    print("History saved:", history_path)


    # Evaluate in Test
    print(f"\nLoading best checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    with torch.no_grad():
        test_loss, test_metrics = run_batches(
            model=model,
            loader=test_loader,
            loss_fn=loss_fn,
            optimizer=None,
            max_batches=max_batches
        )

    test_accuracy = test_metrics["accuracy"]
    test_macro_f1 = test_metrics["macro_f1"]

    num_params = count_parameters(model)
    inference_time = measure_time(model, test_loader, device)

    print(
        f"Test  |"
        f"Loss: {test_loss:.4f} |   "
        f"Accuracy: {test_accuracy:.4f} | "
        f"F1: {test_macro_f1:.4f}"
    )

    summary_path = results_dir / "summary.csv"
    file_exists = summary_path.exists()

    summary_data = {
        "model": model_name,
        "num_params": num_params,
        "best_val_f1": best_val_macro_f1,
        "test_accuracy": test_accuracy,
        "test_f1": test_macro_f1,
        "training_time": training_time,
        "inference_time": inference_time,
    }

    with summary_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_data.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(summary_data)

    print(f"Summary appended to: {summary_path}")




if __name__ == "__main__":
    total_start = time.perf_counter()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=models,
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=200
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=666
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None
    )

    args = parser.parse_args()

    train_model(
        model_name=args.model,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        random_seed=args.seed,
        max_batches=args.max_batches
    )
        # Make sure all GPU operations are completed
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Calculate total execution time
    total_time = time.perf_counter() - total_start

    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    print("\n========== TOTAL EXECUTION TIME ==========")
    print(f"Total seconds: {total_time:.2f} s")
    print(f"Total minutes: {total_time / 60:.2f} min")
    print(f"Total runtime: {hours:02d}:{minutes:02d}:{seconds:05.2f}")
    print("==========================================")