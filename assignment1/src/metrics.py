#just eg metrics make by Phu for call func after Thanh make it again or  change plz keep the same  name 
from sklearn.metrics import accuracy_score, f1_score
import time
import torch


def calculate_metrics(labels, predictions):
    accuracy = accuracy_score(
        labels,
        predictions
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0, # avoid error
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_time(model, dataloader, device):
    if "cuda" in str(device):
        torch.cuda.synchronize()

    model.eval()
    model.to(device)

    start_time = time.time()
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
            inputs = inputs.to(device)

            _ = model(inputs)

    if "cuda" in str(device):
        torch.cuda.synchronize()

    total_time = time.time() - start_time

    print(f"Total time: {total_time}")

    return total_time
        