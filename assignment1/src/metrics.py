#just eg metrics make by Phu for call func after Thanh make it again or  change plz keep the same  name 
from sklearn.metrics import accuracy_score, f1_score


def calculate_metrics(labels, predictions):
    accuracy = accuracy_score(
        labels,
        predictions
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro"
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }