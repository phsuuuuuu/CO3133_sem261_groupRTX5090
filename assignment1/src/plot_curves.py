import pandas as pd 
import matplotlib.pyplot as plt


df_linear = pd.read_csv("results/linear/linear_ep10_history.csv")

df_mlp = pd.read_csv("results/mlp/mlp_ep10_history.csv")

df_cnn = pd.read_csv("results/cnn/cnn_ep10_history.csv")

df_lstm = pd.read_csv("results/lstm/lstm_ep10_history.csv")

df_transformer = pd.read_csv("results/transformer/transformer_ep10_history.csv")

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

metrics = [
    ("loss", "Loss"),
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro F1-Score"),
]

for ax, (metric, title) in zip(axes, metrics):
    ax.plot(df_linear["epoch"], df_linear[f"train_{metric}"], label="Linear Train", color="tab:blue")
    ax.plot(df_mlp["epoch"], df_mlp[f"train_{metric}"], label="MLP Train", color="tab:orange")
    ax.plot(df_cnn["epoch"], df_cnn[f"train_{metric}"], label="CNN Train", color="tab:green")
    ax.plot(df_lstm["epoch"], df_lstm[f"train_{metric}"], label="LSTM Train", color="tab:red")
    ax.plot(df_transformer["epoch"], df_transformer[f"train_{metric}"], label="Transformer Train", color="tab:cyan")

    ax.plot(df_linear["epoch"],df_linear[f"val_{metric}"],label=f"Linear Val",color="tab:blue",linestyle="--")
    ax.plot(df_mlp["epoch"], df_mlp[f"val_{metric}"],label=f"MLP Val",color="tab:orange",linestyle="--")
    ax.plot(df_cnn["epoch"], df_cnn[f"val_{metric}"],label=f"CNN Val",color="tab:green",linestyle="--")
    ax.plot(df_lstm["epoch"], df_lstm[f"val_{metric}"],label=f"LSTM Val",color="tab:red",linestyle="--")
    ax.plot(df_transformer["epoch"], df_transformer[f"val_{metric}"],label=f"Transformer Val",color="tab:cyan",linestyle="--")

    ax.set_title(f"Comparison of {title}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

plt.tight_layout()
plt.savefig("results/figures/training_curves.png", dpi=300)
print(f"The graph has been saved at results/figures/training_curves.png")