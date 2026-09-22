import pandas as pd 
import matplotlib.pyplot as plt


df_linear = pd.read_csv("results/linear_history.csv")

df_mlp = pd.read_csv("results/mlp_history.csv")

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

metrics = [
    ("loss", "Loss"),
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro F1-Score"),
]

for ax, (metric, title) in zip(axes, metrics):
    ax.plot(df_linear["epoch"], df_linear[f"train_{metric}"], label="Linear Train", color="tab:blue")
    ax.plot(df_mlp["epoch"], df_mlp[f"train_{metric}"], label="MLP Train", color="tab:orange")

    ax.plot(df_linear["epoch"],df_linear[f"val_{metric}"],label=f"Linear Val",color="tab:blue",linestyle="--")
    ax.plot(df_mlp["epoch"], df_mlp[f"val_{metric}"],label=f"MLP Val",color="tab:orange",linestyle="--")

    ax.set_title(f"Comparison of {title}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

plt.tight_layout()
plt.savefig("results/figures/training_curves.png", dpi=300)
print(f"The graph has been saved at results/figures/training_curves.png")