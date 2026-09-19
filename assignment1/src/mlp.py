import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=256, output_dim=10, dropout_rate=0.2):
        super().__init__()
        self.flatten = nn.Flatten()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits
    