import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.linear = nn.Linear(28 * 28,10) #28*28  input dimension, 10 output dimension for 10 classes

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)

        return x