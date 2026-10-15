import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.35):
        super().__init__()

        # Convolution layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, padding=1)

        #  Third convolution
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classification
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 14 * 14 , 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Block 1: Convolution -> ReLU -> Pooling
        x = self.conv1(x)
        x = self.relu(x)

        # Block 2: Convolution -> ReLU -> Convolution -> ReLU -> Pooling
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        # Classifier: Flatten -> FC1 -> ReLU -> Dropout -> FC2
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = CNN()
    x = torch.randn(4, 1, 28, 28)

    print("Output:", model(x).shape)