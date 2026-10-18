import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim=28, hidden_dim=2048, num_classes=10):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim,
            batch_first=True)
        
        self.fc = nn.Linear(
            in_features=hidden_dim,
            out_features=num_classes)

    def forward(self, inputs, H_C=None):
        if inputs.dim() == 4:
            x = inputs.squeeze(1)

        else:
            x = inputs

        out, (hn, cn) = self.rnn(x, H_C)

        last_stage = out[:, -1, :]

        logits = self.fc(last_stage)

        return logits
