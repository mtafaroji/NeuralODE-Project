import torch
import torch.nn as nn

class FTheta(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer1: 3 input → 4 output
        self.fc1 = nn.Linear(3, 4)
        # Layer2: 4 input → 4 output
        self.fc2 = nn.Linear(4, 4)
        # Layer3: 4 input → 3 output
        self.fc3 = nn.Linear(4, 3)
        # Activation function: Tanh
        self.act = torch.tanh

    def forward(self, h, t=None):
        x = self.act(self.fc1(h))
        x = self.act(self.fc2(x))
        return self.fc3(x)
