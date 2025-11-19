import torch
import torch.nn as nn

class FTheta(nn.Module):
    """
    Neural ODE function f_theta(h, t)
    Defines the derivative dh/dt = f_theta(h, t)
    """
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, t, h):  
      out = self.net(h)
      return out