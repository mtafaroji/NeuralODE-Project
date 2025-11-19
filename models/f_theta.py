import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class FTheta(nn.Module):
    """
    Neural ODE function f_theta(h, t)
    Defines the derivative dh/dt = f_theta(h)
    (autonomous model: no explicit time dependence)
    """
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, t, h):
        # odeint passes t as the first argument but we don't use it here
        # because our phenomenon in Anylogic time is autonomous.
        if h.dim() == 1:
            h = h.unsqueeze(0)          # (1, input_dim)
        return self.net(h)
