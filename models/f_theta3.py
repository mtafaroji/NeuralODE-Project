import torch
import torch.nn as nn

# تعریف device (GPU اگر موجود باشد، در غیر اینصورت CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class FTheta(nn.Module):
    """
    Neural ODE function f_theta(h, t)
    Defines the derivative dh/dt = f_theta(h, t)
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, t, h): 
              # مطمئن شو که h هم روی همان device است
        h = h.to(device) 
        out = self.net(h)
        return out