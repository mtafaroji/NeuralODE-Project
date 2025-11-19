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
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim +1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, t, h): 

        # h can be (input_dim,) or (batch, input_dim)
        if h.dim() == 1:
            h = h.unsqueeze(0)          # -> (1, input_dim)

        # t as feature with same batch size
        t_tensor = t.expand(h.size(0), 1)   # (batch, 1)


        x_in = torch.cat([h, t_tensor], dim=-1) # (batch, input_dim+1)

        return self.net(x_in)