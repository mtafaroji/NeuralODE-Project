import torch
from models.f_theta import FTheta

# ساخت مدل
model = FTheta()

# نمونه ورودی (یک بردار 3 بعدی)
h0 = torch.randn(1, 3)

# اجرای مدل
out = model(h0)
print("Input:", h0)
print("Output:", out)
print("Output shape:", out.shape)
