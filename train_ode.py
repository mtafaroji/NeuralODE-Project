import torch
from torchdiffeq import odeint
from models.f_theta import FTheta
import matplotlib.pyplot as plt

# ----------------------------
# 1. داده‌های واقعی (synthetic)
# ----------------------------
t = torch.linspace(0, 5, steps=100)  # بازه زمانی
h_true = torch.stack([
    torch.sin(t),        # h1 واقعی
    torch.cos(t),        # h2 واقعی
    torch.sin(0.5*t)     # h3 واقعی
], dim=1).unsqueeze(0)   # batch=1, shape: (1,100,3)

# ----------------------------
# 2. وضعیت اولیه
# ----------------------------
h0 = h_true[:,0,:]  # h(0)

# ----------------------------
# 3. مدل و optimizer
# ----------------------------
model = FTheta()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# ----------------------------
# 4. آموزش (یک epoch ساده)
# ----------------------------
for epoch in range(200):
    optimizer.zero_grad()

    # حل ODE
    def f_ode(t, h):
        return model(h)

    h_pred = odeint(f_ode, h0, t)  # shape: (100, batch, 3)

    # محاسبه loss
    h_pred = h_pred.permute(1,0,2)  # (batch, 100, 3)
    loss = loss_fn(h_pred, h_true)

    # backpropagation
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# ----------------------------
# 5. رسم نتایج
# ----------------------------
h_pred = h_pred.detach().squeeze().numpy()
plt.plot(t.numpy(), h_true.squeeze().numpy()[:,0], label='h1 true')
plt.plot(t.numpy(), h_pred[:,0], '--', label='h1 pred')
plt.plot(t.numpy(), h_true.squeeze().numpy()[:,1], label='h2 true')
plt.plot(t.numpy(), h_pred[:,1], '--', label='h2 pred')
plt.plot(t.numpy(), h_true.squeeze().numpy()[:,2], label='h3 true')
plt.plot(t.numpy(), h_pred[:,2], '--', label='h3 pred')
plt.xlabel('Time')
plt.ylabel('h(t)')
plt.title('Neural ODE training example')
plt.legend()
plt.show()
