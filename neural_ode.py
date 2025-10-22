import torch
from torchdiffeq import odeint
from models.f_theta import FTheta
import matplotlib.pyplot as plt

# نمونه تابع f_theta
f_theta = FTheta()

# وضعیت اولیه h(0)
h0 = torch.tensor([[1.0, 0.0, 0.0]])  # یک batch، 3 بعد

# بازه زمانی که می‌خواهیم ODE را حل کنیم
t = torch.linspace(0, 5, steps=100)  # از t=0 تا t=5 با 100 گام

# تابعی که odeint می‌خواهد
def f_ode(t, h):
    return f_theta(h)

# حل ODE
h_t = odeint(f_ode, h0, t)

# نمایش نتایج
h_t = h_t.squeeze().detach().numpy()  # تبدیل به numpy برای رسم

plt.plot(t.numpy(), h_t[:,0], label='h1')
plt.plot(t.numpy(), h_t[:,1], label='h2')
plt.plot(t.numpy(), h_t[:,2], label='h3')
plt.xlabel('Time')
plt.ylabel('h(t)')
plt.title('Neural ODE trajectory')
plt.legend()
plt.show()
