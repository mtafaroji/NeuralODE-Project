import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# پارامترهای معادله دیفرانسیل
# --------------------------
a = 3.0
b = 6.0

def f(x, y):
    dx = y
    dy = b * x - x**3 - a * y
    return dx, dy

# --------------------------
# ساخت گرید در بازه [-10, 10]
# --------------------------
x_min, x_max = -4.0, 4.0
y_min, y_max = -4.0, 4.0

num_points = 45  # تعداد نقاط روی هر محور؛ اگر گرید ریزتر می‌خواهی، مثلاً 30 یا 40 بگذار

x_vals = np.linspace(x_min, x_max, num_points)
y_vals = np.linspace(y_min, y_max, num_points)

X, Y = np.meshgrid(x_vals, y_vals)  # هر دو (num_points, num_points)

# --------------------------
# محاسبه‌ی بردارها روی گرید
# --------------------------
DX, DY = f(X, Y)  # هر دو با شکل (num_points, num_points)

# --------------------------
# نرمال‌سازی جهت بردارها برای visualization
# (طول ثابت، جهت واقعی)
# --------------------------
mag = np.sqrt(DX**2 + DY**2) + 1e-12
DX_dir = 3*DX / mag
DY_dir = 3*DY / mag

# --------------------------
# رسم میدان برداری با quiver
# --------------------------
plt.figure(figsize=(8, 8))

# هرچه scale بزرگ‌تر باشد فلش‌ها کوتاه‌تر می‌شوند
scale_value = 10  # در صورت نیاز این عدد را کم/زیاد کن

plt.quiver(
    X, Y,              # نقاط شروع فلش‌ها
    DX_dir, DY_dir,    # جهت فلش‌ها (نرمال‌شده)
    angles="xy",
    scale_units="xy",
    scale=scale_value,
    width=0.003,
    color="blue",
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Vector Field of the Real ODE: dx/dt = y, dy/dt = 6x - x^3 - 3y")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.show()
