import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Thông số sóng
z = np.linspace(0, 2*np.pi, 200)
E0, H0 = 1, 1

# Tạo figure 3D
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')

line_E, = ax.plot([], [], [], 'r', label='Điện trường E (x)')
line_H, = ax.plot([], [], [], 'b', label='Từ trường H (y)')
line_k, = ax.plot([], [], [], 'k--', label='Hướng truyền sóng k (z)')

# Giới hạn trục
ax.set_xlim(0, 2*np.pi)
ax.set_ylim(-1.2, 1.2)
ax.set_zlim(-1.2, 1.2)

ax.set_xlabel('Trục z (hướng truyền)')
ax.set_ylabel('Trục x (E)')
ax.set_zlabel('Trục y (H)')
ax.set_title('Sóng điện từ: E ⟂ H ⟂ k (Animation 3D)')
ax.legend()

# Hàm cập nhật
def update(frame):
    t = frame * 0.1
    Ex = E0 * np.cos(z - t)
    Hy = H0 * np.cos(z - t)
    
    # Cập nhật sóng E
    line_E.set_data(z, Ex)
    line_E.set_3d_properties(np.zeros_like(z))
    
    # Cập nhật sóng H
    line_H.set_data(z, np.zeros_like(z))
    line_H.set_3d_properties(Hy)
    
    # Cập nhật hướng k
    line_k.set_data(z, np.zeros_like(z))
    line_k.set_3d_properties(np.zeros_like(z))
    
    return line_E, line_H, line_k

# Tạo animation
ani = FuncAnimation(fig, update, frames=100, interval=100, blit=True)

# Xuất ra dạng HTML5 video để hiển thị
from matplotlib.animation import HTMLWriter
from IPython.display import HTML
plt.close(fig)
HTML(ani.to_jshtml())
