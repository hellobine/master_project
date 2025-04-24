import numpy as np
import matplotlib.pyplot as plt

# Academic color scheme using BMH style
plt.style.use('bmh')
plt.rcParams.update({'font.size': 12})

# Parameter setup
T = 1
t = np.linspace(0, T, 1000)
x = np.cos(2 * np.pi * t / T)
y = 0.5 * np.sin(4 * np.pi * t / T)

# Plotting
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
ax.set_title('Trajectory Projection in xy-plane')
ax.set_aspect('equal', 'box')
plt.tight_layout()
plt.savefig("eight-figure.jpg",dpi=800)
plt.show()
