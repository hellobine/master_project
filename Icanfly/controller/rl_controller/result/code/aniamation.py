import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------- CONFIG ----------
CSV_PATH = 'arm_sim_data.csv'   # 改成你的实际文件名 / 路径
r1, r2 = 1.0, 0.8               # 两段连杆长度
# -----------------------------

# 1) 读取 CSV
if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(f"找不到文件 {CSV_PATH}，请检查路径或文件名。")
df = pd.read_csv(CSV_PATH)

# 2) 提取关节角
q1 = df['q1_sim'].to_numpy()
q2 = df['q2_sim'].to_numpy()

# 3) 末端轨迹与连杆坐标
x1 = r1 * np.cos(q1)
y1 = r1 * np.sin(q1)

x2 = x1 + r2 * np.cos(q1 + q2)
y2 = y1 + r2 * np.sin(q1 + q2)

# 4) 绘制最终结果
plt.figure(figsize=(6, 6))
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Final Pose & End‑Effector Path')

# 轨迹
ax.plot(x2, y2)

# 最后一帧连杆
ax.plot([0, x1[-1]], [0, y1[-1]], linewidth=2)
ax.plot([x1[-1], x2[-1]], [y1[-1], y2[-1]], linewidth=2)

# 基座和末端
ax.plot(0, 0, marker='o')
ax.plot(x2[-1], y2[-1], marker='o')

plt.tight_layout()
plt.show()
