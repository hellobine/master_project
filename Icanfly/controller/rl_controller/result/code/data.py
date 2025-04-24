import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams.update({
    "font.size": 12,                 # 和 LaTeX 文档字体一致（12pt）
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "font.family": "serif",          # 字体风格更接近 LaTeX
    "figure.figsize": (6.0, 3.6),    # 子图合计占宽度 6 inch（≈0.45 * 2）
    "savefig.dpi": 800,               # 保证高清
})

# ----------- 读取数据 -----------
df = pd.read_csv('arm_sim_data.csv')

t_s     = df['t_s']
q1_sim  = df['q1_sim']
q1_ref  = df['q1_ref']
q2_sim  = df['q2_sim']
q2_ref  = df['q2_ref']
tau1    = df['tau1']
tau2    = df['tau2']

# ----------- 文件保存路径设置 -----------
os.makedirs('figures', exist_ok=True)

# ----------- 全局风格 -----------
plt.rcParams['font.size'] = 12
sns.set_style('whitegrid')
colors = sns.color_palette("tab10")

# ----------- q1 对比图 -----------
plt.figure()
plt.plot(t_s, q1_sim, label='q1_real', color="red", linewidth=1.8)
plt.plot(t_s, q1_ref, label='q1_ref', color="green", linestyle='--', linewidth=1.8)


def find_first_cross(t, sim, ref):
    """
    返回 sim - ref 第一次过零的线性插值时间，
    如果根本没交点则返回 None。
    """
    err = sim - ref
    sign_changes = np.where(np.diff(np.sign(err)))[0]
    if sign_changes.size == 0:
        return None
    i = sign_changes[0]
    t0, t1 = t[i],   t[i+1]
    e0, e1 = err[i], err[i+1]
    # 线性插值
    return t0 - e0 * (t1 - t0) / (e1 - e0)

# 示例用法（假设你已经有 t_s, q1_sim, q1_ref）：
t_cross_q1 = find_first_cross(t_s, q1_sim, q1_ref)
if t_cross_q1 is not None:
    print(f'q1 第一次交点在 t = {t_cross_q1:.3f} s')
else:
    print('q1 无交点')


y_cross = np.interp(t_cross_q1, t_s+0.4, q1_sim)
plt.axvline(x=t_cross_q1, color='red', linestyle=':', linewidth=1)
plt.text(t_cross_q1, y_cross, f't = {t_cross_q1:.2f}s' ,
         color='red', ha='left', va='bottom', rotation=0)

plt.xlabel('Time (s)')
plt.ylabel('Joint Angle (rad)')
plt.title('q1: Real vs Reference')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/q1_comparison.png', dpi=800)
plt.show()

# ----------- q2 对比图 -----------
plt.figure()
plt.plot(t_s, q2_sim, label='q2_real', color="red", linewidth=1.8)
plt.plot(t_s, q2_ref, label='q2_ref', color="blue", linestyle='--', linewidth=1.8)



t_cross_q2 = find_first_cross(t_s, q2_sim, q2_ref)
if t_cross_q2 is not None:
    print(f'q1 第一次交点在 t = {t_cross_q2:.3f} s')
else:
    print('q1 无交点')

y_cross = np.interp(t_cross_q2, t_s+0.6, q1_sim + 0.9)
plt.axvline(x=t_cross_q2, color='red', linestyle=':', linewidth=1)
plt.text(t_cross_q2, y_cross, f't = {t_cross_q2:.2f}s' ,
         color='red', ha='left', va='bottom', rotation=0)


plt.xlabel('Time (s)')
plt.ylabel('Joint Angle (rad)')
plt.title('q2: Real vs Reference')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/q2_comparison.png', dpi=800)
plt.show()

# ----------- 力矩输入图 -----------
plt.figure()
plt.plot(t_s, tau1, label='tau1', color="orange", linewidth=1.8)
plt.plot(t_s, tau2, label='tau2', color="purple", linewidth=1.8)
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.title('Control Torque Inputs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/torque_inputs.png', dpi=800)
plt.show()

# ----------- 跟踪误差箱型图（不加点） -----------
df['err_q1'] = q1_sim - q1_ref
df['err_q2'] = q2_sim - q2_ref
err_long = pd.melt(df[['err_q1', 'err_q2']], var_name='Joint', value_name='Error')

plt.figure(figsize=(6, 4))
sns.set_palette('pastel')

ax = sns.boxplot(
    data=err_long,
    x='Joint',
    y='Error',
    showfliers=False,
    width=0.5,
    linewidth=1.2,
    color='purple',
)

ax.set(
    xlabel='',
    ylabel='Tracking Error (rad)',
    title='Tracking Error Distribution'
)
sns.despine(offset=5, trim=True)
plt.tight_layout()
plt.savefig('figures/tracking_error_boxplot.png', dpi=800)
plt.show()
