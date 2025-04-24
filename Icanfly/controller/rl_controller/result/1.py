import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取 CSV 并合并
df1 = pd.read_csv('PPO_L2_1.csv')
df2 = pd.read_csv('PPO_L2_2.csv')

max_step1 = df1['Step'].max()
df2['Step'] = df2['Step'] + max_step1

# 按行追加，并重置索引
df = pd.concat([df1, df2], ignore_index=True)


# 3. 绘图
plt.figure(figsize=(8, 5))
plt.plot(
    df['Step'],
    df['Value'],
    color="orange",
    linewidth=2,
    markerfacecolor="white",
    markeredgecolor="orange"
)
plt.xlabel('Step')
plt.ylabel('Value (grouped mean)')
plt.title('Episode Reward vs Step')
plt.grid(True)

plt.savefig('PPO_baseline_L2.png', dpi=800)
plt.show()