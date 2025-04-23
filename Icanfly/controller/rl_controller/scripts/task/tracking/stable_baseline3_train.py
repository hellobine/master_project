import csv 
import csv 
import os
import datetime
import numpy as np
import torch
import matplotlib
# 使用 Agg 后端，适用于无图形界面的环境
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from stable_baseline3_ppo import PPO
from stable_baselines3.commons.callbacks import BaseCallback
from stable_baselines3.commons.vec_env import DummyVecEnv, VecEnv

# ----------------------------------------------------------------------------
# ⬇️ 目录结构按【日期】存放一次训练所有输出
# ----------------------------------------------------------------------------
current_date = datetime.datetime.now().strftime("%Y%m%d")
file_dir = f"/home/hello/catkin_ws_rotors/rl_trajectory_run/result/task/tracking/result/{current_date}"
checkpoints_file_dir = os.path.join(file_dir, "sb3_checkpoints")
tensorboard_file_dir = os.path.join(file_dir, "sb3_tensorboard")
reward_file_dir = os.path.join(file_dir, "reward")

for _dir in (checkpoints_file_dir, tensorboard_file_dir, reward_file_dir):
    os.makedirs(_dir, exist_ok=True)


# ----------------------------------------------------------------------------
# ⬇️ gymnasium → gym 兼容包装
# ----------------------------------------------------------------------------
class GymnasiumWrapper(gym.Wrapper):
    """把 (obs, reward, terminated, truncated, info) 改成 gym 风格"""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs, _info = self.env.reset(**kwargs)
        return obs


# ----------------------------------------------------------------------------
# ⬇️ 自定义回调：只保存纯 PyTorch 权重 + 绘图 + TensorBoard
# ----------------------------------------------------------------------------
class SB3CustomCallback(BaseCallback):
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        writer: SummaryWriter,
        ax: plt.Axes,
        fig: plt.Figure,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.writer = writer
        self.ax = ax
        self.fig = fig
        self.episode_rewards = []
        self.steps = []
        os.makedirs(self.save_path, exist_ok=True)

    # ------------------------------------------------------------------
    # 🔽 只保存纯 PyTorch 权重（.pt）——推理/微调足够
    # ------------------------------------------------------------------
    def save_model(self, num_timesteps: int):
        model_prefix = os.path.join(self.save_path, f"model_{num_timesteps}_steps")

        torch.save(
            {
                "policy": self.model.policy.state_dict(),
                "optimizer": self.model.policy.optimizer.state_dict(),
            },
            f"{model_prefix}.pt",
        )
        if self.verbose:
            print(f"[Callback] Saved weights to {model_prefix}.pt")

    # ------------------------------------------------------------------
    # 🔽 每步调用
    # ------------------------------------------------------------------
    def _on_step(self) -> bool:
        # 1) 收集 episode reward 输出到 TensorBoard
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                r = info["episode"]["r"]
                self.writer.add_scalar("Reward/Episode", r, self.num_timesteps)
                self.episode_rewards.append(r)
                self.steps.append(self.num_timesteps)

        # 2) 到 save_freq 时保存模型 + 更新折线图
        if self.num_timesteps % self.save_freq == 0 and self.num_timesteps > 0:
            self.save_model(self.num_timesteps)
            self._update_plot(self.num_timesteps)
            csv_file_path = os.path.join(reward_file_dir, f"training_data_{self.num_timesteps}.csv")
            with open(csv_file_path, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Step", "Episode Reward"])
                for step, reward in zip(self.steps, self.episode_rewards):
                    writer.writerow([step, reward])

        return True

    # ------------------------------------------------------------------
    # 🔽 更新并保存训练曲线
    # ------------------------------------------------------------------
    def _update_plot(self, num_timesteps: int):
        step_dict = {}
        for s, r in zip(self.steps, self.episode_rewards):
            step_dict.setdefault(s, []).append(r)
        xs = sorted(step_dict)
        ys = [np.mean(step_dict[x]) for x in xs]

        self.ax.clear()
        self.ax.plot(xs, ys, label="Avg Episode Reward")
        self.ax.set_xlabel("Global Step")
        self.ax.set_ylabel("Reward")
        self.ax.legend()
        self.ax.set_title(f"Training Reward @ {num_timesteps}")

        fig_path = os.path.join(reward_file_dir, f"training_reward_{num_timesteps}.png")
        # self.fig.savefig(fig_path, dpi=300)
        if self.verbose:
            print(f"[Callback] Plot saved to {fig_path}")


# ----------------------------------------------------------------------------
# ⬇️ 训练封装
# ----------------------------------------------------------------------------
class PPOTrainer:
    def __init__(
        self,
        env,
        total_timesteps=1e7,
        batch_size=64,
        n_steps=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        learning_rate=1e-4,
        device="cuda:0"
    ):
        # ----------------- 环境向量化 -----------------
        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: GymnasiumWrapper(env)])
        self.env = env
        self.total_timesteps = int(total_timesteps)

        # ----------------- SB3 模型 ------------------
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            policy_kwargs={"net_arch": dict(pi=[128, 128], vf=[128, 128])},
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            verbose=1,
            seed=42,
            device=device,
            tensorboard_log=tensorboard_file_dir,
        )

        # ----------------- 绘图 / TensorBoard ------------------
        self.fig, self.ax = plt.subplots()
        self.ax.plot([], [], label="Episode Reward")
        self.ax.set_xlabel("Global Step")
        self.ax.set_ylabel("Reward")
        self.ax.legend()

        writer = SummaryWriter(log_dir=tensorboard_file_dir)
        self.callback = SB3CustomCallback(
            save_freq=10_000,
            save_path=checkpoints_file_dir,
            writer=writer,
            ax=self.ax,
            fig=self.fig,
            verbose=1,
        )

    # ----------------- 开始训练 ------------------
    def train(self):
        print("Starting training…")
        self.model.learn(
            total_timesteps=int(self.total_timesteps),  # 可按需改
            callback=self.callback,
            progress_bar=True,
        )
        self._finalize_training()

    # ----------------- 结束善后 ------------------
    def _finalize_training(self):
        self.callback._update_plot(self.callback.num_timesteps)
        self.callback.save_model(self.callback.num_timesteps)
        self.callback.writer.close()
        plt.close(self.fig)
        print("Training finished, final weights saved.")

    # ----------------- 加载权重（纯 PyTorch） ------------------
    def load_weights(self, pt_path):
        state = torch.load(pt_path, map_location="cuda", weights_only=True)
        self.model.policy.load_state_dict(state["policy"])
        if "optimizer" in state:
            self.model.policy.optimizer.load_state_dict(state["optimizer"])
        print(f"Weights loaded from {pt_path}")
