# import csv 
# import csv 
# import os
# import datetime
# import numpy as np
# import torch
# import matplotlib
# # 使用 Agg 后端，适用于无图形界面的环境
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
# import gymnasium as gym
# from stable_baseline3_ppo import PPO
# from stable_baselines3.commons.callbacks import BaseCallback
# from stable_baselines3.commons.vec_env import DummyVecEnv, VecEnv

# # ----------------------------------------------------------------------------
# # ⬇️ 目录结构按【日期】存放一次训练所有输出
# # ----------------------------------------------------------------------------
# current_date = datetime.datetime.now().strftime("%Y%m%d")
# file_dir = f"/home/hello/catkin_ws_rotors/rl_trajectory_run/result/task/tracking/result/{current_date}"
# checkpoints_file_dir = os.path.join(file_dir, "sb3_checkpoints")
# tensorboard_file_dir = os.path.join(file_dir, "sb3_tensorboard")
# reward_file_dir = os.path.join(file_dir, "reward")

# for _dir in (checkpoints_file_dir, tensorboard_file_dir, reward_file_dir):
#     os.makedirs(_dir, exist_ok=True)


# # ----------------------------------------------------------------------------
# # ⬇️ gymnasium → gym 兼容包装
# # ----------------------------------------------------------------------------
# class GymnasiumWrapper(gym.Wrapper):
#     """把 (obs, reward, terminated, truncated, info) 改成 gym 风格"""

#     def step(self, action):
#         obs, reward, terminated, truncated, info = self.env.step(action)
#         done = terminated or truncated
#         return obs, reward, done, info

#     def reset(self, **kwargs):
#         obs, _info = self.env.reset(**kwargs)
#         return obs


# # ----------------------------------------------------------------------------
# # ⬇️ 自定义回调：只保存纯 PyTorch 权重 + 绘图 + TensorBoard
# # ----------------------------------------------------------------------------
# class SB3CustomCallback(BaseCallback):
#     def __init__(
#         self,
#         save_freq: int,
#         save_path: str,
#         writer: SummaryWriter,
#         ax: plt.Axes,
#         fig: plt.Figure,
#         verbose: int = 0,
#     ):
#         super().__init__(verbose)
#         self.save_freq = save_freq
#         self.save_path = save_path
#         self.writer = writer
#         self.ax = ax
#         self.fig = fig
#         self.episode_rewards = []
#         self.steps = []
#         os.makedirs(self.save_path, exist_ok=True)

#     # ------------------------------------------------------------------
#     # 🔽 只保存纯 PyTorch 权重（.pt）——推理/微调足够
#     # ------------------------------------------------------------------
#     def save_model(self, num_timesteps: int):
#         model_prefix = os.path.join(self.save_path, f"model_{num_timesteps}_steps")

#         # torch.save(
#         #     {
#         #         "policy": self.model.policy.state_dict(),
#         #         "optimizer": self.model.policy.optimizer.state_dict(),
#         #     },
#         #     f"{model_prefix}.pt",
#         # )
#         self.model.save(f"{model_prefix}") 
#         if self.verbose:
#             print(f"[Callback] Saved weights to {model_prefix}.pt")

#     # ------------------------------------------------------------------
#     # 🔽 每步调用
#     # ------------------------------------------------------------------
#     def _on_step(self) -> bool:
#         # 1) 收集 episode reward 输出到 TensorBoard
#         infos = self.locals.get("infos", [])
#         for info in infos:
#             if "episode" in info:
#                 r = info["episode"]["r"]
#                 self.writer.add_scalar("Reward/Episode", r, self.num_timesteps)
#                 self.episode_rewards.append(r)
#                 self.steps.append(self.num_timesteps)

#         # 2) 到 save_freq 时保存模型 + 更新折线图
#         if self.num_timesteps % self.save_freq == 0 and self.num_timesteps > 0:
#             self.save_model(self.num_timesteps)
#             self._update_plot(self.num_timesteps)
#             csv_file_path = os.path.join(reward_file_dir, f"training_data_{self.num_timesteps}.csv")
#             with open(csv_file_path, mode='w', newline='') as csv_file:
#                 writer = csv.writer(csv_file)
#                 writer.writerow(["Step", "Episode Reward"])
#                 for step, reward in zip(self.steps, self.episode_rewards):
#                     writer.writerow([step, reward])

#         return True

#     # ------------------------------------------------------------------
#     # 🔽 更新并保存训练曲线
#     # ------------------------------------------------------------------
#     def _update_plot(self, num_timesteps: int):
#         step_dict = {}
#         for s, r in zip(self.steps, self.episode_rewards):
#             step_dict.setdefault(s, []).append(r)
#         xs = sorted(step_dict)
#         ys = [np.mean(step_dict[x]) for x in xs]

#         self.ax.clear()
#         self.ax.plot(xs, ys, label="Avg Episode Reward")
#         self.ax.set_xlabel("Global Step")
#         self.ax.set_ylabel("Reward")
#         self.ax.legend()
#         self.ax.set_title(f"Training Reward @ {num_timesteps}")

#         fig_path = os.path.join(reward_file_dir, f"training_reward_{num_timesteps}.png")
#         # self.fig.savefig(fig_path, dpi=300)
#         if self.verbose:
#             print(f"[Callback] Plot saved to {fig_path}")


# # ----------------------------------------------------------------------------
# # ⬇️ 训练封装
# # ----------------------------------------------------------------------------
# class PPOTrainer:
#     def __init__(
#         self,
#         env,
#         total_timesteps=1e7,
#         batch_size=64,
#         n_steps=128,
#         gamma=0.99,
#         gae_lambda=0.95,
#         clip_range=0.1,
#         learning_rate=1e-4
#         # device="cuda:0"
#     ):
#         # ----------------- 环境向量化 -----------------
#         if not isinstance(env, VecEnv):
#             env = DummyVecEnv([lambda: GymnasiumWrapper(env)])
#         self.env = env
#         self.total_timesteps = int(total_timesteps)

#         # ----------------- SB3 模型 ------------------
#         self.model = PPO(
#             policy="MlpPolicy",
#             env=self.env,
#             policy_kwargs={"net_arch": dict(pi=[128, 128], vf=[128, 128])},
#             learning_rate=learning_rate,
#             n_steps=n_steps,
#             batch_size=batch_size,
#             gamma=gamma,
#             gae_lambda=gae_lambda,
#             clip_range=clip_range,
#             verbose=1,
#             seed=42,
#             device="cpu",  # 根据需求设置设备
#             tensorboard_log=tensorboard_file_dir,
#         )

#         # ----------------- 绘图 / TensorBoard ------------------
#         self.fig, self.ax = plt.subplots()
#         self.ax.plot([], [], label="Episode Reward")
#         self.ax.set_xlabel("Global Step")
#         self.ax.set_ylabel("Reward")
#         self.ax.legend()

#         writer = SummaryWriter(log_dir=tensorboard_file_dir)
#         self.callback = SB3CustomCallback(
#             save_freq=100,
#             save_path=checkpoints_file_dir,
#             writer=writer,
#             ax=self.ax,
#             fig=self.fig,
#             verbose=1,
#         )

#     # ----------------- 开始训练 ------------------
#     def train(self):
#         print("Starting training…")
#         self.model.learn(
#             total_timesteps=int(self.total_timesteps),  # 可按需改
#             callback=self.callback,
#             progress_bar=True,
#         )
#         self._finalize_training()

#     # ----------------- 结束善后 ------------------
#     def _finalize_training(self):
#         self.callback._update_plot(self.callback.num_timesteps)
#         self.callback.save_model(self.callback.num_timesteps)
#         self.callback.writer.close()
#         plt.close(self.fig)
#         print("Training finished, final weights saved.")

#     # ----------------- 加载权重（纯 PyTorch） ------------------
#     def load_weights(self, pt_path):
#         state = torch.load(pt_path)
#         self.model.policy.load_state_dict(state["policy"])
#         if "optimizer" in state:
#             self.model.policy.optimizer.load_state_dict(state["optimizer"])
#         print(f"Weights loaded from {pt_path}")

import os
import datetime
import numpy as np
import torch
import matplotlib
# 使用 Agg 后端，适用于无图形界面的环境
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import nn 
from torch.utils.tensorboard import SummaryWriter
import csv 
import re
from pathlib import Path
import gymnasium as gym
from stable_baseline3_ppo import PPO
from stable_baselines3.commons.callbacks import BaseCallback
from stable_baselines3.commons.vec_env import DummyVecEnv, VecEnv

current_date = datetime.datetime.now().strftime("%Y%m%d")
task_name = "tracking_standard_ppo"
file_dir = f"/home/hello/catkin_ws_rotors/rl_trajectory_run/result/task/tracking/{task_name}/result/{current_date}"
checkpoints_file_dir = file_dir+"/sb3_checkpoints/"
tensorboard_file_dir = file_dir+"/sb3_tensorboard/"
reward_file_dir = file_dir+"/reward/"

if not os.path.exists(reward_file_dir):
    os.makedirs(reward_file_dir)

if not os.path.exists(checkpoints_file_dir):
    os.makedirs(checkpoints_file_dir)

if not os.path.exists(tensorboard_file_dir):
    os.makedirs(tensorboard_file_dir)



def get_next_run_dir(base_dir: str = "runs",
                     prefix: str = "exp_",
                     ) -> str:
    """
    在 base_dir 下找所有形如 prefix<number> 的子目录，
    取最大的数字 +1，返回新的 run 目录路径。
    """
    base = Path(base_dir)
    base.mkdir(exist_ok=True)

    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    nums = []
    for d in base.iterdir():
        if d.is_dir():
            m = pattern.match(d.name)
            if m:
                nums.append(int(m.group(1)))
    next_id = max(nums) + 1 if nums else 1
    return str(base / f"{prefix}{next_id}")


class GymnasiumWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        # gymnasium 的 step 返回 (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def reset(self, **kwargs):
        # gymnasium 的 reset 返回 (obs, info)，这里只返回 obs
        obs, info = self.env.reset(**kwargs)
        return obs
    

class PPOTrainer:
    def __init__(self, env, total_timesteps=1e9, batch_size=64, n_steps=128,
                 gamma=0.99, gae_lambda=0.95, clip_range=0.1, ent_coef=0.0,
                 learning_rate=1e-4,):
        # 如果传入的环境未向量化，则先用 GymnasiumWrapper 包装，再用 DummyVecEnv 包装
        if not isinstance(env, VecEnv):
            self.env = DummyVecEnv([lambda: GymnasiumWrapper(env)])
        else:
            self.env = env
        
        self.total_timesteps = int(total_timesteps)
        self.model_path = checkpoints_file_dir
        
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            # 可以根据需要调整策略网络结构
            policy_kwargs={"net_arch": dict(pi=[128, 128], vf=[128, 128])
                           ,
                           "optimizer_kwargs": {"weight_decay": 1e-5 }
                           
                           },
            # learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            # gamma=gamma,
            # gae_lambda=gae_lambda,
            clip_range=clip_range,
            # ent_coef=ent_coef,
            l2_lambda  = 0.0,
            verbose=1,
            seed=42,
            device="cuda:0",  # 根据需求设置设备
            tensorboard_log=tensorboard_file_dir
        )
        
        # 创建图像用于保存训练曲线，不启用交互模式
        self.fig, self.ax = plt.subplots()
        self.ax.plot([], [], label="Episode Reward")
        self.ax.set_xlabel("Global Step")
        self.ax.set_ylabel("Reward")
        self.ax.legend()
        
        self.episode_rewards = []
        self.steps = []
        
        log_dir = get_next_run_dir(base_dir=tensorboard_file_dir, prefix="episode_")
        self.writer = SummaryWriter(log_dir=log_dir)
        
        
        self.callback = SB3CustomCallback(
            save_freq=10000,
            save_path=checkpoints_file_dir,
            model=self.model,
            writer=self.writer,
            ax=self.ax,
            fig=self.fig,
            episode_rewards=self.episode_rewards,
            steps=self.steps
        )
    

    def train(self):
        print("Starting training...")
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=self.callback,
            progress_bar=True
        )
        self._finalize_training()

    def _finalize_training(self):
        # 最终保存时调用 _update_plot 更新一次曲线并保存图像
        self.callback._update_plot(self.callback.num_timesteps)
        self.model.save(self.model_path)
        self.writer.close()
        # 关闭图像资源
        plt.close(self.fig)
        print(f"Final model saved at {self.model_path}")

    def load(self, path):
        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded from {path}")

    
class SB3CustomCallback(BaseCallback):
    def __init__(self, save_freq, save_path, model, writer, ax, fig, episode_rewards, steps, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.model = model
        self.writer = writer
        self.ax = ax
        self.fig = fig
        self.episode_rewards = episode_rewards
        self.steps = steps
        self.num_timesteps = 0  

    # def __call__(self, locals_: dict, globals_: dict):
    #     # self.num_timesteps = locals_.get("self", self).num_timesteps if "self" in locals_ else self.num_timesteps + 1

    #     # self.num_timesteps += 1
    #     return self._on_step(locals_)
        
    def _on_step(self):
        infos = self.locals.get("infos", None)
        if infos:
            # print("infos:", infos)  # 调试用，查看infos内容
            for info in infos:
                # if "reward" in info:
                    # self.writer.add_scalar("Reward/Step", info["reward"], self.num_timesteps)
                if "episode" in info:
                    # print("self.num_timesteps: ", self.num_timesteps)
                    self.num_timesteps += info["episode"]["l"]
                    self.episode_rewards.append(info["episode"]["r"])
                    self.steps.append(self.num_timesteps)
                    self.writer.add_scalar("Reward/episode", info["episode"]["r"], self.num_timesteps)
         
        # print("self.num_timesteps % self.save_freq: ", self.num_timesteps % self.save_freq)
        if self.num_timesteps % self.save_freq == 0:
            save_path = f"{self.save_path}/ppo_quad_{self.num_timesteps}"
            self._update_plot(self.num_timesteps)
            self.model.save(save_path)

        # if self.num_timesteps % self.save_freq < 100:
            csv_file_path = os.path.join(reward_file_dir, f"training_data_{self.num_timesteps}.csv")
            with open(csv_file_path, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Step", "Episode Reward"])
                for step, reward in zip(self.steps, self.episode_rewards):
                    writer.writerow([step, reward])

        return True

    def _update_plot(self, num_timesteps):
        
        step_reward_dict = {}
        for s, r in zip(self.steps, self.episode_rewards):
            if s in step_reward_dict:
                step_reward_dict[s].append(r)
            else:
                step_reward_dict[s] = [r]
        # 获取排好序的唯一 step 值
        unique_steps = sorted(step_reward_dict.keys())
        # 对每个 step 计算对应奖励的平均值
        avg_rewards = [sum(step_reward_dict[s]) / len(step_reward_dict[s]) for s in unique_steps]

        # 清除之前的绘图，然后用去重后的数据绘制
        self.ax.clear()
        self.ax.plot(unique_steps, avg_rewards, label="Episode Reward")
        self.ax.set_xlabel("Global Step")
        self.ax.set_ylabel("Reward")
        self.ax.legend()

        # 保存图像
        file_path = os.path.join(reward_file_dir, f"training_reward_{num_timesteps}.png")
        self.fig.savefig(file_path, dpi=600)
        
        
