# # import numpy as np
# # import torch
# # # import gym
# # import gymnasium as gym
# # # from stable_baselines3 import PPO
# # # from stable_baselines3.common.callbacks import BaseCallback
# # # from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

# # from stable_baseline3_ppo import PPO
# # from commons.callbacks import BaseCallback
# # from commons.vec_env import DummyVecEnv, VecEnv

# # from torch.utils.tensorboard import SummaryWriter
# # import matplotlib.pyplot as plt
# # from torch import nn 
# # import datetime
# # import os


# # # 定义一个包装器，将 gymnasium 的新 API 转换为 gym 的 API
# # class GymnasiumWrapper(gym.Wrapper):
# #     def __init__(self, env):
# #         super().__init__(env)
    
# #     def step(self, action):
# #         # gymnasium 的 step 返回 (obs, reward, terminated, truncated, info)
# #         obs, reward, terminated, truncated, info = self.env.step(action)
# #         done = terminated or truncated
# #         return obs, reward, done, info

# #     def reset(self, **kwargs):
# #         # gymnasium 的 reset 返回 (obs, info)，这里只返回 obs
# #         obs, info = self.env.reset(**kwargs)
# #         return obs
    

# # class PPOTrainer:
# #     def __init__(self, env, total_timesteps=1e9, batch_size=64, n_steps=128,
# #                  gamma=0.99, gae_lambda=0.95, clip_range=0.08, ent_coef=0.0,
# #                  learning_rate=1e-4, model_path="./run/sb3_ppo_quadrotor"):
# #         # clip_range can decline zaosheng
# #         # ent_coef = 0 is good performance
# #         # 如果传入的环境未向量化，则先用 GymnasiumWrapper 包装，再用 DummyVecEnv 包装
# #         if not isinstance(env, VecEnv):
# #             self.env = DummyVecEnv([lambda: GymnasiumWrapper(env)])
# #         else:
# #             self.env = env
        
# #         self.total_timesteps = int(total_timesteps)
# #         self.model_path = model_path
        
# #         # 使用内置的 MlpPolicy，不再引用自定义策略
# #         self.model = PPO(
# #             policy="MlpPolicy",
# #             env=self.env,
# #             # policy_kwargs={"net_arch": [dict(pi=[256, 256 , 128], vf=[256,256,128])]},
# #             policy_kwargs={"net_arch": dict(pi=[128, 128], vf=[128, 128])
# #                         #    ,
# #                         #    "optimizer_kwargs": {"weight_decay": 1e-5 },
# #                         #     "log_std_init": -1
# #                         #    "activation_fn": nn.Tanh
# #                            },

# #             learning_rate=learning_rate,
# #             n_steps=n_steps,
# #             batch_size=batch_size,
# #             gamma=gamma,
# #             gae_lambda=gae_lambda,
# #             clip_range=clip_range,
# #             ent_coef=ent_coef,
# #             verbose=1,
# #             seed=42,
# #             device="cpu",  # 设置使用 GPU
# #             tensorboard_log="./rl_trajectory_run/sb3_tensorboard/"
# #         )
        
# #         # plt.ion()
# #         self.fig, self.ax = plt.subplots()
# #         self.ax.plot([], [], label="Episode Reward")
# #         self.ax.set_xlabel("Global Step")
# #         self.ax.set_ylabel("Reward")
# #         self.ax.legend()
# #         # self.fig.canvas.draw()
        
# #         self.episode_rewards = []
# #         self.steps = []
# #         self.writer = SummaryWriter(log_dir="./rl_trajectory_run/sb3_tensorboard/")
        
# #         self.callback = SB3CustomCallback(
# #             save_freq=10000,
# #             save_path="./rl_trajectory_run/sb3_checkpoints/",
# #             model=self.model,
# #             writer=self.writer,
# #             ax=self.ax,
# #             fig=self.fig,
# #             episode_rewards=self.episode_rewards,
# #             steps=self.steps
# #         )
    

# #     def train(self):
# #         print("Starting training...")
# #         self.model.learn(
# #             total_timesteps=self.total_timesteps,
# #             callback=self.callback,
# #             progress_bar=True
# #         )
# #         self._finalize_training()

# #     def _finalize_training(self):
# #         plt.ioff()
# #         self.ax.plot(self.steps, self.episode_rewards, label="Episode Reward")
# #         self.fig.canvas.draw()
# #         plt.show(block=True)
# #         self.model.save(self.model_path)
# #         self.writer.close()
# #         print(f"Final model saved at {self.model_path}")

# #     def load(self, path):
# #         self.model = PPO.load(path, env=self.env)
# #         print(f"Model loaded from {path}")

# # class SB3CustomCallback(BaseCallback):
# #     def __init__(self, save_freq, save_path, model, writer, ax, fig, episode_rewards, steps, verbose=0):
# #         super().__init__(verbose)
# #         self.save_freq = save_freq
# #         self.save_path = save_path
# #         self.model = model
# #         self.writer = writer
# #         self.ax = ax
# #         self.fig = fig
# #         self.episode_rewards = episode_rewards
# #         self.steps = steps


# #     def __call__(self, locals_: dict, globals_: dict):
# #         # This makes the callback callable as required
# #         return self._on_step()
        
# #     def _on_step(self) -> bool:
# #         if self.locals.get("infos"):
# #             for info in self.locals["infos"]:
                
# #                 if "reward" in info:
# #                     self.writer.add_scalar("Reward/Step", info["reward"], self.num_timesteps)
# #                 if "episode" in info:
# #                     # average_10_reward=0
# #                     # if len(self.episode_rewards) >= 10:
# #                     #     recent_10 = self.episode_rewards[-9:]
# #                     #     recent_10.append(info["episode"]["r"])
# #                     #     average_10_reward = sum(recent_10) / 10.0
# #                     #     self.episode_rewards.append(average_10_reward)
# #                     # else:
# #                     #     # continue
# #                     #     # average_10_reward = sum(self.episode_rewards) / len(self.episode_rewards)
# #                     #     self.episode_rewards.append(info["episode"]["r"])
                    
# #                     self.episode_rewards.append(info["episode"]["r"])
# #                     self.steps.append(self.num_timesteps)
# #                     # print(f"Episode ended at step {self.num_timesteps}, reward: {info['episode']['r']}")
                    
# #         if self.num_timesteps % self.save_freq == 0:
# #             save_path = f"{self.save_path}/ppo_quad_{self.num_timesteps}"
# #             self._update_plot(self.num_timesteps)
# #             self.model.save(save_path)
# #         return True

# #     def _update_plot(self, num_timesteps):
# #         self.ax.clear()

# #         self.ax.plot(self.steps, self.episode_rewards, label="Episode Reward")
# #         self.ax.set_xlabel("Global Step")
# #         self.ax.set_ylabel("Reward")
# #         self.ax.legend()
# #         # self.fig.canvas.draw()
# #         # self.fig.canvas.flush_events()
# #         current_date = datetime.datetime.now().strftime("%Y%m%d")
# #         # 拼接文件名，将当前日期插入到保存路径中
# #         file_path = f"/home/hello/catkin_ws_rotors/src/Icanfly/controller/rl_controller/result/{current_date}/"
# #         if not os.path.exists(file_path):
# #             os.makedirs(file_path)
# #         # 保存图像到文件
# #         self.fig.savefig(file_path + f"training_reward_{num_timesteps}.png", dpi=300)



# import os
# import datetime
# import numpy as np
# import torch
# import matplotlib
# # 使用 Agg 后端，适用于无图形界面的环境
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from torch import nn 
# from torch.utils.tensorboard import SummaryWriter

# import gymnasium as gym
# # 使用自定义改版的 PPO 和其他工具包
# from stable_baseline3_ppo import PPO
# from commons.callbacks import BaseCallback
# from commons.vec_env import DummyVecEnv, VecEnv


# # 定义一个包装器，将 gymnasium 的新 API 转换为 gym 的 API
# class GymnasiumWrapper(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
    
#     def step(self, action):
#         # gymnasium 的 step 返回 (obs, reward, terminated, truncated, info)
#         obs, reward, terminated, truncated, info = self.env.step(action)
#         done = terminated or truncated
#         return obs, reward, done, info

#     def reset(self, **kwargs):
#         # gymnasium 的 reset 返回 (obs, info)，这里只返回 obs
#         obs, info = self.env.reset(**kwargs)
#         return obs
    

# class PPOTrainer:
#     def __init__(self, env, total_timesteps=1e9, batch_size=64, n_steps=128,
#                  gamma=0.99, gae_lambda=0.95, clip_range=0.08, ent_coef=0.0,
#                  learning_rate=1e-4, model_path="./run/sb3_ppo_quadrotor"):
#         # 如果传入的环境未向量化，则先用 GymnasiumWrapper 包装，再用 DummyVecEnv 包装
#         if not isinstance(env, VecEnv):
#             self.env = DummyVecEnv([lambda: GymnasiumWrapper(env)])
#         else:
#             self.env = env
        
#         self.total_timesteps = int(total_timesteps)
#         self.model_path = model_path
        
#         # 使用内置的 MlpPolicy，不再引用自定义策略
#         self.model = PPO(
#             policy="MlpPolicy",
#             env=self.env,
#             # 可以根据需要调整策略网络结构
#             policy_kwargs={"net_arch": dict(pi=[128, 128], vf=[128, 128])},
#             learning_rate=learning_rate,
#             n_steps=n_steps,
#             batch_size=batch_size,
#             gamma=gamma,
#             gae_lambda=gae_lambda,
#             clip_range=clip_range,
#             ent_coef=ent_coef,
#             verbose=1,
#             seed=42,
#             device="cpu",  # 根据需求设置设备
#             tensorboard_log="./rl_trajectory_run/sb3_tensorboard/"
#         )
        
#         # 创建图像用于保存训练曲线，不启用交互模式
#         self.fig, self.ax = plt.subplots()
#         self.ax.plot([], [], label="Episode Reward")
#         self.ax.set_xlabel("Global Step")
#         self.ax.set_ylabel("Reward")
#         self.ax.legend()
        
#         self.episode_rewards = []
#         self.steps = []
#         self.writer = SummaryWriter(log_dir="./rl_trajectory_run/sb3_tensorboard/")
        
#         self.callback = SB3CustomCallback(
#             save_freq=100,
#             save_path="./rl_trajectory_run/sb3_checkpoints/",
#             model=self.model,
#             writer=self.writer,
#             ax=self.ax,
#             fig=self.fig,
#             episode_rewards=self.episode_rewards,
#             steps=self.steps
#         )
    

#     def train(self):
#         print("Starting training...")
#         self.model.learn(
#             total_timesteps=self.total_timesteps,
#             callback=self.callback,
#             progress_bar=True
#         )
#         self._finalize_training()

#     def _finalize_training(self):
#         # 最终保存时调用 _update_plot 更新一次曲线并保存图像
#         self.callback._update_plot(self.callback.num_timesteps)
#         self.model.save(self.model_path)
#         self.writer.close()
#         # 关闭图像资源
#         plt.close(self.fig)
#         print(f"Final model saved at {self.model_path}")

#     def load(self, path):
#         self.model = PPO.load(path, env=self.env)
#         print(f"Model loaded from {path}")

    
# class SB3CustomCallback(BaseCallback):
#     def __init__(self, save_freq, save_path, model, writer, ax, fig, episode_rewards, steps, verbose=0):
#         super().__init__(verbose)
#         self.save_freq = save_freq
#         self.save_path = save_path
#         self.model = model
#         self.writer = writer
#         self.ax = ax
#         self.fig = fig
#         self.episode_rewards = episode_rewards
#         self.steps = steps
#         self.num_timesteps = 0  # 初始化步数计数

#     def __call__(self, locals_: dict, globals_: dict):
#         # 每步回调时更新内部计数，并调用 _on_step
#         self.num_timesteps = locals_.get("self", self).num_timesteps if "self" in locals_ else self.num_timesteps + 1
#         return self._on_step()
        
#     def _on_step(self) -> bool:
#         # 检查是否有 info 信息记录奖励数据
#         if self.locals.get("infos"):
#             for info in self.locals["infos"]:
#                 if "reward" in info:
#                     self.writer.add_scalar("Reward/Step", info["reward"], self.num_timesteps)
#                 if "episode" in info:
#                     self.episode_rewards.append(info["episode"]["r"])
#                     self.steps.append(self.num_timesteps)
                    
#         # 每达到保存频率时更新图像并保存模型
#         if self.num_timesteps % self.save_freq == 0:
#             save_path = f"{self.save_path}/ppo_quad_{self.num_timesteps}"
#             self._update_plot(self.num_timesteps)
#             self.model.save(save_path)
#         return True

#     def _update_plot(self, num_timesteps):
#         self.ax.clear()
#         self.ax.plot(self.steps, self.episode_rewards, label="Episode Reward")
#         self.ax.set_xlabel("Global Step")
#         self.ax.set_ylabel("Reward")
#         self.ax.legend()

#         self.ax.show()
        
#         # 构造保存图片的目录及文件名（确保文件夹存在）
#         current_date = datetime.datetime.now().strftime("%Y%m%d")
#         file_dir = f"/home/hello/catkin_ws_rotors/src/Icanfly/controller/rl_controller/result/{current_date}/"
#         if not os.path.exists(file_dir):
#             os.makedirs(file_dir)
#         file_path = os.path.join(file_dir, f"training_reward_{num_timesteps}.png")
        
#         # 保存图像
#         self.fig.savefig(file_path, dpi=300)



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

import gymnasium as gym
from stable_baseline3_ppo import PPO
from commons.callbacks import BaseCallback
from commons.vec_env import DummyVecEnv, VecEnv

current_date = datetime.datetime.now().strftime("%Y%m%d")
file_dir = f"/home/hello/catkin_ws_rotors/rl_trajectory_run/result/{current_date}/"
checkpoints_file_dir = file_dir+"/sb3_checkpoints/"
tensorboard_file_dir = file_dir+"/sb3_tensorboard/"
reward_file_dir = file_dir+"/reward/"

if not os.path.exists(reward_file_dir):
    os.makedirs(reward_file_dir)

if not os.path.exists(checkpoints_file_dir):
    os.makedirs(checkpoints_file_dir)

if not os.path.exists(tensorboard_file_dir):
    os.makedirs(tensorboard_file_dir)



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
                 gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
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
            policy_kwargs={"net_arch": dict(pi=[128, 128], vf=[128, 128])},
            # learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            # gamma=gamma,
            # gae_lambda=gae_lambda,
            # clip_range=clip_range,
            # ent_coef=ent_coef,
            verbose=1,
            seed=42,
            device="cpu",  # 根据需求设置设备
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
        self.writer = SummaryWriter(log_dir=tensorboard_file_dir)
        
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
        self.num_timesteps = 0  # 初始化步数计数

    def __call__(self, locals_: dict, globals_: dict):
        self.num_timesteps = locals_.get("self", self).num_timesteps if "self" in locals_ else self.num_timesteps + 1
        return self._on_step(locals_)
        
    def _on_step(self, locals_):
        infos = locals_.get("infos", None)
        if infos:
            # print("infos:", infos)  # 调试用，查看infos内容
            for info in infos:
                if "reward" in info:
                    self.writer.add_scalar("Reward/Step", info["reward"], self.num_timesteps)
                if "episode" in info:
                    # print("self.num_timesteps: ",self.num_timesteps, "episode reward: " ,info["episode"]["r"])
                    self.episode_rewards.append(info["episode"]["r"])
                    self.steps.append(self.num_timesteps)
                
        # 每达到保存频率时更新图像并保存模型
        if self.num_timesteps % self.save_freq == 0:
            save_path = f"{self.save_path}/ppo_quad_{self.num_timesteps}"
            self._update_plot(self.num_timesteps)
            self.model.save(save_path)
        return True

    def _update_plot(self, num_timesteps):
        self.ax.clear()
        self.ax.plot(self.steps, self.episode_rewards, label="Episode Reward")
        self.ax.set_xlabel("Global Step")
        self.ax.set_ylabel("Reward")
        self.ax.legend()

        file_path = os.path.join(reward_file_dir, f"training_reward_{num_timesteps}.png")
        
        # 保存图像
        self.fig.savefig(file_path, dpi=600)
