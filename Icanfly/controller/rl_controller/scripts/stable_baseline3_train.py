import numpy as np
import torch
# import gym
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


# seed = 42
# np.random.seed(seed)
# torch.manual_seed(seed)


# 定义一个包装器，将 gymnasium 的新 API 转换为 gym 的 API
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
    

class SB3PPOTrainer:
    def __init__(self, env, total_timesteps=1e9, batch_size=64, n_steps=128,
                 gamma=0.99, gae_lambda=0.95, clip_range=0.15, ent_coef=0.12,
                 learning_rate=1e-4, model_path="./run/sb3_ppo_quadrotor"):
        
        
        # 如果传入的环境未向量化，则先用 GymnasiumWrapper 包装，再用 DummyVecEnv 包装
        if not isinstance(env, VecEnv):
            self.env = DummyVecEnv([lambda: GymnasiumWrapper(env)])
        else:
            self.env = env
        
        self.total_timesteps = int(total_timesteps)
        self.model_path = model_path
        
        # 使用内置的 MlpPolicy，不再引用自定义策略
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            # policy_kwargs={"net_arch": [dict(pi=[256, 256 , 128], vf=[256,256,128])]},
            policy_kwargs={"net_arch": dict(pi=[128, 128], vf=[128, 128])
                        #    "optimizer_kwargs": {"weight_decay": 1e-6 }
                           },

            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=1,
            # seed=seed,
            # device="cpu",  # 设置使用 GPU
            tensorboard_log="./rl_trajectory_run/sb3_tensorboard/"
        )
        
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.plot([], [], label="Episode Reward")
        self.ax.set_xlabel("Global Step")
        self.ax.set_ylabel("Reward")
        self.ax.legend()
        self.fig.canvas.draw()
        
        self.episode_rewards = []
        self.steps = []
        self.writer = SummaryWriter(log_dir="./rl_trajectory_run/sb3_tensorboard/")
        
        self.callback = SB3CustomCallback(
            save_freq=5000,
            save_path="./rl_trajectory_run/sb3_checkpoints/",
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
        plt.ioff()
        self.ax.plot(self.steps, self.episode_rewards, label="Episode Reward")
        self.fig.canvas.draw()
        plt.show(block=True)
        self.model.save(self.model_path)
        self.writer.close()
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
        
    def _on_step(self) -> bool:
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                
                if "reward" in info:
                    self.writer.add_scalar("Reward/Step", info["reward"], self.num_timesteps)
                if "episode" in info:
                    average_10_reward=0
                    if len(self.episode_rewards) >= 10:
                        recent_10 = self.episode_rewards[-9:]
                        recent_10.append(info["episode"]["r"])
                        average_10_reward = sum(recent_10) / 10.0
                        self.episode_rewards.append(average_10_reward)
                    else:
                        # average_10_reward = sum(self.episode_rewards) / len(self.episode_rewards)
                        self.episode_rewards.append(info["episode"]["r"])
                    
                    # self.episode_rewards.append(info["episode"]["r"])
                    self.steps.append(self.num_timesteps)
                    print(f"Episode ended at step {self.num_timesteps}, reward: {info['episode']['r']}")
                    
        if self.num_timesteps % self.save_freq == 0:
            save_path = f"{self.save_path}/ppo_quad_{self.num_timesteps}"
            self._update_plot()
            self.model.save(save_path)
        return True

    def _update_plot(self):
        self.ax.clear()

        self.ax.plot(self.steps, self.episode_rewards, label="Episode Reward")
        self.ax.set_xlabel("Global Step")
        self.ax.set_ylabel("Reward")
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
