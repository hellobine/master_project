import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO, TD3, DDPG, A2C, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class GymnasiumWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs


class SB3Trainer:
    def __init__(self, env, algorithm='PPO', total_timesteps=1e6, buffer_size=1e6, batch_size=256*2,
                 gamma=0.95, tau=0.005, learning_rate=2e-4, n_steps=256, gae_lambda=0.95,
                 clip_range=0.2, ent_coef=0.05, model_path="./run/sb3_quadrotor"):

        if not isinstance(env, VecEnv):
            self.env = DummyVecEnv([lambda: GymnasiumWrapper(env)])
        else:
            self.env = env

        self.total_timesteps = int(total_timesteps)
        self.model_path = model_path
        self.algorithm = algorithm

        algo_dict = {
            'PPO': PPO(policy="MlpPolicy", env=self.env, n_steps=n_steps, gamma=gamma,
                       gae_lambda=gae_lambda, clip_range=clip_range, ent_coef=ent_coef,
                       learning_rate=learning_rate, verbose=1, policy_kwargs={"net_arch":
                        dict(pi=[128, 128], vf=[128, 128]), "optimizer_kwargs": {"weight_decay": 1e-4 }},
                       tensorboard_log="./rl_trajectory_run/sb3_tensorboard/"),

            'TD3': TD3(policy="MlpPolicy", env=self.env, buffer_size=int(buffer_size),
                       batch_size=batch_size, gamma=gamma, tau=tau, learning_rate=learning_rate,
                       policy_kwargs={"net_arch": dict(pi=[128, 128], qf=[128, 128])},
                       verbose=1, tensorboard_log="./rl_trajectory_run/sb3_tensorboard/"),

            'DDPG': DDPG(policy="MlpPolicy", env=self.env, buffer_size=int(buffer_size),
                         batch_size=batch_size, gamma=gamma, tau=tau, learning_rate=learning_rate,
                         policy_kwargs={"net_arch": dict(pi=[128, 128], qf=[128, 128])},
                         verbose=1, tensorboard_log="./rl_trajectory_run/sb3_tensorboard/"),

            'A2C': A2C(policy="MlpPolicy", env=self.env, n_steps=n_steps, gamma=gamma,
                       learning_rate=learning_rate, policy_kwargs={"net_arch": [128, 128]},
                       verbose=1, tensorboard_log="./rl_trajectory_run/sb3_tensorboard/"),

            'SAC': SAC(policy="MlpPolicy", env=self.env, buffer_size=int(buffer_size), batch_size=batch_size,
                         gamma=gamma, tau=tau, ent_coef=ent_coef, learning_rate=learning_rate,
                         policy_kwargs={"net_arch": dict(pi=[128, 128], qf=[128, 128]),}, verbose=1, 
                         tensorboard_log="./rl_trajectory_run/sb3_tensorboard/"
                         )
        }

        self.model = algo_dict[self.algorithm]

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
            save_freq=10000,
            save_path="./rl_trajectory_run/sb3_checkpoints/",
            model=self.model,
            writer=self.writer,
            ax=self.ax,
            fig=self.fig,
            episode_rewards=self.episode_rewards,
            steps=self.steps
        )

    def train(self):
        print(f"Starting {self.algorithm} training...")
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
        print(f"Final {self.algorithm} model saved at {self.model_path}")

    def load(self, path):
        algo_class = {'PPO': PPO, 'TD3': TD3, 'DDPG': DDPG, 'A2C': A2C}
        self.model = algo_class[self.algorithm].load(path, env=self.env)
        print(f"{self.algorithm} Model loaded from {path}")


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
                    self.episode_rewards.append(info["episode"]["r"])
                    self.steps.append(self.num_timesteps)
                    print(f"Episode ended at step {self.num_timesteps}, reward: {info['episode']['r']}")

        if self.num_timesteps % self.save_freq == 0:
            self._update_plot()
            self.model.save(f"{self.save_path}/{self.model.__class__.__name__}_{self.num_timesteps}")
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
