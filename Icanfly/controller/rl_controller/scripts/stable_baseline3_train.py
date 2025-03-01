
import numpy as np
import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class SB3PPOTrainer:
    def __init__(self, env, total_timesteps=1e6, batch_size=1024, n_steps=1024,
                 gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.05,
                 learning_rate=1e-4, model_path="sb3_ppo_quadrotor"):
        
        self.mass = 0.716
        self.min_thrust = 0
        self.max_thrust = 4 * self.mass * 9.81
        # self.angular_scale = 3.0
        
        if not isinstance(env, DummyVecEnv):
            self.env = DummyVecEnv([lambda: env])
        else:
            self.env = env
        
        self.total_timesteps = int(total_timesteps)
        self.model_path = model_path
        
        from stable_baseline3_policy import CustomActorCriticPolicy
        self.model = PPO(
            policy=CustomActorCriticPolicy,
            env=self.env,
            policy_kwargs={
                "min_thrust": self.min_thrust,
                "max_thrust": self.max_thrust,
                # "angular_scale": self.angular_scale,
                "net_arch": []
            },
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=1,
            tensorboard_log="./sb3_tensorboard/"
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
        self.writer = SummaryWriter(log_dir="./sb3_tensorboard/")
        
        self.callback = SB3CustomCallback(
            save_freq=10000,
            save_path="./sb3_checkpoints/",
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
               episode_reward = 0
               for info in self.locals["infos"]:
                    if "reward" in info:
                        self.writer.add_scalar("Reward/Step", info["reward"], self.num_timesteps)

                    if "episode" in info:
                        # 直接用 info["episode"]["r"] 获取完整 episode 奖励
                        self.episode_rewards.append(info["episode"]["r"])
                        self.steps.append(self.num_timesteps)
                        print(f"Episode ended at step {self.num_timesteps}, reward: {info['episode']['r']}")

                        self._update_plot()
  
          if self.num_timesteps % self.save_freq == 0:
               save_path = f"{self.save_path}/ppo_quad_{self.num_timesteps}"
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