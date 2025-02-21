import numpy as np
import torch
import torch.optim as optim
from environment import QuadrotorEnv
from network import Actor, Critic
import concurrent.futures
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard

import matplotlib.pyplot as plt

class PPOTrainer:
    def __init__(self, env, gamma=0.99, lambda_=0.95, clip_eps=0.2, 
                 actor_lr=3e-4, critic_lr=3e-4, batch_size=64, epochs=30, 
                 train_steps=100000, std=0.5, model_path="ppo_quadrotor.pth"):

        mass = 0.68 + 0.009 * 4.0    # 约 0.716 kg
        g = 9.8
        lower_bound = mass * g + 3        # ≈ 7.02 N
        upper_bound = 4 * mass * g   # ≈ 21.06 N

        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.lambda_ = lambda_
        self.clip_eps = clip_eps
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_steps = train_steps
        self.std = std
        self.model_path = model_path

        # 初始化 Actor 和 Critic 网络
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.actor = Actor(state_dim, action_dim, lower_bound, upper_bound, angular_scale=1.0).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 在线绘图相关设置：使用 Matplotlib 的交互模式
        plt.ion()  # 打开交互模式
        self.fig, self.ax = plt.subplots()
        self.episode_rewards = []  # 用于保存每个 episode 的总 reward
        self.steps = []  # 记录 global step 数
     

        # 新增：创建一个 SummaryWriter 用于记录训练指标
        #self.writer = SummaryWriter(log_dir="runs/ppo_experiment")

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mu = self.actor(state_tensor)
        dist = torch.distributions.Normal(mu, self.std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1)
        return action.detach().cpu().numpy()[0], log_prob.detach().cpu().numpy()[0]

    def compute_gae(self, rewards, dones, values, next_values):
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        gae = 0
        advs = []
        for delta, done in zip(deltas[::-1], dones[::-1]):
            gae = delta + self.gamma * self.lambda_ * gae * (1 - done)
            advs.insert(0, gae)
        return np.array(advs, dtype=np.float32)

    def ppo_update(self, states, actions, advantages, returns, old_log_probs):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(states, actions, advantages, returns, old_log_probs)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        count = 0

        for _ in range(self.epochs):
            for batch_states, batch_actions, batch_adv, batch_returns, batch_old_log_probs in loader:
                mu = self.actor(batch_states)
                dist = torch.distributions.Normal(mu, self.std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=1)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                actor_loss = -torch.mean(torch.min(surr1, surr2))
                
                values = self.critic(batch_states).squeeze()
                critic_loss = torch.mean((batch_returns - values) ** 2)
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                count += 1
         #返回平均损失用于日志记录
        avg_actor_loss = total_actor_loss / count
        avg_critic_loss = total_critic_loss / count
        return avg_actor_loss, avg_critic_loss


    def train_single(self):
        state = self.env.reset()
        episode_reward = 0
        ep_rewards = []
        buffer_states, buffer_actions, buffer_rewards, buffer_dones, buffer_values, buffer_log_probs = [], [], [], [], [], []
        global_step = 0

        while global_step < self.train_steps:
            for _ in range(self.batch_size):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                value = self.critic(state_tensor).item()
                action, log_prob = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                buffer_states.append(state)
                buffer_actions.append(action)
                buffer_rewards.append(reward)
                buffer_dones.append(float(done))
                buffer_values.append(value)
                buffer_log_probs.append(log_prob)
                
                state = next_state
                episode_reward += reward
                global_step += 1

                if done:
                    state = self.env.reset()
                    ep_rewards.append(episode_reward)
                    print("Episode finished. Total reward:", episode_reward)
                    # 记录每个 episode 的 reward 和 global step
                    self.episode_rewards.append(episode_reward)
                    self.steps.append(global_step)
                    # 在线绘制 reward 曲线
                    self.ax.clear()
                    self.ax.plot(self.steps, self.episode_rewards, label="Episode Reward")
                    self.ax.set_xlabel("Global Step")
                    self.ax.set_ylabel("Reward")
                    self.ax.legend()
                    plt.pause(0.01)

                    episode_reward = 0

            # 计算 GAE 和 Returns
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_value = self.critic(state_tensor).item()
            buffer_values = np.array(buffer_values, dtype=np.float32)
            buffer_rewards = np.array(buffer_rewards, dtype=np.float32)
            buffer_dones = np.array(buffer_dones, dtype=np.float32)
            advantages = self.compute_gae(buffer_rewards, buffer_dones, buffer_values, np.append(buffer_values[1:], next_value))
            returns = advantages + buffer_values

            # PPO 更新
            #self.ppo_update(buffer_states, buffer_actions, advantages, returns, buffer_log_probs)

            avg_actor_loss, avg_critic_loss = self.ppo_update(buffer_states, buffer_actions, advantages, returns, buffer_log_probs)

            print("Global step:", global_step, "Avg actor loss:", avg_actor_loss, "Avg critic loss:", avg_critic_loss)

            buffer_states, buffer_actions, buffer_rewards, buffer_dones, buffer_values, buffer_log_probs = [], [], [], [], [], []

            if global_step % (self.batch_size * 10) == 0:
                avg_reward = np.mean(ep_rewards[-10:]) if len(ep_rewards) >= 10 else np.mean(ep_rewards)
                print("Global step:", global_step, "Average episode reward:", avg_reward)

    def save(self):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, self.model_path)
        print(f"Model saved at {self.model_path}")

    def load(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Model loaded from {self.model_path}")



    def rollout(self, rollout_steps):
        """
        单个 rollout 过程，采样 rollout_steps 步数据。
        每个 rollout 内部创建自己的环境实例，避免线程间干扰。
        """
        # 每个线程自己创建环境实例
        local_env = QuadrotorEnv()
        states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
        state = local_env.reset()
        for _ in range(rollout_steps):
            # 获取当前状态对应的值函数估计
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            value = self.critic(state_tensor).item()
            action, log_prob = self.get_action(state)
            next_state, reward, done, _ = local_env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done))
            values.append(value)
            log_probs.append(log_prob)
            
            state = next_state
            if done:
                state = local_env.reset()
        return states, actions, rewards, dones, values, log_probs


    def train_multi_thread(self):
        state = self.env.reset()
        global_step = 0
        ep_rewards = []
        # 定义并行的 worker 数量
        num_workers = 10
        rollout_steps = self.batch_size // num_workers  # 每个 worker 采样的步数
        
        while global_step < self.train_steps:
            buffer_states, buffer_actions, buffer_rewards, buffer_dones, buffer_values, buffer_log_probs = [], [], [], [], [], []
            # 并行采样
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(self.rollout, rollout_steps) for _ in range(num_workers)]
                for future in concurrent.futures.as_completed(futures):
                    states, actions, rewards, dones, values, log_probs = future.result()
                    buffer_states.extend(states)
                    buffer_actions.extend(actions)
                    buffer_rewards.extend(rewards)
                    buffer_dones.extend(dones)
                    buffer_values.extend(values)
                    buffer_log_probs.extend(log_probs)
                    ep_rewards.append(sum(rewards))
                    global_step += len(states)
            
            # 计算 GAE 和 Returns
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_value = self.critic(state_tensor).item()
            buffer_values = np.array(buffer_values, dtype=np.float32)
            buffer_rewards = np.array(buffer_rewards, dtype=np.float32)
            buffer_dones = np.array(buffer_dones, dtype=np.float32)
            advantages = self.compute_gae(buffer_rewards, buffer_dones, buffer_values, np.append(buffer_values[1:], next_value))
            returns = advantages + buffer_values

            # PPO 更新
            self.ppo_update(buffer_states, buffer_actions, advantages, returns, buffer_log_probs)

            if global_step % (self.batch_size * 10) == 0:
                avg_reward = np.mean(ep_rewards[-10:]) if len(ep_rewards) >= 10 else np.mean(ep_rewards)
                print("Global step:", global_step, "Average episode reward:", avg_reward)
