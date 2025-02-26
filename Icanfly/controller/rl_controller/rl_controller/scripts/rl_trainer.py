import numpy as np
import torch
import torch.optim as optim
from Icanfly.controller.rl_controller.rl_controller.scripts.environment import QuadrotorEnv
from Icanfly.controller.rl_controller.rl_controller.scripts.network import Actor, Critic

class PPOTrainer:
    def __init__(self, env, gamma=0.99, lambda_=0.95, clip_eps=0.2, 
                 actor_lr=3e-4, critic_lr=3e-4, batch_size=64, epochs=10, 
                 train_steps=100000, std=0.1, model_path="ppo_quadrotor.pth"):

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

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mu = self.actor(state_tensor)
        mu = torch.clamp(mu, -1, 1)  # 限制范围，防止 NaN
        dist = torch.distributions.Normal(mu, self.std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1)
        return action.detach().cpu().numpy()[0], log_prob.detach().cpu().numpy()[0]

    def compute_gae(self, rewards, dones, values, next_values):
        """
        使用广义优势估计（GAE）
        """
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

    def train(self):
        state = self.env.reset()
        episode_reward = 0
        ep_rewards = []
        buffer = []
        global_step = 0

        while global_step < self.train_steps:
            for _ in range(self.batch_size):
                action, log_prob = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)

                buffer.append((state, action, reward, log_prob, done))
                state = next_state
                episode_reward += reward
                global_step += 1

                if done:
                    state = self.env.reset()
                    ep_rewards.append(episode_reward)
                    print(f"Episode finished. Total reward: {episode_reward:.2f}")
                    episode_reward = 0

            # 计算 GAE 和 Returns
            states, actions, rewards, log_probs, dones = zip(*buffer)
            buffer.clear()
            states = np.array(states, dtype=np.float32)
            actions = np.array(actions, dtype=np.float32)
            rewards = np.array(rewards, dtype=np.float32)
            dones = np.array(dones, dtype=np.float32)
            values = self.critic(torch.FloatTensor(states).to(self.device)).cpu().detach().numpy().squeeze()
            next_values = np.append(values[1:], self.critic(torch.FloatTensor(state).to(self.device)).cpu().detach().numpy())

            advantages = self.compute_gae(rewards, dones, values, next_values)
            returns = advantages + values

            # 更新 PPO
            self.ppo_update(states, actions, advantages, returns, log_probs)

            if global_step % (self.batch_size * 10) == 0:
                avg_reward = np.mean(ep_rewards[-10:]) if len(ep_rewards) >= 10 else np.mean(ep_rewards)
                print(f"Step {global_step}, Avg Reward: {avg_reward:.2f}")

    def save(self):
        """
        保存模型
        """
        try:
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
            }, self.model_path)
            print(f"✅ Model saved at {self.model_path}")
        except Exception as e:
            print(f"❌ Model saving failed: {e}")

    def load(self):
        """
        加载已训练模型
        """
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            print(f"✅ Model loaded from {self.model_path}")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
