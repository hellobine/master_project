#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import gym
# from environment import QuadrotorEnv
# from rl_trainer import PPOTrainer
# from stable_baseline3_train import SB3PPOTrainer
# from stable_baseline3_env import QuadrotorEnv
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv


# def main():
#     rospy.init_node('quadrotor_rl_node', anonymous=True)  # 启动 ROS 训练节点
#     rospy.loginfo("🚀 Quadrotor RL Node Started...")

#     # 创建 Gym 强化学习环境
#     # env = QuadrotorEnv()

#     # 初始化 PPO 训练器
#     # trainer = PPOTrainer(env, train_steps=2000000, model_path="ppo_quadrotor.pth")
    
#     env = QuadrotorEnv()
#     # env = DummyVecEnv([lambda: env])


#     trainer = SB3PPOTrainer(
#         env=env,
#         total_timesteps=1_000_000,
#         batch_size=64,
#         n_steps=2048,        # 增加采样步数
#         gamma=0.99,
#         gae_lambda=0.95,
#         clip_range=0.2,
#         ent_coef=0.01,       # 鼓励探索
#         learning_rate=3e-4,  # 降低学习率
#         model_path="sb3_quadrotor_hover"
#     )
#     # 训练 RL 模型
#     try:
#         trainer.train()
#         # trainer.save()  # 训练结束后保存模型
        
#     except rospy.ROSInterruptException:
#         rospy.logwarn("训练中断，保存模型...")
#         # trainer.save()

#     rospy.spin()  # 保持 ROS 运行

# if __name__ == "__main__":
#     try:
#         main()
#     except rospy.ROSInterruptException:
#         pass


#!/usr/bin/env python3
from stable_baseline3_env import QuadrotorEnv
from stable_baseline3_train import SB3PPOTrainer
from stable_baselines3.common.vec_env import DummyVecEnv

if __name__ == "__main__":
    env = QuadrotorEnv()  # 直接传入原始环境
    
    trainer = SB3PPOTrainer(
        env=env,
        total_timesteps=1_000_0000,
        batch_size=64,
        n_steps=2048,
        learning_rate=3e-4,
        ent_coef=0.01,
        model_path="sb3_quadrotor_hover"
    )
    
    trainer.train()