#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import sys
import os
import re
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

def get_latest_checkpoint(checkpoint_dir):
    """
    在 checkpoint_dir 中查找最新的 checkpoint 文件，文件名格式应为 "ppo_quad_{step}.zip"
    返回最新文件的完整路径，若无则返回 None。
    """
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith("ppo_quad_") and f.endswith(".zip"):
            match = re.search(r"ppo_quad_(\d+)\.zip", f)
            if match:
                step = int(match.group(1))
                checkpoint_files.append((step, f))
    if not checkpoint_files:
        return None
    latest = max(checkpoint_files, key=lambda x: x[0])
    return os.path.join(checkpoint_dir, latest[1])


if __name__ == "__main__":

    # wind_speed
    
    # train_flag=False
    
    train_flag=True
    
    env = QuadrotorEnv()  # 直接传入原始环境
    
    trainer = SB3PPOTrainer(
        env=env,
        total_timesteps=1_000_0000,
        batch_size=64,
        n_steps=2048,
        learning_rate=1e-3,
        # ent_coef=0.01,
        model_path="sb3_quadrotor_hover"
    )
    
    # 优先查找最新保存的 checkpoint 文件
    checkpoint_path = get_latest_checkpoint("./sb3_checkpoints/")
    if checkpoint_path is not None:
        print(f"Found latest checkpoint: {checkpoint_path}")
        trainer.load(checkpoint_path)
    elif os.path.exists(trainer.model_path + ".zip"):
        # 若没有 checkpoint 文件，检查最终保存的模型文件是否存在
        print(f"Found final model file: {trainer.model_path + '.zip'}")
        trainer.load(trainer.model_path)
    else:
        print("No saved model found, starting fresh training.")
    
    if train_flag==True:
        trainer.train()
    else:
        # 重置环境，获取初始状态
        obs = env.reset()
        
        rate = rospy.Rate(50)  # 控制频率，建议与环境内控制频率一致（例如 50Hz）
        rospy.loginfo("Entering control loop...")
        
        while not rospy.is_shutdown():
            # 获取动作，使用 deterministic 模式以获得稳定控制
            action, _ = trainer.model.predict(obs, deterministic=True)
            # 环境 step 内部会发布 ROS 控制消息给无人机
            obs, reward, done, info = env.step(action)
            
            if done:
                rospy.loginfo("Episode finished, resetting environment.")
                obs = env.reset()
            
            rate.sleep()