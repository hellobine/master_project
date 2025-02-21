#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import QuadrotorEnv
from rl_trainer import PPOTrainer

def main():
    rospy.init_node('quadrotor_rl_node', anonymous=True)  # 启动 ROS 训练节点
    rospy.loginfo("🚀 Quadrotor RL Node Started...")

    # 创建 Gym 强化学习环境
    env = QuadrotorEnv()

    # 初始化 PPO 训练器
    trainer = PPOTrainer(env, train_steps=2000000, model_path="ppo_quadrotor.pth")

    # 训练 RL 模型
    try:
        trainer.train_single()
        trainer.save()  # 训练结束后保存模型
        
    except rospy.ROSInterruptException:
        rospy.logwarn("训练中断，保存模型...")
        trainer.save()

    rospy.spin()  # 保持 ROS 运行

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
