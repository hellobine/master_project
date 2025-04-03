#!/usr/bin/env python3
import os
import re
import rospy
from stable_baseline3_env import QuadrotorEnv
from stable_baseline3_train import SB3PPOTrainer
from stable_baselines3.common.vec_env import SubprocVecEnv

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

def make_env(rank, base_namespace="hummingbird"):
    def _init():
        ns = f"{base_namespace}{rank}"
        env = QuadrotorEnv(namespace=ns)
        return env
    return _init

if __name__ == "__main__":
    rospy.init_node('quadrotor_rl_node', anonymous=True)
    
    train_flag = True
    num_envs = 10  # 根据需求调整并行环境数量
    env_fns = [make_env(i) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)
    
    trainer = SB3PPOTrainer(
        env=vec_env,
        total_timesteps=1_000_000_000,
        batch_size= 64*num_envs,#256
        n_steps=64, #256
        model_path="./rl_trajectory_run/sb3_quadrotor_hover"
    )
    
    checkpoint_path = get_latest_checkpoint("./rl_trajectory_run/sb3_checkpoints/")
    if checkpoint_path is not None:
        print(f"Found latest checkpoint: {checkpoint_path}")
        trainer.load(checkpoint_path)
    # elif os.path.exists(trainer.model_path + ".zip"):
    #     print(f"Found final model file: {trainer.model_path + '.zip'}")
    #     trainer.load(trainer.model_path)
    else:
        print("No saved model found, starting fresh training.")
    
    if train_flag:
        rospy.loginfo("Entering train control loop...")
        trainer.train()
    else:
        # 如果需要运行控制模式，则使用其中一个实例（例如 drone_0）
        env = QuadrotorEnv(namespace="hummingbird0")
        obs, _ = env.reset()
        rate = rospy.Rate(50)
        rospy.loginfo("Entering test control loop...")
        obs,_ = env.reset()
        while not rospy.is_shutdown():
            
            action, _ = trainer.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if truncated:
                rospy.loginfo("Episode finished, resetting environment.")
                obs,_ = env.reset()
            rate.sleep()
