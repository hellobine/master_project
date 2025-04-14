#!/usr/bin/env python3
import os
import re
import rospy
from stable_baseline3_env import QuadrotorEnv
from stable_baseline3_train import PPOTrainer
from commons.vec_env import SubprocVecEnv

import os
import datetime

current_date = datetime.datetime.now().strftime("%Y%m%d")
file_dir = f"/home/hello/catkin_ws_rotors/rl_trajectory_run/result/{current_date}/"
checkpoints_file_dir = file_dir+"/sb3_checkpoints/"

def get_latest_checkpoint(checkpoint_dir):
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
    num_envs = 10   
    env_fns = [make_env(i) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)
    
    trainer = PPOTrainer(
        env=vec_env,
        total_timesteps=1_000_000_00,
        batch_size= 128*num_envs,#256
        n_steps=128 #256
    )
    
    checkpoint_path = get_latest_checkpoint(checkpoints_file_dir)
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
 
        env = QuadrotorEnv(namespace="hummingbird0")
        obs, _ = env.reset()
        rate = rospy.Rate(100)
        rospy.loginfo("Entering test control loop...")
    
        while not rospy.is_shutdown():
            action, _ = trainer.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            print("reward: ", reward)
            # if truncated:
            #     rospy.loginfo("Episode finished, resetting environment.")
            #     obs,_ = env.reset()
            rate.sleep()