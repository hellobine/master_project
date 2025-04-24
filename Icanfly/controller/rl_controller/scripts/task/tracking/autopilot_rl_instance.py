# #!/usr/bin/env python3
# import os
# import re
# import rospy
# from stable_baseline3_env import QuadrotorEnv
# from stable_baseline3_train import PPOTrainer
# from stable_baselines3.commons.vec_env import SubprocVecEnv

# import os
# import datetime

# current_date = datetime.datetime.now().strftime("%Y%m%d")
# file_dir = f"/home/hello/catkin_ws_rotors/rl_trajectory_run/result/{current_date}/"
# checkpoints_file_dir = file_dir+"sb3_checkpoints/"

# def get_latest_checkpoint(checkpoint_dir):
#     if not os.path.exists(checkpoint_dir):
#         return None
#     checkpoint_files = []
#     for f in os.listdir(checkpoint_dir):
#         if f.startswith("model_") and f.endswith(".zip"):
#             match = re.search(r"model_(\d+)_steps\.zip", f)
#             if match:
#                 step = int(match.group(1))
#                 checkpoint_files.append((step, f))
#     if not checkpoint_files:
#         return None
#     latest = max(checkpoint_files, key=lambda x: x[0])
#     return os.path.join(checkpoint_dir, latest[1])

# def make_env(rank, base_namespace="hummingbird"):
#     def _init():
#         ns = f"{base_namespace}{rank}"
#         env = QuadrotorEnv(namespace=ns)
#         return env
#     return _init

# if __name__ == "__main__":
#     rospy.init_node('quadrotor_rl_node', anonymous=True)
    
#     num_envs = 1   
#     env_fns = [make_env(i) for i in range(num_envs)]
#     vec_env = SubprocVecEnv(env_fns)


#     trainer = PPOTrainer(
#         env=vec_env,
#         total_timesteps=1_000_000_00,
#         batch_size= 1000*num_envs, #256
#         n_steps=1000 #256
#     )
    
#     checkpoint_path = get_latest_checkpoint(checkpoints_file_dir)
#     if checkpoint_path is not None:
#         print(f"Found latest checkpoint: {checkpoint_path}")
#         trainer.load(checkpoint_path)
#     else:
#         print("No saved model found, starting fresh training.")
    

#     rospy.loginfo("Entering train control loop...")
#     trainer.train()
#     rospy.loginfo("Exiting train control loop...")


# #!/usr/bin/env python3
# """Run‑time launcher: 构建向量化环境，加载最近的 .pt 权重（若有），然后继续训练。"""

# import os
# import re
# import datetime
# import rospy
# from stable_baseline3_env import QuadrotorEnv  # 你的自定义 env
# from stable_baseline3_train import PPOTrainer  # 上一步 Canvas 里的训练封装
# from stable_baselines3.commons.vec_env import SubprocVecEnv

# # ---------------------------------------------------------------------------
# # ⬇️ 路径准备（与训练脚本保持一致的日期目录）
# # ---------------------------------------------------------------------------
# current_date = datetime.datetime.now().strftime("%Y%m%d")
# file_dir = f"/home/hello/catkin_ws_rotors/rl_trajectory_run/result/task/tracking/result/{current_date}"
# checkpoints_file_dir = os.path.join(file_dir, "sb3_checkpoints")

# # ---------------------------------------------------------------------------
# # ⬇️ 找到最新的 .pt 权重文件
# # ---------------------------------------------------------------------------

# def get_latest_weights(checkpoint_dir: str):
#     """返回 {checkpoint_dir}/model_<steps>_steps.pt 中 steps 最大的文件完整路径。"""
#     print(f"Looking for latest weights in {checkpoint_dir}…")
#     if not os.path.exists(checkpoint_dir):
#         return None

#     ckpts = []
#     for fname in os.listdir(checkpoint_dir):
#         if fname.startswith("model_") and fname.endswith(".pt"):
#             m = re.search(r"model_(\d+)_steps\.pt", fname)
#             if m:
#                 ckpts.append((int(m.group(1)), fname))

#     if not ckpts:
#         return None

#     _, latest_fname = max(ckpts, key=lambda x: x[0])
#     return os.path.join(checkpoint_dir, latest_fname)

# # ---------------------------------------------------------------------------
# # ⬇️ 构造向量化环境
# # ---------------------------------------------------------------------------

# def make_env(rank: int, base_namespace: str = "hummingbird"):
#     def _init():
#         ns = f"{base_namespace}{rank}"
#         return QuadrotorEnv(namespace=ns)

#     return _init


# if __name__ == "__main__":
#     rospy.set_param('use_sim_time', True)
#     rospy.init_node("quadrotor_rl_node", anonymous=True)

#     num_envs = 5  # 改成 >1 可并行
#     vec_env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

#     # ----------------- 创建 Trainer（确保超参数与训练时相同） -----------------
#     trainer = PPOTrainer(
#         env=vec_env,
#         total_timesteps=100_000_000_0,  # 后续追加 timesteps
#         batch_size= num_envs*256,  # 256
#         n_steps=256
#         # device="cuda:0",  # 或 "cpu"
#     )

#     # ----------------- 尝试加载最新权重 -----------------
#     weights_path = get_latest_weights(checkpoints_file_dir)
#     if weights_path is not None:
#         print(f"Found latest weights: {weights_path}")
#         trainer.load_weights(weights_path)
#     else:
#         print("No saved weights found, starting fresh training.")

#     # ----------------- 开始训练 -----------------
#     rospy.loginfo("Entering train control loop…")
#     trainer.train()
#     rospy.loginfo("Exiting train control loop…")



#!/usr/bin/env python3
import os
import re
import rospy
from stable_baseline3_env import QuadrotorEnv
from stable_baseline3_train import PPOTrainer
from stable_baselines3.commons.vec_env import SubprocVecEnv

import os
import datetime

current_date = datetime.datetime.now().strftime("%Y%m%d")
task_name = "tracking_standard_ppo"
file_dir = f"/home/hello/catkin_ws_rotors/rl_trajectory_run/result/task/hovering/{task_name}/result/{current_date}"
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
        batch_size= 516*num_envs,#256
        n_steps=516 #256
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
 
        env = QuadrotorEnv(namespace="hummingbird3")
        obs, _ = env.reset()
        rate = rospy.Rate(100)
        rospy.loginfo("Entering test control loop...")
    
        while not rospy.is_shutdown():
            action, _ = trainer.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            # print("reward: ", reward)
            # if truncated:
            #     rospy.loginfo("Episode finished, resetting environment.")
            #     obs,_ = env.reset()
            rate.sleep()