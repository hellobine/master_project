import rospy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from drone_rl.env import UAVEnv

def main():
    # rospy.init_node("train_ppo", anonymous=True)

    env = make_vec_env(UAVEnv, n_envs=4)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_logs/")
    
    model.learn(total_timesteps=100000)
    model.save("ppo_uav")

    print("训练完成，模型已保存！")

if __name__ == "__main__":
    main()
