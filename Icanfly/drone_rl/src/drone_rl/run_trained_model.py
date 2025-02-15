import rospy
from stable_baselines3 import PPO
from drone_rl.env import UAVEnv

def main():
    rospy.init_node("test_ppo", anonymous=True)
    
    env = UAVEnv()
    model = PPO.load("ppo_uav")

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        if done:
            obs = env.reset()

if __name__ == "__main__":
    main()
