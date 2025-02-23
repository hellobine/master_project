

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gym
import numpy as np
from gym import spaces
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import ControlCommand
from gazebo_msgs.msg import ModelState
import threading

class QuadrotorEnv(gym.Env):
    def __init__(self):
        super(QuadrotorEnv, self).__init__()
        
        # 物理参数
        self.mass = 0.716  # kg (无人机 + 电池)
        self.gravity = 9.81
        self.min_thrust = 0  # 7.02 N
        self.max_thrust = 4 * self.mass * self.gravity  # 28.1 N
        self.max_angular_rate = 3.0  # rad/s

        # Gym 空间定义
        self.state_dim = 13  # [px, py, pz, qx, qy, qz, qw, vx, vy, vz]
        self.action_dim = 4  # [推力, ωx, ωy, ωz]
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([self.min_thrust, -self.max_angular_rate, -self.max_angular_rate, -self.max_angular_rate], dtype=np.float32),
            high=np.array([self.max_thrust, self.max_angular_rate, self.max_angular_rate, self.max_angular_rate], dtype=np.float32),
            dtype=np.float32
        )

        # ROS 初始化
        if not rospy.get_node_uri():
            rospy.init_node('quadrotor_sb3_env', anonymous=True)
        
        # 订阅/发布器
        self.odom_sub = rospy.Subscriber('/hummingbird/ground_truth/odometry', Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher('/hummingbird/control_command', ControlCommand, queue_size=1)
        self.state_lock = threading.Lock()
        
        # 控制频率
        self.control_hz = 100.0
        self.rate = rospy.Rate(self.control_hz)
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        self.desired_state = np.array([0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        
        # 安全参数
        self.max_position_error = 5.0  # 米
        self.max_episode_steps = 1024  # 10秒 @ 20Hz
        self.step_count = 0

        self.episode_reward=0

        self.prev_action_ = None

    def odom_callback(self, msg):
        with self.state_lock:
            self.current_state = np.array([
                msg.pose.pose.position.x, 
                msg.pose.pose.position.y, 
                msg.pose.pose.position.z,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z
            ], dtype=np.float32)

    # def step(self, action):
    #     self.step_count += 1
    #     self._publish_action(action)
    #     self.rate.sleep()
        
    #     with self.state_lock:
    #         obs = self.current_state.copy()
        
    #     reward = self._compute_reward(obs)
    #     done = self._check_done(obs) or self.step_count >= self.max_episode_steps
    #     info = {"reward": reward}  # 每步记录奖励
    #     if done:
    #         info["episode"] = {"r": reward, "l": self.step_count}  # 回合结束时记录奖励和长度
        
    #     return obs, reward, done, info

    def step(self, action):
        self.step_count += 1

        self._publish_action(action)
        self.rate.sleep()
        
        with self.state_lock:
            obs = self.current_state.copy()
        
        reward = self._compute_reward(obs, action)

        self.prev_action_ = action
        
        self.episode_reward += reward  # 逐步累积 reward

        done = self._check_done(obs) or self.step_count >= self.max_episode_steps

        info = {"reward": reward}  # 每步记录奖励

        if done:
            info["episode"] = {"r": self.episode_reward, "l": self.step_count}  # 存整个 episode 的 reward
            self.episode_reward = 0  # 记得重置

        return obs, reward, done, info

    def reset(self):
        self.step_count = 0
        # 重置无人机姿态
        self._reset_drone_pose()
        rospy.sleep(1.5)  # 等待复位
        
        # 等待有效状态
        while True:
            with self.state_lock:
                if np.linalg.norm(self.current_state[7:10]) < 0.01:
                    break
            rospy.sleep(0.1)
        
        return self.current_state.copy()

    def _reset_drone_pose(self):
        state = ModelState()
        state.model_name = "hummingbird"
        # state.pose.position.x = np.random.uniform(-1, 1)
        # state.pose.position.y = np.random.uniform(-1, 1)
        state.pose.position.x = 0
        state.pose.position.y = 0
        state.pose.position.z = 0.1
        # state.pose.orientation.w = 1.0
        pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        pub.publish(state)

    def _publish_action(self, action):
        cmd = ControlCommand()
        cmd.armed = True
        cmd.control_mode = ControlCommand.BODY_RATES
        cmd.collective_thrust = np.clip(action[0], self.min_thrust, self.max_thrust) 
        cmd.bodyrates.x = np.clip(action[1], -self.max_angular_rate, self.max_angular_rate)
        cmd.bodyrates.y = np.clip(action[2], -self.max_angular_rate, self.max_angular_rate)
        cmd.bodyrates.z = np.clip(action[3], -self.max_angular_rate, self.max_angular_rate)
        # cmd.bodyrates.x = 0
        # cmd.bodyrates.y = 0
        # cmd.bodyrates.z = 0
        self.cmd_pub.publish(cmd)

    def _compute_reward(self, obs, curr_action):
        # 假设观测包含：[位置(3), 姿态四元数(4), 线速度(3), 角速度(3), 前一个动作(4)]
        # 分解观测数据
        pos = obs[0:3]          # 当前位置
        quat = obs[3:7]         # 当前姿态（四元数）
        lin_vel = obs[7:10]     # 线速度
        ang_vel = obs[10:13]    # 角速度
        # prev_act = prev_action  # 前一个动作
        
        # 目标状态（假设self.desired_state包含目标高度和姿态）
        z_d = self.desired_state[2]              # 目标高度
        target_quat = self.desired_state[3:7]    # 目标姿态四元数
        
        # 1. 高度奖励 (beta1 = -2e-3)
        height_error = pos[2] - z_d
        r_height = -1e-2 * (height_error ** 2)
    
        
        # 2. 姿态奖励：计算姿态矩阵的误差（假设用四元数差代替）
        att_error = 2 * np.arccos(np.clip(np.abs(quat.dot(target_quat)), -1.0, 1.0))
        r_attitude = -1e-3 * (att_error **2)
        
        # 3. 线速度惩罚：使用平方误差，系数设为 -1e-3
        vel_penalty = -1e-3 * (np.linalg.norm(lin_vel) ** 2)
        
        # 4. 角速度惩罚：使用平方误差，系数设为 -1e-3
        ang_vel_penalty = -1e-3 * (np.linalg.norm(ang_vel) ** 2)
        
        # 5. 动作平滑惩罚：使用平方误差，系数设为 -1e-3
        if self.prev_action_ is not None:
            action_diff = np.linalg.norm(curr_action - self.prev_action_)
            r_act_smooth = -1e-3 * (action_diff ** 2)
        else:
            r_act_smooth = 0.0
    
        # 6. 成功悬停奖励：当达到所有条件时，给予持续的正奖励
        success_reward = 0.0
        if (abs(height_error) < 0.1 and
            np.linalg.norm(lin_vel) < 0.5 and
            np.linalg.norm(ang_vel) < 0.1):
            # 可以给一个较小的正奖励，例如每步 + 1
            success_reward = 1
            
        
        # 总奖励 = 各项加权和 + 成功奖励
        total_reward = (
            r_height + 
            r_attitude + 
            vel_penalty + 
            ang_vel_penalty + 
            r_act_smooth + 
            success_reward
        )
        
        return total_reward

    def _check_done(self, obs):
        pos_error = np.linalg.norm(obs[0:3] - self.desired_state[0:3])

        return pos_error > self.max_position_error  

    def render(self, mode='human'):
        pass  # Gazebo自带可视化

    def close(self):
        rospy.signal_shutdown("Environment closed")