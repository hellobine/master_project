#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import ControlCommand
from gazebo_msgs.msg import ModelState
from rotors_comm.msg import WindSpeed
import threading
from std_msgs.msg import Bool
import random

class QuadrotorEnv(gym.Env):
    def __init__(self, namespace="drone"):
        super(QuadrotorEnv, self).__init__()
        self.state_lock = threading.Lock()  # 立即初始化
    
        self.namespace = namespace  # ROS 命名空间，用于区分不同无人机实例

        # 物理参数
        self.mass = 0.68  # kg
        self.gravity = 9.81
        self.min_thrust = 9.2
        self.max_thrust = 40
        self.max_angular_rate = 3.0

        # Gym 空间定义（这里只使用位置信息）
        self.state_dim = 13  # [px, py, pz, qx, qy, qz, qw, vx, vy, vz, ang_x, ang_y, ang_z]
        self.action_dim = 4  # [推力, ωx, ωy, ωz]
        
        
        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf, 
        #     shape=(self.state_dim,), dtype=np.float32
        # )

        self.observation_space = spaces.Box(
        low=np.array([0, 0, 0, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5], dtype=np.float32),
        high=np.array([5, 5, 5, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5], dtype=np.float32),
        dtype=np.float32
        )

        # self.action_space = spaces.Box(
        #     low=np.array([self.min_thrust, -self.max_angular_rate, -self.max_angular_rate, -self.max_angular_rate], dtype=np.float32),
        #     high=np.array([self.max_thrust, self.max_angular_rate, self.max_angular_rate, self.max_angular_rate], dtype=np.float32),
        #     dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-1]*self.action_dim, dtype=np.float32),
            high=np.array([1]*self.action_dim, dtype=np.float32),
            dtype=np.float32
        )

        # ROS 初始化（每个实例使用独特的节点名）
        if not rospy.core.is_initialized():
            rospy.init_node(f'{self.namespace}_quadrotor_sb3_env', anonymous=True)
        
        # 订阅和发布（话题前缀使用命名空间）
        self.odom_sub = rospy.Subscriber(f'/{self.namespace}/ground_truth/odometry', Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher(f'/{self.namespace}/control_command', ControlCommand, queue_size=1)
        self.arm_pub = rospy.Publisher(f'/{self.namespace}/bridge/arm', Bool, queue_size=1)
        
        self.windspeed_pub = rospy.Publisher(f'/{self.namespace}/wind_speed', WindSpeed, queue_size=1)
        # self.state_lock = threading.Lock()

        # 控制频率
        self.control_hz = 50.0
        self.rate = rospy.Rate(self.control_hz)
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        self.desired_state = np.array([0, 0, 3, 0, 0, 0, 1, 0, 0, 0,], dtype=np.float32)

        # 安全和步数设置
        self.max_position_error = 5.0  # 米
        self.max_episode_steps = 256
        self.step_count = 0
        self.episode_reward = 0

        
        self.prev_action_ = None
        self.prev_position = None  # 保存上一时刻位置信息


        self.angular_x = 0
        self.angular_y = 0
        self.angular_z = 0
        rospy.sleep(1.0)
    
        if not rospy.is_shutdown():
            msg = Bool(data=True)
            self.arm_pub.publish(msg)
            rospy.loginfo("Published arm message: true")

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




            # wind_speed_msg = WindSpeed()
            # wind_speed_msg.header.stamp = rospy.Time.now()
            # wind_speed_msg.header.frame_id = "world"
            # wind_speed_msg.velocity = Vector3(0.0, 0.0, -100.0)
            # 可根据需要发布风速消息，此处暂未调用：
            # self.windspeed_pub.publish(wind_speed_msg)

    def step(self, action):
        self.step_count += 1
        self._publish_action(action)
        self.rate.sleep()
        
        with self.state_lock:
            obs = self.current_state.copy()

        reward = self._compute_reward(obs, action)
        self.episode_reward += reward

        # 判断是否结束
        if self.step_count >= self.max_episode_steps or self._check_done(obs):
            # 如果达到最大步数，则标记为截断，否则标记为终止
            if self.step_count >= self.max_episode_steps:
                terminated = False  # 非任务失败
                truncated = True   # 由于达到最大步数
            else:
                terminated = True  # 任务失败，如超出范围
                truncated = False

            # 在所有结束条件下，都返回 episode 信息
            info = {"reward": reward, "episode": {"r": self.episode_reward, "l": self.step_count}}
            # 重置计数器和累计奖励
            self.episode_reward = 0
            self.step_count = 0

        else:
            terminated = False
            truncated = False
            info = {"reward": reward}

        # 重置计数器和累计奖励
        if terminated or truncated:
            self.episode_reward = 0
            self.step_count = 0

        error_obs = obs.copy()
        error_obs[0] = abs(obs[0]-self.desired_state[0])
        error_obs[1] = abs(obs[1]-self.desired_state[1])
        error_obs[2] = abs(obs[2]-self.desired_state[2])
      
        return error_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_count = 0
        self.episode_reward = 0
        self.prev_action_ = None
        self._reset_drone_pose()
        rospy.sleep(0.001)  # 等待复位完成
        with self.state_lock:

            obs = self.current_state.copy()
            error_obs = obs.copy()
            error_obs[0] = abs(obs[0]-self.desired_state[0])
            error_obs[1] = abs(obs[1]-self.desired_state[1])
            error_obs[2] = abs(obs[2]-self.desired_state[2])
        return obs, {}
    

    def _reset_drone_pose(self):
        init_position = np.array([
            np.random.uniform(-1.0, 1.0),  # x 范围 [-1, 1] m
            np.random.uniform(-1.0, 1.0),  # y 范围 [-1, 1] m
            np.random.uniform(0, 4)    # z 范围 [0.5, 1.5] m，避免过高或过低
        ])

        init_orientation = np.random.uniform(-0.4, 0.4, size=3)  # 近似水平的随机扰动
        init_velocity = np.random.uniform(-1, 1, size=3)  # 低速随机初始化 #init_velocity = np.random.uniform(-3, 3, size=3)  # 低速随机初始化
        init_angular_velocity = np.random.uniform(-0.1, 0.1, size=3)  # 低速角速度初始化

        # 发布到 Gazebo 进行物理仿真复位
        state = ModelState()
        state.model_name = self.namespace  # 确保命名空间匹配 Gazebo 的无人机模型
        state.pose.position.x = init_position[0]
        state.pose.position.y = init_position[1]
        state.pose.position.z = 0.2
        state.pose.orientation.x = 0
        state.pose.orientation.y = 0
        state.pose.orientation.z = 0
        state.pose.orientation.w = 1.0  # 保持单位四元数
        
        # state.model_name = self.namespace  # 确保命名空间匹配 Gazebo 的无人机模型
        # state.pose.position.x = 0
        # state.pose.position.y = 0
        # state.pose.position.z = 3
        # state.pose.orientation.x = 0
        # state.pose.orientation.y = 0
        # state.pose.orientation.z = 0
        # state.pose.orientation.w = 1.0  # 保持单位四元数


        state.twist.linear.x=init_velocity[0]
        state.twist.linear.y=init_velocity[1]
        state.twist.linear.z=init_velocity[2]

        state.twist.angular.x=init_angular_velocity[0]
        state.twist.angular.y=init_angular_velocity[1]
        state.twist.angular.z=init_angular_velocity[2]

        pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        pub.publish(state)


    def _publish_action(self, action):
        cmd = ControlCommand()
        cmd.armed = True
        cmd.control_mode = ControlCommand.BODY_RATES
        # cmd.collective_thrust = float(np.clip(action[0], self.min_thrust, self.max_thrust))
        # cmd.collective_thrust = action[0]
           # 推力归一化映射到 [min_thrust, max_thrust]
        thrust = ((action[0] + 1) / 2) * (self.max_thrust - self.min_thrust) + self.min_thrust
        cmd.collective_thrust = thrust

        # 角速度归一化映射到 [-max_angular_rate, max_angular_rate]
        bodyrates = np.clip(action[1:], -1, 1) * self.max_angular_rate
        # 分别赋值给 x, y, z
        cmd.bodyrates.x = bodyrates[0]
        cmd.bodyrates.y = bodyrates[1]
        cmd.bodyrates.z = bodyrates[2]

        # cmd.bodyrates.x = np.clip(action[1], -self.max_angular_rate, self.max_angular_rate)
        # cmd.bodyrates.y = np.clip(action[2], -self.max_angular_rate, self.max_angular_rate)
        # cmd.bodyrates.z = np.clip(action[3], -self.max_angular_rate, self.max_angular_rate)

        self.cmd_pub.publish(cmd)


    def _quaternion_to_euler(self, x, y, z, w):
        """将四元数转换为欧拉角 (roll, pitch, yaw)"""
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        
        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.arcsin(t2)
        
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        return roll, pitch, yaw

    def _compute_reward(self, obs, curr_action):
        # 目标位置（参考轨迹点），例如 [0, 0, 3]
        target_pos = self.desired_state[0:3]

        # 当前位置信息
        curr_pos = obs[0:3]
        
        # 如果没有记录上一时刻位置，则将其初始化为当前位置信息
        if not hasattr(self, 'prev_position') or self.prev_position is None:
            self.prev_position = curr_pos.copy()

        # 进度奖励：上一时刻与目标的距离减去当前时刻与目标的距离
        prev_dist = np.linalg.norm(target_pos - self.prev_position)
        curr_dist = np.linalg.norm(target_pos - curr_pos)
        r_progress = prev_dist - curr_dist

        # 角速度惩罚：取 obs[10:13]（假设这部分为角速度）
        angular_vel = obs[10:13]
        b = 0.01  # 角速度惩罚系数
        r_ang = - b * np.linalg.norm(angular_vel)

        # 动作平滑奖励：如果有上一时刻的动作，则计算动作变化量惩罚（取负值）
        if self.prev_action_ is not None:
            action_diff = np.linalg.norm(curr_action - self.prev_action_)
            r_smooth = - (action_diff ** 2)
        else:
            r_smooth = 0.0

        # 平滑奖励系数
        lambda_val = 0.4

        # 最终奖励为进度奖励 + 角速度惩罚 + 平滑奖励
        reward = r_progress + r_ang + lambda_val * r_smooth

        # 更新上一时刻动作和位置信息
        self.prev_action_ = curr_action.copy()
        self.prev_position = curr_pos.copy()

        return reward


    def _check_done(self, obs):
        pos_error = np.linalg.norm(obs[0:3] - self.desired_state[0:3])
        # roll, pitch, _ = self._quaternion_to_euler(obs[3], obs[4], obs[5], obs[6])

        return pos_error > self.max_position_error

    def render(self, mode='human'):
        pass  # Gazebo 自带可视化

    def close(self):
        rospy.signal_shutdown("Environment closed")

    # def seed(self, seed=None):
    #     self._seed = seed
    #     np.random.seed(seed)
    #     random.seed(seed)
    #     return [seed]