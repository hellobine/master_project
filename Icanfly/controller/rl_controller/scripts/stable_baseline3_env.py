#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gym
import numpy as np
from gym import spaces
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import ControlCommand
from gazebo_msgs.msg import ModelState
from rotors_comm.msg import WindSpeed
import threading
from std_msgs.msg import Bool

class QuadrotorEnv(gym.Env):
    def __init__(self, namespace="drone"):
        super(QuadrotorEnv, self).__init__()
        self.namespace = namespace  # ROS 命名空间，用于区分不同无人机实例

        # 物理参数
        self.mass = 0.68  # kg
        self.gravity = 9.81
        self.min_thrust = 8
        self.max_thrust = 40
        self.max_angular_rate = 2.0

        # Gym 空间定义（这里只使用位置信息）
        self.state_dim = 10  # [px, py, pz, qx, qy, qz, qw, vx, vy, vz, ang_x, ang_y, ang_z]
        self.action_dim = 4  # [推力, ωx, ωy, ωz]
        
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([self.min_thrust, -self.max_angular_rate, -self.max_angular_rate, -self.max_angular_rate], dtype=np.float32),
            high=np.array([self.max_thrust, self.max_angular_rate, self.max_angular_rate, self.max_angular_rate], dtype=np.float32),
            dtype=np.float32)

        # ROS 初始化（每个实例使用独特的节点名）
        if not rospy.core.is_initialized():
            rospy.init_node(f'{self.namespace}_quadrotor_sb3_env', anonymous=True)
        
        # 订阅和发布（话题前缀使用命名空间）
        self.odom_sub = rospy.Subscriber(f'/{self.namespace}/ground_truth/odometry', Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher(f'/{self.namespace}/control_command', ControlCommand, queue_size=1)
        self.arm_pub = rospy.Publisher(f'/{self.namespace}/bridge/arm', Bool, queue_size=1)
        
        self.windspeed_pub = rospy.Publisher(f'/{self.namespace}/wind_speed', WindSpeed, queue_size=1)
        self.state_lock = threading.Lock()

        # 控制频率
        self.control_hz = 50.0
        self.rate = rospy.Rate(self.control_hz)
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        self.desired_state = np.array([0, 0, 3, 0, 0, 0, 1, 0, 0, 0,], dtype=np.float32)

        # 安全和步数设置
        self.max_position_error = 5.0  # 米
        self.max_episode_steps = 1024
        self.step_count = 0
        self.episode_reward = 0
        self.prev_action_ = None

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
                msg.twist.twist.linear.z
            ], dtype=np.float32)



            wind_speed_msg = WindSpeed()
            wind_speed_msg.header.stamp = rospy.Time.now()
            wind_speed_msg.header.frame_id = "world"
            wind_speed_msg.velocity = Vector3(0.0, 0.0, -100.0)
            # 可根据需要发布风速消息，此处暂未调用：
            # self.windspeed_pub.publish(wind_speed_msg)

    def step(self, action):
        self.step_count += 1
        self._publish_action(action)
        self.rate.sleep()
        
        with self.state_lock:
            obs = self.current_state.copy()

        roll, pitch, _ = self._quaternion_to_euler(obs[3], obs[4], obs[5], obs[6])
        if abs(roll) > np.pi/2 or abs(pitch) > np.pi/2:
            done = True
            reward = -100.0  # 侧翻时给予较大的惩罚
            info = {"reason": "side flip", "reward": reward}
            return obs, reward, done, info

        
        reward = self._compute_reward(obs, action)
        self.prev_action_ = action
        self.episode_reward += reward

        done = self._check_done(obs) or self.step_count >= self.max_episode_steps
        info = {"reward": reward}
        if done:
            info["episode"] = {"r": self.episode_reward, "l": self.step_count}
            self.episode_reward = 0
        return obs, reward, done, info

    def reset(self):
        self.step_count = 0
        self.episode_reward = 0
        self.prev_action_ = None
        self._reset_drone_pose()
        rospy.sleep(1.0)  # 等待复位完成
        with self.state_lock:
            return self.current_state.copy()

    def _reset_drone_pose(self):
        init_position = np.array([
            np.random.uniform(-1.0, 1.0),  # x 范围 [-1, 1] m
            np.random.uniform(-1.0, 1.0),  # y 范围 [-1, 1] m
            np.random.uniform(0.5, 1.5)    # z 范围 [0.5, 1.5] m，避免过高或过低
        ])

        init_orientation = np.random.uniform(-0.1, 0.1, size=3)  # 近似水平的随机扰动
        init_velocity = np.random.uniform(-0.5, 0.5, size=3)  # 低速随机初始化
        init_angular_velocity = np.random.uniform(-0.1, 0.1, size=3)  # 低速角速度初始化

        # 发布到 Gazebo 进行物理仿真复位
        state = ModelState()
        state.model_name = self.namespace  # 确保命名空间匹配 Gazebo 的无人机模型
        state.pose.position.x = init_position[0]
        state.pose.position.y = init_position[1]
        state.pose.position.z = 0.01
        state.pose.orientation.x = init_orientation[0]
        state.pose.orientation.y = init_orientation[1]
        state.pose.orientation.z = init_orientation[2]
        state.pose.orientation.w = 1.0  # 保持单位四元数

        pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        pub.publish(state)


    def _publish_action(self, action):
        cmd = ControlCommand()
        cmd.armed = True
        cmd.control_mode = ControlCommand.BODY_RATES
        # cmd.collective_thrust = float(np.clip(action[0], self.min_thrust, self.max_thrust))
        cmd.collective_thrust = action[0]
        cmd.bodyrates.x = np.clip(action[1], -self.max_angular_rate, self.max_angular_rate)
        cmd.bodyrates.y = np.clip(action[2], -self.max_angular_rate, self.max_angular_rate)
        cmd.bodyrates.z = np.clip(action[3], -self.max_angular_rate, self.max_angular_rate)

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
        target_pos = np.array([0.0, 0.0, 3.0])
        pos = obs[0:3]
        lin_vel = obs[7:10]
        ang_vel = obs[10:13]

        # 加权位置误差
        pos_error = pos - target_pos
        r_pos = -(1.0*(pos_error[0]**2 + pos_error[1]**2) 
                + 2.0*(pos_error[2]**2))

        # 速度和角速度惩罚
        r_vel = -0.5 * np.sum(lin_vel ** 2)
        r_ang = -0.5 * np.sum(ang_vel ** 2)

        # 动作平滑性
        action_smooth = 0.0
        if self.prev_action_ is not None:
            action_diff = np.linalg.norm(curr_action - self.prev_action_)
            action_smooth = -1e-3 * (action_diff ** 2)

        # 成功条件（放宽阈值）
        success = (
            np.linalg.norm(pos_error) < 0.2 and
            np.linalg.norm(lin_vel) < 0.5 and
            np.linalg.norm(ang_vel) < 0.5
        )
        success_reward = 100.0 if success else 0.0

        # 自适应时间惩罚
        time_penalty = -1.0 if not success else 0.0

        return r_pos + r_vel + r_ang + action_smooth + success_reward + time_penalty
    
    def _check_done(self, obs):
        pos_error = np.linalg.norm(obs[0:3] - self.desired_state[0:3])
        return pos_error > self.max_position_error  

    def render(self, mode='human'):
        pass  # Gazebo 自带可视化

    def close(self):
        rospy.signal_shutdown("Environment closed")
