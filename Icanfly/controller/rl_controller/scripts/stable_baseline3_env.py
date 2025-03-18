#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import ControlCommand, TrajectoryPoint 

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
        self.gravity = 9.8066
        self.min_thrust = 0 * self.mass * self.gravity
        self.max_thrust =  4 * self.mass * self.gravity
        self.max_angular_rate = 3.0

       # 位置 (3), 旋转矩阵 (3x3=9), 线速度 (3), 上一个动作 (4) -> 共 19 维
        self.state_dim = 25  
        self.action_dim = 4  # [推力, ωx, ωy, ωz]
        

        self.observation_space = spaces.Box(
            low=np.concatenate((
                np.full(3, -10),          # 当前位置
                np.full(9, -1),          # 旋转矩阵
                np.full(3, -5),          # 当前线速度
                np.full(3, -5),          # 目标位置
                np.full(3, -5),          # 目标速度
                np.full(4, -1)           # 上一步动作
            )),
            high=np.concatenate((
                np.full(3, 10),
                np.full(9, 1),
                np.full(3, 5),
                np.full(3, 5),
                np.full(3, 5),
                np.full(4, 1)
            )),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-1]*self.action_dim, dtype=np.float32),
            high=np.array([1]*self.action_dim, dtype=np.float32),
            dtype=np.float32
        )

        # ROS 初始化（每个实例使用独特的节点名）
        if not rospy.core.is_initialized():
            rospy.init_node(f'{self.namespace}_quadrotor_sb3_env', anonymous=True)
        
    
        # 控制频率
        self.control_hz = 100.0
        self.rate = rospy.Rate(self.control_hz)
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        self.desired_state = np.array(
            [0, 0, 2] + [1, 0, 0, 0, 1, 0, 0, 0, 1] + [0, 0, 0] + [0, 0, 0, 0],
            dtype=np.float32
        )
        # 安全和步数设置
        self.max_position_error = 4.0  # 米
        self.max_velocity_error = 5.0
        self.max_episode_steps = 256
        self.step_count = 0
        self.episode_reward = 0

        self.prev_pos_ = None

        
        self.prev_action_ = None
        self.prev_position = None 


        self.angular_x = 0
        self.angular_y = 0
        self.angular_z = 0

         # 订阅和发布（话题前缀使用命名空间）
        self.odom_sub = rospy.Subscriber(f'/{self.namespace}/ground_truth/odometry', Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher(f'/{self.namespace}/control_command', ControlCommand, queue_size=1)
        self.arm_pub = rospy.Publisher(f'/{self.namespace}/bridge/arm', Bool, queue_size=1)
        self.reference_state_sub = rospy.Subscriber('/autopilot/reference_state', TrajectoryPoint, self.reference_state_callback)

        
        self.windspeed_pub = rospy.Publisher(f'/{self.namespace}/wind_speed', WindSpeed, queue_size=1)
   

        rospy.sleep(1.0)
    
        if not rospy.is_shutdown():
            msg = Bool(data=True)
            self.arm_pub.publish(msg)
            rospy.loginfo("Published arm message: true")
    
    
    def _quaternion_to_rotation_matrix(self, x, y, z, w):
        """将四元数转换为旋转矩阵（3x3），注意四元数顺序为 (x, y, z, w)"""
        R = np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - z*w),         2*(x*z + y*w)],
            [2*(x*y + z*w),           1 - 2*(x**2 + z**2),    2*(y*z - x*w)],
            [2*(x*z - y*w),           2*(y*z + x*w),         1 - 2*(x**2 + y**2)]
        ], dtype=np.float32)
        return R
        
    def odom_callback(self, msg):
        with self.state_lock:
            # 获取位置
            pos = [msg.pose.pose.position.x, 
                   msg.pose.pose.position.y,
                   msg.pose.pose.position.z]
            # 从四元数转换为旋转矩阵
            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w
            rot_mat = self._quaternion_to_rotation_matrix(qx, qy, qz, qw).flatten()
            # 获取线速度（不包含角速度）
            lin_vel = [msg.twist.twist.linear.x,
                       msg.twist.twist.linear.y,
                       msg.twist.twist.linear.z]
            # 获取上一个动作，如果未记录则用全零向量
            prev_action = self.prev_action_ if self.prev_action_ is not None else np.zeros(self.action_dim, dtype=np.float32)
            # 组装状态：位置 (3) + 旋转矩阵 (9) + 线速度 (3) + 上一个动作 (4)
            target_pos = self.desired_state[0:3]
            target_vel = self.desired_state[12:15]
            # target_pos = [0,0,2]
            # target_vel = [0,0,0]
            self.current_state = np.array(
                pos + rot_mat.tolist() + lin_vel + 
                target_pos.tolist() + target_vel.tolist() + 
                prev_action.tolist(), dtype=np.float32
            )

    def reference_state_callback(self, msg):
        # 获取位置
        # pos = [msg.pose.position.x, 
        #        msg.pose.position.y,
        #        msg.pose.position.z]
        pos = [0, 
               0,
               3]
        # 将四元数转换为旋转矩阵
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        rot_mat = self._quaternion_to_rotation_matrix(qx, qy, qz, qw).flatten()
        # 获取线速度
        lin_vel = [msg.velocity.linear.x,
                   msg.velocity.linear.y,
                   msg.velocity.linear.z]
        # 期望状态中，上一个动作部分固定为零
        
        self.desired_state = np.array(pos + rot_mat.tolist() + lin_vel, dtype=np.float32)
        print(self.desired_state)

    def step(self, action):
        self.step_count += 1
        self._publish_action(action)
        self.rate.sleep()
        
        with self.state_lock:
            obs = self.current_state.copy()

        reward = self._compute_reward(obs, action)
        self.episode_reward += reward

        if self.step_count >= self.max_episode_steps:
            # 达到最大步数：回合被截断，不算任务失败
            terminated = False  
            truncated = True   
            info = {"reward": reward, "episode": {"r": self.episode_reward, "l": self.step_count}}
            # 重置计数器和累计奖励
            self.episode_reward = 0
            self.step_count = 0
        elif self._check_done(obs):
            # 状态条件满足：回合终止，任务失败（例如超出范围）
            terminated = True  
            truncated = False
            info = {"reward": reward, "episode": {"r": self.episode_reward, "l": self.step_count}}
            # 重置计数器和累计奖励
            self.episode_reward = 0
            self.step_count = 0
        else:
            # 正常状态下：回合继续
            terminated = False
            truncated = False
            info = {"reward": reward}
        # print(self.episode_reward)


        # # 重置计数器和累计奖励
        # if terminated or truncated:
        #     self.episode_reward = 0
        #     self.step_count = 0


        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_count = 0
        self.episode_reward = 0
        self.prev_action_ = None
        self._reset_drone_pose()
        rospy.sleep(0.01)  # 等待复位完成
        with self.state_lock:
            obs = self.current_state.copy()
        return obs, {}
    

    def _reset_drone_pose(self):
        init_position = np.array([
            np.random.uniform(-1, 1),  # x 范围 [-1, 1] m
            np.random.uniform(-1, 1),  # y 范围 [-1, 1] m
            np.random.uniform(0.11, 3.2)    # z 范围 [0.5, 1.5] m，避免过高或过低
        ])

        init_orientation = np.random.uniform(-1, 1, size=3)  # 近似水平的随机扰动
        init_velocity = np.random.uniform(-2, 2, size=3)  # 低速随机初始化 #init_velocity = np.random.uniform(-3, 3, size=3)  # 低速随机初始化
        init_angular_velocity = np.random.uniform(-1, 1, size=3)  # 低速角速度初始化

        state = ModelState()
        state.model_name = self.namespace  # 确保命名空间匹配 Gazebo 的无人机模型
        state.pose.position.x = 0
        state.pose.position.y = 0
        state.pose.position.z = 0.11
        state.pose.orientation.x = 0
        state.pose.orientation.y = 0
        state.pose.orientation.z = 0
        state.pose.orientation.w = 1  # 保持单位四元数

        # state.twist.linear.x=init_velocity[0]
        # state.twist.linear.y=init_velocity[1]
        # state.twist.linear.z=init_velocity[2]

        # state.twist.angular.x=init_angular_velocity[0]
        # state.twist.angular.y=init_angular_velocity[1]
        # state.twist.angular.z=init_angular_velocity[2]

        pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        pub.publish(state)

    def _publish_action(self, action):
        # print(action)
        cmd = ControlCommand()
        cmd.armed = True
        cmd.control_mode = ControlCommand.BODY_RATES

        thrust = ((action[0] + 1) / 2) * (self.max_thrust - self.min_thrust) + self.min_thrust
        cmd.collective_thrust = thrust

        bodyrates = np.clip(action[1:], -1, 1) * self.max_angular_rate

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

    def _compute_reward_error(self, obs, curr_action):
 
        curr_pos = obs[0:3]
        target_pos = self.desired_state[0:3]
        curr_vel = obs[12:15]
        target_vel = self.desired_state[12:15]
        
        # 1. 位置跟踪奖励（指数衰减）
        pos_error = np.linalg.norm(curr_pos - target_pos)
        
        r_position = np.exp(-0.3 * pos_error)
 
        # 2. 速度跟踪奖励
        vel_error = np.linalg.norm(curr_vel - target_vel)
        r_velocity = np.exp(-0.3 * vel_error)
        print("vel_error: ", vel_error)
        # 3. 姿态稳定奖励（旋转矩阵与目标差异）
        # current_rot = obs[3:12].reshape(3,3)
        # target_rot = self.desired_state[3:12].reshape(3,3)
        # rot_diff = np.arccos((np.trace(current_rot @ target_rot.T) - 1)/2)
        # r_attitude = -0.1 * (np.abs(rot_diff/np.pi)**2)

        # 4. 动作平滑惩罚
        r_smooth = 0.0
        if self.prev_action_ is not None:
            action_diff = np.linalg.norm(curr_action - self.prev_action_)
            r_smooth = np.exp(-1 * action_diff)
        
            print("action_diff: ", action_diff)

        # 5. 能量效率惩罚
        # thrust = ((curr_action[0] + 1)/2) * (self.max_thrust - self.min_thrust) + self.min_thrust
        # r_energy = np.exp(-0.5 * (thrust/(self.mass*self.gravity))**2)


        final_reward=0
        if r_position<0.05 and vel_error<0.2:
           final_reward=1

        if r_position<0.05:
            final_reward=0.8

        r_attitude_pen=0

        current_rot = obs[3:12].reshape(3,3)
        # 直接从旋转矩阵中计算 roll: roll = arctan2(R[2,1], R[2,2])
        roll = np.arctan2(current_rot[2,1], current_rot[2,2])
        if np.abs(roll) > 1.3:  # 阈值 1.0 弧度（约57°），可根据需求调整
            r_attitude_pen=-0.5

        # 8. 路径对齐惩罚
        # 根据上一次的位置（self.prev_pos_）和当前的位置构成向量，
        # 计算目标点到该向量的垂直距离，距离越大则惩罚越多，鼓励无人机沿目标方向飞行。
        dist_to_line = 0.0
        if self.prev_pos_ is not None:
            movement = curr_pos - self.prev_pos_
            if np.linalg.norm(movement) > 1e-6:
                # 使用叉积计算目标点到无人机运动方向的垂直距离
                cross = np.cross(target_pos - self.prev_pos_, movement)
                dist_to_line = np.linalg.norm(cross) / np.linalg.norm(movement)
            else:
                dist_to_line = 0.0

            # 计算目标方向向量
            desired_vector = target_pos - self.prev_pos_
            # 防止除以零
            if np.linalg.norm(desired_vector) > 1e-6 and np.linalg.norm(movement) > 1e-6:
                dot = np.dot(movement, desired_vector) / (np.linalg.norm(movement) * np.linalg.norm(desired_vector))
            else:
                dot = 1.0  # 若无明显运动，则视为朝向正确

            # 如果无人机朝向目标飞行（dot >= 0），则给予正奖励；反之给予负奖励
            if dot < 0:
                r_path = -np.exp(-0.5 * abs(dist_to_line))
            else:
                r_path = np.exp(-0.5 * abs(dist_to_line))
        else:
            r_path = 0.0

        
        
        # 合成总奖励
        total_reward = (
            (
            3*r_position +
            #  r_path+
            0.2*r_velocity +
            # r_attitude +
            0.3*r_smooth+ 
            # r_energy+  
            final_reward)
        )
        # print(total_reward, r_position, r_velocity, r_smooth, final_reward)
        self.prev_action_ = curr_action.copy()
        self.prev_pos_ = curr_pos.copy() 
        return total_reward

    def _compute_reward(self, obs, curr_action):
        # 状态分解
        curr_pos = obs[0:3]
        target_pos = self.desired_state[0:3]
        curr_vel = obs[12:15]
        target_vel = self.desired_state[12:15]
        
        # 1. 位置跟踪奖励（指数衰减）
        pos_error = np.linalg.norm(curr_pos - target_pos)

        r_position = np.exp(-2 * pos_error)

        # print(r_position)

        # 2. 速度跟踪奖励
        vel_error = np.linalg.norm(curr_vel - target_vel)
        r_velocity = np.exp(-2 * vel_error)

        # 3. 姿态稳定奖励（旋转矩阵与目标差异）
        current_rot = obs[3:12].reshape(3,3)
        target_rot = self.desired_state[3:12].reshape(3,3)
        rot_diff = np.arccos((np.trace(current_rot @ target_rot.T) - 1)/2)
        r_attitude = 0.1 * (1 - np.abs(rot_diff/np.pi))

        # 4. 动作平滑惩罚
        if self.prev_action_ is not None:
            action_diff = np.linalg.norm(curr_action - self.prev_action_)
            r_smooth = np.exp(-1 * action_diff)
        else:
            r_smooth = 0.0

        # 5. 能量效率惩罚
        thrust = ((curr_action[0] + 1)/2) * (self.max_thrust - self.min_thrust) + self.min_thrust
        r_energy = -0.05 * (thrust/(self.mass*self.gravity))**2

        # 合成总奖励
        total_reward = (
            r_position 
            # r_velocity +
            # r_attitude +
            # r_smooth
        )
        # print("pos_error: ",pos_error, ", r_position: ",r_position, ", 0.4*r_velocity: ", 0.4*r_velocity, ", r_attitude: ", r_attitude, ", r_smooth: ", r_smooth)

        # 更新历史数据
        self.prev_action_ = curr_action.copy()
        return total_reward


    def _check_done(self, obs):
        # 位置失稳检查
        pos_error = np.linalg.norm(obs[0:3] - self.desired_state[0:3])
        if pos_error > self.max_position_error or obs[2]<0:
            return True
        
        # 姿态异常检查
        # 姿态检查：检测侧翻（roll 超过阈值）
        current_rot = obs[3:12].reshape(3,3)
        # # 直接从旋转矩阵中计算 roll: roll = arctan2(R[2,1], R[2,2])
        roll = np.arctan2(current_rot[2,1], current_rot[2,2])
        if np.abs(roll) > 1.2:  # 阈值 1.0 弧度（约57°），可根据需求调整
            return True
        
        # # # 速度失控检查
        # if np.linalg.norm(obs[12:15]) > np.sqrt(self.max_velocity_error):  # 最大允许速度
        #     return True
        
        return False
    
    def render(self, mode='human'):
        pass  # Gazebo 自带可视化

    def close(self):
        rospy.signal_shutdown("Environment closed")

