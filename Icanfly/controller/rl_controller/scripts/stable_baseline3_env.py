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
        self.namespace = namespace  #区分不同无人机实例

        self.state_lock = threading.Lock()
        
        self.mass = 0.73  # kg
        self.gravity = 9.8066

        self.min_thrust = 0.5 * self.mass * self.gravity
        self.max_thrust =  5 * self.mass * self.gravity
        self.max_angular_rate = 5.0

 
        self.state_dim = 17  # [px, py, pz, qx, qy, qz, qw, vx, vy, vz, ang_x, ang_y, ang_z]
        self.action_dim = 4  # [thrust, ωx, ωy, ωz]
        
        self.max_position_error = 5.0  # 米
        self.max_velocity_error = 5.0  # 米

        
        self.observation_space = spaces.Box(
            low=np.concatenate((
                np.full(3, -1.0, dtype=np.float32),  # Position difference
                np.full(4, -1.0, dtype=np.float32),  # Quaternion w,x,y,z
                np.full(3, -1.0, dtype=np.float32),  # Velocity
                np.full(3, -1.0, dtype=np.float32),  # Angular rate
                np.full(1, -1.0, dtype=np.float32),  # Previous action
                np.full(3, -1.0, dtype=np.float32)   # Previous action
            )),
            high=np.concatenate((
                np.full(3, 1.0, dtype=np.float32),
                np.full(4, 1.0, dtype=np.float32),
                np.full(3, 1.0, dtype=np.float32),
                np.full(3, 1.0, dtype=np.float32),
                np.full(1, 1.0, dtype=np.float32),
                np.full(3, 1.0, dtype=np.float32)   # Previous action
            )),
            dtype=np.float32
        )

        self.obs_real_low = np.concatenate((
            np.full(3, -self.max_position_error),    # rel_pos
            np.full(4, -1.0),                        # base_quat（假设已归一化）
            np.full(3, -self.max_velocity_error),    # base_lin_vel
            np.full(3, -self.max_angular_rate),      # base_ang_vel
            np.full(1, self.min_thrust),             # last_actions
            np.full(3, -self.max_angular_rate)
        ))

        self.obs_real_high = np.concatenate((
            np.full(3, self.max_position_error),    # rel_pos
            np.full(4, 1.0),                        # base_quat（假设已归一化）
            np.full(3, self.max_velocity_error),    # base_lin_vel
            np.full(3, self.max_angular_rate),      # base_ang_vel
            np.full(1, self.max_thrust),             # last_actions
            np.full(3, self.max_angular_rate),
        ))

        self.action_real_low = np.concatenate((
            np.full(1, self.min_thrust),
            np.full(3, -self.max_angular_rate)
        ))

        self.action_real_high = np.concatenate((
            np.full(1, self.max_thrust),
            np.full(3, self.max_angular_rate)
        ))

        self.action_space = spaces.Box(
            low=np.array([-1]*self.action_dim, dtype=np.float32),
            high=np.array([1]*self.action_dim, dtype=np.float32),
            dtype=np.float32
        )


        # ROS 
        if not rospy.core.is_initialized():
            rospy.init_node(f'{self.namespace}_quadrotor_sb3_env', anonymous=True)
        
        self.control_hz = 100.0
        self.dt = 0.01
        self.rate = rospy.Rate(self.control_hz)


        self.odom_sub = rospy.Subscriber(f'/{self.namespace}/ground_truth/odometry', Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher(f'/{self.namespace}/control_command', ControlCommand, queue_size=1)
        self.arm_pub = rospy.Publisher(f'/{self.namespace}/bridge/arm', Bool, queue_size=1)
        self.windspeed_pub = rospy.Publisher(f'/{self.namespace}/wind_speed', WindSpeed, queue_size=1)
        

        # env parm
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        self.desired_state = np.array([0, 0, 3, 0, 0, 0, 1, 0, 0, 0,], dtype=np.float32)

        # 安全和步数设置
        self.max_episode_steps = 64

        self.step_count = 0
        self.episode_reward = 0
        self.prev_action_ = None


        self.prev_action_ = [0, 0, 0, 0]
        self.prev_rel_pos = None  # 上一步的相对位置

        # reward weight
        self.s_target = 0.3
        self.s_pos = 1.2
        self.s_smooth = -0.4 #-0.05
        self.s_yaw = 0.01
        self.s_angular =  -0.01
        self.s_crash = -1
        self.yaw_lambda =  0.3
        self.s_vel = 0.05
        self.s_angle_diff = -0.4 # -0.1

        self.s_r_step = 0.1


        self.prev_pos_error = None
        self.observation_space_norm = np.array(
            [0, 0, 2] + [1, 0, 0, 0] + [0, 0, 0] + [0, 0, 0] + [0, 0, 0, 0],
            dtype=np.float32
        )

        self.prev_position = [0,0,0]
        self.prev_orientation_q = [0,0,0,0]           # 用于记录上一步的4 yuanshu
   

        rospy.sleep(1.0)
    
        if not rospy.is_shutdown():
            msg = Bool(data=True)
            self.arm_pub.publish(msg)
            rospy.loginfo("Published arm message: true")

    def odom_callback(self, msg):
        with self.state_lock:
            pos = [msg.pose.pose.position.x, 
                   msg.pose.pose.position.y,
                   msg.pose.pose.position.z]

            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w
            rot_mat = [qw,qx,qy,qz]

            lin_vel = [msg.twist.twist.linear.x,
                       msg.twist.twist.linear.y,
                       msg.twist.twist.linear.z]
            
            ang_vel = [msg.twist.twist.angular.x,
                       msg.twist.twist.angular.y,
                       msg.twist.twist.angular.z]

            self.current_state = np.array(
                pos + rot_mat + lin_vel + ang_vel, dtype=np.float32
            )


    def step(self, action):
        self.step_count += 1
        self._publish_action(action)
        self.rate.sleep()
        
        with self.state_lock:
            current_state = self.current_state.copy()

        observation_space_norm, reward = self._compute_reward(current_state, action)
    
        self.episode_reward += reward

        check_done = True if self._check_done(current_state) else False

        if self.step_count >= self.max_episode_steps or check_done:
            terminated = False
            truncated = True
            info = {"reward": reward, "episode": {"r": self.episode_reward, "l": self.step_count}}
            # reset
            self.episode_reward = 0
            self.step_count = 0
        else:
            terminated = False
            truncated = False
            info = {"reward": reward}

        return observation_space_norm, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_count = 0
        self.episode_reward = 0
        self.prev_action_ = [0, 0, 0, 0]

        # 重新设定 desired_state 的位置（例如：x,y 在[-1,1]随机，z 固定为3）
        new_desired_position = np.array([np.random.uniform(-1, 1),
                                        np.random.uniform(-1, 1),
                                        np.random.uniform(1, 3)], dtype=np.float32)
        
        
        # new_desired_position = np.array([0,
        #                                 0,
        #                                 1], dtype=np.float32)

        
        # 更新 desired_state
        self.desired_state[0:3] = new_desired_position

        # 计算 reset 时的起始位置，设为 desired_state 正下方 1 米（根据实际需求调整）
        reset_position = np.array([np.random.uniform(-1, 1),
                                        np.random.uniform(-1, 1),
                                        np.random.uniform(0, 3)], dtype=np.float32)
        
        # reset_position = np.array([0,
        #                         0,
        #                         0.1], dtype=np.float32)

        # 重置无人机位置
        self._reset_drone_pose(reset_position)
        
        self.prev_position = self.current_state[0:3]
        self.prev_orientation_q = self.current_state[3:7]
        
        curr_rel_pos = self.current_state[0:3] - self.desired_state[0:3]
        base_quat = self.current_state[3:7]
        base_lin_vel = self.current_state[7:10]
        base_ang_vel = self.current_state[10:13]
        obs = np.concatenate((
            curr_rel_pos,
            base_quat,
            base_lin_vel,
            base_ang_vel,
            np.array(self.prev_action_)
        ))
        norm_obs = self._normalize_obs(obs)
        return norm_obs, {}
    
    def _reset_drone_pose(self, reset_position):

        init_orientation = np.random.uniform(-0.2, 0.2, size=4)  
        state = ModelState()
        state.model_name = self.namespace  # 确保命名空间匹配 Gazebo 的无人机模型

        state.pose.position.x = reset_position[0]
        state.pose.position.y = reset_position[1]
        state.pose.position.z = reset_position[2]

        state.pose.orientation.x = 0
        state.pose.orientation.y = 0
        state.pose.orientation.z = 0
        state.pose.orientation.w = 1 

        init_velocity = np.random.uniform(-2, 2, size=3)  
        init_angular_velocity = np.random.uniform(-0.2, 0.2, size=3) 


        state.twist.linear.x=init_velocity[0]
        state.twist.linear.y=init_velocity[1]
        state.twist.linear.z=init_velocity[2]

        state.twist.angular.x=init_angular_velocity[0]
        state.twist.angular.y=init_angular_velocity[1]
        state.twist.angular.z=init_angular_velocity[2]

        pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        pub.publish(state)
        rospy.sleep(0.05)


    def _publish_action(self, action):
        cmd = ControlCommand()
        cmd.armed = True
        cmd.control_mode = ControlCommand.BODY_RATES

        thrust = ((action[0] + 1) / 2) * (self.max_thrust - self.min_thrust) + self.min_thrust
        cmd.collective_thrust = thrust

        # 角速度归一化映射到 [-max_angular_rate, max_angular_rate]
        bodyrates = np.clip(action[1:], -1, 1) * self.max_angular_rate

        cmd.bodyrates.x = bodyrates[0]
        cmd.bodyrates.y = bodyrates[1]
        cmd.bodyrates.z = bodyrates[2]
        self.cmd_pub.publish(cmd)

    def _compute_reward(self, current_state, curr_action):
        # 1. 当前相对位置误差：当前位置与目标位置之差
        curr_pos = current_state[0:3]

        prev_dist = np.linalg.norm(self.desired_state[0:3] - self.prev_position)
        curr_dist = np.linalg.norm(self.desired_state[0:3] - curr_pos)
        r_progress_dis = prev_dist - curr_dist

        r_target = np.tanh(0.5*r_progress_dis)

        curr_rel_pos = current_state[0:3] - self.desired_state[0:3] 
        curr_rel_dis = np.linalg.norm(current_state[0:3] - self.desired_state[0:3])
        # r_position = -curr_rel_dis/self.max_position_error # np.exp(-1.7 * curr_rel_dis)
        r_position = np.exp(-1.0 * curr_rel_dis)


        curr_vel_error = current_state[7:10] - self.desired_state[7:10]
        curr_vel_error_norm = np.linalg.norm(curr_vel_error)
        r_velocity = np.exp(-0.3 * curr_vel_error_norm)


        # 3. 动作平滑奖励：当前动作与上一次动作的平方差之和
        r_smooth = np.sum((curr_action - np.array(self.prev_action_))**2)/16
        # print("r_smooth: ", r_smooth)

        # 3. 惩罚无人机的姿态变化
        epsilon=1e-6
        q_prev = self.safe_normalize(self.prev_orientation_q, epsilon)
        q_curr = self.safe_normalize(current_state[3:7], epsilon)

        # 计算内积并裁剪到[-1, 1]范围内
        dot_product = np.dot(q_prev, q_curr)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        # 计算旋转角度差
        angle_diff = 2 * np.arccos(abs(dot_product))
        # print("self.prev_orientation_q: ", self.prev_orientation_q)
        # print("current_state[3:7]: ", current_state[3:7])
        # print("angle_diff: ", angle_diff)


        # 4. 偏航角奖励：从四元数（顺序 [qw, qx, qy, qz]）中计算 yaw
        qw, qx, qy, qz = current_state[3:7]
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        theta = yaw * 180 / np.pi  # 转换为角度
        if theta > 180:
            phi = theta - 360
        else:
            phi = theta
        tilde_theta = phi * np.pi / 180  # 转换为弧度
        r_yaw = np.exp(self.yaw_lambda * abs(tilde_theta))

        # 5. 角速度奖励：假定角速度存储在 current_state 的索引 10:13
        omega = current_state[10:13]
        r_angular = np.linalg.norm(omega / np.pi)

        # 6. 撞机奖励：若满足撞机条件，则为 1，否则为 0
        r_crash = 1.0 if self._check_done(current_state) else 0.0


        thrust = ((curr_action[0] + 1)/2) * (self.max_thrust - self.min_thrust) + self.min_thrust
        r_energy = (thrust/(self.mass*self.gravity))**2/16

        r_step = 1
        
        total_reward = self.dt*(
            # self.s_r_step * r_step +
            self.s_pos * r_position  +
            self.s_target  * r_target +
            # self.s_smooth * r_smooth +
            # self.s_angle_diff * angle_diff +

            # self.s_vel * r_velocity+
            # self.s_yaw * r_yaw +
            # self.s_angular * r_angular +
            self.s_crash * r_crash 
            # + r_energy
        )


        base_quat = current_state[3:7]
        base_lin_vel = current_state[7:10]
        base_ang_vel = current_state[10:13]
        

        obs = np.concatenate((
            curr_rel_pos,
            base_quat,
            base_lin_vel,
            base_ang_vel,
            np.array(self.prev_action_)
        ))

        norm_obs = self._normalize_obs(obs)
        self.prev_action_ = curr_action.copy()
        self.prev_rel_pos = curr_rel_pos.copy()
        self.prev_position = curr_pos.copy()
        self.prev_orientation_q = current_state[3:7]

        return norm_obs, total_reward
    

    def _normalize_obs(self, obs):
        """
         norm = 2 * (obs - low) / (high - low) - 1
        """
        # print("原始 obs: ", obs)
        
        # 归一化前13维
        norm_obs_first = (obs[0:13] - self.obs_real_low[0:13]) / (self.obs_real_high[0:13] - self.obs_real_low[0:13] + 1e-8)
        norm_obs_first = norm_obs_first * 2 - 1
        # print("归一化后的前13维: ", norm_obs_first)
        
        # 后面的[13:17]部分直接使用，因为预设动作输出已处于[-1,1]
        norm_obs_second = obs[13:17]
        # print("直接使用的后4维 (动作部分): ", norm_obs_second)
        
        # 拼接两个部分
        norm_obs = np.concatenate((norm_obs_first, norm_obs_second))
        # print("最终归一化后的 obs: ", norm_obs)
        
        return norm_obs.astype(np.float32)
    
    def safe_normalize(self, q, epsilon=1e-6):
        """
        安全归一化四元数，防止除以零
        """
        q = np.array(q)  # 转换为 NumPy 数组
        norm = np.linalg.norm(q)
        if norm < epsilon:
            norm = epsilon
        return q / norm
    


    def _check_done(self, curr_state):
        pos_error = np.linalg.norm(curr_state[0:3] - self.desired_state[0:3])
        if pos_error > self.max_position_error:
            return True
        
        qw = curr_state[3]
        qx = curr_state[4]
        qy = curr_state[5]
        qz = curr_state[6]
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        if np.abs(roll) > 1.8:
            return True

        if np.linalg.norm(curr_state[7:10]) > self.max_velocity_error: 
            return True
        
        return False

    def render(self, mode='human'):
        pass  # Gazebo 自带可视化

    def close(self):
        rospy.signal_shutdown("Environment closed")

