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
from quadrotor_msgs.msg import Trajectory
import threading
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker
import random 

class QuadrotorEnv(gym.Env):
    def __init__(self, namespace="drone"):
        super(QuadrotorEnv, self).__init__()
        self.namespace = namespace    

        # calculate origin offset based on namespace
        self.origin_offset = 4

  
        # parameters
        self.mass = 0.73  
        self.gravity = 9.8066
        self.min_thrust = 0.5 * self.mass * self.gravity
        self.max_thrust = 5 * self.mass * self.gravity
        self.max_angular_rate = 5.0

        # state dims
        self.state_dim = 13  # [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        self.action_dim = 4
        self.max_episode_steps = 516

        # bounds
        self.max_position_error = 3.5
        self.max_velocity_error = 15


        # ROS setup
        if not rospy.core.is_initialized():
            rospy.init_node(f'{self.namespace}_quad_env', anonymous=True)

        self.control_hz = 100.0
        self.rate = rospy.Rate(self.control_hz)

        # subscribers & publishers
        self.odom_sub = rospy.Subscriber(f'/{self.namespace}/ground_truth/odometry', Odometry, self.odom_callback)
        self.trajectory_sub = rospy.Subscriber(f'autopilot/trajectory', Trajectory, self.trajectory_callback)
        
        self.cmd_pub = rospy.Publisher(f'/{self.namespace}/control_command', ControlCommand, queue_size=1)
        self.arm_pub = rospy.Publisher(f'/{self.namespace}/bridge/arm', Bool, queue_size=1)
        self.windspeed_pub = rospy.Publisher(f'/{self.namespace}/wind_speed', WindSpeed, queue_size=1)
        self.desired_state_marker_pub = rospy.Publisher(f'/{self.namespace}/desired_state_marker', Marker, queue_size=1)
        

        # spaces
        self.observation_space = spaces.Box(
            low=np.concatenate((
                np.full(3, -1.0, dtype=np.float32),  # Position difference
                np.full(9, -1.0, dtype=np.float32),  # rotation matrix  
                np.full(3, -1.0, dtype=np.float32),  # Velocity
                np.full(3, -1.0, dtype=np.float32),  # Angular rate
                np.full(1, -1.0, dtype=np.float32),  # Previous thrust
                np.full(3, -1.0, dtype=np.float32)   # Previous bady rates
            )),
            high=np.concatenate((
                np.full(3, 1.0, dtype=np.float32),
                np.full(9, 1.0, dtype=np.float32),
                np.full(3, 1.0, dtype=np.float32),
                np.full(3, 1.0, dtype=np.float32),
                np.full(1, 1.0, dtype=np.float32),
                np.full(3, 1.0, dtype=np.float32)    
            )),
            dtype=np.float32
        )

        self.obs_real_low = np.concatenate((
            np.full(3, -self.max_position_error),    
            np.full(9, -1.0),                         
            np.full(3, -self.max_velocity_error),    
            np.full(3, -self.max_angular_rate),      
            np.full(1, self.min_thrust),           
            np.full(3, -self.max_angular_rate)
        ))

        self.obs_real_high = np.concatenate((
            np.full(3, self.max_position_error),     
            np.full(9, 1.0),                       
            np.full(3, self.max_velocity_error),   
            np.full(3, self.max_angular_rate),    
            np.full(1, self.max_thrust),         
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


        # env parm
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        self.desired_state = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0,], dtype=np.float32) 

        # 安全和步数设置
        self.max_episode_steps = 64

        self.step_count = 0
        self.episode_reward = 0
    

        self.prev_action = [0, 0, 0, 0]
        self.prev_rel_pos = None  # 上一步的相对位置

        # reward weight
        self.w_progress = 1
        self.w_dist = 5
        self.w_vel = 0.0 
        self.w_heading = 0.0
        self.w_control =  0.01
        self.w_crash = 10
   

        self.flag_traj=0
        self.desired_id = 0


        self.prev_pos_error = None
        self.observation_space_norm = np.array(
            [0, 0, 2] + [1, 0, 0, 0] + [0, 0, 0] + [0, 0, 0] + [0, 0, 0, 0],
            dtype=np.float32
        )

        self.desired_state = np.array([1, 0, 1, 1, 0, 0, 0, 0, 0, 0,], dtype=np.float32) 
        self.desired_traj = None

        self.prev_pos = [0,0,0]
        self.prev_orientation_q = [0,0,0,0]           

        rospy.sleep(1)
    
        if not rospy.is_shutdown():
            msg = Bool(data=True)
            self.arm_pub.publish(msg)
            rospy.loginfo("Published arm message: true")

        
        
        rospy.set_param('wind_speed/level', 0) #2 5 10

        self.timer_windspeed = rospy.Timer(rospy.Duration(0.01), self.control_wind_speed)

        
        # 轨迹周期 - velocity
        # self.T = 2.0
        # self.T = 3.5
        self.T = 8.0
        self.t = 0
        self.timer_trajectory = rospy.Timer(rospy.Duration(0.01), self.update_desired_state)



        
        reset_position = np.array([self.origin_offset + np.random.uniform(-1, 1),
                                    np.random.uniform(-1, 1),
                                   np.random.uniform(0.4, 1.8)], dtype=np.float32)
        
        self.marker = Marker()
        self.marker.header.frame_id = "world"   
        self.marker.header.stamp = rospy.Time.now()
        self.marker.ns = "desired_state"
        self.marker.id = 0
        self.marker.type = Marker.SPHERE  
        self.marker.action = Marker.ADD

        self.marker.pose.orientation.x = 0.0
        self.marker.pose.orientation.y = 0.0
        self.marker.pose.orientation.z = 0.0
        self.marker.pose.orientation.w = 1.0  

        self.marker.scale.x = 0.08
        self.marker.scale.y = 0.08
        self.marker.scale.z = 0.08
        
        self.marker.color.a = 1.0  
        self.marker.color.r = 0.0
        self.marker.color.g = 1.0
        self.marker.color.b = 0.0
        self._reset_drone_pose(reset_position)
    
    


    def control_wind_speed(self, event):
        """
        定时器回调函数，每次触发时：
        - 从参数服务器读取三个风速大小的值（低、中、高）。
        - 随机选择一个风速值，并生成一个 0~360 度的随机风向。
        - 计算风速在 x、y 方向的分量（z 分量保持为 0），发布 WindSpeed 消息。
        """

        if rospy.is_shutdown():
            return  # 防止节点关闭时发布消息

        speed = rospy.get_param('wind_speed/level', 0.0)
        
        # 固定主方向为 x 轴正方向 (0°)
        main_angle_deg = 0.0
        # 添加小幅度扰动，例如 ±5°
        disturbance_range = 30.0  
        delta_angle_deg = random.uniform(-disturbance_range, disturbance_range)
        total_angle_deg = main_angle_deg + delta_angle_deg
        angle_rad = np.deg2rad(total_angle_deg)
        
        # 计算 x,y 分量
        vx = speed * np.cos(angle_rad)
        vy = speed * np.sin(angle_rad)
        
        # 在 z 方向加入小扰动，例如 ±10% 的风速
        vz = speed * 0.1 * random.uniform(-1, 1)

        # 构造 WindSpeed 消息并发布
        wind_msg = WindSpeed()
        wind_msg.header.stamp = rospy.Time.now()
        wind_msg.header.frame_id = "world"  # 或根据需要修改
        wind_msg.velocity.x = vx
        wind_msg.velocity.y = vy
        wind_msg.velocity.z = vz

        try:
            self.windspeed_pub.publish(wind_msg)
        except rospy.ROSException as e:
            rospy.logwarn(f"发布风速消息失败: {e}")


    def update_desired_state(self, event):

        # if self.desired_traj==None: return
        # id = self.desired_id % len(self.desired_traj)
        # new_desired_position = np.array([self.desired_traj[id].pose.position.x, 
        #                                  self.origin_offset + self.desired_traj[id].pose.position.y, 
        #                                  self.desired_traj[id].pose.position.z], dtype=np.float32)

        # self.desired_state[0:3] = new_desired_position


        # self.desired_state[3:7] =  [self.desired_traj[id].pose.orientation.w, 
        #                             self.desired_traj[id].pose.orientation.x, 
        #                             self.desired_traj[id].pose.orientation.y, 
        #                             self.desired_traj[id].pose.orientation.z]
        
        # self.desired_state[7:10] = [self.desired_traj[id].velocity.linear.x, 
        #                             self.desired_traj[id].velocity.linear.y, 
        #                             self.desired_traj[id].velocity.linear.z]
        
        # self.marker.pose.position.x = new_desired_position[0]
        # self.marker.pose.position.y = new_desired_position[1]
        # self.marker.pose.position.z = new_desired_position[2]

        # self.desired_state_marker_pub.publish(self.marker)
        # self.desired_id += 1

        # print(len(self.desired_traj))


        t= rospy.get_time()
        new_position = np.array([
            self.origin_offset + np.cos(2 * np.pi * t / self.T),
            np.sin(4 * np.pi * t / self.T) / 2,
            1.0
        ], dtype=np.float32)

        self.desired_state[0:3] = new_position

        self.desired_state[3:7] =  [1, 
                            0, 
                            0, 
                            0]

        self.desired_state[7:10] = [1, 
                            1, 
                            0]
        
        self.marker.pose.position.x = new_position[0]
        self.marker.pose.position.y = new_position[1]
        self.marker.pose.position.z = new_position[2]

        self.desired_state_marker_pub.publish(self.marker)


    def trajectory_callback(self, msg):
        if not self.desired_traj:
            self.desired_traj = msg.points


    def odom_callback(self, msg):
        with threading.Lock():
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

    def quaternion_to_rot_matrix(self, quat):
        """
        将四元数 [qw, qx, qy, qz] 转换为 3x3 旋转矩阵，并平铺成长度为 9 的向量
        """
        qw, qx, qy, qz = quat
        R = np.array([
            [1 - 2*qy*qy - 2*qz*qz,   2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw,       1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw,       2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy]
        ], dtype=np.float32)
        return R.flatten()

    def step(self, action):
        self.step_count += 1
        self._publish_action(action)
        self.rate.sleep()
        
        # with self.state_lock:
        current_state = self.current_state 

        observation_space_norm, reward = self._compute_reward(current_state, action)
    
        self.episode_reward += reward

        check_done = True if self._check_done(current_state) else False

        if check_done:
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
        self.prev_action = [0, 0, 0, 0]


        reset_position = np.array([self.origin_offset + 0,
                                    0,
                                   1], dtype=np.float32)


        self._reset_drone_pose(reset_position)
        
        self.prev_pos = self.current_state[0:3]
        self.prev_orientation_q = self.current_state[3:7]
        
        curr_rel_pos = self.current_state[0:3] - self.desired_state[0:3]
        rot_mat = self.quaternion_to_rot_matrix(self.current_state[3:7])
        base_lin_vel = self.current_state[7:10]

        base_ang_vel = self.current_state[10:13]
        obs = np.concatenate((
            curr_rel_pos,
            rot_mat,
            base_lin_vel,
            base_ang_vel,
            np.array(self.prev_action)
        ))

        norm_obs = self._normalize_obs(obs)
        self.desired_id = 0

        return norm_obs, {}
    
    def _pub_reset(self, reset_position):
        state = ModelState()
        state.model_name = self.namespace  

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
    
    def _reset_drone_pose(self, reset_position):
        tolerance = 0.1  
        timeout = rospy.Duration(10.0)  
        start_time = rospy.Time.now()
        
        while not rospy.is_shutdown():
            self._pub_reset(reset_position)
            current_pos = self.current_state[0:3]
            if np.linalg.norm(np.array(current_pos) - reset_position) < tolerance:
                # rospy.logwarn("success drone to return to reset position.")
                break
            if rospy.Time.now() - start_time > timeout:
                rospy.logwarn("Timeout reached while waiting for drone to return to reset position.")
                break


    def _publish_action(self, action):
        cmd = ControlCommand()
        cmd.armed = True
        cmd.control_mode = ControlCommand.BODY_RATES

        thrust = ((action[0] + 1) / 2) * (self.max_thrust - self.min_thrust) + self.min_thrust
        cmd.collective_thrust = thrust

  
        bodyrates = np.clip(action[1:], -1, 1) * self.max_angular_rate

        cmd.bodyrates.x = bodyrates[0]
        cmd.bodyrates.y = bodyrates[1]
        cmd.bodyrates.z = bodyrates[2]
        self.cmd_pub.publish(cmd)

    def _tanh_clip(self, x):
        """软剪切到[-1,1]。"""
        return np.tanh(x)
    
    def _normalize(self, x, xmax):
        """线性缩放到[0,1]，并裁剪。"""
        return np.clip(x / xmax, 0.0, 1.0)

    def _tanh_clip(self, x):
        """软剪切到[-1,1]。"""
        return np.tanh(x)
    
    def _quat_yaw(self, q):
        """从 [qw, qx, qy, qz] 中提取 yaw（[-π, π]）。"""
        qw, qx, qy, qz = q
        return np.arctan2(2*(qw*qz + qx*qy),
                          1 - 2*(qy*qy + qz*qz))

    def _yaw_error(self, q_curr, q_des):
        """
        计算当前航向与目标航向的差值，带符号，范围在 [-π, π]。
        返回绝对误差（正数，≤π）。
        """
        yaw1 = self._quat_yaw(q_curr)
        yaw2 = self._quat_yaw(q_des)
        diff = yaw1 - yaw2
        # wrap 到 [-π, π]
        diff = (diff + np.pi) % (2*np.pi) - np.pi
        return abs(diff)

    def _quat_angle(self, q1, q2):
        d = np.clip(np.dot(q1, q2), -1.0, 1.0)
        return 2 * np.arccos(abs(d))  # [0, π]

    def _compute_reward(self, state, action):
        
        pos, ori_q, vel, ang = state[:3], state[3:7], state[7:10], state[10:13]
        target_pos, target_vel, target_q = self.desired_state[:3], self.desired_state[7:10], self.desired_state[3:7]

        # print("target_vel:", np.linalg.norm(pos- self.prev_pos)/0.01)


        # --- 1. 位置进展奖励（前后距离差） ---
        prev_to_goal = np.linalg.norm(target_pos - self.prev_pos)
        curr_to_goal = np.linalg.norm(target_pos - pos)
        raw_progress = prev_to_goal - curr_to_goal
        norm_progress = self._tanh_clip(raw_progress)  # 防止过大


        # ---- 2. 位置误差 (越小越好) ----
        raw_dist = np.linalg.norm(pos - target_pos)
        norm_dist = 1 - self._normalize(raw_dist, self.max_position_error)  # 1 越接近目标，越大

        # ---- 3. 速度误差 ----
        raw_vel_err = np.linalg.norm(vel - target_vel)
        norm_vel = 1 - self._normalize(raw_vel_err, self.max_velocity_error)

        # ---- 4. 姿态差异 heading 误差 ----
        q_curr = state[3:7] / (np.linalg.norm(state[3:7]) + 1e-6)
        q_des  = self.desired_state[3:7] / (np.linalg.norm(self.desired_state[3:7]) + 1e-6)
        heading_err = self._yaw_error(q_curr, q_des)  # ∈ [0, π]
        r_heading = 1.0 - (heading_err / np.pi)


        # ---- 6. 动作变化成本 ----
        raw_ctrl = np.linalg.norm(action - self.prev_action)
        norm_ctrl = 1 - self._normalize(raw_ctrl, np.linalg.norm(self.prev_action)+1e-6)

        # ---- 7. 撞机惩罚 ----
        crash = float(self._check_done(state))
        crash_penalty = -1.0 if crash else 0.0

        # ---- 汇总加权 ----
        rewards = {
            "progress":   self.w_progress * norm_progress,
            "distance":   self.w_dist     * norm_dist,
            "velocity":   self.w_vel      * norm_vel,
            "attitude":   self.w_heading  * r_heading,
            "control":    self.w_control  * norm_ctrl,
            "crash":      self.w_crash    * crash_penalty,
        }
        # print("rewards:", rewards)
        total_reward = sum(rewards.values())

        rel_pos = pos - target_pos
        rot_mat = self.quaternion_to_rot_matrix(state[3:7])
        obs = np.concatenate((
            rel_pos,
            rot_mat,
            vel,
            ang,
            np.array(self.prev_action)
        ))

        norm_obs = self._normalize_obs(obs)

        # ---- 更新历史 ----
        self.prev_pos    = pos.copy()
        self.prev_action = action.copy()

        return norm_obs, total_reward
    

    def _normalize_obs(self, obs):
        """
        将观测数据归一化到 [-1, 1] 范围内：
           norm = 2 * (obs - low) / (high - low) - 1
        这里将前 18 维（3+9+3+3）归一化，剩下 4 维（上一次动作）直接使用
        """
        norm_obs_first = (obs[0:18] - self.obs_real_low[0:18]) / (self.obs_real_high[0:18] - self.obs_real_low[0:18] + 1e-8)
        norm_obs_first = norm_obs_first * 2 - 1
        norm_obs_first = np.clip(norm_obs_first, -1, 1)
        norm_obs_second = obs[18:22]
        norm_obs = np.concatenate((norm_obs_first, norm_obs_second))
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
        if pos_error > self.max_position_error or curr_state[2]<0.3:
            return True
        
        qw = curr_state[3]
        qx = curr_state[4]
        qy = curr_state[5]
        qz = curr_state[6]
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        if np.abs(roll) > 2.2:
            return True
        
        if np.linalg.norm(curr_state[7:10]) > self.max_velocity_error: 
            return True
        
        return False

    def render(self, mode='human'):
        pass   

    def close(self):
        rospy.signal_shutdown("Environment closed")
