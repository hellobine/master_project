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
# keep a version: add wind speed and Eight-figure trajectory
class QuadrotorEnv(gym.Env):
    def __init__(self, namespace="drone"):
        super(QuadrotorEnv, self).__init__()
        self.namespace = namespace  #区分不同无人机实例
        self.origin_offset = 0
        for i in range(10):
            if str(i) in namespace:
                self.origin_offset += 2*i


        self.state_lock = threading.Lock()
        
        self.mass = 0.73  # kg
        self.gravity = 9.8066

        self.min_thrust = 0.5 * self.mass * self.gravity
        self.max_thrust =  5 * self.mass * self.gravity
        self.max_angular_rate = 5.0

 
        self.state_dim = 22  # [px, py, pz, qx, qy, qz, qw, vx, vy, vz, ang_x, ang_y, ang_z]
        self.action_dim = 4  # [thrust, ωx, ωy, ωz]
        
        self.max_position_error = 3.5  # 米
        self.max_velocity_error = 6.0  # 米

        
        self.observation_space = spaces.Box(
            low=np.concatenate((
                np.full(3, -1.0, dtype=np.float32),  # Position difference
                np.full(9, -1.0, dtype=np.float32),  # Quaternion w,x,y,z
                np.full(3, -1.0, dtype=np.float32),  # Velocity
                np.full(3, -1.0, dtype=np.float32),  # Angular rate
                np.full(1, -1.0, dtype=np.float32),  # Previous action
                np.full(3, -1.0, dtype=np.float32)   # Previous action
            )),
            high=np.concatenate((
                np.full(3, 1.0, dtype=np.float32),
                np.full(9, 1.0, dtype=np.float32),
                np.full(3, 1.0, dtype=np.float32),
                np.full(3, 1.0, dtype=np.float32),
                np.full(1, 1.0, dtype=np.float32),
                np.full(3, 1.0, dtype=np.float32)   # Previous action
            )),
            dtype=np.float32
        )

        self.obs_real_low = np.concatenate((
            np.full(3, -self.max_position_error),    # rel_pos
            np.full(9, -1.0),                        # base_quat（假设已归一化）
            np.full(3, -self.max_velocity_error),    # base_lin_vel
            np.full(3, -self.max_angular_rate),      # base_ang_vel
            np.full(1, self.min_thrust),             # last_actions
            np.full(3, -self.max_angular_rate)
        ))

        self.obs_real_high = np.concatenate((
            np.full(3, self.max_position_error),    # rel_pos
            np.full(9, 1.0),                        # base_quat（假设已归一化）
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
        self.desired_state_marker_pub = rospy.Publisher(f'/{self.namespace}/desired_state_marker', Marker, queue_size=1)
        self.trajectory_sub = rospy.Subscriber(f'autopilot/trajectory', Trajectory, self.trajectory_callback)


        # env parm
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        self.desired_state = np.array([0, 0, 3, 1, 0, 0, 0, 0, 0, 0,], dtype=np.float32) 

        # 安全和步数设置
        self.max_episode_steps = 516

        self.step_count = 0
        self.episode_reward = 0
        self.prev_action_ = None


        self.prev_action_ = [0, 0, 0, 0]
        self.prev_rel_pos = None  # 上一步的相对位置

        # reward weight
        self.s_target = 0.1
        self.s_pos = 10
        self.s_smooth = 0.001 #-0.4 #-0.05
        self.s_yaw = 0.01
        self.s_angular =  -0.01
        self.s_crash = -10
        self.yaw_lambda =  0.3
        self.s_vel = 0.1 # 0.2
        self.s_angle_diff = -0.001 # -0.1

        self.flag_traj=0

        self.desired_id = 0


        self.prev_pos_error = None
        self.observation_space_norm = np.array(
            [0, 0, 2] + [1, 0, 0, 0] + [0, 0, 0] + [0, 0, 0] + [0, 0, 0, 0],
            dtype=np.float32
        )

        self.desired_state = np.array([1, 0, 1, 1, 0, 0, 0, 0, 0, 0,], dtype=np.float32) 
        self.desired_traj = None

        self.prev_position = [0,0,0]
        self.prev_orientation_q = [0,0,0,0]           

        rospy.sleep(1)
    
        if not rospy.is_shutdown():
            msg = Bool(data=True)
            self.arm_pub.publish(msg)
            rospy.loginfo("Published arm message: true")

        
        
        rospy.set_param('wind_speed/level', 0) #2 5 10

        self.timer_windspeed = rospy.Timer(rospy.Duration(0.01), self.control_wind_speed)

        
        # 轨迹周期 - velocity
        self.T = 2.0
        # self.T = 10.0
        # self.T = 15.0
        self.timer_trajectory = rospy.Timer(rospy.Duration(0.01), self.update_desired_state)

        reset_position = np.array([np.random.uniform(-2, 2),
                                                np.random.uniform(-2, 2),
                                                np.random.uniform(0, 4)], dtype=np.float32)
        
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

        if self.desired_traj==None: return
        

        id = self.desired_id % len(self.desired_traj)
        # print("id: ",  id, "namespace: ", self.namespace)
        
        new_desired_position = np.array([self.origin_offset + self.desired_traj[id].pose.position.x, 
                                         self.desired_traj[id].pose.position.y, 
                                         self.desired_traj[id].pose.position.z], dtype=np.float32)

        self.desired_state[0:3] = new_desired_position
        self.desired_state[3:7] =  [self.desired_traj[id].pose.orientation.w, 
                                    self.desired_traj[id].pose.orientation.x, 
                                    self.desired_traj[id].pose.orientation.y, 
                                    self.desired_traj[id].pose.orientation.z]
        
        self.desired_state[7:10] = [self.desired_traj[id].velocity.linear.x, 
                                    self.desired_traj[id].velocity.linear.y, 
                                    self.desired_traj[id].velocity.linear.z]
 

        self.marker.pose.position.x = new_desired_position[0]
        self.marker.pose.position.y = new_desired_position[1]
        self.marker.pose.position.z = new_desired_position[2]

        self.desired_state_marker_pub.publish(self.marker)

        self.desired_id+=1

    def trajectory_callback(self, msg):
        while self.desired_traj==None:
            self.desired_traj = msg.points


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
        # self.update_desired_state()
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

        # # reset desired_state
        # new_desired_position = np.array([self.origin_offset + self.desired_traj[0].pose.position.x, 
        #                                  self.desired_traj[0].pose.position.y, 
        #                                  self.desired_traj[0].pose.position.z], dtype=np.float32)

        # self.desired_state[0:3] = new_desired_position
        # self.desired_state[3:7] =  [self.desired_traj[0].pose.orientation.w, 
        #                             self.desired_traj[0].pose.orientation.x, 
        #                             self.desired_traj[0].pose.orientation.y, 
        #                             self.desired_traj[0].pose.orientation.z]
        
        # self.desired_state[7:10] = [self.desired_traj[0].velocity.linear.x, 
        #                             self.desired_traj[0].velocity.linear.y, 
        #                             self.desired_traj[0].velocity.linear.z]

    


        reset_position = np.array([self.origin_offset + np.random.uniform(-1, 1),
                                   np.random.uniform(-1, 1),
                                   np.random.uniform(0.4, 1.8)], dtype=np.float32)
        
        # reset_position = np.array([self.origin_offset + 0,
        #                            0,
        #                            1], dtype=np.float32)


        self._reset_drone_pose(reset_position)
        
        self.prev_position = self.current_state[0:3]
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
            np.array(self.prev_action_)
        ))

        norm_obs = self._normalize_obs(obs)
        self.desired_id = 0
        
        return norm_obs, {}
    
    def _pub_reset(self, reset_position):
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
    
    def _reset_drone_pose(self, reset_position):

        init_orientation = np.random.uniform(-0.2, 0.2, size=4)  

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

    def _compute_reward(self, current_state, curr_action):
        # 1. 当前相对位置误差：当前位置与目标位置之差
        curr_pos = current_state[0:3]

        prev_dist = np.linalg.norm(self.desired_state[0:3] - self.prev_position)
        curr_dist = np.linalg.norm(self.desired_state[0:3] - curr_pos)
        r_target = prev_dist - curr_dist

        # r_target = np.tanh(r_progress_dis)
        # print("r_target: ", r_target)

        curr_rel_pos = current_state[0:3] - self.desired_state[0:3] 
        curr_rel_dis = np.clip(np.linalg.norm(current_state[0:3] - self.desired_state[0:3])/self.max_position_error,0,1)

        # r_position = -curr_rel_dis/self.max_position_error # np.exp(-1.7 * curr_rel_dis)
        r_position = np.exp(-1.0 * curr_rel_dis)


        curr_vel_error = current_state[7:10] - self.desired_state[7:10]
        curr_vel_error_norm = np.linalg.norm(curr_vel_error)
        r_velocity = np.exp(-0.3 * curr_vel_error_norm)


        # 3. 动作平滑奖励：当前动作与上一次动作的平方差之和
        r_smooth = np.sum((curr_action - np.array(self.prev_action_))**2)/5
        # print("np.sum((curr_action - np.array(self.prev_action_))**2): ",np.sum((curr_action - np.array(self.prev_action_))**2))
        # print("curr_action: ", curr_action)

        # print("self.prev_action_: ", self.prev_action_)

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

         # 3. 惩罚无人机的姿态
        epsilon=1e-6
        q_des = self.safe_normalize(self.desired_state[3:7], epsilon)
        q_curr = self.safe_normalize(current_state[3:7], epsilon)

        # 计算内积并裁剪到[-1, 1]范围内
        dot_product = np.dot(q_des, q_curr)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        # 计算旋转角度差
        angle_diff = 2 * np.arccos(abs(dot_product))

        r_step = 1
        
        total_reward = (
            # r_step +
            self.s_pos * r_position  +
            self.s_target  * r_target +
            self.s_smooth * r_smooth +
            # self.s_angle_diff * angle_diff+
            # self.s_smooth * r_smooth +
            # self.s_angle_diff * angle_diff +

            # self.s_vel * r_velocity +
            # self.s_yaw * r_yaw +
            # self.s_angular * r_angular +
            self.s_crash * r_crash 
            # + r_energy
        )


        rot_mat = self.quaternion_to_rot_matrix(current_state[3:7])
        base_lin_vel = current_state[7:10]
        base_ang_vel = current_state[10:13]
        obs = np.concatenate((curr_rel_pos, rot_mat, base_lin_vel, base_ang_vel, np.array(self.prev_action_)))
    

        obs = np.concatenate((
            curr_rel_pos,
            rot_mat,
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
        if pos_error > self.max_position_error or self.desired_state[2]<0.1:
            return True
        
        qw = curr_state[3]
        qx = curr_state[4]
        qy = curr_state[5]
        qz = curr_state[6]
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        if np.abs(roll) > 2.2:
            return True

        if np.linalg.norm(curr_state[7:10]) > self.max_velocity_error-0.5: 
            return True
        
        return False

    def render(self, mode='human'):
        pass  # Gazebo 自带可视化

    def close(self):
        rospy.signal_shutdown("Environment closed")
